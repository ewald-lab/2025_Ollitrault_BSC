#### BSC ma_continuous_fit ####
library(purrr)
library(tidyr)
require(dplyr)
require(arrow)
# 0. Functions
well_to_coord <- function(well) {
  row <- substr(well, 1, 1)
  col <- as.integer(sub("^[A-P]", "", well))
  
  row_num <- match(row, LETTERS[1:16])
  
  tibble(row = row_num, col = col)
}
get_closest_dmso <- function(comp_fit_coords, dat_ctrl) {
  
  pmap_dfr(
    comp_fit_coords,
    function(...) {
      this <- tibble(...)
      
      ctrl_same_plate <- dat_ctrl %>%
        filter(Metadata_plate == this$Metadata_plate)
      
      if (nrow(ctrl_same_plate) == 0) {
        return(tibble())
      }
      
      ctrl_same_plate <- ctrl_same_plate %>%
        mutate(
          dist = sqrt(
            (row - this$row)^2 +
              (col - this$col)^2
          )
        )
      
      min_dist <- min(ctrl_same_plate$dist)
      
      ctrl_same_plate %>%
        filter(dist == min_dist)
    }
  ) %>%
    distinct()  # avoid duplicates across wells
}
# 1. Arguments
args <- commandArgs(trailingOnly = TRUE)

input_path <- args[1] #"../outputs/distances/distance_bsc_gmd.parquet" #distances_bsc.parquet
output_path <- args[2] #"../outputs/pods/bayese_inference_bsc_pods.parquet"
ctrl <- args[3] #"s0617825"
dmso_mode <- args[4] #"closest"  # "closest" or "plate" # Improve ? : "rowcol" → same row or column only | "k_closest" → k nearest DMSOs | "radius" → within N wells
cp_plot_path <- args[5] #"../outputs/fig/bayese_inference_bsc.pdf"
log10.dose <- as.logical(args[6]) #TRUE
num.sds <- as.numeric(args[7]) #2
# 2. Calculate BMDs from mahalanobis distances
dat <- read_parquet(input_path) %>% as.data.frame()

feat_cols <- colnames(dat)[!grepl("Metadata", colnames(dat))] # The computed distances

dat <- dat %>%
          bind_cols(map_dfr(.$Metadata_well, well_to_coord))
compounds <- unique(dat$Metadata_compound)
compounds <- compounds[!grepl(ctrl, compounds)]

dat_ctrl <- dat %>%
  filter(Metadata_compound == ctrl)
dat_comp <- dat %>%
  filter(Metadata_compound != ctrl)

# ── Pre-compute plate-split once if needed (avoids repeated filtering) ─────────
if (dmso_mode == "plate") {
  dat_ctrl_by_plate <- split(dat_ctrl, dat_ctrl$Metadata_plate)
} else if (dmso_mode != "closest") {
  stop("dmso_mode must be 'closest' or 'plate'")
}

# ── Loop over compounds ───────────────────────────────────────────────
results <- vector("list", length(compounds))
names(results) <- compounds

for (i in seq_along(compounds)) {
  
  compound <- compounds[i]
  cat("Now fitting:", compound, "\n")
  flush.console()
  fail_msgs <- character(0)  # pre-allocate empty
  # Compound selection 
  dat_comp <- dat %>% filter(Metadata_compound == compound)
  # Control selection
  ctrl_fit <- if (dmso_mode == "closest") {
    get_closest_dmso(comp_fit_coords = dat_comp, dat_ctrl = dat_ctrl)
  } else {
    bind_rows(dat_ctrl_by_plate[
      intersect(names(dat_ctrl_by_plate), unique(dat_comp$Metadata_plate))
    ])
  }
  p(sprintf(
    "Fitting POD for %s with %d compound and %d DMSO points.",
    compound, nrow(dat_comp), nrow(ctrl_fit)
  ))
  #message(sprintf(
  #  "Fitting POD for %s with %d compound and %d DMSO points.",
  #  compound, nrow(dat_comp), nrow(ctrl_fit)
  #))
  
  # Prepare data once per compound
  dat_fit  <- bind_rows(ctrl_fit, dat_comp) %>% arrange(Metadata_log10conc)
  dat_mat  <- t(dat_fit[, feat_cols])
  rownames(dat_mat) <- feat_cols
  doses    <- dat_fit$Metadata_log10conc
  
  # Pre-compute reused values once per compound
  max_dose <- max(doses)
  dose2    <- unique(sort(doses))[2]
  
  # Pre-allocate per-compound results
  res_list  <- vector("list", length(feat_cols))
  plot_list <- vector("list", length(feat_cols))
  names(res_list) <- names(plot_list) <- feat_cols
  
  for (feat_col in feat_cols) {
    fit <- ma_continuous_fit(
      doses, dat_mat[feat_col, ],
      fit_type = "laplace",
      threads  = 1,
      BMR_TYPE="sd", BMR=num.sds,
    )
    
    fit$gene.id   <- feat_col
    fit$conv.pass <- sum(is.na(fit$bmd)) == 0
    fit$hd.pass   <- fit$bmd["BMD"] < max_dose
    
    if (isTRUE(log10.dose)) {
      fit$CI.pass <- fit$bmd["BMDU"] - fit$bmd["BMDL"] < log10(40)
      fit$ld.pass <- fit$bmd["BMDL"] > (dose2 - log10(10))
    } else {
      fit$CI.pass <- fit$bmd["BMDU"] / fit$bmd["BMDL"] < 40
      fit$ld.pass <- fit$bmd["BMDL"] > (dose2 / 10)
    }
    
    fit$all.pass <- fit$conv.pass & fit$hd.pass & fit$CI.pass
    
    if (!fit$all.pass) {
      fail_msgs <- c(fail_msgs, sprintf("  [FAIL] compound: %s | feature: %s", compound, feat_col))
    }
    
    res_list[[feat_col]] <- data.frame(
      compound  = compound,
      gene.id   = feat_col,
      bmd       = fit$bmd["BMD"],
      bmdl      = fit$bmd["BMDL"],
      bmdu      = fit$bmd["BMDU"],
      conv.pass = fit$conv.pass,
      hd.pass   = fit$hd.pass,
      CI.pass   = fit$CI.pass,
      ld.pass   = fit$ld.pass,
      all.pass  = fit$all.pass
    )
    ggsave(
      filename = paste0(compound, "__", feat_col, ".png"),
      plot = plot(suppressWarnings(plot(fit) +
                                     ggtitle(paste(compound,feat_col)) +
                                     theme(plot.title = element_text(size = 40, face = "bold", hjust = 0.5))))
    )
  }
  results[[i]] <- list(
    bmd   = bind_rows(res_list),
    fails = fail_msgs
  )
}

# ── Collect results ────────────────────────────────────────────────────────────
bmd_res   <- bind_rows(lapply(results, `[[`, "bmd"))

# ── Print fail messages ────────────────────────────────────────────────────────
all_fails <- unlist(lapply(results, `[[`, "fails"))
if (length(all_fails) > 0) {
  message("\n── Failed fits ──────────────────────────────")
  message(paste(all_fails, collapse = "\n"))
  message(sprintf(
    "Total failed: %d / %d",
    length(all_fails), length(compounds) * length(feat_cols)
  ))
} else {
  message("All fits passed.")
}

# ── Write Parquet ──────────────────────────────────────────────────────────────
write_parquet(as.data.frame(bmd_res), output_path)
