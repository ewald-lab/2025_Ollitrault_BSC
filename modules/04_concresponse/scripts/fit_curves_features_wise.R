#################
if (!requireNamespace("fastbmdR", quietly = TRUE)) {
  
  # Check if remotes is installed, and install it if not
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes")
  }
  
  # Install fastbmdR from GitHub
  remotes::install_github("jessica-ewald/fastbmdR@v0.0.0.9000")
}
library(fastbmdR)
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
# 0. Parrallelization
library(parallel)
# ── Auto-configure ncpus ─────────────────────────────────────────────────
ncpus <- parallelly::availableCores()
message(sprintf("Using %d cores for curve fitting.", ncpus))
# 1. Arguments
args <- commandArgs(trailingOnly = TRUE)

optional_arg <- function(x) {
  if (length(x) == 0 || is.na(x) || x == "" || x == "NA") {
    return(NA_character_)
  }
  x
}

read_compound_from_list <- function(compound_list_path, compound_line) {
  if (is.na(compound_line) || compound_line < 1) {
    stop("compound_line must be a 1-based integer when compound_list_path is provided.")
  }
  if (!file.exists(compound_list_path)) {
    stop(sprintf("Compound list file does not exist: %s", compound_list_path))
  }

  compound_dt <- read.csv(
    compound_list_path,
    header = FALSE,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    colClasses = "character",
    comment.char = "",
    na.strings = character(),
    blank.lines.skip = FALSE,
    fill = TRUE
  )

  if (compound_line > nrow(compound_dt)) {
    stop(sprintf(
      "Compound line %d is outside compound list with %d rows.",
      compound_line,
      nrow(compound_dt)
    ))
  }

  compound <- compound_dt[[1]][compound_line]
  if (is.na(compound) || compound == "") {
    stop(sprintf("Compound line %d is empty in %s.", compound_line, compound_list_path))
  }

  compound
}

input_path <- args[1] #"../../03_postprocessing/outputs/level4b_bsc_filtered.parquet" #distances_bsc.parquet
output_path <- args[2] #"../outputs/pods/test.parquet"
num.sds <- args[3] #2
ctrl <- args[4] #"s0617825"
#cp_plot_path <- "../outputs/fig/bayese_inference_bsc.pdf"
dmso_mode <- args[5] #"closest"  # "closest" or "plate" # Improve ? : "rowcol" → same row or column only | "k_closest" → k nearest DMSOs | "radius" → within N wells
category <- unlist(strsplit(args[6], ",")) #c("DNA", "ER", "AGP", "RNA", "Mito", "Brightfield")
log10.dose <- as.logical(args[7]) #TRUE
compound_selector <- args[8] #"all" or a 1-based compound-list line number
permutation <- as.logical(args[9]) #FALSE
thresholds_n_active <- suppressWarnings(as.numeric(args[10]))
output_path_all <- args[11] # "../outputs/pods/feature_wise/test_cmpd.parquet"
feature_selection_score_file <- args[12]
compound_list_path <- if (length(args) >= 13) optional_arg(args[13]) else NA_character_

if (!is.na(compound_list_path)) {
  compound_line <- suppressWarnings(as.integer(compound_selector))
  compounds <- read_compound_from_list(compound_list_path, compound_line)
  message(sprintf(
    "Resolved compound list line %d to '%s'.",
    compound_line,
    compounds
  ))
} else {
  compounds <- compound_selector
}

# 2. Calculate BMDs for each features
dat <- read_parquet(input_path) %>% as.data.frame()

if (feature_selection_score_file != "NA") {
  feature_selection_score_dt = read.table(feature_selection_score_file, sep = ",", header = T)
  feat_cols <- feature_selection_score_dt$feature
  if (length(feat_cols) == 0) {
    message("feat_cols is empty — writing empty parquet and exiting.")
    
    empty_df <- data.frame(
      Metadata_compound = compounds[1],
      bmd       = NA_real_,
      bmdl      = NA_real_,
      bmdu      = NA_real_,
      bmd_sd    = NA_real_,
      bmdl_sd   = NA_real_,
      bmdu_sd   = NA_real_,
      n.active  = 0L,
      conv.pass = NA,
      hd.pass   = NA,
      CI.pass   = NA,
      ld.pass   = NA,
      all.pass  = FALSE
    )
    arrow::write_parquet(empty_df, output_path)
    if (!is.null(output_path_all)) arrow::write_parquet(empty_df, output_path_all)
    
    quit(save = "no", status = 0)
  }
} else {
  feat_cols <- colnames(dat)[grepl("Feature", colnames(dat))] # The computed distances
}

dat <- dat %>%
  bind_cols(map_dfr(.$Metadata_well, well_to_coord))

if (identical(compounds, "all")) {
  compounds <- unique(dat$Metadata_compound)
  compounds <- compounds[!grepl(ctrl, compounds)]
}

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
res_list <- vector("list", length(compounds))

for (i in seq_along(compounds)) {
  compound <- compounds[i]
    
  dat_comp <- dat %>% filter(Metadata_compound == compound)
  
  # Control selection
  ctrl_fit <- if (dmso_mode == "closest") {
    get_closest_dmso(comp_fit_coords = dat_comp, dat_ctrl = dat_ctrl)
  } else {
    bind_rows(dat_ctrl_by_plate[
      intersect(names(dat_ctrl_by_plate), unique(dat_comp$Metadata_plate))
    ])
  }
  
  sprintf(
    "Fitting POD for %s with %d compound and %d DMSO points.",
    compound, nrow(dat_comp), nrow(ctrl_fit)
  )
  
  # Prepare data once per compound
  dat_fit <- bind_rows(ctrl_fit, dat_comp) %>% arrange(Metadata_log10conc)
  if (permutation) {
    dat_perm <- dat_fit
    dat_perm[, feat_cols] <- lapply(dat_fit[, feat_cols], sample)
    dat_mat <- t(dat_perm[, feat_cols])
    rownames(dat_mat) <- feat_cols
  } else {
    dat_mat <- t(dat_fit[, feat_cols])
    rownames(dat_mat) <- feat_cols
  }
  doses   <- dat_fit$Metadata_log10conc
  # Shift data per rows to have onyly positiv values
  row_mins <- apply(dat_mat, 1, min)
  shift_vals <- ifelse(row_mins <= 0, -row_mins + 1e-3, 0)
  dat_mat <- dat_mat + shift_vals
  
  # Compute FastBMD
  fit_obj <- PerformCurveFitting(
    data   = dat_mat[, , drop = FALSE],
    dose   = doses,
    ncpus  = ncpus,
    models = c("Exp2", "Exp3", "Exp4", "Exp5", "Poly2", "Lin", "Power", "Hill")
  )
  fit_obj <- FilterDRFit(fit_obj, lof.pval = 0, filt.var = "AIC.model")
  fit_obj <- PerformBMDCalc(fit_obj, ncpus = ncpus, num.sds = num.sds, bmr.method = "ctrl.quantile")
  
  # Filter active and Count valid BMDs
  fit_obj$bmd_res$Metadata_compound = compound
  fit_obj_passed_bmd_res  <- fit_obj$bmd_res[fit_obj$bmd_res[, "all.pass"] == TRUE, ]
  
  if (permutation) {
    n_active_features <- nrow(fit_obj_passed_bmd_res)
  } else {
    n_active_features <- nrow(fit_obj_passed_bmd_res) > thresholds_n_active # TRUE or FALSE as a check
  }
  
  # write all features results
  if (nrow(fit_obj_passed_bmd_res) == 0) {
    fit_obj_passed_bmd_res <- data.frame(
      gene.id           = NA_character_,
      mod.name          = NA_character_,
      b                 = NA_real_,
      c                 = NA_real_,
      d                 = NA_real_,
      e                 = NA_real_,
      f                 = NA_real_,
      SDres             = NA_real_,
      AIC.model         = NA_real_,
      lof.p             = NA_real_,
      ctrl.mod          = NA_real_,
      ctrl.mean         = NA_real_,
      adv.incr          = NA,
      bmd               = NA_real_,
      bmdl              = NA_real_,
      bmdu              = NA_real_,
      bmr               = NA_real_,
      SDctrl            = NA_real_,
      conv.pass         = NA,
      hd.pass           = NA,
      CI.pass           = NA,
      ld.pass           = NA,
      all.pass          = FALSE,
      Metadata_compound = compound
    )
  }
  if (!permutation) {
    arrow::write_parquet(fit_obj_passed_bmd_res, file.path(output_path_all))
  }
  
  # Helper to mean/sd finite BMD values (excluding sentinel 9999)
  bmd_mean <- function(x) mean(x[x != 9999], na.rm = TRUE)
  bmd_sd   <- function(x) sd(x[x   != 9999], na.rm = TRUE)
  
  # Check all.pass
  all.pass = (any(fit_obj_passed_bmd_res$conv.pass, na.rm = TRUE) & 
                any(fit_obj_passed_bmd_res$hd.pass,   na.rm = TRUE) & 
                any(fit_obj_passed_bmd_res$CI.pass,    na.rm = TRUE) &
                n_active_features)

  res_list[[i]] <- data.frame(
    Metadata_compound  = compound,
    bmd       = bmd_mean(fit_obj_passed_bmd_res$bmd),
    bmdl      = bmd_mean(fit_obj_passed_bmd_res$bmdl),
    bmdu      = bmd_mean(fit_obj_passed_bmd_res$bmdu),
    bmd_sd    = bmd_sd(fit_obj_passed_bmd_res$bmd),
    bmdl_sd   = bmd_sd(fit_obj_passed_bmd_res$bmdl),
    bmdu_sd   = bmd_sd(fit_obj_passed_bmd_res$bmdu),
    n.active  = n_active_features,
    conv.pass = any(fit_obj_passed_bmd_res$conv.pass, na.rm = TRUE),
    hd.pass   = any(fit_obj_passed_bmd_res$hd.pass,   na.rm = TRUE),
    CI.pass   = any(fit_obj_passed_bmd_res$CI.pass,    na.rm = TRUE),
    ld.pass   = any(fit_obj_passed_bmd_res$ld.pass,    na.rm = TRUE),
    all.pass  = all.pass
  )
}
results <- bind_rows(res_list)
if (permutation) {
  # at the end of the script
  writeLines(as.character(results$n.active), output_path)
} else {
  # write mean results
  arrow::write_parquet(results, output_path)
}
