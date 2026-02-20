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
library(future)
library(furrr)
library(parallel)
library(progressr)
handlers(global = TRUE)
handlers("progress")  # or "cli" if you have the cli package
# ── Auto-configure parallelism ─────────────────────────────────────────────────
total_cores      <- parallel::detectCores(logical = FALSE)  # physical cores only
threads_per_fit  <- 2L                                       # threads inside ma_continuous_fit

# Leave 1 core for OS/session, divide remaining by per-fit threads
n_workers <- max(1L, floor((total_cores - 1L) / threads_per_fit))

message(sprintf(
  "Detected %d physical cores → %d parallel workers × %d threads/fit = %d cores used",
  total_cores, n_workers, threads_per_fit, n_workers * threads_per_fit
))

plan(multisession, workers = n_workers)
# 1. Arguments
args <- commandArgs(trailingOnly = TRUE)

input_path <- args[1] #"../../03_postprocessing/outputs/level4b_bsc_filtered.parquet" #distances_bsc.parquet
output_path <- args[2] #"../outputs/pods/test.parquet"
num.sds <- args[3] #2
ctrl <- args[4] #"s0617825"
#cp_plot_path <- "../outputs/fig/bayese_inference_bsc.pdf"
dmso_mode <- args[5] #"closest"  # "closest" or "plate" # Improve ? : "rowcol" → same row or column only | "k_closest" → k nearest DMSOs | "radius" → within N wells
category <- unlist(strsplit(args[6], ",")) #c("DNA", "ER", "AGP", "RNA", "Mito", "Brightfield")
log10.dose <- args[7] #TRUE
ncpus <- args[8] #4
compounds <- args[9] #"all"
permutation <- args[10] #FALSE

# 2. Calculate BMDs for each features
dat <- read_parquet(input_path) %>% as.data.frame()

feat_cols <- colnames(dat)[grepl("Feature", colnames(dat))] # The computed distances

dat <- dat %>%
  bind_cols(map_dfr(.$Metadata_well, well_to_coord))

if (compounds == "all") {
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

# ── Parallel loop over compounds ───────────────────────────────────────────────
results <- with_progress({
  res_list <- vector("list", length(compounds))
  names(res_list) <- compounds
  p <- progressor(steps = length(compounds))
  
  res_list <- future_map(compounds, function(compound) {
    
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
    
    # Compute FastBMD
    fit_obj <- PerformCurveFitting(
      data   = dat_mat[, , drop = FALSE],
      dose   = doses,
      ncpus  = ncpus,
      models = c("Hill", "Power", "Lin", "Poly2")
    )
    fit_obj <- FilterDRFit(fit_obj, lof.pval = 0, filt.var = "AIC.model")
    fit_obj <- PerformBMDCalc(fit_obj, ncpus = ncpus, num.sds = num.sds, bmr.method = "ctrl.quantile")
    
    # Filter active and Count valid BMDs
    fit_obj_passed_bmd_res  <- fit_obj$bmd_res[fit_obj$bmd_res[, "all.pass"] == TRUE, ]
    n_active_features <- nrow(fit_obj_passed_bmd_res)
    
    # Helper to mean/sd finite BMD values (excluding sentinel 9999)
    bmd_mean <- function(x) mean(x[x != 9999], na.rm = TRUE)
    bmd_sd   <- function(x) sd(x[x   != 9999], na.rm = TRUE)
    
    data.frame(
      compound  = compound,
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
      all.pass  = any(fit_obj_passed_bmd_res$all.pass,   na.rm = TRUE)
    )
  }, .options = furrr_options(seed = TRUE))
  bind_rows(res_list)
})
boxplot(fit_obj$bmd_res[fit_obj$bmd_res[,"all.pass"] == TRUE,"bmd"])

dim(results)
boxplot(results$bmd[results$n.active > 20])
length(results$bmd[results$n.active > 20])

if (permutation) {
  # at the end of the script
  writeLines(as.character(results$n.active), output_path)
}