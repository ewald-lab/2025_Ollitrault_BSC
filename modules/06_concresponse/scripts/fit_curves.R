require(dplyr)
require(arrow)

######## 0. Make sure fastbmdR is installed
if (!requireNamespace("fastbmdR", quietly = TRUE)) {
  
  # Check if remotes is installed, and install it if not
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes")
  }
  
  # Install fastbmdR from GitHub
  remotes::install_github("jessica-ewald/fastbmdR@v0.0.0.9000")
}
library(fastbmdR)

######## Function
library(purrr)
library(tidyr)

well_to_coord <- function(well) {
  row <- substr(well, 1, 1)
  col <- as.integer(sub("^[A-P]", "", well))
  
  row_num <- match(row, LETTERS[1:16])
  
  tibble(row = row_num, col = col)
}
get_closest_dmso <- function(comp_fit_coords, dat_dmso) {
  
  pmap_dfr(
    comp_fit_coords,
    function(...) {
      this <- tibble(...)
      
      dmso_same_plate <- dat_dmso %>%
        filter(Metadata_plate == this$Metadata_plate)
      
      if (nrow(dmso_same_plate) == 0) {
        return(tibble())
      }
      
      dmso_same_plate <- dmso_same_plate %>%
        mutate(
          dist = sqrt(
            (row - this$row)^2 +
              (col - this$col)^2
          )
        )
      
      min_dist <- min(dmso_same_plate$dist)
      
      dmso_same_plate %>%
        filter(dist == min_dist)
    }
  ) %>%
    distinct()  # avoid duplicates across wells
}

######## 1. Parse arguments and define parameters
args <- commandArgs(trailingOnly = TRUE)

input_path <- args[1]
output_path <- args[2]
num_sds <- args[3]
ctrl <- args[4]
dmso_mode <- args[5]  # "closest" or "plate" # Improve ? : "rowcol" → same row or column only | "k_closest" → k nearest DMSOs | "radius" → within N wells

######## 2. Calculate BMDs from global mahalanobis distances
dat <- read_parquet(input_path) %>% as.data.frame()

feat_cols <- colnames(dat)[grepl("Distance", colnames(dat))] # The computed distances

compounds <- unique(dat$Metadata_compound)
compounds <- compounds[!grepl(ctrl, compounds)]

dat_dmso <- dat %>%
  filter(Metadata_compound == ctrl) %>%
  bind_cols(map_dfr(.$Metadata_well, well_to_coord))
dat_comp <- dat[dat$Metadata_compound != ctrl, ]

bmd_res <- data.frame()

for (compound in compounds) {
  comp_fit <- dat %>%
    filter(Metadata_compound == compound) %>%
    bind_cols(map_dfr(.$Metadata_well, well_to_coord))
  # === NEW: nearest DMSO selection ===
  if (grepl("_ap", input_path)) {
    cmpd_ctrls <- grepl(compound, dat$Metadata_compound) &
      grepl(ctrl, dat$Metadata_compound)
    dmso_fit <- dat[cmpd_ctrls, ]
  } else {
    message("DMSO selection mode: ", dmso_mode)
    if (dmso_mode == "closest") {
      dmso_fit <- get_closest_dmso(
        comp_fit_coords = comp_fit,
        dat_dmso = dat_dmso
      )
    } else if (dmso_mode == "plate") {
      dmso_fit <- dat_dmso %>%
        filter(Metadata_plate %in% unique(comp_fit$Metadata_plate))
    } else {
      stop("dmso_mode must be 'closest' or 'plate'")
    }
  }
  print(
    paste0(
      "Fitting POD for ", compound,
      " with ", nrow(comp_fit),
      " compound and ", nrow(dmso_fit),
      " DMSO points."
    )
  )
  dat_fit <- bind_rows(dmso_fit, comp_fit) %>%
    arrange(Metadata_log10conc)
  
  dat_mat <- t(dat_fit[, feat_cols])
  rownames(dat_mat) <- feat_cols
  
  dose <- dat_fit$Metadata_log10conc
  
  pod <- scoresPOD(
    dat_mat,
    dose,
    log10.dose = TRUE,
    num.sds = num_sds,
    filt.var = "SDres"
  )
  if (!is.null(pod)) {
    #pod$Metadata_compound <- compound
    
    # Extract only the fields you want
    row_pod_res <- dplyr::mutate(
          pod$bmd_res,
          Metadata_compound = compound
        )
    # Append safely
    bmd_res <- dplyr::bind_rows(bmd_res, row_pod_res)
    #bmd_res <- rbind(bmd_res, pod)
  }
}
write_parquet(as.data.frame(bmd_res), output_path)