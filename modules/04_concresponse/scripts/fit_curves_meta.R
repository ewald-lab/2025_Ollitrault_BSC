require(dplyr)
require(arrow)

######## Function
well_to_coord <- function(well) {
  row <- substr(well, 1, 1)
  col <- as.integer(sub("^[A-P]", "", well))
  row_num <- match(row, LETTERS[1:16])
  tibble(row = row_num, col = col)
}

get_closest_dmso <- function(comp_fit_coords, dmso_coords) {
  
  purrr::pmap_dfr(
    comp_fit_coords,
    function(...) {
      this <- tibble(...)
      
      dmso_same_plate <- dmso_coords %>%
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
    distinct()
}

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


######## 1. Parse arguments and define parameters
args <- commandArgs(trailingOnly = TRUE)

dat_path <- args[1]
output_path <- args[2]
num_sds <- args[3]
meta_nm <- args[4]
control <- args[5]
dmso_mode <- args[6]

print(paste0("Control: ", control))
######## 2. Calculate BMDs from cell counts
dat <- read_parquet(dat_path) %>% as.data.frame()

compounds <- unique(dat$Metadata_compound)
compounds <- compounds[compounds != control]
print(paste0("Fitting POD for ", length(compounds), " compounds."))
cc_dmso <- dat %>%
  filter(Metadata_compound == control) %>%
  bind_cols(purrr::map_dfr(.$Metadata_well, well_to_coord))
cc_comp <- dat[dat$Metadata_compound != control, ]
print(paste0("DMSO points: ", nrow(cc_dmso), ", Compound points: ", nrow(cc_comp)))
bmd_res <- data.frame()
for (compound in compounds){
  cc_comp_fit <- dat %>%
      filter(Metadata_compound == compound) %>%
      bind_cols(purrr::map_dfr(.$Metadata_well, well_to_coord))
  if (dmso_mode == "closest") {
    cc_dmso_fit <- get_closest_dmso(
      comp_fit_coords = cc_comp_fit,
      dmso_coords = cc_dmso
    ) %>%
    dplyr::select(-any_of(c("dist")))
  } else if (dmso_mode == "plate") {
    cc_dmso_fit <- cc_dmso %>%
      filter(Metadata_plate %in% unique(cc_comp_fit$Metadata_plate))
  } else {
    stop("dmso_mode must be 'closest' or 'plate'")
  }
  cc_fit <- rbind(cc_dmso_fit, cc_comp_fit)
  cc_fit <- cc_fit[order(cc_fit$Metadata_concentration), ]
  cc <- as.data.frame(matrix(c(cc_fit[, meta_nm]), nrow = 1))
  rownames(cc) <- "cc"
  print(
    paste0(
      "Fitting POD for ", compound,
      " with ", nrow(cc_comp_fit),
      " compound and ", nrow(cc_dmso_fit),
      " DMSO points."
    )
  )
  dose <- c(cc_fit$Metadata_log10conc)
  row_pod_res <- tryCatch(
  {
    pod <- scoresPOD(
      cc["cc", ],
      dose,
      log10.dose = TRUE,
      num.sds = num_sds
    )
    
    if (!is.null(pod)) {
      dplyr::mutate(
        pod$bmd_res,
        Metadata_compound = compound
      )
    } else {
      data.frame(
        Metadata_compound = compound,
        stringsAsFactors = FALSE
      )
    }
  },
  error = function(e) {
    message(
      sprintf(
        "scoresPOD failed for compound '%s': %s",
        compound,
        conditionMessage(e)
      )
    )
    
    data.frame(
      Metadata_compound = compound,
      stringsAsFactors = FALSE
    )
  }
  )
  # Append safely
  bmd_res <- dplyr::bind_rows(bmd_res, row_pod_res)
}

dat = dat[, c("Metadata_compound"), drop=FALSE] #, "Metadata_OASIS_ID")]
dat = distinct(dat)
bmd_res = as.data.frame(bmd_res)
bmd_res = merge(bmd_res, dat, by="Metadata_compound")

write_parquet(bmd_res, output_path)