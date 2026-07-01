require(dplyr)
require(arrow)
require(purrr)

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

  compounds <- readLines(compound_list_path, warn = FALSE)

  if (compound_line > length(compounds)) {
    stop(sprintf(
      "Compound line %d is outside compound list with %d rows.",
      compound_line,
      length(compounds)
    ))
  }

  compound <- compounds[[compound_line]]
  if (is.na(compound) || !nzchar(trimws(compound))) {
    stop(sprintf("Compound line %d is empty in %s.", compound_line, compound_list_path))
  }

  compound
}

empty_pod_result <- function(compound = character()) {
  template <- data.frame(
    Metadata_compound = NA_character_,
    gene.id = NA_character_,
    mod.name = NA_character_,
    b = NA_real_,
    c = NA_real_,
    d = NA_real_,
    e = NA_real_,
    f = NA_real_,
    SDres = NA_real_,
    AIC.model = NA_real_,
    lof.p = NA_real_,
    ctrl.mod = NA_real_,
    ctrl.mean = NA_real_,
    adv.incr = NA,
    bmd = NA_real_,
    bmdl = NA_real_,
    bmdu = NA_real_,
    bmr = NA_real_,
    SDctrl = NA_real_,
    conv.pass = NA,
    hd.pass = NA,
    CI.pass = NA,
    ld.pass = NA,
    all.pass = FALSE,
    stringsAsFactors = FALSE
  )

  if (length(compound) == 0) {
    return(template[0, , drop = FALSE])
  }

  template$Metadata_compound <- compound
  template
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

if (length(args) < 6) {
  stop(paste(
    "Usage: fit_curves_meta.R <dat_path> <output_path> <num_sds>",
    "<meta_nm> <control> <dmso_mode> [compound_line] [compound_list_path]"
  ))
}

dat_path <- args[1]
output_path <- args[2]
num_sds <- as.numeric(args[3])
meta_nm <- args[4]
control <- args[5]
dmso_mode <- args[6]
compound_selector <- if (length(args) >= 7) optional_arg(args[7]) else "all"
compound_list_path <- if (length(args) >= 8) optional_arg(args[8]) else NA_character_

message(sprintf("Control: %s", control))

######## 2. Calculate BMDs from cell counts
dat <- read_parquet(dat_path) %>% as.data.frame()

required_cols <- c(
  "Metadata_compound",
  "Metadata_plate",
  "Metadata_well",
  "Metadata_concentration",
  "Metadata_log10conc",
  meta_nm
)
missing_cols <- setdiff(required_cols, colnames(dat))
if (length(missing_cols) > 0) {
  stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")))
}

if (!is.na(compound_list_path)) {
  compound_line <- suppressWarnings(as.integer(compound_selector))
  compounds <- read_compound_from_list(compound_list_path, compound_line)
  message(sprintf(
    "Resolved compound list line %d to '%s'.",
    compound_line,
    compounds
  ))
} else if (is.na(compound_selector) || identical(compound_selector, "all")) {
  compounds <- unique(dat$Metadata_compound)
} else {
  compounds <- compound_selector
}

compounds <- compounds[compounds != control]
message(sprintf("Fitting POD for %d compound(s).", length(compounds)))

if (length(compounds) == 0) {
  write_parquet(empty_pod_result(), output_path)
  quit(save = "no", status = 0)
}

dat_coords <- dat %>%
  bind_cols(purrr::map_dfr(.$Metadata_well, well_to_coord))

cc_dmso <- dat_coords %>%
  filter(Metadata_compound == control)

if (nrow(cc_dmso) == 0) {
  stop(sprintf("No control rows found for '%s'.", control))
}

cc_comp <- dat_coords %>%
  filter(Metadata_compound != control)

message(sprintf("DMSO points: %d, Compound points: %d", nrow(cc_dmso), nrow(cc_comp)))

bmd_res <- data.frame()
for (compound in compounds) {
  cc_comp_fit <- dat_coords %>%
    filter(Metadata_compound == compound)

  if (nrow(cc_comp_fit) == 0) {
    warning(sprintf("No rows found for compound '%s'; writing an empty POD row.", compound))
    bmd_res <- dplyr::bind_rows(bmd_res, empty_pod_result(compound))
    next
  }

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

  if (nrow(cc_dmso_fit) == 0) {
    warning(sprintf("No DMSO rows found for compound '%s'; writing an empty POD row.", compound))
    bmd_res <- dplyr::bind_rows(bmd_res, empty_pod_result(compound))
    next
  }

  cc_fit <- rbind(cc_dmso_fit, cc_comp_fit)
  cc_fit <- cc_fit[order(cc_fit$Metadata_concentration), ]
  cc <- as.data.frame(matrix(cc_fit[[meta_nm]], nrow = 1))
  rownames(cc) <- "cc"

  message(sprintf(
    "Fitting POD for %s with %d compound and %d DMSO points.",
    compound,
    nrow(cc_comp_fit),
    nrow(cc_dmso_fit)
  ))

  dose <- c(cc_fit$Metadata_log10conc)
  row_pod_res <- tryCatch(
    {
      pod <- scoresPOD(
        cc["cc", ],
        dose,
        log10.dose = TRUE,
        num.sds = num_sds
      )

      if (!is.null(pod) && !is.null(pod$bmd_res) && nrow(pod$bmd_res) > 0) {
        dplyr::mutate(
          pod$bmd_res,
          Metadata_compound = compound
        )
      } else {
        empty_pod_result(compound)
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

      empty_pod_result(compound)
    }
  )
  bmd_res <- dplyr::bind_rows(bmd_res, row_pod_res)
}

dat_compounds <- dat[, c("Metadata_compound"), drop = FALSE]
dat_compounds <- distinct(dat_compounds)
bmd_res <- as.data.frame(bmd_res)
bmd_res <- merge(bmd_res, dat_compounds, by = "Metadata_compound")

write_parquet(bmd_res, output_path)
