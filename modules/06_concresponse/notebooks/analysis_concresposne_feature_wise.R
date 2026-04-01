#### 1. plot_cp_curve feature wise ####
require(dplyr)
require(arrow)
require(ggplot2)
require(ggforce)
library(tidyr)
library(purrr)

if (!requireNamespace("fastbmdR", quietly = TRUE)) {
  
  # Check if remotes is installed, and install it if not
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes")
  }
  
  # Install fastbmdR from GitHub
  remotes::install_github("jessica-ewald/fastbmdR@v0.0.0.9000")
}
library(fastbmdR)

Exp2 <- function(b,c,d,e,f,dose){
  return(e * exp(b * dose))
}

Exp3 <- function(b,c,d,e,f,dose){
  return(e * (exp(sign(b) * (abs(b) * dose)^d)))
}

Exp4 <- function(b,c,d,e,f,dose){
  return(e * (c - (c - 1) * exp((-1) * b * dose)))
}

Exp5 <- function(b,c,d,e,f,dose){
  return(e * (c - (c - 1) * exp((-1) * (b * dose)^d)))
}

Hill <- function(b,c,d,e,f,dose){
  return(c + (d - c)/(1 + (dose/e)^b))
}

Pow <- function(b,c,d,e,f,dose){
  return(e + b * (dose^c))
}

Poly2 <- function(b,c,d,e,f,dose){
  return(b + c * (dose) + d * (dose^2))
}

Lin <- function(b,c,d,e,f,dose){
  return(d + b * (dose))
}

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

#### Arguments ####
args <- commandArgs(trailingOnly = TRUE)
#Rscript /Users/ollitrault/Documents/projects/2025_Ollitrault_BSC/modules/06_concresponse/scripts/plot_cp_curve.R         pods_bsc_cp_filter.parquet         '/Users/ollitrault/Documents/projects/2025_Ollitrault_BSC/work/94/89e6911139e0aa306a2a50d3df1a07/pods_bsc_Metadata_ldh_normalized.parquet,/Users/ollitrault/Documents/projects/2025_Ollitrault_BSC/work/bf/40206d302cb7dd4d1121db24994637/pods_bsc_Metadata_mtt_normalized.parquet'         distances_bsc.parquet         "plots_pods_cp_bsc.pdf"         'Metadata_ldh_normalized,Metadata_mtt_normalized'         s0617825         plate

input_path <- "../../03_postprocessing/outputs/level4b_bsc_filtered.parquet"
cp_pod_path <- "../outputs/pods/feature_wise/pod_bsc_s0654880_feature_wise_original_conc.parquet" # args[1]
compound <- "s0654880"
cp_plot_path <- "../results/plots_pods_cp_bsc.pdf" #args[4]
control <- "s0617825" #args[6]
dmso_mode <- "closest" # args[7]  # "plate" or "closest"

#### Load data ####
dat <- read_parquet(input_path) %>% as.data.frame()
cp_pods <- read_parquet(cp_pod_path) %>% as.data.frame()

# Get pod as the 5th percentile of the cp_POD_um column using linear interpolation all all.pass are already true
pod_value <- quantile(cp_pods$cp_POD, 0.05, type = 7)

# filter features
feat_cols <- cp_pods$gene.id #sample(cp_pods$gene.id, 10)

dat <- dat[, c(feat_cols, colnames(dat)[!grepl("Feature", colnames(dat))])] %>% 
  select(where(~ !all(is.na(.))))

cp_pods <- cp_pods %>% filter(gene.id %in% feat_cols)

dat_dmso <- dat %>%
  filter(Metadata_compound == control) %>%
  bind_cols(map_dfr(.$Metadata_well, well_to_coord))

highest_dose <- max(unique(dat$Metadata_log10conc))
highest_dose <- round(highest_dose + (0.025 * highest_dose), 1)
dose <- seq(0, highest_dose, by = 0.1)

dat <- dat %>%
  bind_cols(map_dfr(.$Metadata_well, well_to_coord))

dat_ctrl <- dat %>%
  filter(Metadata_compound == control)
dat_comp <- dat %>%
  filter(Metadata_compound != control)

# ── Pre-compute plate-split once if needed (avoids repeated filtering) ─────────
if (dmso_mode == "plate") {
  dat_ctrl_by_plate <- split(dat_ctrl, dat_ctrl$Metadata_plate)
} else if (dmso_mode != "closest") {
  stop("dmso_mode must be 'closest' or 'plate'")
}

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
# Calcul du minimum par feature (colonne)
col_mins <- apply(dat_fit[, feat_cols], 2, min)

# Valeur de décalage (shift) pour rendre tout positif
shift_vals <- ifelse(col_mins <= 0, -col_mins + 1e-3, 0)

# Appliquer le shift uniquement aux colonnes features
dat_fit[, feat_cols] <- sweep(dat_fit[, feat_cols], 2, shift_vals, "+")

plot_results <- data.frame(Metadata_log10conc = dose)

for (feat in feat_cols) {
  temp_pod <- cp_pods[cp_pods$gene.id == feat, ] %>% distinct()
  model <- temp_pod$mod.name
  b <- temp_pod$b
  c <- temp_pod$c
  d <- temp_pod$d
  e <- temp_pod$e
  f <- temp_pod$f
  
  f_dose <- switch(model,
                   "Exp2" = Exp2(b, c, d, e, f, dose),
                   "Exp3" = Exp3(b, c, d, e, f, dose),
                   "Exp4" = Exp4(b, c, d, e, f, dose),
                   "Exp5" = Exp5(b, c, d, e, f, dose),
                   "Hill" = Hill(b, c, d, e, f, dose),
                   "Power" = Pow(b, c, d, e, f, dose),
                   "Poly2" = Poly2(b, c, d, e, f, dose),
                   "Lin" = Lin(b, c, d, e, f, dose))
  
  plot_results[, feat] <- f_dose  # fix: use feat instead of compound
}

plot_results <- reshape2::melt(plot_results, id.vars = "Metadata_log10conc")
colnames(plot_results)[2:3] <- c("feat_id", "f_dose")

# Melt dat_fit to long format for points
dat_fit_long <- dat_fit[, c("Metadata_log10conc", feat_cols)] %>%
  reshape2::melt(id.vars = "Metadata_log10conc", variable.name = "feat_id", value.name = "f_dose")

ggplot() +
  geom_line(data = plot_results, aes(x = Metadata_log10conc, y = f_dose, color = feat_id)) +
  geom_point(data = dat_fit_long, aes(x = Metadata_log10conc, y = f_dose, color = feat_id), alpha = 0.5) +
  geom_vline(xintercept = pod_value, color = "red", linetype = "solid") +
  theme_bw() +
  labs(x = "log10(concentration)", y = "Feature value", color = "Feature")

ggplot() +
  geom_line(data = plot_results, aes(x = Metadata_log10conc, y = f_dose, color = feat_id), alpha = 0.3) +
  geom_vline(xintercept = pod_value, color = "red", linetype = "solid") +
  theme_bw() +
  theme(legend.position = "none") +
  labs(x = "log10(concentration)", y = "Feature value")


