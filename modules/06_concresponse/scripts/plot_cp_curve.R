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

#### 0. Define helper functions for plot fits
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


#### 1. Read in data and set common parameters
args <- commandArgs(trailingOnly = TRUE)

cp_pod_path <- args[1]
# meta_pod_path can be multiple files separated by comma
meta_pod_path <- unlist(strsplit(args[2], ","))
dist_path <- args[3]
cp_plot_path <- args[4]
# meta_nm can be multiple values separated by comma
meta_nm <- unlist(strsplit(args[5], ","))
control <- args[6]


cp_pods <- read_parquet(cp_pod_path) %>% as.data.frame()
meta_pods <- map2_df(
            meta_pod_path,
            meta_nm,
            ~ read_parquet(.x) %>%
              mutate(metadata_type = .y)
          )
# Merge metadata bmd values to cp bmd
cp_pods <- cp_pods %>%
  left_join(
    meta_pods %>%
      select(Metadata_compound, metadata_type, meta_bmd = bmd),
    by = "Metadata_compound"
  ) %>%
  pivot_wider(
    names_from  = metadata_type,
    values_from = meta_bmd
  ) %>%
  rename_with(
    ~ paste0(.x, "_POD"),
    .cols = all_of(meta_nm)
  )

dat <- read_parquet(dist_path) %>% as.data.frame()

highest_dose <- max(unique(dat$Metadata_log10conc))
highest_dose <- round(highest_dose + (0.025 * highest_dose), 1)
dose <- seq(0, highest_dose, by = 0.1)

n_cols <- 5
n_rows <- 5
n_per_page <- n_cols * n_rows
pdf_w <- 12
pdf_h <- 10


#### 2. Make cell painting plots
cp_compounds <- unique(cp_pods$Metadata_compound)
plot_results <- data.frame(Metadata_log10conc = dose)
fitted_feats <- data.frame()
for (compound in cp_compounds){
  temp_pod <- cp_pods[cp_pods$Metadata_compound == compound, ] %>% distinct()
  feat_type <- temp_pod$gene.id
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
                   "Pow" = Pow(b, c, d, e, f, dose),
                   "Poly2" = Poly2(b, c, d, e, f, dose),
                   "Lin" = Lin(b, c, d, e, f, dose))
  
  plot_results[, compound] <- f_dose
  
  # get observations
  dat_temp <- dat[dat$Metadata_compound == compound, ]
  if (grepl("_ap", cp_pod_path)) {
    cmpd_ctrls <- grepl(compound, dat$Metadata_compound) & grepl(control, dat$Metadata_compound)
    temp_dmso <- dat[cmpd_ctrls, ]
    temp_dmso <- temp_dmso[sample(1:720, 20), c(feat_type, "Metadata_log10conc")]
  } else {
    temp_plates <- unique(dat_temp$Metadata_plate)
    temp_dmso <- dat[(dat$Metadata_compound == control) & (dat$Metadata_plate %in% temp_plates), 
                     c(feat_type, "Metadata_log10conc")]
  }
  
  feats <- dat_temp[, c(feat_type, "Metadata_log10conc")]
  feats <- rbind(temp_dmso, feats)
  
  colnames(feats)[1] <- "Observations"
  feats$Metadata_compound <- compound
  feats$Observation_Type <- feat_type
  fitted_feats <- rbind(fitted_feats, feats)
}

plot_results <- reshape2::melt(plot_results, id.vars = c("Metadata_log10conc"))
colnames(plot_results)[2:3] <- c("Metadata_compound", "f_dose")

# merge dataframe together
fitted_feats$Metadata_log10conc <- round(fitted_feats$Metadata_log10conc, 1)
plot_results <- merge(plot_results, fitted_feats, by=c("Metadata_compound", "Metadata_log10conc"), all.x = TRUE, all.y = FALSE)

# Add better plotting label
cmpd_type <- plot_results[,c("Metadata_compound", "Observation_Type")] %>% na.omit() %>% distinct()
cmpd_type$Compound_Observation <- paste0(cmpd_type$Metadata_compound, " (", cmpd_type$Observation_Type, ")")
cmpd_type <- cmpd_type[,c("Metadata_compound", "Compound_Observation")]
plot_results <- merge(plot_results, cmpd_type, by="Metadata_compound")

# Add pods
for (m in meta_nm) {
  pod_col <- paste0(m, "_POD")
  cp_pods[[pod_col]][is.na(cp_pods[[pod_col]])] <- 9999
  cp_pods[cp_pods[[pod_col]] == 9999, pod_col] <- NA
}

cp_pods$bmdl <- round(cp_pods$bmdl, 1)
cp_pods$bmd <- round(cp_pods$bmd, 1)
cp_pods$bmdu <- round(cp_pods$bmdu, 1)
for (m in meta_nm) {
  cp_pods[,paste0(m,"_POD")] <- round(cp_pods[,paste0(m,"_POD")], 1)
}
plot_results <- merge(plot_results, cp_pods, by = "Metadata_compound", all.x = TRUE, all.y = FALSE)

# Remove duplicated bmd values
for (compound in cp_compounds){
  inds <- which(plot_results$Metadata_compound == compound)
  plot_results$bmdl[inds[-1]] <- NA
  plot_results$bmd[inds[-1]] <- NA
  plot_results$bmdu[inds[-1]] <- NA
  drop_inds <- inds[-1]
  for (m in meta_nm) {
    plot_results[drop_inds, paste0(m,"_POD")] <- NA
  }
}

plot_compounds <- unique(plot_results$Metadata_compound)
n_pages <- ceiling(length(plot_compounds) / n_per_page)
plot_results <- plot_results[order(plot_results$Metadata_compound), ]

# Define colors - adapt to meta_nm length
n_pods <- length(meta_nm)
base_colors <- c("#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", 
                 "#F0E442", "#999999", "#E6194B", "#3CB44B", "#4363D8")
pod_color_palette <- base_colors[1:n_pods]

pod_colors <- setNames(pod_color_palette, meta_nm)

pod_labels <- setNames(
  gsub("^Metadata_|$", "", meta_nm),
  meta_nm
)

# Create a dummy data frame for legend entries
legend_data <- data.frame(
  meta_name = meta_nm,
  stringsAsFactors = FALSE
)

pdf(cp_plot_path, width = pdf_w, height = pdf_h)

for (i in 1:n_pages) {
  tryCatch({
    
    p <- ggplot(plot_results, aes(x = Metadata_log10conc)) +
      
      geom_point(aes(y = Observations)) +
      geom_line(aes(y = f_dose))
    
    # ---- LEGEND 1: Cell Painting POD ----
    p <- p + geom_vline(
      aes(xintercept = NA_real_, linetype = "Cell Painting POD"),
      color = "#D55E00",
      linewidth = 0.8
    )
    
    # ---- LEGEND 2: META PODs (all at once, not in a loop) ----
    p <- p + geom_vline(
      data = legend_data,
      aes(xintercept = NA_real_, color = meta_name),
      linetype = "solid",
      linewidth = 0.8
    )
    
    # Apply scales
    p <- p + 
      scale_linetype_manual(
        name = "Cell Painting",
        values = c("Cell Painting POD" = "solid"),
        labels = c("Cell Painting POD" = "Cell Painting POD")
      ) +
      scale_color_manual(
        name   = "Metadata PODs",
        values = pod_colors,
        labels = pod_labels,
        drop = FALSE
      )
    
    # ---- NOW ADD ACTUAL PLOT LINES (without aesthetics) ----
    # CP BMD lines
    p <- p +
      geom_vline(aes(xintercept = bmdl), linetype = "dashed", color = "#D55E00", na.rm = TRUE) +
      geom_vline(aes(xintercept = bmd),  linetype = "solid",  color = "#D55E00", na.rm = TRUE) +
      geom_vline(aes(xintercept = bmdu), linetype = "dashed", color = "#D55E00", na.rm = TRUE)
    
    # ADD ONE POD LINE PER meta_nm
    for (m in meta_nm) {
      pod_col <- paste0(m, "_POD")
      
      if (pod_col %in% colnames(plot_results)) {
        p <- p + geom_vline(
          aes(xintercept = .data[[pod_col]]),
          linetype = "solid",
          color = pod_colors[m],
          na.rm = TRUE
        )
      }
    }
    
    p <- p +
      xlim(0, highest_dose) +
      facet_wrap_paginate(~ Compound_Observation, ncol = n_cols, nrow = n_rows, page = i, scales = "free_y") +
      theme_bw() +
      theme(
        strip.text = element_text(size = 6),
        legend.box = "vertical"  # Stack legends vertically
      )
    
    print(p)
    
  }, error = function(e) {
    message(sprintf("Error in plotting page %d: %s", i, e$message))
  })
}

dev.off()
