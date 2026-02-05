require(dplyr)
require(arrow)
require(ggplot2)
require(ggforce)
require(reshape2)
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

#### 1. Read in data and set common parameters
args <- commandArgs(trailingOnly = TRUE)

cc_pod_path <- args[1]
cc_path <- args[2]
cc_plot_path <- args[3]
meta_nm <- args[4]
control <- args[5]
dmso_mode <- args[6]

cc_pods <- read_parquet(cc_pod_path) %>% as.data.frame()
cc <- read_parquet(cc_path) %>% as.data.frame()

#if (!"Metadata_log10conc" %in% colnames(cc)) {
#  cc$Metadata_log10conc <- log10(cc$Metadata_log10conc + 1e-10)
#}

highest_dose <- max(unique(cc$Metadata_log10conc))
highest_dose <- round(highest_dose + (0.025 * highest_dose), 1)
dose <- seq(0, highest_dose, by = 0.1)

n_cols <- 5
n_rows <- 5
n_per_page <- n_cols * n_rows
pdf_w <- 12
pdf_h <- 10


#### 2. Make cell count plots
cc_compounds <- unique(cc_pods$Metadata_compound)
plot_results <- data.frame(Metadata_log10conc = dose)

for (compound in cc_compounds){
  temp_pod <- cc_pods[cc_pods$Metadata_compound == compound, ]
  
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
}

plot_results <- reshape2::melt(plot_results, id.vars = c("Metadata_log10conc"))
colnames(plot_results)[2:3] <- c("Metadata_compound", "f_dose")

# Add cell count observations
cc_values <- cc[, c("Metadata_compound", "Metadata_log10conc", meta_nm)]
cc_values$Metadata_log10conc <- round(cc_values$Metadata_log10conc, 1)
cc <- cc %>% bind_cols(purrr::map_dfr(.$Metadata_well, well_to_coord))
# Add DMSO
for (compound in cc_compounds[1:5]){
  print(paste0("Processing compound: ", compound))
  cc_temp <- cc[cc$Metadata_compound == compound, ]
  if (dmso_mode == "plate") {
    cc_plates <- unique(cc %>%
        filter(Metadata_compound == control) %>%
        dplyr::select(c("Metadata_plate")))
    
    cc_dmso <- cc[(cc$Metadata_compound == control) & (cc$Metadata_plate %in% cc_plates), ]
    dmso_values <- cc_dmso[, c("Metadata_compound", "Metadata_log10conc", meta_nm)]
  } else if (dmso_mode == "closest") {
    dmso_values <- get_closest_dmso(
      comp_fit_coords = cc %>%
        filter(Metadata_compound == compound),
      dat_dmso = cc %>%
        filter(Metadata_compound == control)
    ) %>%
    dplyr::select(c(meta_nm, "Metadata_compound", "Metadata_log10conc"))
  }
  cc_values <- rbind(cc_temp[c(meta_nm, "Metadata_compound", "Metadata_log10conc")], dmso_values)
}
cc_values$Metadata_log10conc <- round(cc_values$Metadata_log10conc, 1)
plot_results <- merge(plot_results, cc_values,
                      by = c("Metadata_compound", "Metadata_log10conc"),
                      all.x = TRUE, all.y = FALSE)

# Add cell count pods
cc_pods <- cc_pods[cc_pods$all.pass == TRUE,
                   c("Metadata_compound", "bmdl", "bmd", "bmdu")]
cc_pods$bmdl <- round(cc_pods$bmdl, 1)
cc_pods$bmd <- round(cc_pods$bmd, 1)
cc_pods$bmdu <- round(cc_pods$bmdu, 1)
plot_results <- merge(plot_results, cc_pods, by = "Metadata_compound",
                      all.x = TRUE, all.y = FALSE)

# Remove duplicated bmd values
for (compound in cc_compounds){
  inds <- which(plot_results$Metadata_compound == compound)
  plot_results$bmdl[inds[-1]] <- NA
  plot_results$bmd[inds[-1]] <- NA
  plot_results$bmdu[inds[-1]] <- NA
}

# Make plots
plot_compounds <- unique(plot_results$Metadata_compound)
n_pages <- ceiling(length(plot_compounds) / n_per_page)
plot_results <- plot_results[order(plot_results$Metadata_compound), ]
pdf(cc_plot_path, width = pdf_w, height = pdf_h)
for (i in 1:n_pages) {
  # Use tryCatch to handle errors in plot creation
  tryCatch({
    p <- ggplot(plot_results, aes(x = Metadata_log10conc)) +
      geom_point(aes(y = .data[[meta_nm]])) +
      geom_line(aes(y = f_dose)) +
      
      geom_vline(aes(xintercept = bmdl),
                 linetype = "dashed", color = "red", na.rm = TRUE) +
      geom_vline(aes(xintercept = bmd),
                 linetype = "solid", color = "red", na.rm = TRUE) +
      geom_vline(aes(xintercept = bmdu),
                 linetype = "dashed", color = "red", na.rm = TRUE) +
      
      scale_x_continuous(
        trans = scales::pseudo_log_trans(base = 10),
        limits = c(0, highest_dose)
      ) + #xlim(0, highest_dose) +
      facet_wrap_paginate(~ Metadata_compound,
                          ncol = n_cols, nrow = n_rows, page = i,
                          scales = "free_y") +
      theme_bw()
    
    # Print the plot to the PDF
    print(p)
  }, error = function(e) {
    # Handle error: print a message and skip this plot
    message(sprintf("Error in plotting page %d: %s", i, e$message))
  })
}
dev.off()
