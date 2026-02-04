require(dplyr)
require(arrow)
require(ggplot2)
require(ggforce)
require(reshape2)

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

cc_pod_path <- args[1]
cc_path <- args[2]
cc_plot_path <- args[3]
meta_nm <- args[4]
control <- args[5]

if (FALSE) {
  cc_pods <- read_parquet(cc_pod_path) %>% as.data.frame()
  cc <- read_parquet(cc_path) %>% as.data.frame()

  if (!"Metadata_log10conc" %in% colnames(cc)) {
    cc$Metadata_log10conc <- log10(cc$Metadata_concentration + 1e-10)
  }

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
  # Add DMSO
  for (compound in cc_compounds){
    cc_temp <- cc[cc$Metadata_compound == compound, ]
    cc_plates <- unique(cc_temp$Metadata_plate)

    cc_dmso <- cc[(cc$Metadata_compound == control) & (cc$Metadata_plate %in% cc_plates), ]
    dmso_values <- cc_dmso[, c("Metadata_compound", "Metadata_log10conc", meta_nm)]
    dmso_values$Metadata_compound <- compound
    dmso_values$Metadata_log10conc <- round(dmso_values$Metadata_log10conc, 1)

    cc_values <- rbind(cc_values, dmso_values)
  }
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

        xlim(0, highest_dose) +
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
}

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
# Add DMSO
for (compound in cc_compounds){
  cc_temp <- cc[cc$Metadata_compound == compound, ]
  cc_plates <- unique(cc_temp$Metadata_plate)
  
  cc_dmso <- cc[(cc$Metadata_compound == control) & (cc$Metadata_plate %in% cc_plates), ]
  dmso_values <- cc_dmso[, c("Metadata_compound", "Metadata_log10conc", meta_nm)]
  dmso_values$Metadata_compound <- compound
  dmso_values$Metadata_log10conc <- round(dmso_values$Metadata_log10conc, 1)
  
  cc_values <- rbind(cc_values, dmso_values)
}
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
