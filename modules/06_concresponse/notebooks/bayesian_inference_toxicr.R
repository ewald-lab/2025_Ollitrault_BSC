# Script to try the bayesian inference of ToxicR package





#### Tutorial ####
library(tidyverse)
library(ToxicR)
require(gridExtra)
# Ex 1
data_example <- ToxicR::dichotomousDR %>% filter(ID==695) %>% select(dose,n,obs)
#remove intermediate doses
data_example2 <- data_example[-(3:4),]
# Fit the default Bayesian model
fit_a <- single_dichotomous_fit(data_example$dose,data_example$obs,data_example$n,
                                model_type="weibull")
# Fit the default MLE
fit_aa <- single_dichotomous_fit(data_example$dose,data_example$obs,data_example$n,
                                 model_type="weibull",fit_type="mle")
summary(fit_aa)
summary(fit_a)
grid.arrange(plot(fit_a)+scale_x_continuous(trans="pseudo_log") + ggtitle("Bayesian") ,
             plot(fit_aa)+scale_x_continuous(trans="pseudo_log")+ ggtitle("MLE"), ncol=2)
# Ex 2
data_one <- ToxicR::dichotomousDR %>% group_by(ID) %>% filter(length(ID)>5)
data_example <- ToxicR::dichotomousDR %>% filter(ID==642) %>% select(dose,n,obs)
fit_a <- single_dichotomous_fit(data_example$dose,data_example$obs,data_example$n,
                                model_type="log-probit", fit_type ="laplace")
summary(fit_a)
fit_aa <- single_dichotomous_fit(data_example$dose,data_example$obs,data_example$n,
                                 model_type="log-probit",fit_type ="mle")
summary(fit_aa)
grid.arrange(plot(fit_a)+scale_x_continuous(trans="pseudo_log") + ggtitle("Bayesian") ,
             plot(fit_aa)+scale_x_continuous(trans="pseudo_log")+ ggtitle("MLE"), ncol=2)
data_example2 <- data_example[-(2:3),]
fit_b <- single_dichotomous_fit(data_example2$dose,data_example2$obs,data_example2$n,
                                model_type="log-probit", fit_type ="laplace")
summary(fit_b)
fit_c <- single_dichotomous_fit(data_example2$dose,data_example2$obs,data_example2$n,
                                model_type="log-probit",fit_type = "mle")
summary(fit_c)
grid.arrange(plot(fit_a)+scale_x_continuous(trans="pseudo_log") + ggtitle("Full data - Log-Probit") ,
             plot(fit_b)+scale_x_continuous(trans="pseudo_log")+ ggtitle("Removed data - Log-Probit"), ncol=2)
# Ex 3
resp  <- Y <- c(5.890,6.740,4.976,5.990,6.608,6.527,6.783,5.360,6.606,
                6.272,5.909,4.912,5.486,3.860,4.877,4.641,5.670,
                5.591,4.840,5.801,5.822,6.413,5.708,7.372,6.552,
                5.954,6.253,7.415,7.541,7.490,9.825,9.391,9.469,
                10.966,10.785,10.704,11.716,10.584,10.883)

doses <- c(0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,    
           0.0000,0.0000,0.0000,9.4905,9.4905,9.4905,30.0116,30.0116,
           30.0116,94.9050,94.9050,94.9050,300.1160, 300.1160,300.1160,
           949.0500,949.0500,949.0500,3001.1597,3001.1597,3001.1597,9490.5002,
           9490.5002,  9490.5002, 30011.5969, 30011.5969, 30011.5969, 94905.0023, 
           94905.0023, 94905.0023)

udose <- unique(doses)[-(5:8)]

bayes_a <- single_continuous_fit(doses,resp,model_type = "hill",distribution="normal",fit_type="laplace",BMR=1)
mle_a   <- single_continuous_fit(doses,resp,model_type = "hill",distribution="normal",fit_type="mle",BMR=1)

sub_resp <- resp[doses %in% udose]
sub_dose <- doses[doses %in% udose]

bayes_b <- single_continuous_fit(sub_dose,sub_resp,model_type = "hill",distribution="normal",fit_type="laplace",BMR=1)
mle_b   <- single_continuous_fit(sub_dose,sub_resp,model_type = "hill",distribution="normal",fit_type="mle",BMR=1)

grid.arrange(plot(bayes_a)+scale_x_continuous(trans="pseudo_log"),
             plot(bayes_b)+scale_x_continuous(trans="pseudo_log"))
grid.arrange(plot(mle_a)+scale_x_continuous(trans="pseudo_log"),
             plot(mle_b)+scale_x_continuous(trans="pseudo_log"))

bayes_p <- cbind(bayes_a$parameters,bayes_b$parameters)
mle_p <- cbind(mle_a$parameters,mle_b$parameters)
rownames(bayes_p) <- c("a","b","c","d","log(sigma)")
rownames(mle_p) <- c("a","b","c","d","log(sigma)")
bayes_p[3:4,]
# Why
require(numDeriv)
set.seed(123)
#Randomly Sample 20 data points
gamma_data <- rgamma(20,10,3)

gamma_log_like <-function(beta){
  like_vals <- dgamma(gamma_data,10,beta,log=TRUE)
  return (-sum(like_vals))
}
log_like <- Vectorize(gamma_log_like)
beta_values <- seq(3.08-0.5,3.08+0.5,0.02)
log_l_values <- log_like(beta_values)
plot(beta_values,log_l_values,type='l',col=1,lwd=2,lty=2)
#
# Find the Optimum
MLE <- optim(c(3),log_like,method="Brent",lower = c(0), upper = c(10))$par
VAR <- 1/hessian(log_like,MLE)
theory <- 1/(2*VAR)*(beta_values-MLE)^2 + optim(c(3),log_like,method="Brent",lower = c(0), upper = c(10))$value
lines(beta_values,theory,col=2)
#
obs <- c(8,8,8,8)
y   <- c(6,5,7,8)
doses <- c(0,0.5,1,2)/2

weibull_model <- function(parms,doses){
  g <- parms[1]
  b <- parms[2]
  d <- parms[3]
  
  p <- g + (1-g)*(1-exp(-b*doses^d))
  return(p)
}

log_likelihood <- function(parms){
  p <- weibull_model(parms,doses)
  log_like <- dbinom(y,obs,p,log=TRUE)
  return(-sum(log_like))
}

opt_p <- optim(c(0.5,1.22,0.4181133),log_likelihood, method="L-BFGS-B",
               lower=c(1e-8,0.001,0.001),upper=c(1-1e-8,200,18))
MLE <- opt_p$par
OPTV <- opt_p$value
values <- seq(0.001,18,0.001)*0
ii = 1
for (profile_value in seq(0.001,18,0.001)){
  
  prof_log_likelihood <- function(parms){
    p <- weibull_model(c(parms,profile_value),doses)
    log_like <- dbinom(y,obs,p,log=TRUE)
    return(-sum(log_like))
  }
  opt_p <- optim(c(0.5,1.22),prof_log_likelihood, method="L-BFGS-B",
                 lower=c(1e-8,0.00001),upper=c(1-1e-8,200))
  
  values[ii] <- opt_p$value
  ii <- ii + 1
}
plot(seq(0.001,18,0.001),values,type='l',ylab="-log-likelihood",xlab="values of 'd'")
abline(v=MLE[3],col=2,lty=2)
require(ToxicR)
model_fit <- single_dichotomous_fit(doses,y,obs,model_type="weibull",fit_type='mcmc',samples=50000)
summary(model_fit)
plot(model_fit)
model_fit <- ma_dichotomous_fit(doses,y,obs,fit_type='mcmc',samples=50000,
                                threads = 12)
summary(model_fit)
plot(model_fit)

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
input_path <- "../outputs/distances/distance_bsc_gmd.parquet"
output_path <- NA
ctrl <- "s0617825"
dmso_mode <- "closest"  # "closest" or "plate" # Improve ? : "rowcol" → same row or column only | "k_closest" → k nearest DMSOs | "radius" → within N wells
# 2. Calculate BMDs from global mahalanobis distances
dat <- read_parquet(input_path) %>% as.data.frame()
dat <- dat %>%
          bind_cols(map_dfr(.$Metadata_well, well_to_coord))
compounds <- unique(dat$Metadata_compound)
compounds <- compounds[!grepl(ctrl, compounds)]

dat_ctrl <- dat %>%
  filter(Metadata_compound == ctrl)
dat_comp <- dat %>%
  filter(Metadata_compound != ctrl)

feat_cols <- colnames(dat)[grepl("Feature", colnames(dat))] # The computed distances

bmd_res <- data.frame()

for (compound in compounds) {
  dat_comp <- dat %>%
    filter(Metadata_compound == compound)
  if (dmso_mode == "closest") {
    ctrl_fit <- get_closest_dmso(
      comp_fit_coords = dat_comp,
      dat_ctrl = dat_ctrl
    )
  } else if (dmso_mode == "plate") {
    ctrl_fit <- dat_ctrl %>%
      filter(Metadata_plate %in% unique(dat_comp$Metadata_plate))
  } else {
    stop("dmso_mode must be 'closest' or 'plate'")
  }
}
