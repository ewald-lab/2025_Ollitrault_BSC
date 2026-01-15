# 🧪 Ewald Lab Data Analysis of BSC

 Cell Painting dataset to assess the bioactivity caused by botanicals in cells. Botanicals are typically mixtures of 100s-1000s of compounds, including “natural” compounds, and are sold to consumers with no/very little toxicity testing or safety evaluation as dietary supplements or vitamins. The OASIS members tested the botanicals themselves, as well as the individual compounds advertised as “active ingredients” within each botanical. The goal is to see if the active ingredient is the main thing that causes activity in cells, or if additional bioactivity is being triggered by the other added ingredients.


TODO : add sitecustomize to the environment

# For HPC computing
module load r/4.5.1
create ~/.Renviron file in home directory
write in .Renviron : R_LIBS_USER=/nfs/research/ewald/softwares/R/x86_64-pc-linux-gnu-library/4.5
write in .Renviron : R_LIBS=/nfs/research/ewald/softwares/R/x86_64-pc-linux-gnu-library/4.5
restart session
module load r/4.5.1
open R and check that the path is in it:
module load r/4.5.1
R
.libPaths()
You should see :
[1] "/usr/local/lib/R/site-library"                                  
[2] "/usr/local/lib/R/library"                                       
[3] "/nfs/research/ewald/softwares/R/x86_64-pc-linux-gnu-library/4.5"
Install in specified place : 
install.packages(
  c("dplyr", "arrow", "foreach", "doParallel",
    "ggplot2", "ggforce", "tidyr", "purrr", "data.table"),
  repos = "https://cloud.r-project.org",
  lib = "/nfs/research/ewald/softwares/R/x86_64-pc-linux-gnu-library/4.5"
)