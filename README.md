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

# replace all instances of 'mod-00-env' with the name of your environment
uv venv mod-00-env --python=python3.11
source mod-00-env/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name=mod-00-env --display-name "Python (mod-00-env)"

echo "To activate later: source mod-00-env/bin/activate"
echo "To install a package: uv pip install pkg-name"
echo "To update requirements.txt after installing packages: uv pip freeze > requirements.txt"


# Python Base Environment

Contains core dependencies shared by all analysis modules.
Python 3.11

# [If necessary] Python ML Environment

Extension of python-base for ML-heavy modules.
Includes packages like xgboost, lightgbm, tensorflow, torch.

# R Base Environment

Used for R scripts in concentration response and plots.
Includes tidyverse, ggplot2, dplyr, data.table.