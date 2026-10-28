# BSC and Axiom Cell Painting Analysis

This repository contains the workflows and notebooks used to process the BSC and Axiom OASIS Cell Painting datasets.

## Requirements

- Python 3.11
- R >= 4.5
- [Nextflow](https://www.nextflow.io/)

## Download the data

Axiom OASIS data can be downloaded from:

https://github.com/jessica-ewald/2024_09_09_Axiom_OASIS/tree/main

The default config expects the Axiom data repository to be next to this repository:

```bash
git clone https://github.com/jessica-ewald/2024_09_09_Axiom_OASIS.git ../2024_09_09_Axiom_OASIS
```

BSC data are included in this Git repository, mainly under:

```text
modules/01_preprocessing/inputs/
```

Input paths are configured in `configs/config_base_axiom.json` and `configs/config_base_bsc.json`.

## Set up the Python environment

Install the Python dependencies in `envs/python-base` using `requirements-dev.txt`:

```bash
uv venv envs/python-base --python=python3.11
source envs/python-base/bin/activate
uv pip install -r requirements-dev.txt
```

If `uv` is not available, use standard `venv` and `pip`:

```bash
python3.11 -m venv envs/python-base
source envs/python-base/bin/activate
pip install -r requirements-dev.txt
```
## Set up the R environment

The concentration-response step (`modules/04_concresponse/`) uses R (>= 4.5). Install the packages from an R session:

```r
install.packages(c(
  "arrow", "data.table", "dplyr", "tidyr", "purrr", "reshape2",
  "ggplot2", "ggforce", "scales", "foreach", "doParallel", "parallelly",
  "remotes"
))
remotes::install_github("jessica-ewald/fastbmdR@v0.0.0.9000")
```

## Run the analysis

Launch the full workflow with `main.nf`:

```bash
source envs/python-base/bin/activate
nextflow run main.nf -profile local --entrypoint_profile slurm -resume
```

For a fully local run, use:

```bash
nextflow run main.nf -profile local --entrypoint_profile local -resume
```

## Explore the results

Manuscript notebooks:

```text
modules/07_downstream_analysis/notebook_manuscript/
```

Exported HTML files:

```text
modules/07_downstream_analysis/notebook_manuscript/__marimo__/
```

## Citation

If you use this code or data, please cite:

> Ollitrault G., Mitchell C.A., Ferguson S.S., Roe A.L., White B., Titterton K., Ewald J. (2026). Cell Painting and machine learning for mechanistic profiling of hepatotoxic botanicals and their constituents. *bioRxiv*. https://doi.org/10.64898/2026.10.05.756246

```bibtex
@article{Ollitrault2026.10.05.756246,
  author = {Ollitrault, Guillaume and Mitchell, Constance A. and Ferguson, Stephen S. and Roe, Amy L. and White, Brandon and Titterton, Katherine and Ewald, Jessica},
  title = {Cell Painting and machine learning for mechanistic profiling of hepatotoxic botanicals and their constituents},
  elocation-id = {2026.10.05.756246},
  year = {2026},
  doi = {10.64898/2026.10.05.756246},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2026/10/12/2026.10.05.756246},
  journal = {bioRxiv}
}
```

## License

This project is released under the [CC0 1.0 Universal](LICENSE) license.
