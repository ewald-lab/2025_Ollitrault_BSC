# BSC and Axiom Cell Painting Analysis

This repository contains the workflows and notebooks used to process the BSC and Axiom OASIS Cell Painting datasets.

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

Figures:

```text
modules/07_downstream_analysis/figures/
```
