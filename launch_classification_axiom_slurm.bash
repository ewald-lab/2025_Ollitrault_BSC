#!/bin/bash
#SBATCH --job-name=axiom_classification
#SBATCH --output=logs/classification_%j.log
#SBATCH --error=logs/classification_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

run_id="${SLURM_JOB_ID:-manual}"

#source envs/python-base/bin/activate

nextflow run entrypoints/06_classification_axiom.nf \
    -params-file configs/config_base_axiom.json \
    -profile slurm 
