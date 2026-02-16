#!/bin/bash
#SBATCH --job-name=BSC_axiom
#SBATCH --output=logs/output_%j.log      # %j = SLURM job ID
#SBATCH --error=logs/output_%j.err
#SBATCH --time=12:00:00               # adjust as needed
#SBATCH --cpus-per-task=32      # number of CPUs
#SBATCH --mem=64G               # total memory for the job

nextflow run main.nf -params-file configs/config_base_bsc.json -profile slurm -with-report report_bsc.html
nextflow run main.nf -params-file configs/config_base_axiom.json -profile slurm -with-report report_axiom.html