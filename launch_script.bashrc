#!/bin/bash
#SBATCH --job-name=BSC_axiom
#SBATCH --output=logs/output_%j.log      # %j = SLURM job ID
#SBATCH --error=logs/output_%j.err
#SBATCH --time=12:00:00               # adjust as needed
#SBATCH --cpus-per-task=32      # number of CPUs
#SBATCH --mem=64G               # total memory for the job

nextflow run main.nf -profile local --entrypoint_profile slurm -with-report report_entrypoints.html -resume
