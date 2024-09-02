#!/bin/bash
#SBATCH --job-name=quarto
#SBATCH --output=quarto.out
#SBATCH --error=quarto.err
#SBATCH --time=00:30:00
#SBATCH --partition=umformation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=f_miashs

HOME=~/scratch
srun quarto render --profile notebooks
srun quarto render --profile slides
srun quarto render --profile book