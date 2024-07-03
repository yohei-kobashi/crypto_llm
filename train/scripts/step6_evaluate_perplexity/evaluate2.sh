#!/bin/bash

# Command line options go here
#SBATCH --time=24:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-6
#SBATCH --job-name=evaluate_perplexity2
#SBATCH --output=evaluate_perplexity2.out
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --ntasks=208

source ~/miniconda3/etc/profile.d/conda.sh
conda activate .venv_train
cd ~/ucllm_nedo_dev/train/scripts/step6_evaluate_perplexity/
python evaluate_perplexity2.py