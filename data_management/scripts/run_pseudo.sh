#!/bin/bash

#$ -l rt_C.large=1
#$ -l h_rt=48:00:00
#$ -j y
#$ -cwd

source ~/miniconda3/etc/profile.d/conda.sh
conda activate .venv_data
cd ~/crypto_llm/data_management/scripts/

python extract_pseudo_privacy_sentences.py