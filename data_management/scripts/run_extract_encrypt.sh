#!/bin/bash

#$ -l rt_C.large=1
#$ -l h_rt=48:00:00
#$ -j y
#$ -cwd

source ~/miniconda3/etc/profile.d/conda.sh
conda activate .venv_data
cd ~/crypto_llm/data_management/scripts/

python extract_random_jsonl_lines.py \
        --input_file /groups/gcf51099/crypto_llm/data/test/pseudo_privacy_wikipedia_latin_no_encryption_000000_1234_True.jsonl \
        --output_file /groups/gcf51099/crypto_llm/data/test/pseudo_privacy_test.jsonl \
        --num_lines 1000 \
        --seed 1234