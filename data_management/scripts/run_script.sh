#!/bin/bash

#$ -l rt_C.large=1
#$ -l h_rt=12:00:00
#$ -j y
#$ -cwd

source ~/miniconda3/etc/profile.d/conda.sh
conda activate .venv_data
cd ~/crypto_llm/data_management/scripts/
# # no encryption
# python merge_and_split_jsonls.py \
#         --lang latin \
#         --split_rate 0.9 \
#         --input_dir /groups/gcf51099/crypto_llm/data/datasets/wikipedia/20240501/en/ \
#         --output_dir /groups/gcf51099/crypto_llm/data/encrypted/ \
#         --seed 1234
# # poly, key length == 1
# python encrypt_jsonl.py \
#         --lang latin \
#         --encryption_type poly \
#         --input_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True.jsonl \
#         --output_dir /groups/gcf51099/crypto_llm/data/encrypted/ \
#         --key_length 1 \
#         --seed 1234
# # poly, key length == 10
# python encrypt_jsonl.py \
#         --lang latin \
#         --encryption_type poly \
#         --input_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True.jsonl \
#         --output_dir /groups/gcf51099/crypto_llm/data/encrypted/ \
#         --key_length 10 \
#         --seed 1234
# # poly, key length == 100
# python encrypt_jsonl.py \
#         --lang latin \
#         --encryption_type poly \
#         --input_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True.jsonl \
#         --output_dir /groups/gcf51099/crypto_llm/data/encrypted/ \
#         --key_length 100 \
#         --seed 1234
# # poly, key length == 1000
# python encrypt_jsonl.py \
#         --lang latin \
#         --encryption_type poly \
#         --input_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True.jsonl \
#         --output_dir /groups/gcf51099/crypto_llm/data/encrypted/ \
#         --key_length 1000 \
#         --seed 1234
# # poly, key length == 10000
# python encrypt_jsonl.py \
#         --lang latin \
#         --encryption_type poly \
#         --input_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True.jsonl \
#         --output_dir /groups/gcf51099/crypto_llm/data/encrypted/ \
#         --key_length 10000 \
#         --seed 1234
# poly, key length == 100000
python encrypt_jsonl.py \
        --lang latin \
        --encryption_type poly \
        --input_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True.jsonl \
        --output_dir /groups/gcf51099/crypto_llm/data/encrypted/ \
        --key_length 100000 \
        --seed 1234
# poly, key length == 0 (max)
python encrypt_jsonl.py \
        --lang latin \
        --encryption_type poly \
        --input_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True.jsonl \
        --output_dir /groups/gcf51099/crypto_llm/data/encrypted/ \
        --key_length 0 \
        --seed 1234
