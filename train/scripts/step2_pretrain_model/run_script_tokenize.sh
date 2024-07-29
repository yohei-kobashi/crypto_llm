#!/bin/bash

#$ -l rt_G.large=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh && module purge
module load python/3.11 cuda/11.8 hpcx/2.12
source ~/ucllm_nedo_dev/train/.venv_train/bin/activate

cd ~/crypto_llm/train/scripts/step2_pretrain_model/
# no encryption
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_wikipedia_latin_no_encryption_000000_1234_True.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True_no_encryption.jsonl \
    --data_name wikipedia-no_encryption \
    --workers 20
# poly, key length == 1
encryption="poly_000001"
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_wikipedia_latin_${encryption}_1234_True.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_${encryption}_1234_True.jsonl \
    --data_name wikipedia-${encryption} \
    --workers 20
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_wikipedia_latin_${encryption}_1234_True.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_${encryption}_1234_True_no_encryption.jsonl \
    --data_name wikipedia-${encryption}-no_encryption \
    --workers 20
# poly, key length == 10
encryption="poly_000010"
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_wikipedia_latin_${encryption}_1234_True.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_${encryption}_1234_True.jsonl \
    --data_name wikipedia-${encryption} \
    --workers 20
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_wikipedia_latin_${encryption}_1234_True.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_${encryption}_1234_True_no_encryption.jsonl \
    --data_name wikipedia-${encryption}-no_encryption \
    --workers 20
# poly, key length == 100
encryption="poly_000100"
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_wikipedia_latin_${encryption}_1234_True.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_${encryption}_1234_True.jsonl \
    --data_name wikipedia-${encryption} \
    --workers 20
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_wikipedia_latin_${encryption}_1234_True.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_${encryption}_1234_True_no_encryption.jsonl \
    --data_name wikipedia-${encryption}-no_encryption \
    --workers 20
# poly, key length == 1000
encryption="poly_001000"
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_wikipedia_latin_${encryption}_1234_True.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_${encryption}_1234_True.jsonl \
    --data_name wikipedia-${encryption} \
    --workers 20
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_wikipedia_latin_${encryption}_1234_True.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_${encryption}_1234_True_no_encryption.jsonl \
    --data_name wikipedia-${encryption}-no_encryption \
    --workers 20
# poly, key length == 10000
encryption="poly_010000"
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_wikipedia_latin_${encryption}_1234_True.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_${encryption}_1234_True.jsonl \
    --data_name wikipedia-${encryption} \
    --workers 20
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_wikipedia_latin_${encryption}_1234_True.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_${encryption}_1234_True_no_encryption.jsonl \
    --data_name wikipedia-${encryption}-no_encryption \
    --workers 20
# poly, key length == 100000
encryption="poly_100000"
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_wikipedia_latin_${encryption}_1234_True.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_${encryption}_1234_True.jsonl \
    --data_name wikipedia-${encryption} \
    --workers 20
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_wikipedia_latin_${encryption}_1234_True.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_${encryption}_1234_True_no_encryption.jsonl \
    --data_name wikipedia-${encryption}-no_encryption \
    --workers 20
# poly, key length == max
encryption="poly_000000"
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_wikipedia_latin_${encryption}_1234_True.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_${encryption}_1234_True.jsonl \
    --data_name wikipedia-${encryption} \
    --workers 20
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_wikipedia_latin_${encryption}_1234_True.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_${encryption}_1234_True_no_encryption.jsonl \
    --data_name wikipedia-${encryption}-no_encryption \
    --workers 20



