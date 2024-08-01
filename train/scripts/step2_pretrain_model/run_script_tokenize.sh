#!/bin/bash

#$ -l rt_G.large=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh && module purge
module load python/3.11 cuda/11.8 hpcx/2.12
source ~/ucllm_nedo_dev/train/.venv_train/bin/activate

cd ~/crypto_llm/train/scripts/step2_pretrain_model/

data="wikipedia"
lang="latin"
plain="no_encryption_000000_1234_True"

# no encryption
encryption="no_encryption_000000_1234_True"
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}_${lang}_${encryption}.jsonl \
    --data_name ${data}_${lang}_${encryption} \
    --workers 20
# no encryption (continual pretraining)
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}_${lang}_${plain}_no_encryption.jsonl \
    --data_name ${data}_${lang}_${encryption}_no_encryption \
    --workers 20

# poly, key length == 1
encryption="poly_000001_1234_True"
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}_${lang}_${encryption}.jsonl \
    --data_name ${data}_${lang}_${encryption} \
    --workers 20
# no encryption (continual pretraining)
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}_${lang}_${plain}_no_encryption.jsonl \
    --data_name ${data}_${lang}_${encryption}_no_encryption \
    --workers 20

# poly, key length == 10
encryption="poly_000010_1234_True"
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}_${lang}_${encryption}.jsonl \
    --data_name ${data}_${lang}_${encryption} \
    --workers 20
# no encryption (continual pretraining)
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}_${lang}_${plain}_no_encryption.jsonl \
    --data_name ${data}_${lang}_${encryption}_no_encryption \
    --workers 20

# poly, key length == 100
encryption="poly_000100_1234_True"
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}_${lang}_${encryption}.jsonl \
    --data_name ${data}_${lang}_${encryption} \
    --workers 20
# no encryption (continual pretraining)
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}_${lang}_${plain}_no_encryption.jsonl \
    --data_name ${data}_${lang}_${encryption}_no_encryption \
    --workers 20

# poly, key length == 1000
encryption="poly_001000_1234_True"
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}_${lang}_${encryption}.jsonl \
    --data_name ${data}_${lang}_${encryption} \
    --workers 20
# no encryption (continual pretraining)
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}_${lang}_${plain}_no_encryption.jsonl \
    --data_name ${data}_${lang}_${encryption}_no_encryption \
    --workers 20

# poly, key length == 10000
encryption="poly_010000_1234_True"
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}_${lang}_${encryption}.jsonl \
    --data_name ${data}_${lang}_${encryption} \
    --workers 20
# no encryption (continual pretraining)
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}_${lang}_${plain}_no_encryption.jsonl \
    --data_name ${data}_${lang}_${encryption}_no_encryption \
    --workers 20

# poly, key length == 100000
encryption="poly_100000_1234_True"
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}_${lang}_${encryption}.jsonl \
    --data_name ${data}_${lang}_${encryption} \
    --workers 20
# no encryption (continual pretraining)
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}_${lang}_${plain}_no_encryption.jsonl \
    --data_name ${data}_${lang}_${encryption}_no_encryption \
    --workers 20

# poly, key length == max
encryption="poly_000000_1234_True"
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}_${lang}_${encryption}.jsonl \
    --data_name ${data}_${lang}_${encryption} \
    --workers 20
# no encryption (continual pretraining)
bash ./gcp_honban_node-1_gpu-8/tokenization.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}_${lang}_${plain}_no_encryption.jsonl \
    --data_name ${data}_${lang}_${encryption}_no_encryption \
    --workers 20
