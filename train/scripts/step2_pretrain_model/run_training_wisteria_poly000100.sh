#!/bin/bash

#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=24:00:00
#PJM -g gb20
#PJM -j

###PJM -L node=1
set -e

echo "* Loading modules"
module load gcc/8.3.1
module load cuda/11.8
module load cudnn/8.8.0  
module list

export HOME=/work/gb20/b20048
# export WORK_HOME=$HOME/crypto_llm/train

# NOTE if needed, conda installation
# conda env remove -n deep -y
#mkdir -p ~/miniconda3
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
#bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
#rm ~/miniconda3/miniconda.sh

echo "* Creating conda environment"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate deep

cd ~/crypto_llm/train/scripts/step2_pretrain_model/

# model_num and jobname
model_num="1"
jobname="tinyllama-1.1B"

# hyper parameters
global_batch_size=4096
batch_size=1
seq_len=2048
precision="fp32"
mp_size=1
pp_size=1
zero_stage=0
min_lr=1.0e-6

host="${HOSTNAME}"
current_time=$(date "+%Y.%m.%d_%H.%M.%S")

# # no encryption
# data="wikipedia"
# lang="latin"
# encryption="no_encryption_000000_1234_True"
# # poly_000001
# data="wikipedia"
# lang="latin"
# encryption="poly_000001_1234_True"
# poly_000100
data="wikipedia"
lang="latin"
encryption="poly_000100_1234_True"
# poly_010000
# data="wikipedia"
# lang="latin"
# encryption="poly_010000_1234_True"
# # poly_000000
# data="wikipedia"
# lang="latin"
# encryption="poly_000000_1234_True"

unique_id="${model_num}.${lang}_${data}_${encryption}"
output_model_dir="/work/gb20/b20048/crypto_llm_data/models/${unique_id}/"
# bash ./gcp_honban_node-1_gpu-8/llama2_flashattn2-on.sh \
#     --input_tokenizer_file /work/gb20/b20048/crypto_llm_data/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
#     --output_model_dir ${output_model_dir} \
#     --save_interval 100 \
#     --train_iters 1000 \
#     --wandb_entity yohei-kobashi \
#     --wandb_project encrypted_data_LLM \
#     --data_base_dir /work/gb20/b20048/crypto_llm_data/data \
#     --train_data_file /work/gb20/b20048/crypto_llm_data/data/encrypted/${data}-${lang}-${encryption}.jsonl \
#     --data_name ${data}_${lang}_${encryption} \
#     --unique_id ${unique_id} \
#     --global_batch_size ${global_batch_size} \
#     --batch_size ${batch_size} \
#     --seq_len ${seq_len} \
#     --precision ${precision} \
#     --mp_size ${mp_size} \
#     --pp_size ${pp_size} \
#     --zero_stage ${zero_stage} \
#     --min_lr ${min_lr} \
#     --jobname ${jobname} \
#     --train_data_exact_num_epochs 2 \
#     --untie_embeddings_and_output_weights true \
#     --tinyllama true

# bash ./gcp_honban_node-1_gpu-8/llama2_flashattn2-on.sh \
#     --input_tokenizer_file /work/gb20/b20048/crypto_llm_data/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
#     --output_model_dir ${output_model_dir} \
#     --save_interval 10 \
#     --train_iters 1010 \
#     --wandb_entity yohei-kobashi \
#     --wandb_project encrypted_data_LLM \
#     --data_base_dir /work/gb20/b20048/crypto_llm_data/data/test \
#     --train_data_file /work/gb20/b20048/crypto_llm_data/data/encrypted/${data}_${lang}_${encryption}.jsonl \
#     --data_name ${data}_${lang}_${encryption} \
#     --unique_id ${unique_id} \
#     --global_batch_size ${global_batch_size} \
#     --batch_size ${batch_size} \
#     --seq_len ${seq_len} \
#     --precision ${precision} \
#     --mp_size ${mp_size} \
#     --pp_size ${pp_size} \
#     --zero_stage ${zero_stage} \
#     --min_lr ${min_lr} \
#     --jobname ${jobname} \
#     --split 100,0,0 \
#     --untie_embeddings_and_output_weights true \
#     --tinyllama true

# bash ./gcp_honban_node-1_gpu-8/llama2_flashattn2-on.sh \
#     --input_tokenizer_file /work/gb20/b20048/crypto_llm_data/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
#     --output_model_dir ${output_model_dir} \
#     --save_interval 100 \
#     --train_iters 1400 \
#     --wandb_entity yohei-kobashi \
#     --wandb_project encrypted_data_LLM \
#     --wandb_tag other_gpu \
#     --data_base_dir /work/gb20/b20048/crypto_llm_data/data \
#     --train_data_file /work/gb20/b20048/crypto_llm_data/data/encrypted/${data}_${lang}_${encryption}_no_encryption.jsonl \
#     --data_name ${data}_${lang}_${encryption}_no_encryption \
#     --unique_id ${unique_id} \
#     --global_batch_size ${global_batch_size} \
#     --batch_size ${batch_size} \
#     --seq_len ${seq_len} \
#     --precision ${precision} \
#     --mp_size ${mp_size} \
#     --pp_size ${pp_size} \
#     --zero_stage ${zero_stage} \
#     --min_lr ${min_lr} \
#     --jobname ${jobname} \
#     --untie_embeddings_and_output_weights true \
#     --tinyllama true

bash ./gcp_honban_node-1_gpu-8/llama2_flashattn2-on.sh \
    --input_tokenizer_file /work/gb20/b20048/crypto_llm_data/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --output_model_dir ${output_model_dir} \
    --save_interval 100 \
    --train_iters 1500 \
    --wandb_entity yohei-kobashi \
    --wandb_project encrypted_data_LLM \
    --wandb_tag other_gpu \
    --data_base_dir /work/gb20/b20048/crypto_llm_data/data \
    --train_data_file /work/gb20/b20048/crypto_llm_data/data/encrypted/${data}_${lang}_${encryption}_no_encryption.jsonl \
    --data_name ${data}_${lang}_${encryption}_no_encryption \
    --unique_id ${unique_id} \
    --global_batch_size ${global_batch_size} \
    --batch_size ${batch_size} \
    --seq_len ${seq_len} \
    --precision ${precision} \
    --mp_size ${mp_size} \
    --pp_size ${pp_size} \
    --zero_stage ${zero_stage} \
    --min_lr ${min_lr} \
    --jobname ${jobname} \
    --tinyllama true