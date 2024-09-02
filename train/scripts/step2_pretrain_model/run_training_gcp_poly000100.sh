#!/bin/bash

# Command line options go here
#SBATCH --time=30:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-3
#SBATCH --job-name=training_poly000100-pre
#SBATCH --output=run_training_poly000100-pre.out
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --ntasks=208

# model_num and jobname
model_num="1"
jobname="tinyllama-1.1B"

# hyper parameters
global_batch_size=4096
batch_size=8
seq_len=2048
precision="fp32"
mp_size=1
pp_size=1
zero_stage=0
min_lr=1.0e-6

# conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate .venv_train2
cd ~/crypto_llm/train/scripts/step2_pretrain_model/

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
output_model_dir="/storage9/crypto_llm/models/${unique_id}/"
# bash ./gcp_honban_node-1_gpu-8/flashattn2-on.sh \
#     --input_tokenizer_file /storage9/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
#     --output_model_dir ${output_model_dir} \
#     --save_interval 100 \
#     --train_iters 2000 \
#     --wandb_entity yohei-kobashi \
#     --wandb_project encrypted_data_LLM \
#     --data_base_dir /storage9/crypto_llm/data \
#     --train_data_file /storage9/crypto_llm/data/encrypted/${data}-${lang}-${encryption}.jsonl \
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
#     --train_data_exact_num_epochs 1

bash ./gcp_honban_node-1_gpu-8/flashattn2-on.sh \
    --input_tokenizer_file /storage9/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --output_model_dir ${output_model_dir} \
    --save_interval 10 \
    --train_iters 1010 \
    --wandb_entity yohei-kobashi \
    --wandb_project encrypted_data_LLM \
    --data_base_dir /storage9/crypto_llm/data/test \
    --train_data_file /storage9/crypto_llm/data/encrypted/${data}_${lang}_${encryption}.jsonl \
    --data_name ${data}_${lang}_${encryption} \
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
    --split 100,0,0

bash ./gcp_honban_node-1_gpu-8/flashattn2-on.sh \
    --input_tokenizer_file /storage9/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --output_model_dir ${output_model_dir} \
    --save_interval 100 \
    --train_iters 1500 \
    --wandb_entity yohei-kobashi \
    --wandb_project encrypted_data_LLM \
    --data_base_dir /storage9/crypto_llm/data \
    --train_data_file /storage9/crypto_llm/data/encrypted/${data}_${lang}_${encryption}_no_encryption.jsonl \
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
    --jobname ${jobname}
