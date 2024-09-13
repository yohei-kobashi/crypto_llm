#!/bin/bash

#$ -l rt_AF=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh && module purge
module load python/3.11 cuda/11.8 hpcx/2.12
source ~/ucllm_nedo_dev/train/.venv_train/bin/activate

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
# # poly_000100
# data="wikipedia"
# lang="latin"
# encryption="poly_000100_1234_True"
# # poly_010000
# data="wikipedia"
# lang="latin"
# encryption="poly_010000_1234_True"
# poly_000000
data="wikipedia"
lang="latin"
encryption="poly_000000_1234_True"

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

# bash ./gcp_honban_node-1_gpu-8/flashattn2-on.sh \
#     --input_tokenizer_file /storage9/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
#     --output_model_dir ${output_model_dir} \
#     --save_interval 10 \
#     --train_iters 1010 \
#     --wandb_entity yohei-kobashi \
#     --wandb_project encrypted_data_LLM \
#     --data_base_dir /storage9/crypto_llm/data/test \
#     --train_data_file /storage9/crypto_llm/data/encrypted/${data}_${lang}_${encryption}.jsonl \
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

# bash ./gcp_honban_node-1_gpu-8/flashattn2-on.sh \
#     --input_tokenizer_file /storage9/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
#     --output_model_dir ${output_model_dir} \
#     --save_interval 100 \
#     --train_iters 1400 \
#     --wandb_entity yohei-kobashi \
#     --wandb_project encrypted_data_LLM \
#     --data_base_dir /storage9/crypto_llm/data \
#     --train_data_file /storage9/crypto_llm/data/encrypted/${data}_${lang}_${encryption}_no_encryption.jsonl \
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