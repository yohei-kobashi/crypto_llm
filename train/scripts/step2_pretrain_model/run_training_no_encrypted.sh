#!/bin/bash

# Command line options go here
#SBATCH --time=24:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-8
#SBATCH --job-name=training_no_encrypted
#SBATCH --output=run_training_no_encrypted.out
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --ntasks=208

source ~/miniconda3/etc/profile.d/conda.sh
conda activate .venv_train
cd ~/ucllm_nedo_dev/train/scripts/step2_pretrain_model/
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
host="${HOSTNAME}"
model_num="1"
lang="en"
encryption="no_encrypted"
output_model_dir="/storage9/encrypted_llm/models/${model_num}.${lang}_${encryption}/"
unique_id="${host}_${current_time}"
bash ./gcp_honban_node-1_gpu-8/zero-0_dp-8_pp-1_tp-1_precision-fp32_flashattn2-on.sh \
    --input_tokenizer_file /storage9/encrypted_llm/tokenizers/tokenizer-wikipedia-${lang}-${encryption}.model \
    --output_model_dir ${output_model_dir} \
    --save_interval 500 \
    --wandb_entity yohei-kobashi \
    --wandb_project encrypted_data_LLM \
    --data_base_dir /storage9/encrypted_llm/datasets \
    --train_data_file /storage9/encrypted_llm/datasets/wikipedia/${lang}_${encryption}.jsonl \
    --data_name wikipedia_${lang}_${encryption} \
    --unique_id ${unique_id}
# bash ./gcp_honban_node-1_gpu-8/zero-0_dp-8_pp-1_tp-1_precision-fp32_flashattn2-on.sh \
#     --input_tokenizer_file /storage9/encrypted_llm/tokenizers/tokenizer-wikipedia-${lang}-${encryption}.model \
#     --output_model_dir ${output_model_dir} \
#     --save_interval 500 \
#     --wandb_entity yohei-kobashi \
#     --wandb_project encrypted_data_LLM \
#     --data_base_dir /storage9/encrypted_llm/datasets \
#     --train_data_file /storage9/encrypted_llm/datasets/encrypted/wikipedia_${lang}_${encryption}_no_encryption.jsonl \
#     --data_name wikipedia_${lang}_${encryption}_no_encryption \
#     --unique_id ${unique_id}