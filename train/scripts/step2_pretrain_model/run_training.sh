#!/bin/bash

#$ -l rt_AF=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -cwd

export LD_LIBRARY_PATH=/home/acf16449gb/miniconda3/envs/.venv_data/lib:/home/acf16449gb/ucllm_nedo_dev/train/.venv_train/lib
source /etc/profile.d/modules.sh
module load python/3.11/3.11.9 cuda/11.8/11.8.0 hpcx/2.12
source ~/ucllm_nedo_dev/train/.venv_train/bin/activate
cd ~/ucllm_nedo_prod/train/scripts/step2_pretrain_model/
host="${HOSTNAME}"
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
model_num="0"
data="wikipedia"
lang="latin"

# hyper parameter tuning
global_batch_size=("384","512","1024","1536","1024","1024","1024","1024")
batch_size=("1","1","1","1","2","4","1","1")
seq_len=("2048","2048","2048","2048","2048","2048","2048","4096")
precision=("fp32","fp32","fp32","fp32","fp32","fp32","tf32","tf32")
len=${#global_batch_size[@]}
encryption="no_encryption_000000_1234_True"
unique_id="${model_num}.${lang}_${data}_${encryption}"
output_model_dir="/groups/gcf51099/crypto_llm/models/${unique_id}/"
for ((i=0; i<$len; i++)); do
    bash ./abci_node-1_gpu-1/zero-0_dp-8_pp-1_tp-1_precision-fp32_flashattn2-on \
        --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
        --output_model_dir ${output_model_dir} \
        --save_interval 500 \
        --train_iters 100 \
        --wandb_entity yohei-kobashi \
        --wandb_project encrypted_data_LLM \
        --wandb_tag other_gpu \
        --data_base_dir /groups/gcf51099/crypto_llm/data \
        --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}-${lang}-${encryption}.jsonl \
        --data_name ${data}_${lang}_${encryption} \
        --unique_id ${unique_id}-${i} \
        --global_batch_size ${global_batch_size[$i]} \
        --batch_size ${batch_size[$i]} \
        --seq_len ${seq_len[$i]} \
        --precision ${precision[$i]}
done