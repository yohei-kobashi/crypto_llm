#!/bin/bash

#$ -l rt_F=8
#$ -l h_rt=24:00:00
#$ -l USE_SSH=1
#$ -j y
#$ -cwd

export LD_LIBRARY_PATH=/home/acf16449gb/miniconda3/envs/.venv_data/lib:/home/acf16449gb/ucllm_nedo_dev/train/.venv_train/lib
source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 hpcx/2.12
source ~/crypto_llm/train/.venv_train/bin/activate
cd ~/crypto_llm/train/scripts/common/
bash ./create_ssh_config_file_for_abci_multi_node_multi_gpu.sh
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd ~/crypto_llm/train/scripts/step2_pretrain_model/

host="${HOSTNAME}"
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
model_num="0"
# # no encryption
# data="wikipedia"
# lang="latin"
# encryption="no_encryption_000000_1234_True"
# poly_000001
data="wikipedia"
lang="latin"
encryption="poly_000001_1234_True"
# # poly_000100
# data="wikipedia"
# lang="latin"
# encryption="poly_000100_1234_True"
# # poly_010000
# data="wikipedia"
# lang="latin"
# encryption="poly_010000_1234_True"
# # poly_000000
# data="wikipedia"
# lang="latin"
# encryption="poly_000000_1234_True"

# hyper parameters
global_batch_size=128
batch_size=1
seq_len=2048
precision="fp16"
mp_size=1
pp_size=8
zero_stage=0

unique_id="${model_num}.${lang}_${data}_${encryption}"
output_model_dir="/groups/gcf51099/crypto_llm/models/${unique_id}/"
bash ./abci_node-8_gpu-32-v100/flashattn2-off.sh \
    --input_tokenizer_file /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
    --output_model_dir ${output_model_dir} \
    --save_interval 1000 \
    --train_iters 10000000 \
    --wandb_entity yohei-kobashi \
    --wandb_project encrypted_data_LLM \
    --wandb_tag other_gpu \
    --data_base_dir /groups/gcf51099/crypto_llm/data \
    --train_data_file /groups/gcf51099/crypto_llm/data/encrypted/${data}-${lang}-${encryption}.jsonl \
    --data_name ${data}_${lang}_${encryption} \
    --unique_id ${unique_id} \
    --global_batch_size ${global_batch_size} \
    --batch_size ${batch_size} \
    --seq_len ${seq_len} \
    --precision ${precision} \
    --mp_size ${mp_size} \
    --pp_size ${pp_size} \
    --zero_stage ${zero_stage}
