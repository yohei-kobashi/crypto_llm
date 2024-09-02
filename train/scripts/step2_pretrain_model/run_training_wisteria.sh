#!/bin/bash

#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=2:00:00
#PJM -g gb20
#PJM -j

###PJM -L node=1
set -e

echo "* Loading modules"
module load gcc/8.3.1
module load cuda/11.8
module load cudnn/8.8.0  
# module load singularity/3.7.3
# module load gcc/8.3.1 python/3.11.7 cuda/11.8 hpcx/2.15.0
# source /work/gb20/b20048/crypto_llm/train/.venv_train/bin/activate

# condaによる環境起動
export WORK_HOME=/work/gb20/b20048
export CONDARC=$WORK_HOME/miniconda3/.condarc
export CONDA_ROOT=$WORK_HOME/miniconda3
export CONDA_PREFIX=$WORK_HOME/miniconda3
export HOME=$WORK_HOME
export CONDA_ENVS_PATH=$WORK_HOME/miniconda3
source /work/gb20/b20048/miniconda3/etc/profile.d/conda.sh
conda activate deep

cd /work/gb20/b20048/crypto_llm/train/scripts/step2_pretrain_model

export TRITON_CACHE_DIR=/work/gb20/b20048/data_cache
export HF_HOME=/work/gb20/b20048/data_cache
# export NCCL_SOCKET_NTHREADS=8
# export NCCL_SOCKET_IFNAME=ens85f0
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=0
# export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO

# --- デバッグ用コマンドを追加 ---
echo "Checking NCCL configuration..."
python -c "import torch; print(torch.version.cuda)"
nvidia-smi
# --- ここまで ---

host="${HOSTNAME}"
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
model_num="0"
data="wikipedia"
lang="latin"

encryption="poly_000001_1234_True"
unique_id="${model_num}.${lang}_${data}_${encryption}"
output_model_dir="/work/gb20/b20048/crypto_llm/models/${unique_id}/"

cd /work/gb20/b20048/crypto_llm/train/Megatron-DeepSpeed/examples_deepspeed/rebase/
bash ds_pretrain_gpt_125M.sh
# hyper parameter tuning
# global_batch_size=(512 1024 1536)
# batch_size=(1 2 4)
# seq_len=(2048)
# precision=("fp32" "tf32")
# mp_size=(1 2)
# pp_size=(1 2)
# global_batch_size=(512)
# batch_size=(1)
# seq_len=(2048)
# precision=("fp32")
# mp_size=(1)
# pp_size=(1)
# for ((i=0; i<${#global_batch_size[@]}; i++)); do
#     for ((j=0; j<${#batch_size[@]}; j++)); do
#         for ((k=0; k<${#seq_len[@]}; k++)); do
#             for ((l=0; l<${#precision[@]}; l++)); do
#                 for ((m=0; m<${#mp_size[@]}; m++)); do
#                     for ((n=0; n<${#pp_size[@]}; n++)); do
#                         bash ./gcp_honban_node-1_gpu-8/llama2_flashattn2-on.sh \
#                             --input_tokenizer_file /work/gb20/b20048/crypto_llm/tokenizers/tokenizer_${data}_${lang}_${encryption}.model \
#                             --output_model_dir ${output_model_dir} \
#                             --save_interval 500 \
#                             --train_iters 50 \
#                             --wandb_entity yohei-kobashi \
#                             --wandb_project encrypted_data_LLM \
#                             --wandb_tag other_gpu \
#                             --data_base_dir /work/gb20/b20048/crypto_llm/data \
#                             --train_data_file /work/gb20/b20048/crypto_llm/data/encrypted/${data}-${lang}-${encryption}.jsonl \
#                             --data_name ${data}_${lang}_${encryption} \
#                             --unique_id ${unique_id}-${i}-${j}-${k}-${l}-${m}-${n} \
#                             --global_batch_size ${global_batch_size[$i]} \
#                             --batch_size ${batch_size[$j]} \
#                             --seq_len ${seq_len[$k]} \
#                             --precision ${precision[$l]} \
#                             --mp_size ${mp_size[$m]} \
#                             --pp_size ${pp_size[$n]}
#                     done
#                 done
#             done
#         done
#     done
# done
