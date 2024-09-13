#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -cwd

export LD_LIBRARY_PATH=/home/acf16449gb/miniconda3/envs/.venv_data/lib:/home/acf16449gb/crypto_llm/train/.venv_train/lib
source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 hpcx/2.12
source ~/crypto_llm/train/.venv_train/bin/activate

set -e
echo ""

python /home/acf16449gb/crypto_llm/train/Megatron-DeepSpeed/tools/convert_checkpoint/inspect_deepspeed_checkpoint.py \
    --folder /groups/gcf51099/crypto_llm/models/1.latin_wikipedia_no_encryption_000000_1234_True/checkpoint/tinyllama-1.1B/global_step1500/
