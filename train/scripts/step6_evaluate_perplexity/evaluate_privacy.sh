#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=0:02:00
#$ -j y
#$ -cwd

export LD_LIBRARY_PATH=/home/acf16449gb/miniconda3/envs/.venv_data/lib:/home/acf16449gb/crypto_llm/train/.venv_train/lib
source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 hpcx/2.12
source ~/crypto_llm/train/.venv_train/bin/activate

set -e
echo ""

python evaluate_privacy.py \
    --model_dir /groups/gcf51099/crypto_llm/models/megatron/1.latin_wikipedia_poly_000000_1234_True/iter_0001400/mp_rank_00/