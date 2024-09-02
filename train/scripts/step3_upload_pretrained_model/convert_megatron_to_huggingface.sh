#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=0:10:00
#$ -j y
#$ -cwd

export LD_LIBRARY_PATH=/home/acf16449gb/miniconda3/envs/.venv_data/lib:/home/acf16449gb/crypto_llm/train/.venv_train/lib
source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 hpcx/2.12
source ~/crypto_llm/train/.venv_train/bin/activate

set -e
echo ""

python /home/acf16449gb/crypto_llm/train/.venv_train/lib/python3.11/site-packages/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py \
    /groups/gcf51099/crypto_llm/models/megatron/1.latin_wikipedia_poly_000000_1234_True/iter_0001400/mp_rank_00/model_optim_rng.pt \
    --print-checkpoint-structure \
    --activation_function swiglu \
    --tokenizer_path /groups/gcf51099/crypto_llm/tokenizers/tokenizer_wikipedia_latin_poly_000000_1234_True.model \
    --model_max_length 2048