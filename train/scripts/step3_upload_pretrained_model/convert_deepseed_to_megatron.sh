#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=2:00:00
#$ -j y
#$ -cwd

export LD_LIBRARY_PATH=/home/acf16449gb/miniconda3/envs/.venv_data/lib:/home/acf16449gb/crypto_llm/train/.venv_train/lib
source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 hpcx/2.12
source ~/crypto_llm/train/.venv_train/bin/activate

set -e
echo ""

lang="latin"
corpus="wikipedia"
encryption="poly_000000_1234_True"

# Converts the pretrained model from Megatron-DeepSpeed format to HuggingFace Transformers format.
python /home/acf16449gb/crypto_llm/train/Megatron-DeepSpeed/tools/convert_checkpoint/deepspeed_to_megatron.py --target_tp 1 --target_pp 1 \
    --input_folder /groups/gcf51099/crypto_llm/models/1.latin_wikipedia_poly_000000_1234_True/checkpoint/tinyllama-1.1B/global_step1400 \
    --output_folder /groups/gcf51099/crypto_llm/models/megatron/1.latin_wikipedia_poly_000000_1234_True \
    --activation_function swiglu
