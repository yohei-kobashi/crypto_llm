#!/bin/bash

# Command line options go here
#SBATCH --time=12:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-4
#SBATCH --job-name=tokenize_poly_00s
#SBATCH --output=run_training_poly_00s.out
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --ntasks=208

encryption="en_poly_00s_block_000"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate .venv_train
cd ~/ucllm_nedo_dev/train/scripts/step1_train_tokenizer/
python ./train_sentencepiece_tokenizer.py \
    --input  /storage9/encrypted_llm/datasets/encrypted/wikipedia_${encryption}.jsonl,/storage9/encrypted_llm/datasets/encrypted/wikipedia_${encryption}_no_encryption.jsonl \
    --model_prefix tokenizer-wikipedia-${encryption} \
    --vocab_size 32000 \
    --input_sentence_size 3000000 \
    --shuffle_input_sentence True \
    --num_threads 208