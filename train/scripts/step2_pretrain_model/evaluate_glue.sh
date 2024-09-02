#!/bin/bash

# Command line options go here
#SBATCH --time=30:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-5
#SBATCH --job-name=training_poly000000-eval
#SBATCH --output=run_training_poly000000-eval.out
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --ntasks=208

# conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate .venv_train2
cd ~/crypto_llm/train/scripts/step2_pretrain_model/

# vars
checkpoint="/storage9/crypto_llm/models/1.latin_wikipedia_poly_000000_1234_True/checkpoint/tinyllama-1.1B/global_step1500"

python ~/crypto_llm/crypto_llm/train/Megatron-DeepSpeed/tasks/main.py \
    --task MNLI \
    --epochs 2 \
    --pretrained-checkpoint ${checkpoint}

python ~/crypto_llm/crypto_llm/train/Megatron-DeepSpeed/tasks/main.py \
    --task QQP \
    --epochs 2 \
    --pretrained-checkpoint ${checkpoint}

python ~/crypto_llm/crypto_llm/train/Megatron-DeepSpeed/tasks/main.py \
    --task QNLI \
    --epochs 2 \
    --pretrained-checkpoint ${checkpoint}

python ~/crypto_llm/crypto_llm/train/Megatron-DeepSpeed/tasks/main.py \
    --task SST-2 \
    --epochs 2 \
    --pretrained-checkpoint ${checkpoint}

python ~/crypto_llm/crypto_llm/train/Megatron-DeepSpeed/tasks/main.py \
    --task CoLA \
    --epochs 2 \
    --pretrained-checkpoint ${checkpoint}

python ~/crypto_llm/crypto_llm/train/Megatron-DeepSpeed/tasks/main.py \
    --task STS-B \
    --epochs 2 \
    --pretrained-checkpoint ${checkpoint}

python ~/crypto_llm/crypto_llm/train/Megatron-DeepSpeed/tasks/main.py \
    --task MRPC \
    --epochs 2 \
    --pretrained-checkpoint ${checkpoint}

python ~/crypto_llm/crypto_llm/train/Megatron-DeepSpeed/tasks/main.py \
    --task RTE \
    --epochs 2 \
    --pretrained-checkpoint ${checkpoint}
