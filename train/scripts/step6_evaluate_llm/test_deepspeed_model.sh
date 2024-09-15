#!/bin/bash

#$ -l rt_AG.small=1
#$ -l h_rt=01:00:00
#$ -j y
#$ -cwd

export LD_LIBRARY_PATH=/home/acf16449gb/miniconda3/envs/.venv_data/lib:/home/acf16449gb/crypto_llm/train/.venv_train/lib
source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 hpcx/2.12
source ~/crypto_llm/train/.venv_train/bin/activate

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=6006
export CUDA_DEVICE_MAX_CONNECTIONS=1
export LOCAL_RANK=0

lang="latin"
corpus="wikipedia"
encryption="no_encryption_000000_1234_True"
global_step="global_step1000"

python test_deepspeed_model.py \
    --micro-batch-size 1 \
    --num-layers 22 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --num-key-value-heads 4 \
    --max-position-embeddings 2048 \
    --ffn-hidden-size 5632 \
    --no-query-key-layer-scaling \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --normalization rmsnorm \
    --disable-bias-linear \
    --seq-length 2048 \
    --out-seq-length 50 \
    --load /groups/gcf51099/crypto_llm/models/1.${lang}_${corpus}_${encryption}/checkpoint/tinyllama-1.1B/${global_step} \
    --use-flash-attn-v2 \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${corpus}_${lang}_${encryption}.model \
    --inference
