#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=01:00:00
#$ -j y
#$ -cwd

export LD_LIBRARY_PATH=/home/acf16449gb/miniconda3/envs/.venv_data/lib:/home/acf16449gb/crypto_llm/train/.venv_train/lib
source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 hpcx/2.12
source ~/crypto_llm/train/.venv_train/bin/activate

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=6006

lang="latin"
corpus="wikipedia"
encryption="poly_000000_1234_True"

python test_model.py \
    --override-opt_param-scheduler \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tensor-model-parallel-size 1 \
    --init-method-std 0.013 \
    --lr-decay-tokens $((300 * 1000 * 1000 * 1000)) \
    --lr-warmup-tokens $((3000 * 1000 * 1000)) \
    --micro-batch-size 8 \
    --exit-duration-in-mins 30000000 \
    --global-batch-size 4096 \
    --num-layers 22 \
    --hidden-size 2048 \
    --ffn-hidden-size 5632 \
    --num-attention-heads 16 \
    --num-key-value-heads 4 \
    --no-query-key-layer-scaling \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --normalization rmsnorm \
    --disable-bias-linear \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-tokens $((4096000 * 2048)) \
    --train-samples 4096000 \
    --lr 2.0e-4 \
    --min-lr 1.0e-6 \
    --lr-decay-style cosine \
    --split 949,50,1 \
    --log-interval 10 \
    --eval-interval 100 \
    --eval-iters 1000 \
    --save-interval 100 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --hysteresis 2 \
    --num-workers 0 \
    --seed 1234 \
    --load /groups/gcf51099/crypto_llm/models/megatron/1.${lang}_${corpus}_${encryption} \
    --save /groups/gcf51099/crypto_llm/models/megatron/1.${lang}_${corpus}_${encryption} \
    --no-async-tensor-model-parallel-allreduce \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model /groups/gcf51099/crypto_llm/tokenizers/tokenizer_${corpus}_${lang}_${encryption}.model \
    --sample-input-file /groups/gcf51099/crypto_llm/data/test/test_input.txt \
    --sample-output-file /groups/gcf51099/crypto_llm/data/test/test_output.txt

