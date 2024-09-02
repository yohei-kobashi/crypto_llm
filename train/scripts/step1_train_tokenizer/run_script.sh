#!/bin/bash

#$ -l rt_C.large=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh && module purge
module load python/3.11 cuda/11.8 hpcx/2.12
source ~/ucllm_nedo_dev/train/.venv_train/bin/activate

cd ~/crypto_llm/train/scripts/step1_train_tokenizer/
# no encryption
python ./train_sentencepiece_tokenizer.py \
    --input /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True.jsonl,/groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True_no_encryption.jsonl \
    --model_prefix tokenizer_wikipedia_latin_no_encryption_000000_1234_True \
    --vocab_size 32000 \
    --input_sentence_size 3000000 \
    --shuffle_input_sentence True \
    --num_threads 16
# # poly, key length == 1
# python ./train_sentencepiece_tokenizer.py \
#     --input /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_poly_000001_1234_True.jsonl,/groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True_no_encryption.jsonl \
#     --model_prefix tokenizer_wikipedia_latin_poly_000001_1234_True \
#     --vocab_size 32000 \
#     --input_sentence_size 3000000 \
#     --shuffle_input_sentence True \
#     --num_threads 20
# # poly, key length == 10
# python ./train_sentencepiece_tokenizer.py \
#     --input /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_poly_000010_1234_True.jsonl,/groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True_no_encryption.jsonl \
#     --model_prefix tokenizer_wikipedia_latin_poly_000010_1234_True \
#     --vocab_size 32000 \
#     --input_sentence_size 3000000 \
#     --shuffle_input_sentence True \
#     --num_threads 20
# # poly, key length = 100
# python ./train_sentencepiece_tokenizer.py \
#     --input /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_poly_000100_1234_True.jsonl,/groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True_no_encryption.jsonl \
#     --model_prefix tokenizer_wikipedia_latin_poly_000100_1234_True \
#     --vocab_size 32000 \
#     --input_sentence_size 3000000 \
#     --shuffle_input_sentence True \
#     --num_threads 20
# # poly, key length = 1000
# python ./train_sentencepiece_tokenizer.py \
#     --input /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_poly_001000_1234_True.jsonl,/groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True_no_encryption.jsonl \
#     --model_prefix tokenizer_wikipedia_latin_poly_001000_1234_True \
#     --vocab_size 32000 \
#     --input_sentence_size 3000000 \
#     --shuffle_input_sentence True \
#     --num_threads 20
# # poly, key length = 10000
# python ./train_sentencepiece_tokenizer.py \
#     --input /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_poly_010000_1234_True.jsonl,/groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True_no_encryption.jsonl \
#     --model_prefix tokenizer_wikipedia_latin_poly_010000_1234_True \
#     --vocab_size 32000 \
#     --input_sentence_size 3000000 \
#     --shuffle_input_sentence True \
#     --num_threads 20
# # poly, key length = 100000
# python ./train_sentencepiece_tokenizer.py \
#     --input /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_poly_100000_1234_True.jsonl,/groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True_no_encryption.jsonl \
#     --model_prefix tokenizer_wikipedia_latin_poly_100000_1234_True \
#     --vocab_size 32000 \
#     --input_sentence_size 3000000 \
#     --shuffle_input_sentence True \
#     --num_threads 20
# # poly, key length = 0 (max)
# python ./train_sentencepiece_tokenizer.py \
#     --input /groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_poly_000000_1234_True.jsonl,/groups/gcf51099/crypto_llm/data/encrypted/wikipedia_latin_no_encryption_000000_1234_True_no_encryption.jsonl \
#     --model_prefix tokenizer_wikipedia_latin_poly_000000_1234_True \
#     --vocab_size 32000 \
#     --input_sentence_size 3000000 \
#     --shuffle_input_sentence True \
#     --num_threads 20