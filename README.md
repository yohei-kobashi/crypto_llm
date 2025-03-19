# Crypto LLM: Two-Stage Language Model Pre-training with Ciphered and Natural Language Data

This repository has scripts to prepare training subset for each condition, train tokenizer used in encoding plain or encrypted text and extract presudo-pii data from existing training subset. We also included configuration files of [Meta Lingua](https://github.com/facebookresearch/lingua)

## Preliminaries
As a training data, we choose sample 10BT of [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

## Prepare training subset
### Collect person name from the entire data
### Extract presudo-pii samples
### Split into pre-training and continual pre-training data & Encryption
We recommend nchunks=1 because meta lingua ignores jsonl files if the number of jsonl files is bigger than world size at training.
```
python3 script/preprocess/preprocess_dropped.py data_dropped --data_dir /home/uchiyama.fumiya/ucllm/cryptollm/working_dir/datatrove/extract_names_from_fwe10b_n1 --nchunks 1

python3 script/preprocess/preprocess_final.py data_final --data_dir /home/uchiyama.fumiya/ucllm/cryptollm/working_dir/datatrove/extract_names_from_fwe10b_n1 --sample_ratio 0.5 --nchunks 1
```
## Train sentencepiece tokenizers
### Concat into raw txt
#### Tokenizer for plain text (Plain-LLM 1&2, Continual pre-training of Crypto-LLM)
```
python script/tokenizer/make_raw_text.py \
    --input_dir working_dir/datatrove/extract_names_from_fwe10b/data_dropped_chunked working_dir/datatrove/extract_names_from_fwe10b/data_final_0.5_pretrain_chunked \
    --probabilities 1.0 1.0 \
    --output working_dir/tokenizer/raw_plain_text.txt
```
### Train spm
#### Crypto-LLM 1
```
python script/tokenizer/train_spm.py \
    working_dir/tokenizer/raw_poly1_text.txt \
    --model_prefix sp_model_pt_poly1_v2 \
    --model_type unigram \
    --byte_fallback \
    --split_digits \
    --allow_whitespace_only_pieces
```
#### Crypto-LLM 2
```
python script/tokenizer/train_spm.py \
    working_dir/tokenizer/raw_poly10_text.txt \
    --model_prefix sp_model_pt_poly10_v2 \
    --model_type unigram \
    --byte_fallback \
    --split_digits \
    --allow_whitespace_only_pieces
```
#### Crypto-LLM 3
```
python script/tokenizer/train_spm.py \
    working_dir/tokenizer/raw_poly100_text.txt \
    --model_prefix sp_model_pt_poly100_v2 \
    --model_type unigram \
    --byte_fallback \
    --split_digits \
    --allow_whitespace_only_pieces
```
#### Tokenizer for plain text (Plain-LLM 1&2, Continual pre-training of Crypto-LLM)
```
python script/tokenizer/train_spm.py \
    working_dir/tokenizer/raw_plain_text.txt \
    --model_prefix sp_model_pt_plain_v2 \
    --model_type unigram \
    --byte_fallback \
    --split_digits \
    --allow_whitespace_only_pieces
```
### Eval tokenizer
```
python script/tokenizer/eval_tokenization.py \
    --input_dir working_dir/datatrove/extract_names_from_fwe10b/data_dropped_encrypted_1_chunked working_dir/datatrove/extract_names_from_fwe10b/data_final_0.5_pretrain_encrypted_1_chunked \
    --probabilities 1.0 1.0 \
    --model script/tokenizer/poly1/sp_model_pt_poly1_v2.model
```
## Training configuration of Meta Lingua
| Model Name | Configuration Name | Key Length |
|----|----|----|
| Plain-LLM 1 | b | - |
| Plain-LLM 2 | c | - |
| Crypto-LLM 1 | a | 1 |
| Crypto-LLM 2 | a | 10 |
| Crypto-LLM 3 | a | 100 |


## Extract presudo-pii samples from continual pre-training data
```
python script/train/extract_another_pii.py \
    /home/uchiyama.fumiya/ucllm/cryptollm/working_dir/datatrove/extract_names_from_fwe10b_n1/data_final_0.5_continual_chunked \
    extract_names_from_fwe10b \
    --skip 1426808 --limit 1436808 \
    --drop_ratio 0.8 --seed 1
```