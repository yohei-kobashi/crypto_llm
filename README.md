# Crypto LLM: Two-Stage Language Model Pre-training with Ciphered and Natural Language Data

This repository has scripts to prepare training subset for each condition, train tokenizer used in encoding plain or encrypted text and extract presudo-pii data from existing training subset. We also included configuration files of [Meta Lingua](https://github.com/facebookresearch/lingua)

## Preliminaries
 - Training data:  sample 10BT of [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
   - downloaded parquet files under `./fineweb_edu_10bt`
 - Computational Resource: 2 node with 8 H200 GPUs each

## Prepare training subset
### Collect person name from the entire data
```
python script/preprocess/extract_name.py fineweb_edu_10bt/ extract_names_from_fwe10b
```
### Extract presudo-pii samples
```
python script/preprocess/filter_name.py fineweb_edu_10bt/ extract_names_from_fwe10b
```
### Split into pre-training and continual pre-training data & Encryption
We recommend nchunks=1 because meta lingua ignores jsonl files if the number of jsonl files is bigger than world size at training.
```
python3 script/preprocess/preprocess_dropped.py data_dropped --data_dir working_dir/datatrove/extract_names_from_fwe10b_n1 --nchunks 1

python3 script/preprocess/preprocess_final.py data_final --data_dir working_dir/datatrove/extract_names_from_fwe10b_n1 --sample_ratio 0.5 --nchunks 1
```
So far, we obtain following subsets on `./working_dir`
| Name | Train type | Text Type | Presudo-PII? | Target |
|----|----|----|----|----|
| `data_dropped_chunk` | Pre-training | Plain | Yes | Plain-LLM 1 |
| `data_final_0.5_pretrain_chunk` | Pre-training | Plain | No | Plain-LLM 1 |
| `data_final_0.25_pretrain_chunk_01` | Pre-training` | Plain | No | Plain-LLM 1 |
| `data_final_0.25_continual2_chunk_00` | Continual pre-training | Plain | No | All |
| `encrypted_0.5_pretrain_chunk_00_alpha_poly_000001_1234_True` | Pre-training | Cipher(key_length=1) | No | Crypto-LLM 1 |
| `encrypted_0.5_pretrain_chunk_00_alpha_poly_000010_1234_True` | Pre-training | Cipher(key_length=10) | No | Crypto-LLM 2 |
| `encrypted_0.5_pretrain_chunk_00_alpha_poly_000100_1234_True` | Pre-training | Cipher(key_length=100) | No | Crypto-LLM 3 |
| `encrypted_0.25_pretrain_chunk_01_alpha_poly_000001_1234_True` | Pre-training | Cipher(key_length=1) | No | Crypto-LLM 1 |
| `encrypted_0.25_pretrain_chunk_01_alpha_poly_000010_1234_True` | Pre-training | Cipher(key_length=10) | No | Crypto-LLM 2 |
| `encrypted_0.25_pretrain_chunk_01_alpha_poly_000100_1234_True` | Pre-training | Cipher(key_length=100) | No | Crypto-LLM 3 |
| `encrypted_dropped_chunk.00_alpha_poly_000001_1234_True` | Pre-training | Cipher(key_length=1) | Yes | Crypto-LLM 1 |
| `encrypted_dropped_chunk.00_alpha_poly_000010_1234_True` | Pre-training | Cipher(key_length=10) | Yes | Crypto-LLM 2 |
| `encrypted_droppeds_chunk.00_alpha_poly_000100_1234_True` | Pre-training | Cipher(key_length=100) | Yes | Crypto-LLM 3 |
## Train sentencepiece tokenizers
Here, we show how to train tokenizer for plain text (Plain-LLM 1&2, Continual pre-training of Crypto-LLM)
Tokenizers for cipher text (key_length=1,10,100) can be trained by same method.
### Concat into raw txt
```
python script/tokenizer/make_raw_text.py \
    --input_dir working_dir/datatrove/extract_names_from_fwe10b/data_dropped_chunked working_dir/datatrove/extract_names_from_fwe10b/data_final_0.5_pretrain_chunked \
    --probabilities 1.0 1.0 \
    --output working_dir/tokenizer/raw_plain_text.txt
```
### Train spm
```
python script/tokenizer/train_spm.py \
    working_dir/tokenizer/raw_plain_text.txt \
    --model_prefix sp_model_pt_plain \
    --model_type unigram \
    --byte_fallback \
    --split_digits \
    --allow_whitespace_only_pieces
```
### Eval tokenizer
```
python script/tokenizer/eval_tokenization.py \
    --input_dir working_dir/datatrove/extract_names_from_fwe10b/data_dropped_chunk working_dir/datatrove/extract_names_from_fwe10b/data_final_0.5_pretrain_chunk \
    --probabilities 1.0 1.0 \
    --model script/tokenizer/poly1/sp_model_pt_plain.model
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