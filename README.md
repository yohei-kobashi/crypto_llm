# Crypto LLM: Two-Stage Language Model Pre-training with Ciphered and Natural Language Data

This repository consists of scripts to prepare training subset for each condition, train tokenizers used for encoding plain or encrypted text, and configuration files for training models. We also included configuration files for [Meta Lingua](https://github.com/facebookresearch/lingua).

## Preliminaries
 - Install libraries
    ```bash
    cd {repository_dir}
    pip install -r requirements.txt
    ```
 - Training data:  sample 10BT of [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
   - Download parquet files under `./fineweb_edu_10bt`
   - You can download it using `download_dataset.py`. This takes a few minutes.
   ```bash
   python3 download_dataset.py .
   ```
 - Computational Resource (for model training): 2 nodes with 8 H200 GPUs each

## Preparing training subset
### Collecting person names from the entire dataset
From fineweb-edu, detect person entities using spaCy, and save all detected names in the dataset into `./logs/datatrove/extract_names_from_fwe10b_extract/stats.json` 
```bash
python script/preprocess/extract_name.py fineweb_edu_10bt/ extract_names_from_fwe10b
```
### Extracting pseudo-PII samples
from `./logs/datatrove/extract_names_from_fwe10b_extract/stats.json` , sample names as pseudo-PII data and split the dataset into two subsets:
 - pseudo-PII (contains specific set of names): `working_dir/datatrove/extract_names_from_fwe10b/data_dropped`
 - remaining dataset: `working_dir/datatrove/extract_names_from_fwe10b/data_final`


```bash
python script/preprocess/filter_name.py fineweb_edu_10bt/ extract_names_from_fwe10b
```

The set of pseudo-PII names are saved into `./logs/datatrove/extract_names_from_fwe10b_extract/stats.json_drop_0.json`
### Spliting into pre-training and continual pre-training data & Chunking
We recommend `nchunks=1` because Meta Lingua ignores jsonl files if the number of jsonl files is bigger than world size at training.
```bash
python3 script/preprocess/preprocess_dropped.py data_dropped --data_dir working_dir/datatrove/extract_names_from_fwe10b --nchunks 1

python3 script/preprocess/preprocess_final.py data_final --data_dir working_dir/datatrove/extract_names_from_fwe10b --sample_ratio 0.75 --nchunks 1
```
For `preprocess_final.py`, `--sample_ratio` specifies how much data to use for pre-training. The above example utilizes 75% of data under `data_final` for pre-training and the remaining 25% for continual pre-training.

We can obtain the following subsets:
```bash
# pseudo-PII data
working_dir/datatrove/extract_names_from_fwe10b/data_dropped_chunked

# normal pre-training and continual pre-training dataset
working_dir/datatrove/extract_names_from_fwe10b/data_final_0.75_pretrain_chunked
working_dir/datatrove/extract_names_from_fwe10b/data_final_0.75_continual_chunked
```
This script generates encrypted data. However, this generated encrypted data by this script are deprecated.
### Encrypting training data
Run the following commands to encrypt training data:

```bash
# for key_length=1
python3 script/preprocess/encrypt_jsonl.py \
    --lang alpha \
    --input_file working_dir/datatrove/extract_names_from_fwe10b/data_dropped_chunked/data_dropped.chunk.00.jsonl \
    --output_dir working_dir/datatrove/extract_names_from_fwe10b/encrypted_dropped_chunk_00_alpha_poly_000001_1234_True \
    --key_length 1 --seed 1234

python3 script/preprocess/encrypt_jsonl.py \
    --lang alpha \
    --input_file working_dir/datatrove/extract_names_from_fwe10b/data_final_0.75_pretrain_chunked/data_final.chunk.00.jsonl \
    --output_dir working_dir/datatrove/extract_names_from_fwe10b/encrypted_0.75_pretrain_chunk_00_alpha_poly_000001_1234_True \
    --key_length 1 --seed 1234

# for key_length=10
python3 script/preprocess/encrypt_jsonl.py \
    --lang alpha \
    --input_file working_dir/datatrove/extract_names_from_fwe10b/data_dropped_chunked/data_dropped.chunk.00.jsonl \
    --output_dir working_dir/datatrove/extract_names_from_fwe10b/encrypted_dropped_chunk_00_alpha_poly_000010_1234_True \
    --key_length 10 --seed 1234

python3 script/preprocess/encrypt_jsonl.py \
    --lang alpha \
    --input_file working_dir/datatrove/extract_names_from_fwe10b/data_final_0.75_pretrain_chunked/data_final.chunk.00.jsonl \
    --output_dir working_dir/datatrove/extract_names_from_fwe10b/encrypted_0.75_pretrain_chunk_00_alpha_poly_000010_1234_True \
    --key_length 10 --seed 1234

# for key_length=100
python3 script/preprocess/encrypt_jsonl.py \
    --lang alpha \
    --input_file working_dir/datatrove/extract_names_from_fwe10b/data_dropped_chunked/data_dropped.chunk.00.jsonl \
    --output_dir working_dir/datatrove/extract_names_from_fwe10b/encrypted_dropped_chunk_00_alpha_poly_000100_1234_True \
    --key_length 100 --seed 1234

python3 script/preprocess/encrypt_jsonl.py \
    --lang alpha \
    --input_file working_dir/datatrove/extract_names_from_fwe10b/data_final_0.75_pretrain_chunked/data_final.chunk.00.jsonl \
    --output_dir working_dir/datatrove/extract_names_from_fwe10b/encrypted_0.75_pretrain_chunk_00_alpha_poly_000100_1234_True \
    --key_length 100 --seed 1234

```
**[IMPORTANT]** Meta Lingua recognize files matching `*.chunk.*.jsonl` as a training data. After this process, please rename the jsonl file satisfying this.

So far, we have obtained following subsets on `./working_dir/datatrove/extract_names_from_fwe10b`
| Name | Train type | Text Type | Pseudo-PII? | Target |
|----|----|----|----|----|
| `data_dropped_chunked` | Pre-training | Plain | Yes | Plain-LLM 1 |
| `encrypted_dropped_chunk_00_alpha_poly_000001_1234_True` | Pre-training | Cipher(key_length=1) | Yes | Crypto-LLM 1 |
| `encrypted_dropped_chunk_00_alpha_poly_000010_1234_True` | Pre-training | Cipher(key_length=10) | Yes | Crypto-LLM 2 |
| `encrypted_dropped_chunk_00_alpha_poly_000100_1234_True` | Pre-training | Cipher(key_length=100) | Yes | Crypto-LLM 3 |
| `data_final_0.75_pretrain_chunked` | Pre-training | Plain | No | Plain-LLM 1 |
| `encrypted_0.75_pretrain_chunk_00_alpha_poly_000001_1234_True` | Pre-training | Cipher(key_length=1) | No | Crypto-LLM 1 |
| `encrypted_0.75_pretrain_chunk_00_alpha_poly_000010_1234_True` | Pre-training | Cipher(key_length=10) | No | Crypto-LLM 2 |
| `encrypted_0.75_pretrain_chunk_00_alpha_poly_000100_1234_True` | Pre-training | Cipher(key_length=100) | No | Crypto-LLM 3 |
| `data_final_0.75_continual_chunked` | Continual pre-training | Plain | No | All 

## Training sentencepiece tokenizers
Here, we show how to train a tokenizer for plain text (Plain-LLM 1&2, Continual pre-training of Crypto-LLM)
Tokenizers for encrypted text (key_length=1,10,100) can be trained using the same method.
### Concatenating data into a raw text file
SentencePiece tokenizer can be trained by multiple sources. However, we need to concat jsonl files into a single text file so we can avoid the command-line argument length constraints.
```
python script/tokenizer/make_raw_text.py \
    --input_dir working_dir/datatrove/extract_names_from_fwe10b/data_dropped_chunked working_dir/datatrove/extract_names_from_fwe10b/data_final_0.75_pretrain_chunked \
    --probabilities 1.0 1.0 \
    --output working_dir/tokenizer/raw_plain_text.txt
```
### Training spm
```
python script/tokenizer/train_spm.py \
	working_dir/tokenizer/raw_plain_text.txt \
	--model_prefix plain \
	--vocab_size 32000 \
	--input_sentence_size 3000000 \
	--shuffle_input_sentence \
	--model_type bpe \
	--num_threads 16
```
### Evaluating tokenizer
This script evaluates tokenizers by counting tokens and characters.
```
python script/tokenizer/eval_tokenization.py \
    --input_dir working_dir/datatrove/extract_names_from_fwe10b/data_dropped_chunked working_dir/datatrove/extract_names_from_fwe10b/data_final_0.75_pretrain_chunked \
    --probabilities 1.0 1.0 \
    --model plain.model
```
## Training configuration of Meta Lingua
Move all training subset and tokenizer under `./data/cryptollm_exp5`.

Following the official Meta Lingua repository, locate configuration files from `lingua_config` to `{lingua-repo-dir}/apps/main/config`. The letters appearing in the configuration filenames under `lingua_config` correspond to the following model types:
| Model Name | Configuration Label | Key Length |
|----|----|----|
| Plain-LLM 1 | b | - |
| Plain-LLM 2 | c | - |
| Crypto-LLM 1 | a | 1 |
| Crypto-LLM 2 | a | 10 |
| Crypto-LLM 3 | a | 100 |

`pt` means pre-training and `ft` means continual pre-training. 

For continual pre-training, please edit `checkpoint.init_ckpt_path` in the configuration file to the correct path. Specifying `cryptollm_llama_*.yaml` configuration file when running Meta Lingua's `python -m apps.main.train` trains Crypto-LLM and Plain-LLM.

## Extract pseudo-PII samples from continual pre-training data
To compare the difficulty of restoring encrypted versus plain data, we extract additional pseudo-PII samples from the continual pre-training dataset. Since extracting names from the entire dataset is time-consuming, we extract samples from the first 20,000 entries only.
```
python script/train/extract_another_pii.py \
    {cryptollm_path}/working_dir/datatrove/extract_names_from_fwe10b/data_final_0.75_continual_chunked \
    extract_names_from_fwe10b \
    --skip 0 --limit 20000 \
    --drop_ratio 0.8 --seed 1
```
After this, move `working_dir/datatrove/extract_names_from_fwe10b/extract_names_from_fwe10b_b0_e20000_seed1` under `./data/cryptollm_exp5`.

# Evaluate trained models
## PII Perplexity
To evaluate Crypto-LLM and Plain-LLM, modify the `eval_*.yaml` configuration file. Especially, set the `ckpt_dir` variable to the correct checkpoint directory. Then, run `python -m apps.main.eval` on Meta Lingua to evaluate perplexity.

## Reconstruction Attack
### Place scripts into lingua's repository
Copy `lingua_apps/crypto_llm` into Meta Lingua's `apps`. `lingua_apps/crypto_llm` contains scripts to conduct reconstruction attack.
### Convert PII jsonl
This script splits each sentence in pii_jsonl_path into prefix, PII entiry and suffix. The splited sentences are dumped into a new jsonl file.
```bash
python -m apps.crypto_llm.separate_pii \
    --input_path {pii_jsonl_path} \
    --output_path {converted_pii_jsonl_path} \
    --sp_model_path data/cryptollm_exp5/plain.model
```
### Running the reconstruction attack
Once you convert PII data, you can evaluate pretrained model by reconstruction attack. To reproduce the results of the paper, run the following command:
```bash
python -m apps.crypto_llm.eval_reconstruction \
    ckpt={ckpt_path}/consolidated \
    pii_jsonl_path={converted_pii_jsonl_path} \
    pii_num=1000
```

## True-Prefix Attack
### Running the true-prefix attack
You can evaluate pretrained model by true-prefix attack with the same convert PII data. To reproduce the results of the paper, run the following command:
```bash
python -m apps.crypto_llm.eval_true_prefix_attack \
    --input_path {converted_pii_jsonl_path} \
    --output_path {results_of_true-prefix_attack_jsonl} \
    --model {model_id} \
    --pii_num=1000 \
    --N_sampling 64
```
The "model_id" must be chosen from ["a_1", "a_10", "a_100", "b", "c"].

### Preparing Not trained PII data
First, download the 100BT sample dataset from [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
Next, run the following script to extract texts that were not included in the 10BT subset:
```bash
cd {repository_dir} 
python script/preprocess/get_not_learned_texts_from_fwe_100B.py \
    --dir100bt {dir_containing_100BT_parquet_files} \
    --dir10bt {dir_containing_10BT_parquet_files} \
    --num-procs 20 \
    --sampling-rate 0.01 \
    --seed 42
```
Finally, you can extract and convert pseudo-PII data from these texts using the same procedure as applied to the 10BT data.