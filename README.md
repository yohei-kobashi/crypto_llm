# Crypto LLM: Two-Stage Language Model Pre-training with Ciphered and Natural Language Data

This repository has scripts to prepare training subset for each condition, train tokenizer used in encoding plain or encrypted text and extract presudo-pii data from existing training subset. We also included configuration files of [Meta Lingua](https://github.com/facebookresearch/lingua)

## Preliminaries
As a training data, we choose sample 10BT of [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

## Prepare training subset
### Collect person name from the entire data
### Extract presudo-pii samples
### Split into pre-training and continual pre-training data & Encryption

## Train sentencepiece tokenizers


## Training configuration of Meta Lingua
| Model Name | Configuration Name | Key Length |
|----|----|----|
| Plain-LLM 1 | b | - |
| Plain-LLM 2 | c | - |
| Crypto-LLM 1 | a | 1 |
| Crypto-LLM 2 | a | 10 |
| Crypto-LLM 3 | a | 100 |


## Extract presudo-pii samples from continual pre-training data