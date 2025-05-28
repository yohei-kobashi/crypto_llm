# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import List, Optional

import torch
from torch import nn
from tqdm import tqdm

from omegaconf import OmegaConf
from torch.nn import functional as F
import xformers

from apps.main.transformer import LMTransformer, LMTransformerArgs
from lingua.args import dataclass_from_dict
# from lingua.checkpoint import CONSOLIDATE_NAME
from lingua.tokenizer import Tokenizer, build_tokenizer
from lingua.transformer import (
    Attention,
    causal_mask,
    generate_doc_mask_mod,
    lengths_to_local_ids,
    lengths_to_start_ids,
)
from torch.nn.attention.flex_attention import create_block_mask

import json
import argparse

import numpy as np
# from Levenshtein import distance as levenshtein_distance

import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


def extract_sentences_with_person(text):
    """
    Extract sentences containing PERSON named entities and split them
    into S0 (before name), name, and S1 (after name) segments. Also extract S1 keywords.

    Args:
        text (str): Input document text

    Returns:
        List[dict]: List of dictionaries containing S0, name, S1, S1_keywords
    """
    results = []
    doc = nlp(text)

    for sent in doc.sents:
        for ent in sent.ents:
            if ent.label_ == "PERSON":
                name = ent.text
                start = ent.start_char - sent.start_char
                end = ent.end_char - sent.start_char

                sentence_text = sent.text
                s0 = sentence_text[:start].strip()
                s1 = sentence_text[end:].strip()

                # Extract keywords from S1 using POS tags to exclude function words
                s1_doc = nlp(s1)
                s1_keywords = [
                    token.text for token in s1_doc
                    if not token.is_punct and token.pos_ not in {"AUX", "PRON", "DET", "PART", "ADP", "CCONJ", "SCONJ", "INTJ"} and token.text.strip()
                ]

                results.append({
                    "S0": s0,
                    "name": name,
                    "S1": s1,
                    "S1_keywords": s1_keywords
                })
                break  # only process the first PERSON per sentence

    return results

# Standardized Levenshtein distance

def lds(list1, list2):
    """
    Calculate the normalized Levenshtein distance between two lists of strings,
    treating each string (token) as one unit (i.e., token-level distance).

    Args:
        list1 (List[str]): First list of keywords (e.g., S1_keywords)
        list2 (List[str]): Second list of keywords

    Returns:
        float: Normalized Levenshtein distance between 0.0 and 1.0
    """
    len1 = len(list1)
    len2 = len(list2)

    # Initialize the distance matrix
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Fill in the matrix with edit distances
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if list1[i - 1] == list2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    raw_distance = dp[len1][len2]
    max_len = max(len1, len2)
    if max_len == 0:
        return 0.0  # both lists are empty

    return raw_distance / max_len

def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def sample_top_k(probs, k):
    topk_value, _ = torch.topk(probs, k)  # batch_sz x topk
    min_value_top_k = topk_value[:, [-1]]
    probs[probs < min_value_top_k] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def sample_tokens(logits, temperature=1, top_p=None, top_k=None):
    shape = logits.shape
    logits = logits.flatten(end_dim=-2)
    if temperature > 0.0:
        probs = torch.softmax(logits / temperature, dim=-1)

        if top_p is not None:
            next_token = sample_top_p(probs, top_p)
        elif top_k is not None:
            next_token = sample_top_k(probs, top_k)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1)
    return next_token.view(shape[:-1])


def pack_prompts(prompts: List[int]):
    res = []
    lengths = []
    for i, p in enumerate(prompts):
        p = torch.tensor(p, dtype=torch.long)
        l = p.size(0)
        res.append(p)
        lengths.append(l)
    lengths = torch.tensor(lengths, dtype=torch.long)
    res = torch.cat(res)
    return res, lengths


def batch_prompts(prompts, max_elements, lengths=None):
    batches = []
    current_batch = []
    current_count = 0

    for i in range(len(prompts)):
        prt = prompts[i]
        prompt_size = len(prt) if lengths is None else lengths[i]
        if current_count + prompt_size <= max_elements:
            current_batch.append(prt)
            current_count += prompt_size
        else:
            if current_batch:  # Add the current batch to batches
                batches.append(current_batch)
            # Start a new batch with the current prompt
            current_batch = [prt]
            current_count = prompt_size

    # Add the last batch if it contains any prompts
    if current_batch:
        batches.append(current_batch)

    return batches


class KVCache(nn.Module):
    def __init__(self, bsz, seqlen, n_heads, head_dim, dtype, device):
        super().__init__()
        shape = (bsz, seqlen, n_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.offset = 0

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.offset = 0

    def update(self, k_val, v_val, tok_idx):
        # input_pos: [B], k_val: [B, S, H, D]
        self.k_cache.index_copy_(1, self.offset + tok_idx, k_val)
        self.v_cache.index_copy_(1, self.offset + tok_idx, v_val)
        return self.k_cache, self.v_cache


@dataclass
class PackedCausalTransformerGeneratorArgs:
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = 64
    max_gen_len: int = 20  # Maximum number of tokens to generate
    max_tokens: int = 2048  # Maximum number of tokens that can go through the model
    max_prompt_len: Optional[int] = None
    until: List[str] = field(default_factory=list)
    compile_prefilling: bool = False
    reduce_generation_overhead: bool = False
    show_progress: bool = False
    dtype: Optional[str] = "bf16"
    device: Optional[str] = "cuda"


class PackedCausalTransformerGenerator:
    def __init__(
        self,
        cfg: PackedCausalTransformerGeneratorArgs,
        model: nn.Module,
        tokenizer: Tokenizer,
    ):
        """
        This class wraps a causal transformer model with its corresponding tokenizer
        and provides an efficient way to pack prompts together and do generation on
        the packed sequence.

        For example, if we had the prompts "Hello, I am a " and "Initiating calibration "
        Then this class will concatenate those sequence (pack them together)
        "Hello, I am a Initiating calibration"
        And make the necessary attention masks such that a sequence only attends to itself
        during prefilling and generation.

        This class creates a fixed size cache of size max_tokens or sum of prompt sizes
        + the max number of generated tokens per sequence.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.top_k = cfg.top_k

        self.max_gen_len = cfg.max_gen_len
        self.max_tokens = cfg.max_tokens
        self.max_prompt_len = cfg.max_prompt_len
        self.until = cfg.until
        self.max_until_size = max([len(e) for e in self.until]) if self.until else 1
        self.device = cfg.device

        # Compile if necessary
        self.prefill = torch.compile(self.prefill, disable=not cfg.compile_prefilling)
        self.generate_next_token = torch.compile(
            self.generate_next_token,
            mode="reduce-overhead",
            disable=not cfg.reduce_generation_overhead,
        )

        self.show_progress = cfg.show_progress
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[cfg.dtype]

        self.prefill_doc_id, self.prefill_tok_id = None, None
        self.padded_doc_id, self.padded_tok_id = None, None
        self.current_doc_id, self.current_tok_id = None, None
        self.padded_doc_start = None
        self.prefill_mask = None

    def clear_cache(self, offset):
        for module in self.model.modules():
            if isinstance(module, Attention):
                if not hasattr(module, "kv_cache"):
                    module.kv_cache = KVCache(
                        1,
                        self.max_tokens,
                        module.n_kv_heads,
                        module.head_dim,
                        self.dtype,
                        self.device,
                    )
                module.kv_cache.offset = offset

    @torch.compiler.disable
    def setup_prefilling(self, lengths: torch.Tensor):
        # The KV cache is a fixed size tensor of size max_tokens that we need
        # to update in order to do correct autoregressive generation.

        # Here we will generate token by token but on multiple sequences
        # at once. To do so, we need to have an attention mask that makes
        # each sequence independent.

        # Each sequence will write to its allocated space in the KV Cache.
        # We allocate len(seq) + max_gen_len to each sequence in the cache.

        # We will generate max_gen_len for each document
        padded_lengths = lengths + self.max_gen_len
        max_tokens = self.max_tokens or padded_lengths.sum().item()
        # The last document might have more padding to fill up to max_tokens
        padded_lengths[-1] += max_tokens - padded_lengths.sum()

        # This is the start index in the cache for each document
        self.padded_doc_start = lengths_to_start_ids(padded_lengths)
        # For example with ab--123--cdef--
        # this would be 0, 4, 9 if max_gen_len is 2

        # We repeat interleave to align with tokens for prefilling
        # Ex: ab--123--cdef--
        #     000044444999999
        prefill_offset = torch.repeat_interleave(self.padded_doc_start, lengths)
        # This offset will make sure the tokens are written to the
        # correct positions in the cache during prefilling

        # We either init the cache or clear it by resetting the offset to prefill_offset
        self.clear_cache(prefill_offset)

        # The prefilling mask looks like the following for
        # the two packed sequences ab and 123 : ab123
        # Where spaces are empty cache positions
        #                 keys
        #                ab---123---
        #   queries    a 10000000000
        #              b 11000000000
        #              1 00000100000
        #              2 00000110000
        #              3 00000111000
        # We make sure to skip the empty cache positions
        # and only attend to positions within the same sequence
        doc_mask_mod = generate_doc_mask_mod(causal_mask, lengths, padded_lengths)
        self.prefill_mask = create_block_mask(
            doc_mask_mod, 1, None, lengths.sum(), max_tokens
        )

        # This creates the prefilling token ids which look like
        # the following for the packed sequence abcdefg1234
        # abcdefg1234
        # 01234560123
        # The token id gives us the position within each sequence
        # This is used to compute ROPE and to update the cache
        # At each forward pass the current tokens are written to
        # offset + tok_id
        self.prefill_doc_id, self.prefill_tok_id = lengths_to_local_ids(lengths)

        # This creates the padded token and document ids
        # which look like the following for the packed sequence ab123
        #               ab---123---               ab---123---
        # padded_doc_id 00000111111 padded_tok_id 01234012345
        # This will later be useful for the attention mask at generation
        self.padded_doc_id, self.padded_tok_id = lengths_to_local_ids(padded_lengths)

    @torch.compiler.disable
    def setup_generation(self, lengths):
        # KV Cache offset is set to the start of the padded documents
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.kv_cache.offset = self.padded_doc_start
        # The token ids during generations correspond to the lengths of each doc
        # current_tok_id will be incremented during generation
        self.current_tok_id = lengths.clone()
        # Since we're generating one token per document
        # the document id is just an arange
        self.current_doc_id = torch.arange(lengths.size(0), device=lengths.device)

    # From here on some methods for generation
    def prefill(self, tokens: torch.Tensor, lengths: torch.Tensor):
        # Prefilling is done by taking multiple packed sequences and
        # doing block diagonal attention on them so they remain independent
        self.setup_prefilling(lengths=lengths)
        prefill_out = self.model.forward(
            tokens,
            tok_idx=self.prefill_tok_id,
            mask=self.prefill_mask,
            attn_impl="flex_attention",
        )
        self.setup_generation(lengths=lengths)
        return prefill_out

    def generate_next_token(self, current_token):
        # Since we're doing generation with multiple sequences at once
        # we need to ignore tokens and cache entries from other sequences
        # or in the future.
        # Example mask :
        #                  keys
        #                abc--1234--
        #   queries    c 11100000000
        #              4 00000111100

        # mask shape : (n_seqs, cache_size)
        doc_mask = self.current_doc_id.unsqueeze(1) == self.padded_doc_id.unsqueeze(0)
        caus_mask = self.current_tok_id.unsqueeze(1) >= self.padded_tok_id.unsqueeze(0)
        mask = doc_mask & caus_mask
        out = self.model.forward(
            current_token,
            tok_idx=self.current_tok_id,  # n_seqs
            mask=mask,
            attn_impl="sdpa",
        )
        self.current_tok_id += 1
        return out

    @torch.inference_mode()
    def generate(self, prompts):
        # Tokenize
        prompts = [
            self.tokenizer.encode(p, add_bos=True, add_eos=False) for p in prompts
        ]
        # Truncate
        max_seqlen = (
            self.max_tokens
            if not hasattr(self.model, "max_seqlen")
            else self.model.max_seqlen
        )
        max_prompt_len = self.max_prompt_len or min(
            max_seqlen - self.max_gen_len, self.max_tokens - self.max_gen_len
        )
        prompts = [p[-max_prompt_len:] for p in prompts]
        
        # Account for the generation in lengths
        padded_lengths = [len(p) + self.max_gen_len for p in prompts]
        generation = []
        loglikelihood = []
        greedy = []
        it = batch_prompts(prompts, self.max_tokens, lengths=padded_lengths)
        if self.show_progress:
            it = tqdm(it)
        for batch_idx, batch in enumerate(it):
            n_seqs = len(batch)
            generated_tokens = [[] for _ in range(n_seqs)]
            is_done = [False for _ in range(n_seqs)]
            packed_batch, lengths = pack_prompts(batch)
            packed_batch, lengths = packed_batch.cuda(), lengths.cuda()
            n_seqs = lengths.size(0)

            # Prefilling cache
            prompt_logits = self.prefill(packed_batch.unsqueeze(0), lengths)
            
            # Selecting last token in each prompt
            all_tokens = sample_tokens(
                prompt_logits, self.temperature, self.top_p, self.top_k
            )
            start_token = all_tokens[:, lengths.cumsum(0) - 1]

            for seq_id, tok in enumerate(start_token.squeeze(0).tolist()):
                generated_tokens[seq_id].append(tok)

            current_token = start_token
            for i in range(1, self.max_gen_len):

                next_logits = self.generate_next_token(current_token)
                next_token = sample_tokens(
                    next_logits.clone(), self.temperature, self.top_p, self.top_k
                )

                for seq_id, tok in enumerate(next_token.squeeze(0).tolist()):
                    if not is_done[seq_id]:
                        generated_tokens[seq_id].append(tok)
                        current_end_str = self.tokenizer.decode(
                            generated_tokens[seq_id][-self.max_until_size :]
                        )
                        contains_end_string = any(
                            [e in current_end_str for e in self.until]
                        )
                        is_done[seq_id] = (
                            contains_end_string or tok == self.tokenizer.eos_id
                        )
                if all(is_done):
                    break

                current_token = next_token

            generation.extend([self.tokenizer.decode(g) for g in generated_tokens])

            for p, logit in zip(
                batch, prompt_logits.squeeze(0).split(lengths.tolist())
            ):
                x = logit[:-1]
                y = torch.tensor(p[1:], device=x.device)
                loglikelihood.append(-F.cross_entropy(x, y, reduction="none").cpu())
                greedy.append((x.argmax(dim=-1) == y).cpu())

        return generation, loglikelihood, greedy


def load_consolidated_model_and_tokenizer(
    consolidated_path,
    model_cls=LMTransformer,
    model_args_cls=LMTransformerArgs,
):
    ckpt_path = Path(consolidated_path)
    config = ckpt_path / "params.json"
    config = OmegaConf.load(config)

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    model_args = dataclass_from_dict(model_args_cls, config.model, strict=False)
    tokenizer = build_tokenizer(config.data.tokenizer.name, config.data.tokenizer.path)
    model = model_cls(model_args)
    st_dict = torch.load(ckpt_path / "consolidated.pth", weights_only=True)
    model.load_state_dict(st_dict["model"])
    model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)
    return model, tokenizer, config


def main():
    parser = argparse.ArgumentParser(description="Execute ture-prefix attack against JSONL")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--model", type=str, required=True, choices=["a_1", "a_10", "a_100", "b", "c"])
    parser.add_argument("--N_sentences", type=int, default=2, help="N of target sentences")
    parser.add_argument("--N_sampling", type=int, default=10, help="N of target sentences")
    args = parser.parse_args()
    
    # Load CLI arguments (overrides) and combine with a YAML config
    cfg = OmegaConf.load("lingua_config/eval_{}.yaml".format(args.model))
    
    model, tokenizer, _ = load_consolidated_model_and_tokenizer(cfg.ckpt_dir)
    
    # execute true prefix attack
    output_N = 0
    start_time = time.time()
    with open(args.input_path, "r", encoding="utf-8") as infile, open(args.output_path, "w", encoding="utf-8") as outfile:
        for line_idx, line in tqdm(enumerate(infile)):
            data = json.loads(line)
            for sent_idx, sent in enumerate(data["person_sentences"]):
                sent["S1_keywords"] = [k for k in sent["S1_keywords"] if k.strip()]
                if len(sent["S1_keywords"]):
                    true_prefix = [sent["name"]]
                    if sent["S0"]:
                        true_prefix = [sent["S0"], sent["name"]]
                    prompt = " ".join(true_prefix)
                    prompts = [prompt] * args.N_sampling
                    gen_args = PackedCausalTransformerGeneratorArgs
                    gen_args.max_gen_len = sent["total_token_count"] - sent["prefix_token_count"]
                    gen_cfg = dataclass_from_dict(
                        gen_args, cfg, strict=False
                    )
                    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)
                    generation, loglikelihood, greedy = generator.generate(prompts)
                    estimated_PIIs = []
                    for i, gen in enumerate(generation):
                        try:
                            estimated_PII = extract_sentences_with_person(" ".join([prompt, gen]))[0]
                        except Exception as e:
                            continue
                        estimated_PII["lds"] = lds(sent["S1_keywords"], estimated_PII["S1_keywords"])
                        estimated_PIIs.append(estimated_PII)
                    if not len(estimated_PIIs):
                        continue
                    min_i = int(np.argmin([row["lds"] for row in estimated_PIIs]))
                    sent["estimated_S1"] = estimated_PIIs[min_i]["S1"]
                    sent["estimated_S1_keywords"] = estimated_PIIs[min_i]["S1_keywords"]
                    sent["lds"] = estimated_PIIs[min_i]["lds"]
                    outfile.write(json.dumps(sent, ensure_ascii=False) + "\n")
                    output_N += 1
                    break
                
            if output_N >= args.N_sentences:
                break

    # # Allow multiple prompts
    # prompts = []
    # while True:
    #     prompt = input("Enter a prompt (or press enter to finish): ")
    #     if not prompt:
    #         break
    #     prompts.append(prompt)

    # Start generation
    
    end_time = time.time()

    # Calculate tokens per second
    # total_tokens = sum(len(tokenizer.encode(gen, False, False)) for gen in generation)
    # tokens_per_second = total_tokens / (end_time - start_time)

    # # Display the results
    # for i, gen in enumerate(generation):
    #     print(f"\nPrompt {i+1}: {prompts[i]}")
    #     print(f"Generated Text: {gen}")
    
    print(f"\nTotal time: {end_time - start_time:.2f}")


if __name__ == "__main__":
    main()
