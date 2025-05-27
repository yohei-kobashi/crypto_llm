from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import List, Optional, Tuple, Set
from enum import Enum

import torch
from torch import nn
from tqdm import tqdm
import random

from omegaconf import OmegaConf
from torch.nn import functional as F
import xformers
import spacy
from dataclasses import dataclass, field
import json
from typing import List, Optional
from logging import getLogger, INFO
from transformers import pipeline

from apps.main.transformer import LMTransformer, LMTransformerArgs
from lingua.args import dataclass_from_dict
from lingua.checkpoint import CONSOLIDATE_NAME
from lingua.tokenizer import Tokenizer, build_tokenizer
from lingua.transformer import (
    Attention,
    causal_mask,
    generate_doc_mask_mod,
    lengths_to_local_ids,
    lengths_to_start_ids,
)
from lingua.logger import init_logger
from apps.main.generate import (
    PackedCausalTransformerGeneratorArgs,
    PackedCausalTransformerGenerator,
    load_consolidated_model_and_tokenizer,
)

logger = getLogger(__name__)
logger.setLevel(INFO)


class MaskMode(Enum):
    ALL_PERSON = "all_person"
    ONLY_FIRST = "only_first"

@dataclass
class PIIReconstructionArgs:
    gen_cfg: PackedCausalTransformerGeneratorArgs = field(
        default_factory=PackedCausalTransformerGeneratorArgs
    )
    ckpt: str = field(
        default_factory=lambda: str(Path(__file__).parent.parent.parent / "ckpt" / CONSOLIDATE_NAME)
    )
    pii_jsonl_path: str = ""
    pii_mask_mode: MaskMode = MaskMode.ONLY_FIRST
    pii_num: Optional[int] = None
    cand_num: Optional[int] = None


MASK_TOKEN = "<mask>"
def fill_mask_helper(
        pipeline,
        text: str,
        pii_idx_list: List[Tuple[int, int]],
    ):
    i = len(pii_idx_list)
    split_text = []
    buffer = 0
    for start, end in pii_idx_list:
        assert start >= buffer, f"start: {start}, buffer: {buffer}"
        split_text.append(text[buffer:start])
        buffer = end

    split_text.append(text[buffer:])

    output = split_text[0]
    for j in range(i):
        result = pipeline(output + MASK_TOKEN + split_text[j + 1])
        predicted_word = result[0]["token_str"].strip()
        output += predicted_word + split_text[j + 1]

    return output


def main():
    init_logger()
    # Load CLI arguments (overrides) and combine with a YAML config
    cfg = OmegaConf.from_cli()
    cfg = dataclass_from_dict(
        PIIReconstructionArgs, cfg, strict=False
    )
    logger.info(cfg)

    model, tokenizer, _ = load_consolidated_model_and_tokenizer(cfg.ckpt)

    generator = PackedCausalTransformerGenerator(cfg.gen_cfg, model, tokenizer)

    logger.info("Model loaded successfully.")


    nlp = spacy.load("en_core_web_sm")

    separated_text_list: List[Tuple[str, str, str]] = [] # prefix, name, suffix
    pii_idx_list: List[List[Tuple[int, int]]] = []
    done = False
    # Load PII data
    logger.info("Loading PII data...")
    with open(cfg.pii_jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f)):
            data = json.loads(line)

            assert "person_sentences" in data.keys()
            if len(data["person_sentences"]) == 0:
                logger.warning(f"Empty person_sentences in {idx}th line")
                continue
            else:
                for d in data["person_sentences"]:
                    prefix = d["S0"]
                    name = d["name"]
                    suffix = d["S1"]
                    separated_text_list.append((prefix, name, suffix))

                    list_idx_to_mask = []
                    # assume reconstruct the first name, 
                    # and remaining pii entities are filled with fill_mask_helper()
                    doc_nlp = nlp(suffix)
                    for ent in doc_nlp.ents:
                        if ent.label_ == "PERSON":
                            if cfg.pii_mask_mode == MaskMode.ONLY_FIRST:
                                if ent.text != name:
                                    continue
                            elif cfg.pii_mask_mode == MaskMode.ALL_PERSON:
                                pass
                            else:
                                raise ValueError(f"Unknown mask mode: {cfg.pii_mask_mode}")
                            list_idx_to_mask.append((ent.start_char, ent.end_char))

                    pii_idx_list.append(list_idx_to_mask)

                    if cfg.pii_num is not None:
                        if len(separated_text_list) >= cfg.pii_num:
                            done = True
                            break
                if done:
                    break
    logger.info(f"Number of PII entities: {len(separated_text_list)}")

    logger.info(f"Filling the mask...")

    # Load fill-mask pipeline
    fill_mask_pipeline = pipeline(
        "fill-mask",
        model="FacebookAI/roberta-large",
    )

    filled_separated_text_list: List[Tuple[str, str, str]] = []
    for i, (prefix, name, suffix) in tqdm(enumerate(separated_text_list)):
        list_idx_to_mask = pii_idx_list[i]
        # fill the mask
        filled_suffix = fill_mask_helper(fill_mask_pipeline, suffix, list_idx_to_mask)
        filled_separated_text_list.append((prefix, name, filled_suffix))

    # gather all PII entities saved in pii_person_list into a set
    pii_name_set = set()
    for _, name, _ in filled_separated_text_list:
        pii_name_set.add(name)

    candidate_name_list = list(pii_name_set)
    
    # Remove duplicates
    logger.info(f"Number of unique PII entities: {len(candidate_name_list)}")
    logger.info(f"PII entities: {candidate_name_list}")

    logger.info("Calculating extractability...")

    generator.max_gen_len = 1

    success_count = 0
    count = 0
    for i, (prefix, name, suffix) in tqdm(enumerate(filled_separated_text_list)):
        prompts = []
        if cfg.cand_num is not None:
            cand_list = [name]
            remains = pii_name_set - {name}
            assert len(remains) > cfg.cand_num - 1
            sampled = random.sample(list(remains), cfg.cand_num - 1)
            cand_list += sampled
            assert len(cand_list) == cfg.cand_num
        else:
            cand_list = candidate_name_list

        for cand_name in cand_list:
            p = prefix + cand_name + suffix
            prompts.append(p)
        max_pii = float("-inf")
        cand_with_max_pii = ""
        _, loglikelihoods, _ = generator.generate(prompts)
        ans_l = 0
        for k, loglikelihood in enumerate(loglikelihoods):
            tmp = loglikelihood.sum().item()
            if tmp > max_pii:
                max_pii = tmp
                cand_with_max_pii = cand_list[k]
            if cand_list[k] == name:
                ans_l = tmp
        assert ans_l != 0, f"Answer {name} not in candidates {cand_list}"
        if cand_with_max_pii == name:
            success_count += 1
            logger.info(f"Answer: {name} ({ans_l})\t\tCandidate: {cand_with_max_pii} ({max_pii}) (success)")
        else:
            logger.info(f"Answer: {name} ({ans_l})\t\tCandidate: {cand_with_max_pii} ({max_pii}) (fail)")
        
        count += 1

    # Calculate success rate
    logger.info(f"Success rate: {success_count / len(filled_separated_text_list)}")
                

if __name__ == "__main__":
    main()