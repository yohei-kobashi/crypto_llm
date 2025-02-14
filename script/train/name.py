# https://github.com/huggingface/datatrove/blob/main/examples/filter_hf_dataset.py
"""
This file contains code to:
1 - Load a parquet-format Hugging Face dataset from the hub.
2 - Filter the dataset (to include only entries that contain the word 'hugging' in the text column).
3 - Push the filtered dataset back to the hub.
"""

import argparse
import spacy
import json
import re
import random
import contextlib
from typing import List, Tuple

from loguru import logger
from datatrove.pipeline.base import PipelineStep
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.batching import batched
from datatrove.utils.typeshelper import StatHints
from datatrove.pipeline.filters.base_filter import BaseFilter

person_entity = "person_entity"

def normalize_name(raw_name: str) -> str:
    # 小文字に変換
    name = raw_name.lower()
    # 余分な空白を正規化
    name = re.sub(r'\s+', ' ', name.strip())
    return name

def canonicalize_name(raw_name: str) -> str:
    """
    姓名の並び順が変わっても同一とみなせるように標準化する。
    ここでは「名 姓」「姓 名」の順番を考慮し、ソートした状態で統一的なキーを返す。
    """
    name = normalize_name(raw_name)
    parts = name.split(' ')
    
    # partsが1つの場合はそのまま返す（ただし人名としては単一ワードは異例）
    # partsが2つの場合、並び順をnormalize（sort）する
    if len(parts) == 2:
        # 名前のリストをソートして、"名_姓"のような形で結合（順番固定）
        sorted_parts = sorted(parts)
        return "_".join(sorted_parts)
    else:
        # それ以外の場合は単純に正規化済みの文字列を返す
        return ""
    
def get_filter_result(res):
    result, reason = res, None
    if isinstance(result, tuple):
        result, reason = res
    return result, reason

class ExtractName(PipelineStep):
    def __init__(self, some_folder: DataFolderLike):
        super().__init__()
        self.some_folder = get_datafolder(some_folder)
        self.nlp = spacy.load("en_core_web_sm")

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        # name_freq = {}
        for doc in data:
            with self.track_time():
                name_set = set()
                result = self.nlp(doc.text)
                for ent in result.ents:
                    if ent.label_ == "PERSON":
                        name_set.add(ent.text)
                
                canonical_name_set = set()
                for name in name_set:
                    name_canonical = canonicalize_name(name)
                    if name_canonical!="":
                        canonical_name_set.add(name_canonical)

                for name_canonical in canonical_name_set:
                    self.stat_update(f"{person_entity}/{canonicalize_name(name)}")
            yield doc

class FilterName(BaseFilter):
    def __init__(
            self, 
            stat_path: str,
            drop_ratio: float = 0.01,
            max_freq_to_drop: int = None,
            seed: int = 0,
            exclusion_writer: DiskWriter = None, 
            batch_size: int = 1,
            ):
        super().__init__(exclusion_writer=exclusion_writer, batch_size=batch_size)
        self.stat_path = stat_path
        self.drop_ratio = drop_ratio
        self.max_freq_to_drop = max_freq_to_drop
        self.seed = seed
        self.drop_names_dict = {}
        self.nlp = spacy.load("en_core_web_sm")

    def filter(self, doc: Document) -> bool | Tuple[bool, str]:
        name_set = set()
        result = self.nlp(doc.text)
        for ent in result.ents:
            if ent.label_ == "PERSON":
                name_set.add(ent.text)
        
        canonical_name_set = set()
        for name in name_set:
            name_canonical = canonicalize_name(name)
            if name_canonical!="":
                canonical_name_set.add(name_canonical)

        for name_canonical in canonical_name_set:
            if (f"{person_entity}/{canonicalize_name(name)}" in self.drop_names_dict.keys()):
                return False
        return True

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        # define self.drop_names_dict
        ## load stat file
        with open(self.stat_path, "r") as f:
            full_names_dict = json.load(f)[-1]['stats']
            assert "person_entity" in list(full_names_dict.keys())[0]
        ## sort by frequency
        sorted_full_names = sorted(full_names_dict.items(), key=lambda x: x[1], reverse=True)
        ## exclude names with frequency over max_freq_to_drop
        if self.max_freq_to_drop:
            sorted_full_names = [x for x in sorted_full_names if x[1]<=self.max_freq_to_drop]
        ## choice which to drop
        random.seed(self.seed)
        drop_names = random.choices(sorted_full_names, k=int(len(sorted_full_names)*self.drop_ratio))
        self.drop_names_dict = dict(drop_names)
        
        # save drop_names_dict
        with open(f"{self.stat_path}_drop_{str(self.seed)}.json", "w") as f:
            json.dump(self.drop_names_dict, f)
        
        # filter
        with self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext() as writer:
            for batch in batched(data, self.batch_size):
                if self.batch_size > 1:
                    self.stat_update("batches")
                with self.track_time("batch" if self.batch_size > 1 else None):
                    batch_filter_result = self.filter_batch(batch)
                for doc, doc_filter_result in zip(batch, batch_filter_result):
                    self.stat_update(StatHints.total)
                    filter_result, reason = get_filter_result(doc_filter_result)
                    if filter_result:
                        self.stat_update(StatHints.forwarded)
                        self.update_doc_stats(doc)
                        yield doc
                    else:
                        self.stat_update(StatHints.dropped)
                        if reason:
                            self.stat_update(f"dropped_{reason}")
                        if self.exclusion_writer:
                            if reason:
                                doc.metadata["filter_reason"] = reason
                            writer.write(doc, rank)
