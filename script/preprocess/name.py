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

# Constant key for person entities
person_entity = "person_entity"


def normalize_name(raw_name: str) -> str:
    """
    Normalize a name string.

    Args:
        raw_name (str): Original name string (e.g., "John   Smith").

    Returns:
        str: Lowercased and whitespace-normalized version (e.g., "john smith").
    """
    name = raw_name.lower()
    name = re.sub(r'\s+', ' ', name.strip())
    return name


def canonicalize_name(raw_name: str) -> str:
    """
    Canonicalize a name to avoid differences due to order (e.g., first last vs last first).

    Args:
        raw_name (str): Normalized full name (e.g., "john smith").

    Returns:
        str: Canonical name string in sorted order with underscore delimiter (e.g., "john_smith" or "smith_john" -> "john_smith"),
             or empty string if not applicable (e.g., single word).
    """
    name = normalize_name(raw_name)
    parts = name.split(' ')

    if len(parts) == 2:
        sorted_parts = sorted(parts)
        return "_".join(sorted_parts)
    else:
        return ""


def get_filter_result(res):
    """
    Helper function to unpack filter result.

    Args:
        res (bool | Tuple[bool, str]): Result of filtering step.

    Returns:
        Tuple[bool, str | None]: Tuple containing boolean filter result and optional reason string.
    """
    result, reason = res, None
    if isinstance(result, tuple):
        result, reason = res
    return result, reason


class ExtractName(PipelineStep):
    """
    A pipeline step that extracts PERSON entities from document text using spaCy
    and records normalized canonical names to statistics.

    Args:
        some_folder (DataFolderLike): Target folder for pipeline metadata or IO setup.

    Returns:
        DocumentsPipeline: Unchanged documents with internal stats updated.
    """
    def __init__(self, some_folder: DataFolderLike):
        super().__init__()
        self.some_folder = get_datafolder(some_folder)
        self.nlp = spacy.load("en_core_web_sm")

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            with self.track_time():
                name_set = set()
                result = self.nlp(doc.text)

                # Extract PERSON named entities
                for ent in result.ents:
                    if ent.label_ == "PERSON":
                        name_set.add(ent.text)

                # Normalize and canonicalize entity names
                canonical_name_set = set()
                for name in name_set:
                    name_canonical = canonicalize_name(name)
                    if name_canonical != "":
                        canonical_name_set.add(name_canonical)

                # Record each canonicalized name to stats
                for name_canonical in canonical_name_set:
                    self.stat_update(f"{person_entity}/{canonicalize_name(name)}")
            yield doc


class FilterName(BaseFilter):
    """
    A document filter that excludes documents based on extracted PERSON entity names.

    This filter loads a name frequency stats file, selects a portion of the most common names,
    and filters out documents that include those names.

    Args:
        stat_path (str): Path to a statistics file containing person name frequencies.
        drop_ratio (float): Proportion of names to drop from the dataset.
        max_freq_to_drop (int | None): Maximum allowed frequency to consider a name for dropping.
        seed (int): Random seed for reproducibility.
        exclusion_writer (DiskWriter | None): Optional writer to save filtered documents.
        batch_size (int): Batch size for document processing.

    Returns:
        DocumentsPipeline: Stream of documents with filtered items removed.
    """
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
        """
        Apply filtering logic for a single document.

        Args:
            doc (Document): A document containing text to scan for person names.

        Returns:
            bool | Tuple[bool, str]: True to keep, False to drop (with optional reason).
        """
        name_set = set()
        result = self.nlp(doc.text)
        for ent in result.ents:
            if ent.label_ == "PERSON":
                name_set.add(ent.text)

        canonical_name_set = set()
        for name in name_set:
            name_canonical = canonicalize_name(name)
            if name_canonical != "":
                canonical_name_set.add(name_canonical)

        for name_canonical in canonical_name_set:
            if (f"{person_entity}/{canonicalize_name(name)}" in self.drop_names_dict.keys()):
                return False
        return True

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
        Load name frequency statistics, select names to drop, and filter incoming documents.

        Returns:
            DocumentsPipeline: Stream of filtered documents.
        """
        # Load stats from file
        with open(self.stat_path, "r") as f:
            full_names_dict = json.load(f)[-1]['stats']
            assert "person_entity" in list(full_names_dict.keys())[0]

        # Sort by frequency
        sorted_full_names = sorted(full_names_dict.items(), key=lambda x: x[1], reverse=True)

        # Optionally exclude very frequent names
        if self.max_freq_to_drop:
            sorted_full_names = [x for x in sorted_full_names if x[1] <= self.max_freq_to_drop]

        # Randomly choose names to drop
        random.seed(self.seed)
        drop_names = random.choices(sorted_full_names, k=int(len(sorted_full_names) * self.drop_ratio))
        self.drop_names_dict = dict(drop_names)

        # Save selected drop names
        with open(f"{self.stat_path}_drop_{str(self.seed)}.json", "w") as f:
            json.dump(self.drop_names_dict, f)

        # Perform filtering in batches
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
