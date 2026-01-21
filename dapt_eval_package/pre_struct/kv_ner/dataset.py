# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .data_utils import Entity, Relation, Sample, generate_char_labels
from .chunking import chunk_text_by_tokens
from .noise_utils import NoiseFeatureProcessor, PERFECT_VALUES


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    labels: torch.Tensor
    offset_mapping: torch.Tensor
    texts: List[str]
    task_ids: List[str]
    titles: List[str]
    entities: List[List[Entity]]
    relations: List[List[Relation]]
    original_task_ids: List[str]
    chunk_indices: List[int]
    chunk_spans: List[Tuple[int, int]]
    noise_ids: Optional[torch.Tensor] = None


class TokenClassificationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        tokenizer: PreTrainedTokenizerBase,
        label2id: Dict[str, int],
        *,
        max_seq_length: int,
        label_all_tokens: bool = True,
        include_labels: bool = True,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        enable_chunking: bool = False,
        noise_processor: Optional[NoiseFeatureProcessor] = None,
    ) -> None:
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_length = int(max_seq_length)
        self.label_all_tokens = bool(label_all_tokens)
        self.include_labels = bool(include_labels)
        self.enable_chunking = bool(enable_chunking)
        self.noise_processor = noise_processor
        if self.enable_chunking:
            self.chunk_size = int(chunk_size or self.max_seq_length)
            self.chunk_overlap = int(chunk_overlap or 0)
            if self.chunk_size > self.max_seq_length:
                self.chunk_size = self.max_seq_length
        else:
            self.chunk_size = None
            self.chunk_overlap = None
        if "O" not in self.label2id:
            raise ValueError("label2id must include 'O'")
        self._features: List[dict] = []
        self._prepare()

    def _prepare(self) -> None:
        for sample in self.samples:
            text = sample.text or ""
            spans = self._chunk_sample(text)

            full_char_labels: Optional[List[str]] = None
            if self.include_labels:
                full_char_labels = generate_char_labels(len(text), sample.entities)

            for chunk_idx, (chunk_text, char_start, char_end) in enumerate(spans):
                encoded = self.tokenizer(
                    chunk_text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_offsets_mapping=True,
                    return_attention_mask=True,
                )
                offset_mapping = encoded.pop("offset_mapping")
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                token_type_ids = encoded.get(
                    "token_type_ids", [0] * len(input_ids)
                )

                if not self.include_labels:
                    labels = [self.label2id["O"]] * len(input_ids)
                else:
                    assert full_char_labels is not None
                    chunk_char_labels = full_char_labels[char_start:char_end]
                    labels = self._align_labels(offset_mapping, chunk_char_labels)

                chunk_entities = self._slice_entities(
                    sample.entities, char_start, char_end, sample.text
                )

                feature = {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
                    "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "offset_mapping": torch.tensor(offset_mapping, dtype=torch.long),
                    "text": chunk_text,
                    "task_id": sample.task_id,
                    "title": sample.title,
                    "entities": chunk_entities,
                    "relations": sample.relations,
                    "original_task_id": sample.task_id,
                    "chunk_index": chunk_idx,
                    "chunk_span": (char_start, char_end),
                }
                # 生成每个token的noise_ids（如果提供了noise_processor且样本含有noise_values）
                if self.noise_processor is not None and getattr(sample, "noise_values", None):
                    nv = sample.noise_values or []
                    noise_ids_per_token: List[List[int]] = []
                    for (s, e) in offset_mapping:
                        s = int(s); e = int(e)
                        if e <= s:
                            noise_ids_per_token.append(self.noise_processor.values_to_bin_ids(PERFECT_VALUES))
                            continue
                        vecs = []
                        abs_s = char_start + s
                        abs_e = char_start + e
                        for ci in range(abs_s, abs_e):
                            if 0 <= ci < len(nv):
                                v = nv[ci]
                                if isinstance(v, (list, tuple)) and len(v) == 7:
                                    vecs.append(v)
                        if vecs:
                            avg = [sum(col) / len(col) for col in zip(*vecs)]
                            noise_ids_per_token.append(self.noise_processor.values_to_bin_ids(avg))
                        else:
                            noise_ids_per_token.append(self.noise_processor.values_to_bin_ids(PERFECT_VALUES))
                    feature["noise_ids"] = torch.tensor(noise_ids_per_token, dtype=torch.long)
                self._features.append(feature)

    def _chunk_sample(self, text: str) -> List[Tuple[str, int, int]]:
        if not self.enable_chunking:
            return [(text, 0, len(text))]
        spans = chunk_text_by_tokens(
            text,
            self.tokenizer,
            chunk_size=self.chunk_size or self.max_seq_length,
            overlap=self.chunk_overlap or 0,
        )
        return spans or [(text, 0, len(text))]

    def _align_labels(self, offsets, char_labels: List[str]) -> List[int]:
        labels: List[int] = []
        previous_entity: Optional[str] = None
        o_id = self.label2id["O"]
        for start, end in offsets:
            if end <= start:
                labels.append(o_id)
                continue
            start = int(start)
            end = int(end)
            if start >= len(char_labels):
                labels.append(o_id)
                previous_entity = None
                continue
            slice_labels = char_labels[start:end]
            target = "O"
            for lab in slice_labels:
                if lab == "O":
                    continue
                prefix = lab.split("-", 1)[0] if "-" in lab else lab
                if prefix == "B":
                    target = lab
                    break
                if prefix == "E":
                    target = lab
                elif prefix == "I" and target == "O":
                    target = lab
            if target.startswith("I-"):
                entity = target[2:]
                if previous_entity != entity:
                    target = f"B-{entity}"
            if target == "O" and not self.label_all_tokens:
                previous_entity = None
                labels.append(o_id)
                continue

            labels.append(self.label2id.get(target, o_id))
            if target == "O":
                previous_entity = None
            else:
                entity = target.split("-", 1)[1] if "-" in target else None
                if target.startswith("E-"):
                    previous_entity = None
                else:
                    previous_entity = entity
        return labels

    def _slice_entities(
        self,
        entities: Sequence[Entity],
        char_start: int,
        char_end: int,
        full_text: str,
    ) -> List[Entity]:
        """
        Slice entities for a given chunk [char_start, char_end).

        Important change (do-not-drop policy):
        - Previously we only kept entities fully contained in the chunk, which
          dropped gold labels that straddled chunk boundaries.
        - Now we KEEP ANY OVERLAPPING ENTITY by CLIPPING it to the chunk range.
          This prevents losing labeled content at boundaries during training.

        Notes:
        - Char-level labels for the chunk are generated separately by
          generate_char_labels(...) and then sliced to this same range in
          _prepare(). The alignment logic (_align_labels) already converts a
          leading I-* to B-* when an entity begins inside a chunk, so training
          remains consistent even for partial spans.
        - We use strict overlap (ent.end > char_start and ent.start < char_end)
          and then clip to [char_start, char_end) to produce local offsets.
        """
        sliced: List[Entity] = []
        for ent in entities:
            if ent.end > char_start and ent.start < char_end:
                s = max(char_start, ent.start)
                e = min(char_end, ent.end)
                if e <= s:
                    continue
                sliced.append(
                    Entity(
                        start=s - char_start,
                        end=e - char_start,
                        label=ent.label,
                        result_id=ent.result_id,
                        text=(full_text[s:e] if full_text else ent.text),
                    )
                )
        return sliced

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> dict:
        return self._features[idx]


def collate_batch(batch: Sequence[dict]) -> Batch:
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    token_type_ids = torch.stack([item["token_type_ids"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    offset_mapping = torch.stack([item["offset_mapping"] for item in batch], dim=0)
    texts = [item["text"] for item in batch]
    task_ids = [item["task_id"] for item in batch]
    titles = [item["title"] for item in batch]
    entities = [item["entities"] for item in batch]
    relations = [item["relations"] for item in batch]
    original_task_ids = [item.get("original_task_id", item["task_id"]) for item in batch]
    chunk_indices = [item.get("chunk_index", 0) for item in batch]
    chunk_spans = [tuple(item.get("chunk_span", (0, len(item["text"])))) for item in batch]
    noise_ids = None
    if "noise_ids" in batch[0]:
        noise_ids = torch.stack([item["noise_ids"] for item in batch], dim=0)
    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=labels,
        offset_mapping=offset_mapping,
        texts=texts,
        task_ids=task_ids,
        titles=titles,
        entities=entities,
        relations=relations,
        original_task_ids=original_task_ids,
        chunk_indices=chunk_indices,
        chunk_spans=chunk_spans,
        noise_ids=noise_ids,
    )
