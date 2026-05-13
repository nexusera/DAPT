"""Decoder-only KV-NSP dataset and collator."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from kv_nsp.negative_sampling import build_negative_sampling_config, sample_kv_nsp_text_pair

from .constants import KV_NSP_SEP_TOKEN, PERFECT_VALUES
from .data import find_json_files, read_json_or_jsonl
from noise_feature_processor import NoiseFeatureProcessor
from noise_fusion import needs_bucket_ids, uses_continuous_noise


def extract_label_studio_pairs(sample: dict[str, Any]) -> list[tuple[str, str]]:
    annotations = sample.get("annotations", [])
    valid = [a for a in annotations if not a.get("was_cancelled")]
    if not valid:
        return []
    results = valid[-1].get("result", [])
    entities: dict[str, dict[str, str]] = {}
    relations: list[tuple[str, str]] = []
    for res in results:
        if res.get("type") == "labels":
            labels = res.get("value", {}).get("labels", [])
            if not labels or labels[0] not in ("键名", "值"):
                continue
            text = res.get("value", {}).get("text", "")
            if text:
                entities[str(res.get("id"))] = {"label": labels[0], "text": text}
        elif res.get("type") == "relation":
            from_id = res.get("from_id")
            to_id = res.get("to_id")
            if from_id and to_id:
                relations.append((str(from_id), str(to_id)))
    pairs: list[tuple[str, str]] = []
    for from_id, to_id in relations:
        key = entities.get(from_id)
        value = entities.get(to_id)
        if key and value and key["label"] == "键名" and value["label"] == "值":
            pairs.append((key["text"].strip(), value["text"].strip()))
    return [(k, v) for k, v in pairs if k and v]


def extract_direct_pairs(sample: dict[str, Any]) -> list[tuple[str, str]]:
    if "key" in sample and "value" in sample:
        key = str(sample.get("key", "")).strip()
        value = str(sample.get("value", "")).strip()
        return [(key, value)] if key and value else []
    pairs = sample.get("pairs")
    if isinstance(pairs, list):
        out: list[tuple[str, str]] = []
        for item in pairs:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            value = str(item.get("value", "")).strip()
            if key and value:
                out.append((key, value))
        return out
    return []


class LlmKvnspDataset(Dataset):
    """Key/value match dataset for decoder-only last-token classification."""

    def __init__(
        self,
        data_path: str | Path,
        *,
        negative_prob: float = 0.5,
        reverse_negative_ratio: float = 1.0,
        random_negative_ratio: float = 1.0,
        max_easy_retries: int = 10,
        seed: int = 42,
        max_samples: int | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.sampling_config = build_negative_sampling_config(
            negative_prob=negative_prob,
            reverse_negative_ratio=reverse_negative_ratio,
            random_negative_ratio=random_negative_ratio,
            max_easy_retries=max_easy_retries,
        )
        self.negative_prob = self.sampling_config.negative_prob
        self.reverse_negative_prob = self.sampling_config.reverse_negative_prob
        self.random_negative_prob = self.sampling_config.random_negative_prob
        self.reverse_negative_ratio = self.sampling_config.reverse_negative_ratio
        self.random_negative_ratio = self.sampling_config.random_negative_ratio
        self.max_easy_retries = self.sampling_config.max_easy_retries
        pairs: list[tuple[str, str]] = []
        for path in find_json_files(data_path):
            for record in read_json_or_jsonl(path):
                direct_pairs = extract_direct_pairs(record)
                if direct_pairs:
                    pairs.extend(direct_pairs)
                    continue
                pairs.extend(extract_label_studio_pairs(record))
        pairs = [(k, v) for k, v in pairs if k and v]
        if max_samples is not None:
            pairs = pairs[: int(max_samples)]
        if not pairs:
            raise ValueError(f"No key/value pairs found in {data_path}")
        self.pairs = pairs
        self.value_pool = [v for _, v in pairs]
        self.valid = set(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        key, value = self.pairs[idx]
        key, value, label, strategy = sample_kv_nsp_text_pair(
            key_text=key,
            value_text=value,
            value_pool=self.value_pool,
            valid_pairs_set=self.valid,
            config=self.sampling_config,
            pair_pool=self.pairs,
            rng=self.rng,
        )
        return {"key": key, "value": value, "nsp_labels": label, "strategy": strategy}


@dataclass
class LlmKvnspCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 256
    sep_token: str | None = None
    noise_mode: str = "bucket"
    noise_processor: NoiseFeatureProcessor | None = None

    def _pair_text(self, key: str, value: str) -> str:
        sep = self.sep_token or getattr(self.tokenizer, "sep_token", None) or KV_NSP_SEP_TOKEN
        return f"{key}{sep}{value}"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [self._pair_text(str(x["key"]), str(x["value"])) for x in features]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["nsp_labels"] = torch.tensor([int(x["nsp_labels"]) for x in features], dtype=torch.long)
        batch["task_type"] = "nsp"
        rows = [[PERFECT_VALUES] * batch["input_ids"].shape[1]] * batch["input_ids"].shape[0]
        mode = str(self.noise_mode or "bucket").lower()
        if needs_bucket_ids(mode):
            processor = self.noise_processor or NoiseFeatureProcessor()
            batch["noise_ids"] = torch.tensor([processor.map_batch(x) for x in rows], dtype=torch.long)
        elif uses_continuous_noise(mode):
            batch["noise_values"] = torch.tensor(rows, dtype=torch.float32)
        return batch
