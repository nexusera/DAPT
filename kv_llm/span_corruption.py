"""Decoder-only span corruption for KV-LLM continued pretraining."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from .constants import PERFECT_VALUES, build_sentinel_tokens
from noise_feature_processor import FEATURES, NoiseFeatureProcessor
from noise_fusion import needs_bucket_ids, uses_continuous_noise


def load_entity_dictionary(path: str | Path | Sequence[str | Path] | None) -> list[str]:
    if not path:
        return []
    paths: list[Path]
    if isinstance(path, (str, Path)):
        paths = [Path(path)]
    else:
        paths = [Path(p) for p in path]
    terms: list[str] = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                term = line.strip()
                if term:
                    terms.append(term)
    terms = sorted(set(terms), key=len, reverse=True)
    return terms


def _fallback_spans(text: str, *, max_spans: int, rng: random.Random) -> list[tuple[int, int]]:
    chunks = [(m.start(), m.end()) for m in re.finditer(r"[\u4e00-\u9fffA-Za-z0-9]{2,12}", text)]
    rng.shuffle(chunks)
    return sorted(chunks[:max_spans])


def select_random_spans(
    text: str,
    *,
    mask_prob: float,
    max_spans: int,
    rng: random.Random,
    span_len_range: tuple[int, int] = (2, 12),
) -> list[tuple[int, int]]:
    """SC2-A random-token / random-span mask: ignore entity dictionary,
    sample non-overlapping spans of length in [lo, hi] uniformly. Used to
    isolate the contribution of 'entity-aware' vs 'just any span' masking
    (plan \u00a710 SC2 setting A)."""
    lo, hi = span_len_range
    n = len(text)
    if n < lo:
        return []
    n_target = max(1, int(round((n * mask_prob) / ((lo + hi) / 2))))
    n_target = min(n_target, max_spans)
    occupied: list[tuple[int, int]] = []
    selected: list[tuple[int, int]] = []
    for _ in range(n_target * 5):  # bounded retry
        if len(selected) >= n_target:
            break
        span_len = rng.randint(lo, hi)
        if span_len > n:
            continue
        start = rng.randint(0, n - span_len)
        end = start + span_len
        if any(start < b and end > a for a, b in occupied):
            continue
        selected.append((start, end))
        occupied.append((start, end))
    return sorted(selected) or _fallback_spans(text, max_spans=max_spans, rng=rng)


def select_entity_spans(
    text: str,
    entities: Sequence[str],
    *,
    mask_prob: float,
    max_spans: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    occupied: list[tuple[int, int]] = []
    for term in entities:
        start = 0
        while True:
            idx = text.find(term, start)
            if idx < 0:
                break
            end = idx + len(term)
            if not any(idx < b and end > a for a, b in occupied):
                candidates.append((idx, end))
                occupied.append((idx, end))
            start = idx + max(1, len(term))
    if not candidates:
        return _fallback_spans(text, max_spans=max_spans, rng=rng)
    selected = [span for span in candidates if rng.random() < mask_prob]
    if not selected:
        selected = [rng.choice(candidates)]
    selected = sorted(selected[:max_spans])
    return selected


def build_span_corruption_text(
    text: str,
    spans: Sequence[tuple[int, int]],
    *,
    sentinels: Sequence[str],
) -> tuple[str, str]:
    """Return `(source, target)` in T5-style sentinel format."""
    if not spans:
        return text, ""
    source_parts: list[str] = []
    target_parts: list[str] = []
    cursor = 0
    for i, (start, end) in enumerate(spans):
        sentinel = sentinels[i]
        source_parts.append(text[cursor:start])
        source_parts.append(sentinel)
        target_parts.append(sentinel)
        target_parts.append(text[start:end])
        cursor = end
    source_parts.append(text[cursor:])
    target_parts.append(sentinels[len(spans)])
    return "".join(source_parts), "".join(target_parts)


@dataclass
class SpanCorruptionCollator:
    tokenizer: PreTrainedTokenizerBase
    entity_terms: Sequence[str]
    max_length: int = 512
    mask_prob: float = 0.15
    max_spans: int = 24
    seed: int = 42
    instruction: str = "Recover the masked medical spans."
    plain_clm: bool = False
    random_mask: bool = False  # SC2-A: random span mask instead of entity-aware
    noise_mode: str = "bucket"
    noise_processor: NoiseFeatureProcessor | None = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.sentinels = build_sentinel_tokens(max(self.max_spans + 2, 100))

    def _build_example(self, text: str) -> tuple[str, str]:
        if self.plain_clm:
            return text, ""
        if self.random_mask:
            spans = select_random_spans(
                text,
                mask_prob=self.mask_prob,
                max_spans=self.max_spans,
                rng=self.rng,
            )
        else:
            spans = select_entity_spans(
                text,
                self.entity_terms,
                mask_prob=self.mask_prob,
                max_spans=self.max_spans,
                rng=self.rng,
            )
        source, target = build_span_corruption_text(text, spans, sentinels=self.sentinels)
        prompt = f"{self.instruction}\n\nInput:\n{source}\n\nAnswer:\n"
        return prompt, target

    def _noise_tensors(self, features: list[dict[str, Any]], seq_len: int) -> dict[str, torch.Tensor]:
        rows: list[list[list[float]]] = []
        for item in features:
            noise = item.get("noise_values")
            if (
                isinstance(noise, list)
                and len(noise) == len(FEATURES)
                and all(not isinstance(x, (list, tuple)) for x in noise)
            ):
                vec = [float(x) for x in noise]
            else:
                vec = list(PERFECT_VALUES)
            rows.append([vec] * seq_len)
        mode = str(self.noise_mode or "bucket").lower()
        if needs_bucket_ids(mode):
            processor = self.noise_processor or NoiseFeatureProcessor()
            ids = [processor.map_batch(x) for x in rows]
            return {"noise_ids": torch.tensor(ids, dtype=torch.long)}
        if uses_continuous_noise(mode):
            return {"noise_values": torch.tensor(rows, dtype=torch.float32)}
        return {}

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompts: list[str] = []
        targets: list[str] = []
        for item in features:
            text = str(item.get("text", ""))
            prompt, target = self._build_example(text)
            prompts.append(prompt)
            targets.append(target)

        if self.plain_clm:
            batch = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = -100
            batch["labels"] = labels
            batch["task_type"] = "clm"
            batch.update(self._noise_tensors(features, batch["input_ids"].shape[1]))
            return batch

        texts = [p + t for p, t in zip(prompts, targets)]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        prompt_lens = [
            len(self.tokenizer(p, add_special_tokens=False)["input_ids"])
            for p in prompts
        ]
        for i, plen in enumerate(prompt_lens):
            labels[i, : min(plen, labels.shape[1])] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        batch["task_type"] = "span"
        batch.update(self._noise_tensors(features, batch["input_ids"].shape[1]))
        return batch
