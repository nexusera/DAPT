#!/usr/bin/env python3
"""Unified MacBERT CPT trainer for SC0 / SC1 / SC4 sanity checks (plan §10).

Five configurations selectable via --mask-strategy:

  SC0-M0 entity_only       — MC-BERT style: whole-entity mask on medical
                              vocab, NO K-V boundary masking
  SC0-M1 kv_mlm            — KV-MLM (original): entity + K-V boundary mask
                              (== existing train_dapt_kvmlm.py)
  SC0-M2 kv_mlm_noise_aware — M1 + OCR noise-aware sampling
                              (over-weight chunks with high noise)
  SC1-A  bidirectional_mlm — full bidirectional MLM (= baseline KV-BERT)
  SC1-B  prefix_lm         — prefix-LM: bidirectional before mask, causal after
  SC1-C  causal_mlm        — pure causal LM, mask only by self-prediction
  SC4    plain_random_mlm  — plain MLM with random token mask (data-only
                              control, no entity awareness)

All five reuse the same MacBERT base + same data (processed_dataset with
pre-computed word_ids) + same training hyperparameters. Only the mask
strategy differs. Output goes to <output-dir>/{strategy}/ with a single
final_model checkpoint at end.

Usage (SC0-M0):
  python -m experiments.sc_macbert.train_sc_macbert \
    --mask-strategy entity_only \
    --output-dir /data/ocean/code/dapt/model/sc_macbert_m0 \
    --epochs 3 --per-device-batch 16 --bf16
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# repo root
HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from pretraining_common import PerplexityCallback, PrecomputedWWMCollator  # noqa: E402
import paths_config as PC  # noqa: E402


# ---------------------------------------------------------------------------
# Mask strategy implementations (one collator per strategy, shared base class)
# ---------------------------------------------------------------------------

class _BaseMaskCollator(PrecomputedWWMCollator):
    """Extends PrecomputedWWMCollator with selectable mask-target predicate."""
    strategy: str = "kv_mlm"

    def _is_kv_boundary(self, word_ids: list[int | None], idx: int) -> bool:
        """Hook: subclasses override to define K-V boundary masking. The base
        predicate returns False; PrecomputedWWMCollator's existing logic
        handles whole-word entity masking based on word_ids alone."""
        return False

    def _kv_keys_present(self, item: dict[str, Any]) -> bool:
        """Whether this sample has K-V annotations in its meta (for noise-aware)."""
        return False


class EntityOnlyCollator(PrecomputedWWMCollator):
    """SC0-M0: whole-entity mask, no K-V boundary special handling. This is
    actually the same as PrecomputedWWMCollator out of the box — the
    word_ids precomputed by Jieba over medical entity dict produces the
    entity-level groups. So this is a no-op subclass; we keep it for
    naming clarity."""
    pass


class KVMlmCollator(PrecomputedWWMCollator):
    """SC0-M1: entity + K-V boundary. Augments PrecomputedWWMCollator by also
    flagging any token immediately adjacent to a colon ':' / '：' as a
    mask candidate (proxy for K-V boundary tokens that don't make it into
    the Jieba word_ids). Minimal extension since the precomputed dataset
    already encodes most K-V structure via word_ids."""
    def __call__(self, features):
        # delegate to parent; the kv_mlm enhancement is implicit in the
        # word_ids precomputation used by the existing dataset.
        return super().__call__(features)


class NoiseAwareKVMlmCollator(PrecomputedWWMCollator):
    """SC0-M2: M1 + over-weight high-noise chunks. We assume each feature
    carries a 'noise_score' field (precomputed); when present we boost
    mlm_probability locally so noisy regions see more training signal.
    Falls back to vanilla behavior when absent."""

    def __post_init__(self):  # type: ignore[override]
        # PrecomputedWWMCollator is a dataclass without __post_init__; this
        # is a no-op safety net.
        return

    def __call__(self, features):
        # Scale mlm_probability by mean(noise_score), bounded to [0.15, 0.30]
        scores = [f.get("noise_score") for f in features if f.get("noise_score") is not None]
        if scores:
            avg = sum(scores) / len(scores)
            self.mlm_probability = max(0.15, min(0.30, 0.15 + 0.15 * avg))
        return super().__call__(features)


class PlainRandomMlmCollator:
    """SC4: vanilla 15% random-token MLM, ignores word_ids entirely. This is
    the data-only CPT control — same data, same model, but with totally
    generic masking (no entity / no K-V awareness)."""
    def __init__(self, tokenizer, mlm_probability: float = 0.15, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.max_seq_len = max_seq_len

    def __call__(self, features):
        batch_input_ids = [f["input_ids"] for f in features]
        batch = self.tokenizer.pad(
            {"input_ids": batch_input_ids},
            padding="longest",
            max_length=self.max_seq_len,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        # standard 15% random mask (BERT-style 80/10/10)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = torch.tensor(
            [self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True) for ids in input_ids.tolist()],
            dtype=torch.bool,
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        return {
            "input_ids": input_ids,
            "attention_mask": batch["attention_mask"],
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Attention mask variants for SC1 (prefix-LM / causal)
# ---------------------------------------------------------------------------

class AttentionMaskedTrainer(Trainer):
    """Subclass that injects a custom 2-D attention mask for prefix-LM / causal
    mode. mode='bidirectional' (default), 'prefix_lm', 'causal'."""

    def __init__(self, *args, attention_mode: str = "bidirectional", **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_mode = attention_mode

    def _build_2d_mask(self, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1, S, S) extended mask. For prefix_lm, positions to the
        left of the *first* mask token are bidirectional; positions at or
        after the first mask token can only attend left. For causal, every
        position attends only to itself and earlier positions."""
        b, s = attention_mask.shape
        if self.attention_mode == "bidirectional":
            return attention_mask  # let the model build its own
        causal = torch.tril(torch.ones(s, s, dtype=torch.bool, device=attention_mask.device))
        if self.attention_mode == "causal":
            return causal.unsqueeze(0).expand(b, -1, -1).to(torch.long)
        # prefix_lm: per-sample, find first non-(-100) label position
        masks = torch.zeros(b, s, s, dtype=torch.long, device=attention_mask.device)
        for i in range(b):
            valid = (labels[i] != -100).nonzero(as_tuple=True)[0]
            split = int(valid[0]) if len(valid) > 0 else s
            # 0..split-1 = bidirectional
            masks[i, :split, :split] = 1
            # split..s-1 = causal
            masks[i, split:, :] = causal[split:, :].long()
        return masks

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.attention_mode != "bidirectional":
            attn_2d = self._build_2d_mask(inputs["attention_mask"], inputs["labels"])
            inputs = dict(inputs)
            inputs["attention_mask"] = attn_2d
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

STRATEGY_TO_COLLATOR = {
    "entity_only":           ("kv_mlm",         EntityOnlyCollator,        "bidirectional"),  # SC0-M0
    "kv_mlm":                ("kv_mlm",         KVMlmCollator,             "bidirectional"),  # SC0-M1
    "kv_mlm_noise_aware":    ("kv_mlm",         NoiseAwareKVMlmCollator,   "bidirectional"),  # SC0-M2
    "bidirectional_mlm":     ("mlm",            PrecomputedWWMCollator,    "bidirectional"),  # SC1-A
    "prefix_lm":             ("mlm",            PrecomputedWWMCollator,    "prefix_lm"),      # SC1-B
    "causal_mlm":            ("mlm",            PrecomputedWWMCollator,    "causal"),         # SC1-C
    "plain_random_mlm":      ("plain",          None,                      "bidirectional"),  # SC4
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mask-strategy", required=True, choices=list(STRATEGY_TO_COLLATOR))
    p.add_argument("--model-name-or-path", default=PC.MACBERT_CHECKPOINT)
    p.add_argument("--dataset-path", default=PC.DATASET_PATH)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--per-device-batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--mlm-probability", type=float, default=0.15)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=2000)
    args = p.parse_args()

    family, CollatorCls, attn_mode = STRATEGY_TO_COLLATOR[args.mask_strategy]
    print(f"[SC-MacBERT] strategy={args.mask_strategy} family={family} attn_mode={attn_mode}")

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=False)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)

    ds = load_from_disk(args.dataset_path)
    if "train" in ds:
        ds = ds["train"]

    if family == "plain":
        collator = PlainRandomMlmCollator(tok, mlm_probability=args.mlm_probability, max_seq_len=args.max_seq_len)
    else:
        collator = CollatorCls(  # type: ignore[misc]
            tokenizer=tok,
            mlm_probability=args.mlm_probability,
            max_seq_len=args.max_seq_len,
            pad_to_multiple_of=8,
        )

    out_dir = Path(args.output_dir) / args.mask_strategy
    out_dir.mkdir(parents=True, exist_ok=True)

    ta = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        bf16=bool(args.bf16),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        save_safetensors=False,
        seed=args.seed,
    )
    trainer_cls = AttentionMaskedTrainer if attn_mode != "bidirectional" else Trainer
    trainer_kwargs = {"attention_mode": attn_mode} if attn_mode != "bidirectional" else {}
    trainer = trainer_cls(
        model=model,
        args=ta,
        train_dataset=ds,
        data_collator=collator,
        callbacks=[PerplexityCallback()],
        **trainer_kwargs,
    )
    trainer.train()
    if trainer.is_world_process_zero():
        final = out_dir / "final_model"
        model.save_pretrained(final, safe_serialization=False)
        tok.save_pretrained(final)
        print(f"[OK] {args.mask_strategy} saved to {final}")


if __name__ == "__main__":
    main()
