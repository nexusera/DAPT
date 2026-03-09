#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Debug KV-NER label alignment for a given config/tokenizer.

Goal: determine whether gold labels are being aligned into token labels correctly.
This helps diagnose cases like variant t2 where KV-NER F1 becomes 0.

Usage:
  python debug_kvner_label_alignment.py \
    --config /data/ocean/DAPT/experiments/downstream_finetune/generated_configs/kv_ner_config_t2.json \
    --split train --max_samples 200

It prints:
- number of samples loaded
- number of chunks (after optional chunking)
- token-level label distribution (excluding padding)
- non-O ratio
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _load_config(path: Path) -> Dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--max_samples", type=int, default=200)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)

    # Import project modules (works when run from repo root or anywhere)
    import os, sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from transformers import AutoTokenizer

    from dapt_eval_package.pre_struct.kv_ner import config_io
    from dapt_eval_package.pre_struct.kv_ner.data_utils import build_bio_label_list
    from dapt_eval_package.pre_struct.kv_ner.train_with_noise import load_jsonl_with_noise
    from dapt_eval_package.pre_struct.kv_ner.dataset import TokenClassificationDataset

    train_block = config_io.ensure_block(cfg, "train")
    label_map = config_io.label_map_from(cfg)
    label_list = build_bio_label_list(label_map)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    tokenizer_name = config_io.tokenizer_name_from(cfg)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if args.split == "train":
        data_path = Path(train_block.get("data_path"))
    elif args.split == "val":
        vp = str(train_block.get("val_data_path") or "").strip()
        if not vp:
            raise SystemExit("val_data_path is empty in config; choose split=train or split=test")
        data_path = Path(vp)
    else:
        tp = str(train_block.get("test_data_path") or "").strip()
        if not tp:
            raise SystemExit("test_data_path is empty in config")
        data_path = Path(tp)

    if not data_path.exists():
        raise SystemExit(f"data_path not found: {data_path}")

    samples = load_jsonl_with_noise(data_path, label_map, include_unlabeled=False)
    samples = [s for s in samples if s.has_labels]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    max_len = config_io.max_seq_length(cfg)
    enable_chunking = bool(cfg.get("enable_chunking", True))
    chunk_size = int(cfg.get("chunk_size", max_len))
    chunk_overlap = int(cfg.get("chunk_overlap", 0))

    ds = TokenClassificationDataset(
        samples,
        tokenizer,
        label2id,
        max_seq_length=max_len,
        label_all_tokens=config_io.label_all_tokens(cfg),
        include_labels=True,
        enable_chunking=enable_chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        noise_processor=None,
    )

    o_id = label2id["O"]
    unk_id = getattr(tokenizer, "unk_token_id", None)
    counts = Counter()
    total_active = 0
    total_non_o = 0
    total_unk = 0
    total_zero_offset = 0

    for feat in getattr(ds, "_features", []):
        attn = feat["attention_mask"].tolist()
        input_ids = feat["input_ids"].tolist()
        labels = feat["labels"].tolist()
        offsets = feat.get("offset_mapping")
        offsets_list = offsets.tolist() if offsets is not None else None
        for idx, (m, tid, y) in enumerate(zip(attn, input_ids, labels)):
            if int(m) != 1:
                continue
            total_active += 1
            counts[int(y)] += 1
            if int(y) != o_id:
                total_non_o += 1
            if unk_id is not None and int(tid) == int(unk_id):
                total_unk += 1
            if offsets_list is not None:
                s, e = offsets_list[idx]
                if int(e) <= int(s):
                    total_zero_offset += 1

    print("=" * 80)
    print(f"config: {cfg_path}")
    print(f"split: {args.split}  max_samples: {args.max_samples}")
    print(f"tokenizer: {tokenizer_name}")
    print(f"tokenizer_vocab: {len(tokenizer)}")
    print(f"samples_loaded(labeled): {len(samples)}")
    print(f"chunks: {len(getattr(ds, '_features', []))}")
    print(f"active_tokens(attention_mask=1): {total_active}")
    print(f"non_O_tokens: {total_non_o}  ratio: {total_non_o / max(1, total_active):.6f}")
    print(f"zero_offset_tokens(end<=start): {total_zero_offset}  ratio: {total_zero_offset / max(1, total_active):.6f}")
    if unk_id is not None:
        print(f"unk_tokens: {total_unk}  ratio: {total_unk / max(1, total_active):.6f}")
    print("label_distribution (top 20):")
    for lid, c in counts.most_common(20):
        print(f"  {lid:>3}  {id2label.get(lid, '??'):<12}  {c}")
    print("=" * 80)


if __name__ == "__main__":
    main()
