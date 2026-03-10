#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Debug KV-NER prediction distribution for a trained model.

This complements debug_kvner_label_alignment.py:
- label_alignment: checks gold label distribution after alignment
- pred_distribution: checks decoded predictions distribution from a trained 'best' model

Usage:
  python debug_kvner_pred_distribution.py \
    --config /data/ocean/DAPT/experiments/downstream_finetune/generated_configs/kv_ner_config_t2.json \
    --split test --max_chunks 200

Defaults align with RUNBOOK_DOWNSTREAM_TOKENIZER_VARIANTS.md.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_config(path: Path) -> Dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return obj


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--max_chunks", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument(
        "--noise_bins",
        type=str,
        default="/data/ocean/DAPT/workspace/noise_bins.json",
        help="Optional; if exists, will generate noise_ids for inference",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)

    import os, sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch
    from transformers import AutoTokenizer

    from dapt_eval_package.pre_struct.kv_ner import config_io
    from dapt_eval_package.pre_struct.kv_ner.data_utils import build_bio_label_list
    from dapt_eval_package.pre_struct.kv_ner.train_with_noise import load_jsonl_with_noise
    from dapt_eval_package.pre_struct.kv_ner.dataset import TokenClassificationDataset
    from dapt_eval_package.pre_struct.kv_ner.modeling import BertCrfTokenClassifier
    from dapt_eval_package.pre_struct.kv_ner.noise_utils import NoiseFeatureProcessor

    train_block = config_io.ensure_block(cfg, "train")
    label_map = config_io.label_map_from(cfg)
    label_list = build_bio_label_list(label_map)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    o_id = label2id["O"]

    # Tokenizer
    tokenizer_name = config_io.tokenizer_name_from(cfg)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Data path
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

    # Noise processor (optional)
    noise_processor = None
    nb_path = Path(args.noise_bins)
    if nb_path.exists():
        try:
            noise_processor = NoiseFeatureProcessor.load(str(nb_path))
        except Exception:
            noise_processor = None

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
        noise_processor=noise_processor,
    )

    feats: List[Dict[str, Any]] = getattr(ds, "_features", [])
    if args.max_chunks > 0:
        feats = feats[: args.max_chunks]

    # Model dir resolution: prefer train.output_dir/best
    out_dir = str(train_block.get("output_dir") or "").strip()
    candidates = []
    if out_dir:
        candidates.append(Path(out_dir) / "best")
        candidates.append(Path(out_dir))
    # Also try canonical runs layout
    # (If output_dir is empty, this will still likely fail clearly)

    model_dir = _first_existing(candidates)
    if model_dir is None:
        raise SystemExit(f"Cannot find trained model dir. Tried: {candidates}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertCrfTokenClassifier.from_pretrained(str(model_dir), map_location=("cpu" if device.type == "cpu" else None))
    model.to(device)
    model.eval()

    pred_counts = Counter()
    total_active = 0
    total_non_o = 0

    def _batch(iterable, n):
        buf = []
        for x in iterable:
            buf.append(x)
            if len(buf) >= n:
                yield buf
                buf = []
        if buf:
            yield buf

    with torch.no_grad():
        for batch_feats in _batch(feats, args.batch_size):
            input_ids = torch.stack([f["input_ids"] for f in batch_feats], dim=0).to(device)
            attention_mask = torch.stack([f["attention_mask"] for f in batch_feats], dim=0).to(device)
            token_type_ids = torch.stack([f["token_type_ids"] for f in batch_feats], dim=0).to(device)
            noise_ids = None
            if "noise_ids" in batch_feats[0]:
                try:
                    noise_ids = torch.stack([f.get("noise_ids") for f in batch_feats], dim=0).to(device)
                except Exception:
                    noise_ids = None

            decoded = model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                noise_ids=noise_ids,
            )
            # decoded: List[List[int]] length B, each length = active tokens length (CRF decode ignores padding)

            # Count per-token labels over active mask positions
            # CRF decode returns variable lengths; rely on attention_mask sum
            lengths = attention_mask.sum(dim=1).tolist()
            for seq, L in zip(decoded, lengths):
                L = int(L)
                seq = seq[:L]
                total_active += L
                for y in seq:
                    pred_counts[int(y)] += 1
                    if int(y) != o_id:
                        total_non_o += 1

    print("=" * 80)
    print(f"config: {cfg_path}")
    print(f"split: {args.split}  max_samples: {args.max_samples}  max_chunks: {args.max_chunks}")
    print(f"tokenizer: {tokenizer_name}")
    print(f"tokenizer_vocab: {len(tokenizer)}")
    print(f"model_dir: {model_dir}")
    print(f"device: {device}")
    print(f"chunks_used: {len(feats)}")
    print(f"active_tokens: {total_active}")
    print(f"pred_non_O_tokens: {total_non_o}  ratio: {total_non_o / max(1, total_active):.6f}")
    print("pred_label_distribution (top 20):")
    for lid, c in pred_counts.most_common(20):
        print(f"  {lid:>3}  {id2label.get(lid, '??'):<12}  {c}")
    print("=" * 80)


if __name__ == "__main__":
    main()
