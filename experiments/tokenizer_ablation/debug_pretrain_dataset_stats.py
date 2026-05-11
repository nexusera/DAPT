#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer


def _iter_dataset(ds, n: int, seed: int):
    rng = random.Random(seed)
    total = len(ds)
    if n <= 0 or n >= total:
        for i in range(total):
            yield ds[i]
        return
    for idx in rng.sample(range(total), n):
        yield ds[idx]


def _simulate_wwm_mask_fraction(word_ids: List[Optional[int]], mlm_probability: float, seed: int) -> float:
    mapping: Dict[int, List[int]] = {}
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        mapping.setdefault(int(wid), []).append(i)

    if not mapping:
        return float(mlm_probability)

    unique_words = list(mapping.keys())
    num_to_mask = max(1, int(len(unique_words) * mlm_probability))

    rng = random.Random(seed)
    masked_words = set(rng.sample(unique_words, min(num_to_mask, len(unique_words))))

    masked_tokens = 0
    total_tokens = 0
    for wid, positions in mapping.items():
        total_tokens += len(positions)
        if wid in masked_words:
            masked_tokens += len(positions)

    return float(masked_tokens) / float(max(1, total_tokens))


def main():
    ap = argparse.ArgumentParser(description="Debug stats for pretrain dataset (MLM) to diagnose collapse")
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument("--base_model", type=str, default="hfl/chinese-macbert-base")
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"])
    ap.add_argument("--num_samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlm_probability", type=float, default=0.15)

    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    cfg = AutoConfig.from_pretrained(args.base_model)
    base_vocab_size = int(getattr(cfg, "vocab_size", 0))

    ds_disk = load_from_disk(args.dataset_path)
    ds = ds_disk[args.split] if isinstance(ds_disk, dict) and args.split in ds_disk else ds_disk

    unk_id = tokenizer.unk_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    lengths: List[int] = []
    unk_counts: List[int] = []
    token_counts: List[int] = []
    new_token_counts: List[int] = []
    unique_word_counts: List[int] = []
    wwm_mask_fracs: List[float] = []
    wid_group_sizes: List[int] = []

    sampled = 0
    for ex in _iter_dataset(ds, args.num_samples, args.seed):
        input_ids = ex.get("input_ids")
        word_ids = ex.get("word_ids")
        if not input_ids or not word_ids:
            continue
        if len(input_ids) != len(word_ids):
            continue

        sampled += 1
        lengths.append(len(input_ids))

        special = {x for x in [cls_id, sep_id, pad_id] if x is not None}
        content_ids = [int(tid) for tid in input_ids if int(tid) not in special]
        token_counts.append(len(content_ids))

        if unk_id is not None:
            unk_counts.append(sum(1 for tid in content_ids if tid == int(unk_id)))
        else:
            unk_counts.append(0)

        if base_vocab_size > 0:
            new_token_counts.append(sum(1 for tid in content_ids if tid >= base_vocab_size))
        else:
            new_token_counts.append(0)

        mapping: Dict[int, int] = {}
        for wid in word_ids:
            if wid is None:
                continue
            mapping[int(wid)] = mapping.get(int(wid), 0) + 1
        unique_word_counts.append(len(mapping))
        wid_group_sizes.extend(list(mapping.values()))

        wwm_mask_fracs.append(_simulate_wwm_mask_fraction(word_ids, args.mlm_probability, seed=args.seed + sampled))

    def pct(a: int, b: int) -> float:
        return 0.0 if b <= 0 else 100.0 * float(a) / float(b)

    total_tokens = int(sum(token_counts))
    total_unk = int(sum(unk_counts))
    total_new = int(sum(new_token_counts))

    print("=" * 80)
    print(f"dataset_path={args.dataset_path}")
    print(f"split={args.split} sampled={sampled} (requested={args.num_samples})")
    print(f"tokenizer_path={args.tokenizer_path}")
    print(f"tokenizer_len={len(tokenizer)} base_model={args.base_model} base_vocab_size={base_vocab_size}")
    print("-" * 80)

    if sampled == 0:
        print("No valid samples found (missing input_ids/word_ids or length mismatch).")
        return

    print(f"seq_len: mean={statistics.mean(lengths):.1f} p50={statistics.median(lengths):.0f} min={min(lengths)} max={max(lengths)}")
    print(f"content_tokens: total={total_tokens} mean={statistics.mean(token_counts):.1f}")
    print(f"UNK: total={total_unk} ratio={pct(total_unk, total_tokens):.3f}%")
    if base_vocab_size > 0:
        print(f"NEW_TOKEN(id>=base_vocab): total={total_new} ratio={pct(total_new, total_tokens):.3f}%")

    print("-" * 80)
    print(f"unique_word_ids: mean={statistics.mean(unique_word_counts):.1f} p50={statistics.median(unique_word_counts):.0f} min={min(unique_word_counts)} max={max(unique_word_counts)}")
    if wid_group_sizes:
        print(f"word_group_size(#subtokens per wid): mean={statistics.mean(wid_group_sizes):.2f} p50={statistics.median(wid_group_sizes):.0f} p95={np.percentile(wid_group_sizes,95):.0f} max={max(wid_group_sizes)}")

    print("-" * 80)
    print(f"simulated_wwm_mask_fraction(p={args.mlm_probability}): mean={statistics.mean(wwm_mask_fracs):.3f} p50={statistics.median(wwm_mask_fracs):.3f} min={min(wwm_mask_fracs):.3f} max={max(wwm_mask_fracs):.3f}")

    c = Counter(wid_group_sizes)
    common = ", ".join([f"{k}:{v}" for k, v in c.most_common(8)])
    print(f"word_group_size histogram(top): {common}")


if __name__ == "__main__":
    main()
