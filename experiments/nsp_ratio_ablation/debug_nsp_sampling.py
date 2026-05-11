#!/usr/bin/env python
import argparse
import os
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kv_nsp"))
from dataset import KVDataset


def parse_args():
    p = argparse.ArgumentParser(description="Debug KV-NSP dynamic sampling distribution")
    p.add_argument("--tokenizer_path", type=str, required=True)
    p.add_argument("--nsp_data_dir", type=str, required=True)
    p.add_argument("--sample_size", type=int, default=200000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative_prob", type=float, default=0.5)
    p.add_argument("--max_easy_retries", type=int, default=10)
    return p.parse_args()


def iter_nsp_files(path_str: str):
    p = Path(path_str)
    if p.is_file():
        return [p]
    return sorted([x for x in p.iterdir() if x.suffix == ".json"])


def run_one(name, reverse_ratio, random_ratio, args, tok):
    ds = KVDataset(
        data_files=iter_nsp_files(args.nsp_data_dir),
        tokenizer=tok,
        negative_prob=args.negative_prob,
        reverse_negative_ratio=reverse_ratio,
        random_negative_ratio=random_ratio,
        max_easy_retries=args.max_easy_retries,
        seed=args.seed,
    )

    n = len(ds)
    sample_n = min(args.sample_size, max(1, n * 2))
    rng = random.Random(args.seed)

    label_counter = Counter()
    strategy_counter = Counter()

    for _ in range(sample_n):
        idx = rng.randrange(n)
        _, _, label, strategy = ds.sample_text_pair(idx)
        label_counter[label] += 1
        strategy_counter[strategy] += 1

    neg = label_counter.get(0, 0)
    pos = label_counter.get(1, 0)
    total = max(1, neg + pos)

    print(f"\n===== {name} (reverse:random={reverse_ratio}:{random_ratio}) =====")
    print(f"pairs={n}, sampled={sample_n}")
    print(f"labels: pos={pos} ({pos/total:.4f}) | neg={neg} ({neg/total:.4f})")
    print("strategies:")
    for k, v in strategy_counter.most_common():
        print(f"  - {k}: {v} ({v/max(1, sample_n):.4f})")


if __name__ == "__main__":
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)

    run_one("r11", 1, 1, args, tok)
    run_one("r31", 3, 1, args, tok)
    run_one("r13", 1, 3, args, tok)
