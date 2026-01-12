#!/usr/bin/env python3
"""
根据分源文件做静态重采样，生成新的 train_resampled.txt。

示例：
  python resample_mix.py \
    --clinical /data/ocean/bpe_workspace/train_clinical.txt \
    --book_core /data/ocean/bpe_workspace/train_book_core.txt \
    --book_old /data/ocean/bpe_workspace/train_book_old.txt \
    --paper /data/ocean/bpe_workspace/train_paper.txt \
    --general /data/ocean/bpe_workspace/train_general.txt \
    --supplement /data/ocean/bpe_workspace/train_supplement.txt \
    --weights 0.25 0.35 0.05 0.1 0.2 0.05 \
    --output /data/ocean/bpe_workspace/train_resampled.txt

说明：
- 按权重从各源均匀随机抽样，抽到指定总行数（默认为各源行数加权和）。
- 若某源不足，其样本会被重复抽（带放回）。
- 输出为 txt（每行一条），可后续用于 chunk_long_lines.py 进一步切分。
"""

import argparse
import os
import random


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", type=str, required=True)
    ap.add_argument("--book_core", type=str, required=True)
    ap.add_argument("--book_old", type=str, required=True)
    ap.add_argument("--paper", type=str, required=True)
    ap.add_argument("--general", type=str, required=True)
    ap.add_argument("--supplement", type=str, required=True)
    ap.add_argument("--weights", type=float, nargs=6, required=True, help="六路权重，按 clinical/book_core/book_old/paper/general/supplement 顺序")
    ap.add_argument("--total_lines", type=int, default=None, help="目标总行数；默认为各源行数加权和的整数")
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    sources = [
        ("clinical", args.clinical),
        ("book_core", args.book_core),
        ("book_old", args.book_old),
        ("paper", args.paper),
        ("general", args.general),
        ("supplement", args.supplement),
    ]
    if len(args.weights) != 6:
        raise ValueError("weights 必须提供 6 个值，对应六路来源")

    data = []
    counts = []
    for name, path in sources:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")
        lines = read_lines(path)
        data.append(lines)
        counts.append(len(lines))
        print(f"{name}: {len(lines)} lines")

    total_weighted = int(sum(c * w for c, w in zip(counts, args.weights)))
    total = args.total_lines or total_weighted
    print(f"目标总行数: {total} (默认加权和 {total_weighted})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for _ in range(total):
            # 按权重选择来源
            src_idx = random.choices(range(6), weights=args.weights, k=1)[0]
            pool = data[src_idx]
            if not pool:
                continue
            line = random.choice(pool)
            fout.write(line + "\n")

    print(f"Done. Saved to {args.output}")


if __name__ == "__main__":
    main()

