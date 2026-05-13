#!/usr/bin/env python3
"""
根据分源文件做静态重采样，生成新的 train_resampled.txt。

示例：
  python scripts/data/resample_mix.py \
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
import gzip


def read_lines(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", type=str, required=True)
    ap.add_argument("--book_core", type=str, required=True)
    ap.add_argument("--book_old", type=str, required=True)
    ap.add_argument("--paper", type=str, required=True)
    ap.add_argument("--general", type=str, required=True)
    ap.add_argument("--supplement", type=str, required=True)
    ap.add_argument("--wiki_med", type=str, required=False)
    ap.add_argument("--wiki_general", type=str, required=False)
    ap.add_argument("--med_book", type=str, required=False)
    ap.add_argument("--general2", type=str, required=False, help="额外通用语料，如 fineweb 抽样")
    ap.add_argument(
        "--weights",
        type=float,
        nargs="+",
        required=True,
        help="按提供的来源顺序给出权重："
             "clinical/book_core/book_old/paper/general/supplement/[wiki_med]/[wiki_general]/[med_book]/[general2]",
    )
    ap.add_argument("--total_lines", type=int, default=None, help="目标总行数；默认为各源行数加权和的整数")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（用于复现实验/论文统计）")
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    sources = [
        ("clinical", args.clinical),
        ("book_core", args.book_core),
        ("book_old", args.book_old),
        ("paper", args.paper),
        ("general", args.general),
        ("supplement", args.supplement),
    ]
    if args.wiki_med:
        sources.append(("wiki_med", args.wiki_med))
    if args.wiki_general:
        sources.append(("wiki_general", args.wiki_general))
    if args.med_book:
        sources.append(("med_book", args.med_book))
    if args.general2:
        sources.append(("general2", args.general2))

    if len(args.weights) != len(sources):
        raise ValueError(f"weights 数量({len(args.weights)})需与来源数量({len(sources)})一致")

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

    sampled_counts = [0 for _ in sources]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for _ in range(total):
            # 按权重选择来源（使用动态来源数量）
            src_idx = random.choices(range(len(data)), weights=args.weights, k=1)[0]
            pool = data[src_idx]
            if not pool:
                continue
            line = random.choice(pool)
            fout.write(line + "\n")
            sampled_counts[src_idx] += 1

    print(f"Done. Saved to {args.output}")

    # 打印实际抽样比例（便于论文直接引用）
    print("\n抽样后来源分布:")
    total_sampled = sum(sampled_counts)
    for (name, _), n in zip(sources, sampled_counts):
        pct = (n / total_sampled * 100.0) if total_sampled else 0.0
        print(f"  - {name}: {n} lines ({pct:.2f}%)")


if __name__ == "__main__":
    main()
