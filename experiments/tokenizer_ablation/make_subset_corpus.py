#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create a smaller corpus file for quick ablation runs.

- 默认取前 N 行（不打乱，保证可复现）
- 可选按固定 seed 做随机采样
"""

import argparse
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--lines", type=int, default=20000)
    ap.add_argument("--random_sample", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    with open(args.input, "r", encoding="utf-8") as f:
        all_lines = [ln for ln in f if ln.strip()]

    if args.lines <= 0:
        raise ValueError("--lines must be > 0")

    if args.random_sample:
        random.seed(args.seed)
        if args.lines < len(all_lines):
            sampled = random.sample(all_lines, args.lines)
        else:
            sampled = all_lines
    else:
        sampled = all_lines[: args.lines]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(sampled)

    print(f"[subset] in={len(all_lines)} out={len(sampled)} -> {args.output}")


if __name__ == "__main__":
    main()
