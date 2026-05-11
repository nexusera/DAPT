#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计：每个 Jieba 词经当前 Tokenizer 切成多少个子词（与 build_dataset_final_slim 对齐逻辑一致）。

用于判断「全词掩码」在数据上是否经常退化为「一词 = 一 token」：
- 若绝大多数 len(tokenize(word)) == 1，则 WWM 与 token 级 MLM 行为接近；
- 若有一定比例 >=2，则 WWM 仍在强制跨子词联合预测。

用法（在 DAPT 目录下）:
  python scripts/stats_jieba_tokenizer_wwm.py \\
    --train_file /data/ocean/DAPT/workspace/train_chunked.txt \\
    --tokenizer_path /data/ocean/DAPT/my-medical-tokenizer \\
    --max_lines 50000
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jieba
from transformers import AutoTokenizer

from build_dataset_final_slim import (
    DEFAULT_KEYS_FILE,
    DEFAULT_TOKENIZER_PATH,
    DEFAULT_TRAIN_FILE,
    DEFAULT_VOCAB_FOR_JIEBA,
    has_chinese,
    init_jieba,
    is_noisy_token,
)

RE_LONG_ALNUM = re.compile(r"^[A-Za-z0-9]{6,}$")


def iter_lines(path: str, max_lines: int | None, seed: int):
    """顺序读取；若 max_lines 给定且文件更长，则均匀下采样（需先数行或两遍扫描）。"""
    if max_lines is None or max_lines <= 0:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line
        return

    # 第一遍计数
    total = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            total += 1

    if total <= max_lines:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line
        return

    rng = random.Random(seed)
    # 水库抽样：从 total 行里抽 max_lines 个下标
    chosen = set(rng.sample(range(total), max_lines))
    idx = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if idx in chosen:
                yield line
            idx += 1


def parse_args():
    p = argparse.ArgumentParser(description="Jieba 词粒度 vs Tokenizer 子词数统计（WWM 诊断）")
    p.add_argument("--train_file", type=str, default=DEFAULT_TRAIN_FILE)
    p.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    p.add_argument("--keys_file", type=str, default=DEFAULT_KEYS_FILE)
    p.add_argument("--vocab_for_jieba", type=str, default=DEFAULT_VOCAB_FOR_JIEBA)
    p.add_argument(
        "--max_lines",
        type=int,
        default=50_000,
        help="最多处理的语料行数；<=0 表示全文件（大文件会较慢）",
    )
    p.add_argument("--seed", type=int, default=42, help="超过 max_lines 时下采样随机种子")
    p.add_argument(
        "--examples",
        type=int,
        default=30,
        help="打印多少条「多子词」的 Jieba 词示例（去重后按首次出现顺序）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    print(f"Tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)

    init_jieba(args.keys_file, args.vocab_for_jieba)

    hist: Counter[int] = Counter()
    lines_used = 0
    jieba_words = 0
    multi_token_words = 0
    examples: list[tuple[str, int, list[str]]] = []
    seen_example: set[str] = set()

    max_lines = None if args.max_lines <= 0 else args.max_lines

    for line in iter_lines(args.train_file, max_lines, args.seed):
        lines_used += 1
        text = line.strip()
        if not text:
            continue
        if not has_chinese(text) and RE_LONG_ALNUM.match(text):
            continue

        for word in jieba.cut(text):
            if is_noisy_token(word):
                continue
            toks = tokenizer.tokenize(word)
            n = len(toks)
            hist[n] += 1
            jieba_words += 1
            if n >= 2:
                multi_token_words += 1
                if word not in seen_example and len(examples) < args.examples:
                    seen_example.add(word)
                    examples.append((word, n, toks))

    print("\n========== Jieba 词 → Tokenizer 子词数 ==========")
    print(f"语料文件: {args.train_file}")
    print(f"处理行数: {lines_used}")
    print(f"计入统计的 Jieba 词总数（过 is_noisy_token 后）: {jieba_words}")
    if jieba_words == 0:
        print("无有效词，请检查路径与编码。")
        return

    print(f"多子词词数 (len>=2): {multi_token_words} ({100.0 * multi_token_words / jieba_words:.2f}%)")
    print(f"单子词占比 (len==1): {100.0 * hist[1] / jieba_words:.2f}%")

    weighted = sum(k * v for k, v in hist.items())
    print(f"平均每 Jieba 词对应子词数: {weighted / jieba_words:.4f}")

    print("\n--- 子词数直方图（前 20 个桶） ---")
    for k in sorted(hist.keys())[:20]:
        bar = "#" * min(50, int(50 * hist[k] / jieba_words) + 1)
        print(f"  len={k:3d}  {hist[k]:10d}  ({100.0 * hist[k] / jieba_words:6.2f}%)  {bar}")
    if len(hist) > 20:
        rest = sum(v for kk, v in hist.items() if kk > 20)
        print(f"  len>20  {rest:10d}  ({100.0 * rest / jieba_words:6.2f}%)")

    if examples:
        print(f"\n--- 多子词示例（最多 {args.examples} 条） ---")
        for word, n, toks in examples:
            print(f"  {word!r}  ->  n={n}  tokens={toks}")


if __name__ == "__main__":
    main()
