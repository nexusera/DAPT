#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build Jieba userdict for each tokenizer variant.

为什么要每个变体一份：
- `build_dataset_final_slim.py` 的 word_ids 依赖 Jieba 分词。
- Tokenizer 变体如果注入了新词，但 Jieba 词典没同步，会造成消融不干净（confound）。

输出格式：每行 `word 99999`，保证 Jieba 优先切分。
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, Set

DEFAULT_VIP_TERMS = {
    "brca1基因", "brca2基因", "her2基因", "fish检测",
    "er阳性", "pr阳性", "p53蛋白", "ptnm分期",
}


def iter_vocab_lines(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip()
            if tok:
                yield tok


def main():
    ap = argparse.ArgumentParser(description="Build jieba userdict for tokenizer ablation")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=6)
    ap.add_argument("--vip_terms", type=str, default=None)
    ap.add_argument("--ocr_vocab", type=str, default=None)
    ap.add_argument("--keys_vocab", type=str, default=None, help="建议用 min5 keys 文件")

    args = ap.parse_args()

    vip_terms: Set[str] = set(DEFAULT_VIP_TERMS)
    if args.vip_terms:
        vip_terms = set(iter_vocab_lines(args.vip_terms))

    sources = []
    if args.ocr_vocab:
        if not os.path.exists(args.ocr_vocab):
            raise FileNotFoundError(f"ocr_vocab not found: {args.ocr_vocab}")
        sources.append(args.ocr_vocab)
    if args.keys_vocab:
        if not os.path.exists(args.keys_vocab):
            raise FileNotFoundError(f"keys_vocab not found: {args.keys_vocab}")
        sources.append(args.keys_vocab)

    words: Set[str] = set(vip_terms)
    vip_lower = {v.lower() for v in vip_terms}

    for src in sources:
        for tok in iter_vocab_lines(src):
            if " " in tok:
                continue
            if tok.lower() in vip_lower:
                words.add(tok)
                continue
            if len(tok) > args.max_len:
                continue
            if len(tok) > 1:
                words.add(tok)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for w in sorted(words):
            f.write(f"{w} 99999\n")

    print(f"[save] {out_path} words={len(words)}")


if __name__ == "__main__":
    main()
