#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a shared Jieba-segmented dataset (words only).

Goal (tokenizer ablation purity):
- Use ONE shared Jieba userdict to segment corpus into `words`.
- This fixes word boundaries across T1~T4.
- Tokenizers are applied later to the same `words` to produce input_ids/word_ids.

Output: HuggingFace DatasetDict saved to disk with columns:
- words: List[str]

Notes:
- We keep OCR/no-OCR split shuffling behavior identical to the existing pipeline:
  - non-OCR: may shuffle split
  - OCR: should disable shuffle to preserve alignment order
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List

import jieba
from datasets import Dataset, DatasetDict, load_dataset

RE_LONG_ALNUM = re.compile(r"^[A-Za-z0-9]{6,}$")
RE_LONG_DIGITS = re.compile(r"\d{6,}")


def has_chinese(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s))


def is_noisy_token(tok: str) -> bool:
    # Only block long codes if there is no Chinese.
    if has_chinese(tok):
        return False
    return bool(RE_LONG_ALNUM.match(tok) or RE_LONG_DIGITS.search(tok))


def init_jieba(userdict_path: str):
    if not os.path.exists(userdict_path):
        raise FileNotFoundError(f"jieba userdict not found: {userdict_path}")
    jieba.initialize()
    jieba.load_userdict(userdict_path)
    print(f"✅ Jieba userdict loaded: {userdict_path}")


def _segment_batch(examples) -> dict:
    out_words: List[List[str]] = []

    for text in examples["text"]:
        if text is None:
            out_words.append([])
            continue
        text = str(text)
        if not text.strip():
            out_words.append([])
            continue

        stripped = text.strip()
        # If the whole line is an ID-like long code without Chinese, skip.
        if not has_chinese(stripped) and RE_LONG_ALNUM.match(stripped):
            out_words.append([])
            continue

        words = list(jieba.cut(text))
        filtered: List[str] = []
        for w in words:
            w = w.strip()
            if not w:
                continue
            if is_noisy_token(w):
                continue
            filtered.append(w)

        out_words.append(filtered)

    return {"words": out_words}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build shared jieba-segmented words dataset")
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--jieba_userdict", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=2000)
    ap.add_argument("--num_proc", type=int, default=1)

    ap.add_argument(
        "--shuffle_split",
        action="store_true",
        default=True,
        help="Whether to shuffle before train/test split (non-OCR may enable; OCR should disable)",
    )
    ap.add_argument(
        "--no_shuffle_split",
        action="store_false",
        dest="shuffle_split",
        help="Disable shuffle before train/test split",
    )

    ap.add_argument("--test_size", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print(f"加载原始语料: {args.train_file}")
    raw = load_dataset("text", data_files={"train": args.train_file})

    init_jieba(args.jieba_userdict)

    map_kwargs = {
        "batched": True,
        "batch_size": args.batch_size,
        "remove_columns": ["text"],
    }
    if args.num_proc and args.num_proc > 1:
        map_kwargs["num_proc"] = args.num_proc

    print("开始 Jieba 分词生成 words...")
    words_ds: Dataset = raw["train"].map(_segment_batch, **map_kwargs)

    # Filter empty samples
    if args.num_proc and args.num_proc > 1:
        words_ds = words_ds.filter(lambda ex: len(ex["words"]) > 0, num_proc=args.num_proc)
    else:
        words_ds = words_ds.filter(lambda ex: len(ex["words"]) > 0)

    print(
        f"开始划分数据集 (Train {int((1-args.test_size)*100)}% / Test {int(args.test_size*100)}%), shuffle={args.shuffle_split} ..."
    )
    split: DatasetDict = words_ds.train_test_split(
        test_size=args.test_size,
        seed=args.seed,
        shuffle=args.shuffle_split,
    )

    os.makedirs(args.output_path, exist_ok=True)
    print(f"保存 words dataset 到: {args.output_path}")
    split.save_to_disk(args.output_path)

    print("\n" + "=" * 30)
    print("✅ words dataset 构建完成")
    print(f"Train: {len(split['train'])}")
    print(f"Test : {len(split['test'])}")
    print("=" * 30)


if __name__ == "__main__":
    main()
