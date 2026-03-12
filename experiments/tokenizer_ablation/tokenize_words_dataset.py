#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tokenize a shared `words` DatasetDict into input_ids/word_ids for one tokenizer.

This is the 2nd stage of the tokenizer ablation data build:
1) Build words dataset with Jieba once (shared boundaries).
2) For each tokenizer variant, convert the same `words` into:
   - input_ids
   - word_ids (per token, aligned to word boundaries)

We intentionally use slow tokenizer by default (use_fast=False) to reduce fast-backend confounds
in pretraining. Downstream tasks can still require fast offsets.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from datasets import DatasetDict, load_from_disk
import transformers
from transformers import AutoTokenizer


_TOKENIZER_CACHE = {}


def _looks_like_tokenizer(obj) -> bool:
    return hasattr(obj, "tokenize") and hasattr(obj, "convert_tokens_to_ids")


def _diagnose_env(tokenizer_path: str) -> str:
    info = {
        "tokenizer_path": tokenizer_path,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "sys_path_0": sys.path[0] if sys.path else None,
        "transformers_file": getattr(transformers, "__file__", None),
        "transformers_version": getattr(transformers, "__version__", None),
        "AutoTokenizer_repr": repr(AutoTokenizer),
    }
    import json

    return json.dumps(info, ensure_ascii=False)


def _get_tokenizer(tokenizer_path: str, use_fast: bool):
    key = (tokenizer_path, bool(use_fast))
    if key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[key]

    tok = None
    if use_fast:
        # Prefer fast backend for downstream needs; here mainly for parity/debug.
        tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        if not _looks_like_tokenizer(tok) or isinstance(tok, bool):
            # Fallback to explicit fast class.
            try:
                from transformers import PreTrainedTokenizerFast

                tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
            except Exception:
                pass
    else:
        # Prefer slow BertTokenizer to avoid env-specific AutoTokenizer(use_fast=False) oddities.
        try:
            from transformers import BertTokenizer

            tok = BertTokenizer.from_pretrained(tokenizer_path)
        except Exception:
            tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    if not _looks_like_tokenizer(tok) or isinstance(tok, bool):
        raise TypeError(
            "Failed to load a valid tokenizer object. "
            f"type={type(tok).__name__}, value={tok!r}. "
            "Diagnose=" + _diagnose_env(tokenizer_path)
        )

    _TOKENIZER_CACHE[key] = tok
    return tok


def _tokenize_words_batch(examples, tokenizer_path: str, use_fast: bool, max_len: int) -> dict:
    tokenizer = _get_tokenizer(tokenizer_path, use_fast)
    batch_input_ids: List[List[int]] = []
    batch_word_ids: List[List[Optional[int]]] = []

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    unk_id = tokenizer.unk_token_id

    if cls_id is None or sep_id is None or unk_id is None:
        raise ValueError(
            f"Tokenizer missing special ids: cls={cls_id} sep={sep_id} unk={unk_id}. "
            "Expecting a BERT-like tokenizer."
        )

    for words in examples["words"]:
        tokens: List[int] = [cls_id]
        word_ids: List[Optional[int]] = [None]

        current_word_index = 0
        for w in words:
            if w is None:
                current_word_index += 1
                continue
            w = str(w).strip()
            if not w:
                current_word_index += 1
                continue

            word_tokens = tokenizer.tokenize(w)
            word_token_ids = tokenizer.convert_tokens_to_ids(word_tokens)

            # For strict boundary consistency across tokenizers, never drop a word.
            # If tokenization unexpectedly yields empty, fall back to [UNK].
            if not word_token_ids:
                word_token_ids = [unk_id]

            tokens.extend(word_token_ids)
            word_ids.extend([current_word_index] * len(word_token_ids))
            current_word_index += 1

        # If everything is empty (should be filtered earlier), keep a minimal sample.
        if current_word_index == 0:
            tokens.append(sep_id)
            word_ids.append(None)
        else:
            tokens.append(sep_id)
            word_ids.append(None)

        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            word_ids = word_ids[:max_len]
            tokens[-1] = sep_id
            word_ids[-1] = None

        batch_input_ids.append(tokens)
        batch_word_ids.append(word_ids)

    return {"input_ids": batch_input_ids, "word_ids": batch_word_ids}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tokenize shared words dataset into input_ids/word_ids")
    ap.add_argument("--words_dataset", type=str, required=True, help="Path to DatasetDict saved_to_disk with column 'words'")
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, required=True)

    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2000)
    ap.add_argument("--num_proc", type=int, default=1)

    ap.add_argument(
        "--use_fast",
        action="store_true",
        default=False,
        help="Use fast tokenizer backend (not recommended for pretrain; downstream may need fast offsets).",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print(f"加载 words dataset: {args.words_dataset}")
    words_dd: DatasetDict = load_from_disk(args.words_dataset)

    # NOTE: do not pass tokenizer object into datasets.map with num_proc>1.
    # It can break multiprocessing serialization and result in weird objects (e.g., bool) in workers.
    print(f"Tokenizer path: {args.tokenizer_path} (use_fast={args.use_fast})")

    map_kwargs = {
        "batched": True,
        "batch_size": args.batch_size,
        "remove_columns": ["words"],
        "fn_kwargs": {"tokenizer_path": args.tokenizer_path, "use_fast": bool(args.use_fast), "max_len": args.max_len},
    }
    if args.num_proc and args.num_proc > 1:
        map_kwargs["num_proc"] = args.num_proc

    print(f"开始分词并生成 input_ids/word_ids -> {args.output_path}")
    out_dd = DatasetDict()
    for split_name, split_ds in words_dd.items():
        out_dd[split_name] = split_ds.map(_tokenize_words_batch, **map_kwargs)

    os.makedirs(args.output_path, exist_ok=True)
    out_dd.save_to_disk(args.output_path)

    print("\n" + "=" * 30)
    print("✅ tokenizer-specific dataset 构建完成")
    for k in out_dd.keys():
        print(f"{k}: {len(out_dd[k])}")
    print("=" * 30)


if __name__ == "__main__":
    main()
