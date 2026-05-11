#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build tokenizer variants for ablation.

设计目标：
- 不改动你现有的 `final_merge_v9_regex_split_slim.py`（该脚本硬编码 /data/ocean 路径）。
- 在实验中用可配置的方式复现同等逻辑：从 OCR vocab / keys vocab 注入新 token。
- 兼容 HuggingFace 名称或本地目录作为 base tokenizer。

输出：`save_pretrained(output_dir)`，并打印新增 token 统计。
"""

import argparse
import os
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Set

from transformers import AutoTokenizer

DEFAULT_VIP_TERMS = {
    "brca1基因", "brca2基因", "her2基因", "fish检测",
    "er阳性", "pr阳性", "p53蛋白", "ptnm分期",
}


def smart_extract_key_core(token: str) -> List[str]:
    """从复杂 Key 中提取核心概念（尽量贴近你 v9 slim 脚本逻辑）。"""
    extracted: Set[str] = set()

    cleaned = re.sub(r"[\(（][^\)）]+[\)）]", "", token)
    if len(cleaned) > 1 and len(cleaned) < len(token):
        extracted.add(cleaned.strip())

    chinese_parts = re.findall(r"[\u4e00-\u9fa5]+", token)
    for part in chinese_parts:
        if len(part) > 1:
            extracted.add(part)

    split_parts = re.split(r"[：:-]", token)
    if len(split_parts) > 1:
        for part in split_parts:
            part = part.strip()
            if len(part) > 1:
                extracted.add(part)

    return list(extracted)


def iter_vocab_lines(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip()
            if not tok:
                continue
            yield tok


def ordered_vocab_tokens(tokenizer: Any) -> List[str]:
    """Return vocab tokens ordered by token id.

    This is important because WordPiece `vocab.txt` order defines token ids.
    """
    vocab = tokenizer.get_vocab()
    return [tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])]


def write_merged_wordpiece_tokenizer(tokenizer: Any, output_dir: Path, new_tokens: List[str]) -> None:
    """Save config files then overwrite vocab.txt with a clean merged vocab.

    Rationale:
    - `tokenizer.add_tokens()` writes runtime added tokens into added_tokens.json.
    - Our ablation wants new tokens to be part of the base WordPiece vocab (vocab.txt)
      so that slow/fast backends can be rebuilt deterministically from vocab.txt.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config/special tokens files first.
    tokenizer.save_pretrained(str(output_dir))

    base_ordered = ordered_vocab_tokens(tokenizer)
    seen = set(base_ordered)
    appended = 0
    for tok in sorted(new_tokens):
        if tok in seen:
            continue
        base_ordered.append(tok)
        seen.add(tok)
        appended += 1

    vocab_txt = output_dir / "vocab.txt"
    with open(vocab_txt, "w", encoding="utf-8") as f:
        for tok in base_ordered:
            f.write(tok + "\n")

    # These describe runtime added-tokens behavior and/or stale fast backend.
    # For ablation we want a clean vocab-only tokenizer directory.
    for stale_name in ("added_tokens.json", "tokenizer.json"):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    print(f"[save] merged WordPiece vocab -> {vocab_txt} (appended={appended})")


def load_candidates(
    ocr_vocab: Optional[str],
    keys_vocab: Optional[str],
    vip_terms: Set[str],
    max_token_length: int,
    enable_key_core_extraction: bool,
) -> Set[str]:
    candidates: Set[str] = set(vip_terms)
    vip_lower = {v.lower() for v in vip_terms}

    def add_token(tok: str):
        if not tok or " " in tok:
            return
        if tok.lower() in vip_lower:
            candidates.add(tok)
            return
        if len(tok) <= 1:
            return
        if len(tok) > max_token_length:
            return
        candidates.add(tok)

    if ocr_vocab:
        for tok in iter_vocab_lines(ocr_vocab):
            add_token(tok)

    if keys_vocab:
        for tok in iter_vocab_lines(keys_vocab):
            if not tok or " " in tok:
                continue
            if tok.lower() in vip_lower:
                candidates.add(tok)
                continue

            if enable_key_core_extraction and len(tok) > max_token_length:
                for core in smart_extract_key_core(tok):
                    add_token(core)
                continue

            add_token(tok)

    return candidates


def main():
    ap = argparse.ArgumentParser(description="Build tokenizer variants (T1~T4) for ablation")
    ap.add_argument("--base_tokenizer", type=str, required=True, help="HF name or local path")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--ocr_vocab", type=str, default=None, help="OCR vocab file (one token per line)")
    ap.add_argument("--keys_vocab", type=str, default=None, help="Keys vocab file (one token per line)")
    ap.add_argument("--max_token_length", type=int, default=7)
    ap.add_argument("--disable_key_core_extraction", action="store_true")
    ap.add_argument("--vip_terms", type=str, default=None, help="Optional VIP terms file")
    ap.add_argument("--lowercase", action="store_true", help="Force lowercase tokenizer if supported")

    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vip_terms = set(DEFAULT_VIP_TERMS)
    if args.vip_terms:
        vip_terms = set()
        for tok in iter_vocab_lines(args.vip_terms):
            vip_terms.add(tok)

    if args.ocr_vocab and not os.path.exists(args.ocr_vocab):
        raise FileNotFoundError(f"OCR vocab not found: {args.ocr_vocab}")
    if args.keys_vocab and not os.path.exists(args.keys_vocab):
        raise FileNotFoundError(f"Keys vocab not found: {args.keys_vocab}")

    print(f"[load] base_tokenizer={args.base_tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=False)

    candidates = load_candidates(
        ocr_vocab=args.ocr_vocab,
        keys_vocab=args.keys_vocab,
        vip_terms=vip_terms,
        max_token_length=args.max_token_length,
        enable_key_core_extraction=(not args.disable_key_core_extraction),
    )

    base_vocab = tokenizer.get_vocab()
    new_tokens = [t for t in candidates if t not in base_vocab]

    print("=" * 60)
    print(f"[stats] candidates={len(candidates)} base_vocab={len(base_vocab)}")
    print(f"[stats] new_tokens={len(new_tokens)}")

    write_merged_wordpiece_tokenizer(tokenizer, output_dir, new_tokens)
    print(f"[save] {output_dir}")

    # 尝试输出 vocab size（不同 tokenizer 保存结构可能不一样）
    vocab_txt = output_dir / "vocab.txt"
    if vocab_txt.exists():
        try:
            vocab_size = sum(1 for _ in open(vocab_txt, "r", encoding="utf-8"))
            print(f"[vocab.txt] lines={vocab_size}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
