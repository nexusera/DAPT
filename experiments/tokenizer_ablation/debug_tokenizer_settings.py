#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from typing import Any, Dict

from transformers import AutoTokenizer


def _get_attr(obj: Any, name: str):
    return getattr(obj, name, None)


def _safe_len(obj: Any):
    try:
        return len(obj)
    except Exception:
        return None


def _try_dump_tokenizer_config(tok: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    out["loaded_type"] = type(tok).__name__
    if isinstance(tok, bool):
        out["error"] = (
            "Loaded object is bool, not a tokenizer. "
            "This usually indicates something is shadowing transformers.AutoTokenizer, "
            "or the load step returned an unexpected object."
        )
        return out

    for k in [
        "name_or_path",
        "do_lower_case",
        "model_max_length",
        "tokenize_chinese_chars",
        "strip_accents",
        "clean_text",
        "never_split",
    ]:
        v = _get_attr(tok, k)
        if v is not None:
            out[k] = v

    # For slow BertTokenizer, basic_tokenizer/wordpiece_tokenizer exists
    bt = _get_attr(tok, "basic_tokenizer")
    if bt is not None:
        out["basic_tokenizer"] = {
            "do_lower_case": _get_attr(bt, "do_lower_case"),
            "tokenize_chinese_chars": _get_attr(bt, "tokenize_chinese_chars"),
            "strip_accents": _get_attr(bt, "strip_accents"),
            "clean_text": _get_attr(bt, "clean_text"),
        }

    out["tokenizer_class"] = tok.__class__.__name__
    is_fast = _get_attr(tok, "is_fast")
    out["is_fast"] = None if is_fast is None else bool(is_fast)

    vocab_size = _get_attr(tok, "vocab_size")
    if vocab_size is None:
        vocab_size = _safe_len(tok)
    out["vocab_size"] = vocab_size

    out["unk_token"] = _get_attr(tok, "unk_token")
    out["unk_token_id"] = _get_attr(tok, "unk_token_id")
    out["mask_token"] = _get_attr(tok, "mask_token")
    out["mask_token_id"] = _get_attr(tok, "mask_token_id")

    return out


def main():
    ap = argparse.ArgumentParser(description="Debug tokenizer settings and tokenization behavior")
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument(
        "--use_fast",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Force loading fast/slow tokenizer. auto=default AutoTokenizer behavior.",
    )
    ap.add_argument(
        "--samples",
        type=str,
        nargs="*",
        default=[
            "肿瘤标志物",
            "胸部X光",
            "BRCA1基因检测",
            "2024-03-10",
            "住院号1234567890",
            "糖化血红蛋白HbA1c",
        ],
    )

    args = ap.parse_args()

    use_fast = None
    if args.use_fast == "true":
        use_fast = True
    elif args.use_fast == "false":
        use_fast = False

    if use_fast is None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=use_fast)

    info = _try_dump_tokenizer_config(tokenizer)

    print("=" * 80)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print("-" * 80)

    if not hasattr(tokenizer, "tokenize") or not hasattr(tokenizer, "convert_tokens_to_ids"):
        raise TypeError(
            "Loaded object does not look like a HuggingFace tokenizer. "
            f"type={type(tokenizer).__name__}, value={tokenizer!r}. "
            "Please check your transformers installation and whether a local file/module is shadowing it."
        )

    for s in args.samples:
        pieces = tokenizer.tokenize(s)
        ids = tokenizer.convert_tokens_to_ids(pieces)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        unk_cnt = sum(1 for i in ids if i == unk_id)
        print(f"[sample] {s}")
        print(f"  pieces({len(pieces)}): {pieces[:50]}")
        print(f"  unk_in_pieces={unk_cnt}/{len(pieces)}")


if __name__ == "__main__":
    main()
