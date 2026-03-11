#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from typing import Any, Dict

from transformers import AutoTokenizer


def _get_attr(obj: Any, name: str):
    return getattr(obj, name, None)


def _try_dump_tokenizer_config(tok: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
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
    out["is_fast"] = bool(_get_attr(tok, "is_fast"))
    out["vocab_size"] = len(tok)
    out["unk_token"] = tok.unk_token
    out["unk_token_id"] = tok.unk_token_id
    out["mask_token"] = tok.mask_token
    out["mask_token_id"] = tok.mask_token_id

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
        tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=use_fast)
    info = _try_dump_tokenizer_config(tok)

    print("=" * 80)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print("-" * 80)

    for s in args.samples:
        pieces = tok.tokenize(s)
        ids = tok.convert_tokens_to_ids(pieces)
        unk_id = tok.unk_token_id
        unk_cnt = sum(1 for i in ids if i == unk_id)
        print(f"[sample] {s}")
        print(f"  pieces({len(pieces)}): {pieces[:50]}")
        print(f"  unk_in_pieces={unk_cnt}/{len(pieces)}")


if __name__ == "__main__":
    main()
