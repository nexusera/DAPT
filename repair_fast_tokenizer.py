#!/usr/bin/env python
"""Repair (re-generate) a HuggingFace *fast* tokenizer file set.

Why this exists
---------------
In this repo we sometimes edit/merge a BERT WordPiece vocab (vocab.txt).
If a stale/incompatible `tokenizer.json` is left in the tokenizer directory,
`AutoTokenizer(..., use_fast=True)` may load the stale backend and produce
pathological tokenization (e.g. Chinese phrases become a single [UNK]).

This script:
- Loads the tokenizer in slow mode (ignores tokenizer.json)
- Backs up and removes tokenizer.json (optional)
- Re-creates a fast tokenizer from the remaining slow assets
- Saves it back to the same directory (writes tokenizer.json)
- Runs a small self-test to catch obviously broken configs

Typical usage (on server):
  python DAPT/repair_fast_tokenizer.py --tokenizer_dir /data/ocean/DAPT/my-medical-tokenizer
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

try:
    from tokenizers import Tokenizer
    from tokenizers.decoders import WordPiece as WordPieceDecoder
    from tokenizers.models import WordPiece
    from tokenizers.normalizers import BertNormalizer
    from tokenizers.pre_tokenizers import BertPreTokenizer
    from tokenizers.processors import BertProcessing
except Exception:  # pragma: no cover
    Tokenizer = None  # type: ignore


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_probably_broken_fast(tokenizer, test_strings: List[str]) -> List[str]:
    """Return a list of failure messages; empty means OK."""
    failures: List[str] = []

    if not getattr(tokenizer, "is_fast", False):
        failures.append("Loaded tokenizer is not fast (is_fast=False).")
        return failures

    for s in test_strings:
        try:
            pieces = tokenizer.tokenize(s)
        except Exception as e:
            failures.append(f"tokenize() failed for {s!r}: {e}")
            continue

        if len(pieces) == 0:
            failures.append(f"tokenize() produced 0 pieces for {s!r}")
            continue

        # Strong red flags:
        # 1) all pieces map to unk_id (even if token strings are not literally '[UNK]')
        # 2) a whole phrase collapses to a single unknown token
        try:
            ids = tokenizer.convert_tokens_to_ids(pieces)
            unk_id = tokenizer.unk_token_id
            unk_cnt = sum(1 for i in ids if i == unk_id)
            if unk_cnt == len(pieces):
                failures.append(f"All pieces are UNK for {s!r} (pieces={pieces}).")
            if len(pieces) == 1 and unk_cnt == 1:
                failures.append(f"Single-UNK tokenization for {s!r} (pieces={pieces}).")
        except Exception as e:
            failures.append(f"convert_tokens_to_ids failed for {s!r}: {e}")

        # Ensure offsets_mapping works (required by downstream NER/QA)
        try:
            enc = tokenizer(s, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc.get("offset_mapping")
            if offsets is None or len(offsets) == 0:
                failures.append(f"No offset_mapping returned for {s!r}.")
        except Exception as e:
            failures.append(f"return_offsets_mapping failed for {s!r}: {e}")

    return failures


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _diagnose_transformers() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "transformers_file": getattr(transformers, "__file__", None),
        "transformers_version": getattr(transformers, "__version__", None),
        "AutoTokenizer_type": type(AutoTokenizer).__name__,
        "AutoTokenizer_repr": repr(AutoTokenizer),
    }
    return info


def _looks_like_tokenizer(obj: Any) -> bool:
    return hasattr(obj, "tokenize") and hasattr(obj, "convert_tokens_to_ids")


def _build_fast_backend_from_vocab(
    tok_dir: Path,
    *,
    do_lower_case: bool,
    tokenize_chinese_chars: bool,
    strip_accents: Optional[bool],
    unk_token: str,
    cls_token: str,
    sep_token: str,
    pad_token: str,
    mask_token: str,
) -> PreTrainedTokenizerFast:
    if Tokenizer is None:
        raise RuntimeError("tokenizers is not available; cannot build fast tokenizer backend")

    vocab_path = tok_dir / "vocab.txt"
    tokens: List[str] = []
    with vocab_path.open("r", encoding="utf-8") as f:
        for line in f:
            # Keep internal spaces if any; only strip newlines.
            t = line.rstrip("\n")
            tokens.append(t)

    vocab = {t: i for i, t in enumerate(tokens)}

    model = WordPiece(vocab=vocab, unk_token=unk_token, continuing_subword_prefix="##")
    backend = Tokenizer(model)
    backend.normalizer = BertNormalizer(
        lowercase=bool(do_lower_case),
        clean_text=True,
        handle_chinese_chars=bool(tokenize_chinese_chars),
        strip_accents=strip_accents,
    )
    backend.pre_tokenizer = BertPreTokenizer()
    backend.decoder = WordPieceDecoder(prefix="##")

    # Add [CLS]/[SEP] in a BERT-compatible way
    cls_id = vocab.get(cls_token, 101)
    sep_id = vocab.get(sep_token, 102)
    backend.post_processor = BertProcessing(
        (sep_token, sep_id),
        (cls_token, cls_id),
    )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=unk_token,
        cls_token=cls_token,
        sep_token=sep_token,
        pad_token=pad_token,
        mask_token=mask_token,
        do_lower_case=bool(do_lower_case),
    )
    return fast


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair fast tokenizer.json to match vocab.txt")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Tokenizer directory (contains vocab.txt)")
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not backup tokenizer.json before overwriting/removing",
    )
    parser.add_argument(
        "--keep_old_json",
        action="store_true",
        help="Keep old tokenizer.json in place (NOT recommended); useful for dry investigation.",
    )
    parser.add_argument(
        "--test_strings",
        type=str,
        nargs="*",
        default=["肿瘤标志物", "住院号", "血常规", "王小明", "CT检查"],
        help="Strings used to sanity-check the regenerated fast tokenizer",
    )

    args = parser.parse_args()
    tok_dir = Path(args.tokenizer_dir)
    if not tok_dir.exists() or not tok_dir.is_dir():
        raise FileNotFoundError(f"tokenizer_dir not found: {tok_dir}")

    vocab = tok_dir / "vocab.txt"
    if not vocab.exists():
        raise FileNotFoundError(f"vocab.txt not found under: {tok_dir}")

    old_json = tok_dir / "tokenizer.json"

    print(f"[1/4] Loading slow tokenizer from: {tok_dir}")
    slow_tok: Optional[Any] = None
    try:
        candidate = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=False)
        if _looks_like_tokenizer(candidate) and not isinstance(candidate, bool):
            slow_tok = candidate
            print(f"  slow tokenizer class={slow_tok.__class__.__name__}")
        else:
            print(f"  slow tokenizer class={type(candidate).__name__}")
            print("  WARNING: slow tokenizer did not load correctly; falling back to config-only mode")
            print("  diagnose=" + json.dumps(_diagnose_transformers(), ensure_ascii=False))
    except Exception as e:
        print(f"  WARNING: failed to load slow tokenizer: {e}")
        print("  diagnose=" + json.dumps(_diagnose_transformers(), ensure_ascii=False))

    # Read config hints (if present) so fast matches slow behavior
    tok_cfg = _read_json(tok_dir / "tokenizer_config.json")
    sp_map = _read_json(tok_dir / "special_tokens_map.json")
    do_lower_case = bool(tok_cfg.get("do_lower_case", getattr(slow_tok, "do_lower_case", True) if slow_tok is not None else True))
    tokenize_chinese_chars = bool(tok_cfg.get("tokenize_chinese_chars", True))
    strip_accents = tok_cfg.get("strip_accents", None)
    if strip_accents not in (None, True, False):
        strip_accents = None

    def _fallback_special(name: str, default: str) -> str:
        if slow_tok is not None:
            v = getattr(slow_tok, name, None)
            if isinstance(v, str) and v:
                return v
        return default

    unk_token = str(sp_map.get("unk_token", _fallback_special("unk_token", "[UNK]")))
    cls_token = str(sp_map.get("cls_token", _fallback_special("cls_token", "[CLS]")))
    sep_token = str(sp_map.get("sep_token", _fallback_special("sep_token", "[SEP]")))
    pad_token = str(sp_map.get("pad_token", _fallback_special("pad_token", "[PAD]")))
    mask_token = str(sp_map.get("mask_token", _fallback_special("mask_token", "[MASK]")))

    if old_json.exists() and not args.keep_old_json:
        if not args.no_backup:
            backup = tok_dir / f"tokenizer.json.bak.{_now_tag()}"
            print(f"[2/4] Backing up existing tokenizer.json -> {backup}")
            shutil.copy2(old_json, backup)
        print(f"[2/4] Removing existing tokenizer.json: {old_json}")
        old_json.unlink()
    else:
        print(f"[2/4] No tokenizer.json removal (exists={old_json.exists()}, keep_old_json={args.keep_old_json})")

    print(f"[3/4] Building fast tokenizer backend from vocab.txt (deterministic WordPiece)...")
    fast_tok = _build_fast_backend_from_vocab(
        tok_dir,
        do_lower_case=do_lower_case,
        tokenize_chinese_chars=tokenize_chinese_chars,
        strip_accents=strip_accents,
        unk_token=unk_token,
        cls_token=cls_token,
        sep_token=sep_token,
        pad_token=pad_token,
        mask_token=mask_token,
    )
    print(f"  fast tokenizer class={fast_tok.__class__.__name__} is_fast={getattr(fast_tok, 'is_fast', False)}")

    print(f"[3/4] Saving fast tokenizer back to: {tok_dir}")
    fast_tok.save_pretrained(str(tok_dir))

    print(f"[4/4] Self-test...")
    failures = _is_probably_broken_fast(fast_tok, list(args.test_strings))
    if failures:
        print("Self-test FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        raise SystemExit(
            "Fast tokenizer still looks broken. "
            "This usually means the directory contains non-BERT assets or incompatible config. "
            "Inspect tokenizer_config.json/special_tokens_map.json and consider rebuilding the tokenizer directory cleanly."
        )

    print("Self-test OK. Fast tokenizer.json regenerated successfully.")


if __name__ == "__main__":
    # Reduce parallelism noise in logs
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
