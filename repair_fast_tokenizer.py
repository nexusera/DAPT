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
import os
import shutil
from pathlib import Path
from typing import List

from transformers import AutoTokenizer


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

        # Strong red flag we previously observed: a whole Chinese phrase -> single [UNK]
        if len(pieces) == 1 and pieces[0] == tokenizer.unk_token:
            failures.append(f"Single UNK piece for {s!r} (pieces={pieces}).")

        # Ensure offsets_mapping works (required by downstream NER/QA)
        try:
            enc = tokenizer(s, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc.get("offset_mapping")
            if offsets is None or len(offsets) == 0:
                failures.append(f"No offset_mapping returned for {s!r}.")
        except Exception as e:
            failures.append(f"return_offsets_mapping failed for {s!r}: {e}")

    return failures


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
    slow_tok = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=False)
    print(f"  slow tokenizer class={slow_tok.__class__.__name__}")

    if old_json.exists() and not args.keep_old_json:
        if not args.no_backup:
            backup = tok_dir / f"tokenizer.json.bak.{_now_tag()}"
            print(f"[2/4] Backing up existing tokenizer.json -> {backup}")
            shutil.copy2(old_json, backup)
        print(f"[2/4] Removing existing tokenizer.json: {old_json}")
        old_json.unlink()
    else:
        print(f"[2/4] No tokenizer.json removal (exists={old_json.exists()}, keep_old_json={args.keep_old_json})")

    print(f"[3/4] Building fast tokenizer from directory assets...")
    fast_tok = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True)
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
