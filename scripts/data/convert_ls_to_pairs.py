#!/usr/bin/env python3
"""Convert MedStruct-S annotation files from transferred_annotations -> pairs.

The KV-LLM SFT pipeline (`kv_llm/fine_tune_sft.py`) expects each record to
have:
    {"ocr_text": str, "pairs": [{"key": str, "value": str}, ...], ...}

The MedStruct-S raw files at
`/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_{train,test}_with_ocr.json`
use the original Label Studio span schema:
    {"record_id": int, "ocr_text": str, "category": str,
     "transferred_annotations": [{"labels": ["键名"|"值"|"医院名称"], "text": str, ...}, ...]}

with alternating 键名/值 pairs (no explicit relation IDs — pairing is by
list order). This script collapses adjacent 键名→值 entries into pair
records and writes JSONL.

Usage:
    python scripts/data/convert_ls_to_pairs.py \
        --input  /data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json \
        --output /data/ocean/code/dapt/data_full/medstruct_train_pairs.jsonl

    python scripts/data/convert_ls_to_pairs.py \
        --input  /data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \
        --output /data/ocean/code/dapt/data_full/medstruct_test_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def extract_pairs(transferred_annotations: list[dict]) -> list[dict]:
    """Collapse adjacent 键名/值 spans into K-V pair dicts; capture 医院名称
    separately under a reserved key prefix."""
    pairs: list[dict] = []
    pending_key: str | None = None
    hospital: str | None = None
    for ann in transferred_annotations:
        labels = ann.get("labels") or []
        text = (ann.get("text") or "").strip()
        if not labels or not text:
            continue
        lbl = labels[0]
        if lbl == "医院名称":
            hospital = text
        elif lbl == "键名":
            pending_key = text
        elif lbl == "值":
            if pending_key:
                pairs.append({"key": pending_key, "value": text})
                pending_key = None
    if hospital:
        pairs.insert(0, {"key": "_医院名称", "value": hospital})
    return pairs


def iter_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            for r in data:
                yield r
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    n_pairs_total = 0
    n_no_pairs = 0
    with dst.open("w", encoding="utf-8") as f:
        for rec in iter_records(src):
            n_in += 1
            ocr_text = rec.get("ocr_text") or rec.get("text") or ""
            if not ocr_text.strip():
                continue
            pairs = extract_pairs(rec.get("transferred_annotations") or [])
            if not pairs:
                n_no_pairs += 1
                continue
            out_rec = {
                "record_id": rec.get("record_id") or rec.get("id"),
                "category": rec.get("category"),
                "ocr_text": ocr_text,
                "pairs": pairs,
                "image": rec.get("relative_image_path"),
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_out += 1
            n_pairs_total += len(pairs)

    print(f"[OK] {n_in} input records, {n_out} written ({n_no_pairs} skipped — no pairs)")
    print(f"     total pairs = {n_pairs_total}, avg = {n_pairs_total / max(n_out,1):.1f} per record")
    print(f"     output: {dst} ({dst.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
