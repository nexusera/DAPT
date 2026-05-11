#!/usr/bin/env python3
"""Make a deterministic OCR json subset for quick-run.

We must preserve order to keep noise alignment valid.

Input supports:
- .json (list or dict with data/ocr_list/items)
- .jsonl (one json obj per line)

Output is a plain JSON list.
"""

import argparse
import json
import os
from typing import Any, List


def load_ocr_list(path: str) -> List[Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"OCR json not found: {path}")

    if path.endswith(".jsonl"):
        out: List[Any] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ["data", "ocr_list", "items"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        return [obj]

    raise ValueError(f"Unsupported OCR JSON format: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to OCR json/jsonl")
    ap.add_argument("--output", required=True, help="output json path (list)")
    ap.add_argument("--n", type=int, required=True, help="keep first N items")
    args = ap.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be > 0")

    items = load_ocr_list(args.input)
    sub = items[: args.n]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False)

    print(f"Done. {len(sub)}/{len(items)} items -> {args.output}")


if __name__ == "__main__":
    main()
