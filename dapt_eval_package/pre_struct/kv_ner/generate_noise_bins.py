#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate noise_bins.json from a JSONL dataset that contains `noise_values`.

- Reads noise_values (7-dim) per token/char from JSONL
- Aggregates per feature across the dataset
- Produces quantile-based bin edges for each feature

Output format matches NoiseFeatureProcessor.load() expectation in noise_utils.py:
{
  "conf_avg": [... edges ...],
  "conf_min": [...],
  "conf_var_log": [...],
  "conf_gap": [...],
  "punct_err_ratio": [...],
  "char_break_ratio": [...],
  "align_score": [...]
}

Usage:
  python pre_struct/kv_ner/generate_noise_bins.py \
    --input /data/ocean/DAPT/biaozhu_with_ocr_noise/train.jsonl \
    --output pre_struct/kv_ner/noise_bins_from_train.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

# Keep feature ordering consistent with noise_utils.py
FEATURES = [
    "conf_avg",
    "conf_min",
    "conf_var_log",
    "conf_gap",
    "punct_err_ratio",
    "char_break_ratio",
    "align_score",
]
NUM_BINS = {
    "conf_avg": 64,
    "conf_min": 64,
    "conf_var_log": 32,
    "conf_gap": 32,
    "punct_err_ratio": 16,
    "char_break_ratio": 32,
    "align_score": 64,
}
# Optional clipping to reduce extreme tails (align with noise_utils defaults)
CLIP = {
    "char_break_ratio": 0.25,
    "align_score": 3500.0,
}


def read_noise_values(jsonl_path: Path) -> Dict[str, List[float]]:
    agg: Dict[str, List[float]] = {k: [] for k in FEATURES}
    n_lines = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_lines += 1
            nv = obj.get("noise_values")
            if not isinstance(nv, list):
                continue
            for item in nv:
                if not (isinstance(item, list) and len(item) == 7):
                    continue
                for i, feat in enumerate(FEATURES):
                    v = item[i]
                    try:
                        x = float(v)
                    except Exception:
                        continue
                    # basic clipping
                    if feat in CLIP:
                        lim = CLIP[feat]
                        if feat == "align_score":
                            x = max(-lim, min(lim, x))
                        else:
                            x = max(0.0, min(lim, x))
                    agg[feat].append(x)
    if n_lines == 0:
        raise RuntimeError(f"No valid lines read from {jsonl_path}")
    return agg


def compute_edges(values: List[float], k_bins: int) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return []
    # Remove exact zeros (reserved for anchor bin id 0)
    nonzero = arr[arr != 0.0]
    if nonzero.size == 0:
        return []
    # Quantile boundaries between (0,1). We use k_bins-1 interior cut points
    q = np.linspace(1.0 / k_bins, (k_bins - 1) / k_bins, k_bins - 1)
    edges = np.quantile(nonzero, q).astype(float).tolist()
    # Ensure strictly increasing
    dedup: List[float] = []
    for e in edges:
        if not dedup or e > dedup[-1]:
            dedup.append(e)
    return dedup


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate noise_bins.json from JSONL noise_values")
    ap.add_argument("--input", required=True, help="Path to train.jsonl with noise_values")
    ap.add_argument("--output", required=True, help="Path to write noise_bins.json")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    agg = read_noise_values(src)
    bins: Dict[str, List[float]] = {}
    for feat in FEATURES:
        k = NUM_BINS[feat]
        edges = compute_edges(agg[feat], k)
        bins[feat] = edges

    dst.write_text(json.dumps(bins, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote noise bins to {dst}")


if __name__ == "__main__":
    main()
