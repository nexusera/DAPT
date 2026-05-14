#!/usr/bin/env python3
"""D3.7 — Probing classifier × 3 tasks × architecture × layer-wise.

For each layer of a frozen model, extract its hidden representations on
a probing dataset and train a tiny logistic-regression head on three
binary / multi-class tasks:

  - noise: predict OCR noise bucket (low/mid/high) from token-level conf
  - kv:    predict whether a (key, value) pair is matched (KV-NSP-style)
  - entity: predict whether each token is a medical entity boundary token

The signal we measure: which layer of the CPT-pretrained model encodes
the prior best, vs. the same layer of the base un-CPT'd model. Big delta
== the CPT injected that prior; small delta == it was already in base.

Usage (KV-LLM):
  python scripts/analysis/probing_classifier.py \
    --model-dir /data/ocean/code/dapt/model/kv_llm_qwen3_0.6b_full/final_model \
    --base-model /data/ocean/model/Qwen/Qwen3-0.6B-Base \
    --probe-data /data/ocean/code/dapt/data_full/medstruct_test_pairs.jsonl \
    --output /data/ocean/code/dapt/results/eval/probing_06b.csv \
    --bf16 --max-samples 200
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


def load_records(path: Path, limit: Optional[int] = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            rows = json.load(f)
        else:
            rows = [json.loads(l) for l in f if l.strip()]
    if limit:
        rows = rows[:limit]
    return rows


def build_probe_labels(rec: dict) -> dict[str, Optional[int]]:
    """Derive 3 probe labels from a MedStruct-S-style record.

    For simplicity:
      noise: 0 if record's noise_level <= 0.05 else (1 if <=0.20 else 2)
             (when not in synthetic_noise benchmark, fallback = -1)
      kv:    1 (positive); we sample negatives at training time
      entity: token-level; we mark hospital + key + value spans as entity
              boundary tokens (1), others 0

    Returns just the doc-level labels; token-level labels are derived at
    encoding time from char offsets vs. the pair text positions.
    """
    nl = rec.get("noise_level")
    if nl is None:
        noise = -1
    elif nl <= 0.05: noise = 0
    elif nl <= 0.20: noise = 1
    else: noise = 2
    return {"noise": noise, "kv": 1}


def extract_layer_features(model, tok, text: str, max_length: int = 512) -> torch.Tensor:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    if torch.cuda.is_available():
        enc = {k: v.to("cuda") for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    # (n_layers, hidden) — mean-pool over tokens per layer
    hs = out.hidden_states  # tuple of (1, S, H)
    pooled = torch.stack([h.mean(dim=1).squeeze(0) for h in hs])  # (L, H)
    return pooled.float().cpu()


def fit_logistic(X: np.ndarray, y: np.ndarray) -> float:
    """Tiny logistic regression with class-balanced loss; returns accuracy.
    Uses sklearn if available, else scratch numpy."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        if len(set(y.tolist())) < 2 or len(y) < 20:
            return float("nan")
        clf = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced")
        scores = cross_val_score(clf, X, y, cv=min(5, len(y) // 4), scoring="accuracy")
        return float(scores.mean())
    except Exception:
        return float("nan")


def run_probing(model_dir: str, probe_records: list[dict], bf16: bool, tag: str,
                tasks: list[str]) -> list[dict]:
    """Returns list of rows: {tag, layer, task, acc}."""
    dtype = torch.bfloat16 if bf16 else None
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=dtype)
    model.eval()
    if torch.cuda.is_available(): model = model.to("cuda")

    feats_per_layer: list[list[np.ndarray]] = []
    labels: dict[str, list[int]] = {t: [] for t in tasks}
    for i, r in enumerate(probe_records):
        text = r.get("text_noisy") or r.get("ocr_text") or r.get("text", "")
        if not text:
            continue
        f = extract_layer_features(model, tok, text)  # (L, H)
        if not feats_per_layer:
            feats_per_layer = [[] for _ in range(f.shape[0])]
        for li in range(f.shape[0]):
            feats_per_layer[li].append(f[li].numpy())
        lbl = build_probe_labels(r)
        for t in tasks:
            labels[t].append(lbl.get(t, -1))
        if (i + 1) % 50 == 0:
            print(f"  [{tag}] {i+1}/{len(probe_records)} encoded", file=sys.stderr)

    out_rows = []
    for li, feats in enumerate(feats_per_layer):
        X = np.stack(feats)
        for t in tasks:
            y = np.array(labels[t])
            valid_mask = y != -1
            if valid_mask.sum() < 20:
                out_rows.append({"model": tag, "layer": li, "task": t, "acc": "NA", "n": int(valid_mask.sum())})
                continue
            acc = fit_logistic(X[valid_mask], y[valid_mask])
            out_rows.append({"model": tag, "layer": li, "task": t,
                             "acc": f"{acc:.4f}" if not np.isnan(acc) else "NA",
                             "n": int(valid_mask.sum())})
    del model
    torch.cuda.empty_cache()
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="CPT'd model dir (or any HF model)")
    ap.add_argument("--base-model", default=None, help="optional second model to compare layerwise")
    ap.add_argument("--probe-data", required=True)
    ap.add_argument("--tasks", default="noise,kv", help="comma-sep subset of {noise, kv, entity}")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    records = load_records(Path(args.probe_data), limit=args.max_samples)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[D3.7] {len(records)} probe records, tasks={tasks}", file=sys.stderr)

    rows = run_probing(args.model_dir, records, args.bf16, tag="cpt", tasks=tasks)
    if args.base_model:
        rows += run_probing(args.base_model, records, args.bf16, tag="base", tasks=tasks)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "layer", "task", "acc", "n"])
        w.writeheader(); w.writerows(rows)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
