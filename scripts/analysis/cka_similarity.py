#!/usr/bin/env python3
"""D3.8 — Centered Kernel Alignment (CKA) layer-wise similarity.

Compare two model checkpoints (e.g., CPT'd vs base, or KV-LLM 0.6B vs
KV-BERT) layer-by-layer using linear CKA on hidden representations
extracted from a fixed probe set. Big drop in CKA at layer k means CPT
modified that layer heavily; small drop means it preserved the base.

Output: CSV layer × cka_similarity, plus a heatmap-friendly N×N matrix
when --pairwise is set (every-layer-of-A × every-layer-of-B).

Usage:
  python scripts/analysis/cka_similarity.py \
    --model-a /data/ocean/code/dapt/model/kv_llm_qwen3_0.6b_full/final_model \
    --model-b /data/ocean/model/Qwen/Qwen3-0.6B-Base \
    --probe-data /data/ocean/code/dapt/data_full/medstruct_test_pairs.jsonl \
    --output /data/ocean/code/dapt/results/eval/cka_06b_vs_base.csv \
    --bf16 --max-samples 200
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two (n_samples, n_features) matrices.
    Equal to (||Y^T X||_F^2) / (||X^T X||_F · ||Y^T Y||_F)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    den = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    if den < 1e-12:
        return float("nan")
    return float(num / den)


def load_records(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            rows = json.load(f)
        else:
            rows = [json.loads(l) for l in f if l.strip()]
    return rows[:limit] if limit else rows


def extract_layer_features(model_dir: str, records: list[dict], bf16: bool,
                            max_length: int = 512) -> np.ndarray:
    """Returns (n_samples, n_layers, hidden_dim) — float32 numpy."""
    dtype = torch.bfloat16 if bf16 else None
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=dtype)
    model.eval()
    if torch.cuda.is_available(): model = model.to("cuda")
    feats_per_sample: list[np.ndarray] = []
    for i, r in enumerate(records):
        text = r.get("text_noisy") or r.get("ocr_text") or r.get("text", "")
        if not text:
            continue
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
        if torch.cuda.is_available():
            enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        # mean-pool tokens per layer → (L, H)
        pooled = torch.stack([h.mean(dim=1).squeeze(0) for h in out.hidden_states])
        feats_per_sample.append(pooled.float().cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f"  encoded {i+1}/{len(records)}", file=sys.stderr)
    del model
    torch.cuda.empty_cache()
    return np.stack(feats_per_sample)  # (N, L, H)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--probe-data", required=True)
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--pairwise", action="store_true", help="full L_A × L_B matrix; default = diagonal only")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    recs = load_records(Path(args.probe_data), args.max_samples)
    print(f"[D3.8] encoding model A ({args.model_a}) on {len(recs)} samples", file=sys.stderr)
    A = extract_layer_features(args.model_a, recs, args.bf16)  # (N, La, H)
    print(f"[D3.8] encoding model B ({args.model_b}) on {len(recs)} samples", file=sys.stderr)
    B = extract_layer_features(args.model_b, recs, args.bf16)  # (N, Lb, H)

    out_rows = []
    if args.pairwise:
        La, Lb = A.shape[1], B.shape[1]
        for i in range(La):
            for j in range(Lb):
                cka = linear_cka(A[:, i, :], B[:, j, :])
                out_rows.append({"layer_a": i, "layer_b": j, "cka": f"{cka:.4f}"})
    else:
        L = min(A.shape[1], B.shape[1])
        for i in range(L):
            cka = linear_cka(A[:, i, :], B[:, i, :])
            out_rows.append({"layer_a": i, "layer_b": i, "cka": f"{cka:.4f}"})

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["layer_a", "layer_b", "cka"])
        w.writeheader(); w.writerows(out_rows)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
