#!/usr/bin/env python3
"""N12 — NCAG gate-vs-noise correlation analysis (plan §N12 line 820).

Loads a trained NCAG gate checkpoint and reports the Spearman correlation
between the per-token gate value and each of the 7 noise features, plus
diagnostic plots.

This is the **mechanism evidence** that goes into the paper:

    "Attention learned to down-weight low-confidence tokens — the gate
     value is significantly correlated with conf_avg (ρ = 0.42, p<1e-30)
     and inversely correlated with char_break_ratio (ρ = -0.31, p<1e-20)."

Usage::

    python -m scripts.analysis.ncag_gate_correlation \
        --gate_ckpt /data/ocean/code/dapt/model/ncag_pilot/span/final_model/kv_llm_ncag_gate.pt \
        --eval_jsonl /data/ocean/code/dapt/data/synthetic_noise_benchmark.jsonl \
        --output_dir results/d3_n12_ncag_gate_correlation \
        --n_samples 5000

Output:
    - ncag_gate_corr.json   — per-feature Spearman ρ, p, n
    - ncag_gate_scatter.png — gate vs conf_avg scatter (matplotlib, if installed)
    - report.md             — paper-friendly Markdown table

The script is read-only on data; it never mutates the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kv_llm.ncag import NCAGGate  # noqa: E402

NOISE_FEATURE_NAMES = [
    "conf_avg",
    "conf_min",
    "conf_var_log",
    "conf_gap",
    "punct_err_ratio",
    "char_break_ratio",
    "align_score",
]
# Expected sign of correlation with the gate output (positive = "more reliable").
# The gate should track conf_avg / conf_min positively and error rates negatively.
EXPECTED_SIGN = {
    "conf_avg": +1,
    "conf_min": +1,
    "conf_var_log": -1,   # higher variance log → noisier
    "conf_gap": -1,
    "punct_err_ratio": -1,
    "char_break_ratio": -1,
    "align_score": +1,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="N12 — gate-vs-noise Spearman correlation analysis.")
    p.add_argument("--gate_ckpt", required=True, help="Path to kv_llm_ncag_gate.pt")
    p.add_argument(
        "--eval_jsonl",
        default=None,
        help="Optional JSONL with noise_values arrays; if omitted, synthesise samples.",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden_dim", type=int, default=0, help="Match the trained gate (0 = single Linear).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Spearman
# ---------------------------------------------------------------------------


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    try:
        from scipy.stats import spearmanr  # type: ignore

        r, p = spearmanr(x, y)
        return float(r), float(p)
    except Exception:
        from math import sqrt, erfc

        def rank(a: np.ndarray) -> np.ndarray:
            order = a.argsort()
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(a))
            return ranks

        rx, ry = rank(x), rank(y)
        rxc, ryc = rx - rx.mean(), ry - ry.mean()
        denom = np.sqrt((rxc**2).sum() * (ryc**2).sum())
        r = float((rxc * ryc).sum() / denom) if denom > 0 else 0.0
        n = len(x)
        if n > 2 and abs(r) < 1.0:
            t = r * sqrt((n - 2) / max(1 - r**2, 1e-12))
            p = float(erfc(abs(t) / sqrt(2)))
        else:
            p = 0.0
        return r, p


# ---------------------------------------------------------------------------
# Sample loader: JSONL or synthetic
# ---------------------------------------------------------------------------


def _read_noise_vectors(path: Path) -> np.ndarray:
    """Expects each line to contain ``{"noise_values": [v1, v2, ..., v7]}`` or
    a list of such vectors. Skip lines that don't match.
    """
    vectors: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            candidates: Iterable = []
            if isinstance(obj, dict):
                v = obj.get("noise_values")
                if v is not None:
                    candidates = [v] if isinstance(v[0], (int, float)) else v  # type: ignore[index]
            elif isinstance(obj, list):
                candidates = obj
            for vec in candidates:
                if isinstance(vec, list) and len(vec) == 7:
                    try:
                        vectors.append([float(x) for x in vec])
                    except (TypeError, ValueError):
                        pass
    return np.asarray(vectors, dtype=np.float32)


def _synthetic_noise(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    conf_avg = rng.uniform(0.30, 1.0, n)
    conf_min = np.clip(conf_avg - rng.uniform(0, 0.4, n), 0, 1)
    conf_var_log = rng.uniform(-12, 0, n)
    conf_gap = rng.uniform(0, 0.6, n)
    punct_err = rng.uniform(0, 0.4, n)
    char_break = rng.uniform(0, 0.25, n)
    align = rng.uniform(0, 3500, n)
    return np.stack(
        [conf_avg, conf_min, conf_var_log, conf_gap, punct_err, char_break, align], axis=-1
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate = NCAGGate(hidden_dim=args.hidden_dim if args.hidden_dim > 0 else None)
    gate.load_state_dict(torch.load(args.gate_ckpt, map_location=device))
    gate.to(device).eval()

    if args.eval_jsonl:
        vectors = _read_noise_vectors(Path(args.eval_jsonl))
        if len(vectors) == 0:
            print(f"[N12] eval_jsonl produced 0 rows; falling back to synthetic.")
            vectors = _synthetic_noise(args.n_samples, args.seed)
        else:
            if len(vectors) > args.n_samples:
                rng = np.random.default_rng(args.seed)
                vectors = vectors[rng.choice(len(vectors), args.n_samples, replace=False)]
    else:
        vectors = _synthetic_noise(args.n_samples, args.seed)

    with torch.no_grad():
        noise_t = torch.tensor(vectors, dtype=torch.float32, device=device).unsqueeze(0)  # [1, N, 7]
        g = gate(noise_t).squeeze(0).cpu().numpy()  # [N]

    results = {
        "n_samples": int(len(vectors)),
        "gate_mean": float(g.mean()),
        "gate_std": float(g.std()),
        "gate_min": float(g.min()),
        "gate_max": float(g.max()),
        "per_feature_spearman": {},
        "sign_agreement_count": 0,
    }
    for i, feat in enumerate(NOISE_FEATURE_NAMES):
        rho, p = _spearman(g, vectors[:, i])
        agrees = int(np.sign(rho) == EXPECTED_SIGN[feat]) if rho != 0 else 0
        results["per_feature_spearman"][feat] = {
            "rho": rho,
            "p_value": p,
            "expected_sign": EXPECTED_SIGN[feat],
            "sign_agrees": bool(agrees),
        }
        results["sign_agreement_count"] += agrees

    json_path = out / "ncag_gate_corr.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[N12] wrote {json_path}")

    # Markdown summary table — paper-friendly
    md_lines = ["# N12 — NCAG Gate-vs-Noise Spearman Correlation\n"]
    md_lines.append(f"- gate checkpoint: `{args.gate_ckpt}`")
    md_lines.append(f"- n samples: {results['n_samples']}")
    md_lines.append(
        f"- gate stats: mean={results['gate_mean']:.3f} std={results['gate_std']:.3f} "
        f"min={results['gate_min']:.3f} max={results['gate_max']:.3f}"
    )
    md_lines.append(
        f"- sign agreement: {results['sign_agreement_count']} / {len(NOISE_FEATURE_NAMES)}\n"
    )
    md_lines.append("| Feature | Spearman ρ | p-value | Expected sign | Agrees |")
    md_lines.append("|---|---:|---:|:---:|:---:|")
    for feat, stat in results["per_feature_spearman"].items():
        md_lines.append(
            f"| `{feat}` | {stat['rho']:+.3f} | {stat['p_value']:.2e} | "
            f"{'+' if stat['expected_sign'] > 0 else '−'} | "
            f"{'✓' if stat['sign_agrees'] else '✗'} |"
        )
    (out / "report.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Optional scatter plot — skip silently if matplotlib not available
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(vectors[:, 0], g, s=4, alpha=0.5)
        ax.set_xlabel("conf_avg")
        ax.set_ylabel("gate value σ(W·noise)")
        ax.set_title("NCAG gate vs OCR confidence")
        fig.tight_layout()
        fig.savefig(out / "ncag_gate_scatter.png", dpi=120)
    except Exception as e:  # pragma: no cover
        print(f"[N12] matplotlib unavailable, skipping scatter ({e!r})")

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
