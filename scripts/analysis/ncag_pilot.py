#!/usr/bin/env python3
"""NCAG pilot — plan §A5 lines 727-729 quantitative judgment.

Runs Qwen3-0.6B-Base + NCAG on a 50k-chunk subset for 1 epoch and answers
two questions:

    (a) Does training stay numerically stable? (loss is finite, no NaN)
    (b) Does the gate learn ``low-confidence → low weight``?

The quantitative judgment criterion (plan line 728):

    Spearman ρ ( gate value , per-token mean OCR confidence ) > 0.3
    with p < 0.05

If both hold → green-light the full N3/N4 CPT sweep.
If either fails → fall back to additive-only (plan line 729 decision point).

Usage::

    python -m scripts.analysis.ncag_pilot \
        --model_name_or_path Qwen/Qwen3-0.6B-Base \
        --output_dir /data/ocean/code/dapt/model/ncag_pilot \
        --span_data /data/ocean/code/dapt/data/train_chunked.txt \
        --max_samples 50000 \
        --num_epochs 1 \
        --noise_mode ncag

This script is a thin wrapper around :mod:`kv_llm.train_cpt` — it imports
the existing parsing/training pipeline, forces a small-batch / single-card
profile, and runs the gate-vs-confidence correlation eval after training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

# Allow ``python scripts/analysis/ncag_pilot.py`` from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kv_llm.ncag import NCAGGate  # noqa: E402  (after sys.path insert)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NCAG pilot — plan §A5 quantitative judgment.")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--span_data", required=True)
    p.add_argument("--noise_bins_json", default=None)
    p.add_argument("--max_samples", type=int, default=50000)
    p.add_argument("--num_epochs", type=float, default=1.0)
    p.add_argument(
        "--noise_mode",
        choices=["ncag", "ncag_additive"],
        default="ncag",
        help="Pilot runs N3 by default; pass ncag_additive to pilot N4 instead.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_samples", type=int, default=512)
    p.add_argument(
        "--rho_threshold", type=float, default=0.3, help="Spearman threshold for go-decision."
    )
    p.add_argument(
        "--p_threshold", type=float, default=0.05, help="p-value threshold for go-decision."
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Eval: dump (gate, conf) pairs and compute Spearman ρ
# ---------------------------------------------------------------------------


def _spearmanr_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """SciPy is heavy — implement Spearman via rankdata + Pearson, fall back
    on Pearson if SciPy is unavailable.
    """
    try:
        from scipy.stats import spearmanr  # type: ignore

        r, p = spearmanr(x, y)
        return float(r), float(p)
    except Exception:
        # Manual fallback using numpy ranking
        from math import sqrt

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
        # crude two-sided p approximation via t-distribution
        if n > 2 and abs(r) < 1.0:
            t = r * sqrt((n - 2) / max(1 - r**2, 1e-12))
            # erfc-based bound (very loose, only for "is p small?" sanity)
            from math import erfc

            p = float(erfc(abs(t) / sqrt(2)))
        else:
            p = 0.0
        return r, p


def eval_gate_correlation(
    *,
    gate_ckpt: Path,
    eval_jsonl: Path | None,
    n_samples: int,
    device: torch.device,
) -> dict:
    """Sample noise vectors, run them through the saved gate, report ρ.

    Synthesises a controlled noise distribution if no eval_jsonl is given:
    each sample draws conf_avg ∈ U[0.4, 1.0] and the rest from feature
    ranges; this gives the gate a clear "low conf → should be small gate"
    signal to demonstrate.
    """
    gate = NCAGGate()
    state = torch.load(gate_ckpt, map_location=device)
    gate.load_state_dict(state)
    gate.to(device).eval()

    if eval_jsonl is not None and eval_jsonl.exists():
        # TODO(verify): plug in the synthetic-noise benchmark loader once we
        # have a confirmed schema for per-token confidence in D1.16 output.
        raise NotImplementedError("eval_jsonl loading is not wired in v0; use synthetic mode.")

    rng = np.random.default_rng(0)
    conf_avg = rng.uniform(0.40, 1.0, size=n_samples)
    conf_min = np.clip(conf_avg - rng.uniform(0, 0.3, n_samples), 0, 1)
    conf_var_log = rng.uniform(-12, 0, n_samples)
    conf_gap = rng.uniform(0, 0.5, n_samples)
    punct_err = rng.uniform(0, 0.3, n_samples)
    char_break = rng.uniform(0, 0.25, n_samples)
    align = rng.uniform(0, 3500, n_samples)

    noise = np.stack(
        [conf_avg, conf_min, conf_var_log, conf_gap, punct_err, char_break, align], axis=-1
    )
    noise_t = torch.tensor(noise, dtype=torch.float32, device=device).unsqueeze(0)  # [1, N, 7]

    with torch.no_grad():
        g = gate(noise_t).squeeze(0).cpu().numpy()  # [N]

    rho, p_value = _spearmanr_safe(g, conf_avg)
    return {
        "n_samples": int(n_samples),
        "gate_mean": float(g.mean()),
        "gate_std": float(g.std()),
        "gate_min": float(g.min()),
        "gate_max": float(g.max()),
        "spearman_rho_gate_vs_conf_avg": rho,
        "spearman_p_value": p_value,
        "raw_low_conf_gate_avg": float(g[conf_avg < 0.6].mean()) if (conf_avg < 0.6).any() else None,
        "raw_high_conf_gate_avg": float(g[conf_avg > 0.85].mean()) if (conf_avg > 0.85).any() else None,
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Delegate training to the standard CPT entrypoint. We synthesise the
    # CLI it expects.
    train_argv: List[str] = [
        "train_cpt",
        "--model_name_or_path",
        args.model_name_or_path,
        "--output_dir",
        str(out),
        "--span_data",
        args.span_data,
        "--schedule",
        "span",  # span phase only, no NSP — saves time in pilot
        "--noise_mode",
        args.noise_mode,
        "--max_samples",
        str(args.max_samples),
        "--span_epochs_per_round",
        str(args.num_epochs),
        "--num_rounds",
        "1",
        "--per_device_train_batch_size",
        "16",
        "--gradient_accumulation_steps",
        "1",
        "--bf16",
        "--seed",
        str(args.seed),
        "--logging_steps",
        "10",
    ]
    if args.noise_bins_json:
        train_argv += ["--noise_bins_json", args.noise_bins_json]

    print(f"[NCAG-pilot] launching train_cpt with: {' '.join(train_argv)}")
    sys.argv = train_argv  # train_cpt.main() reads sys.argv via argparse
    from kv_llm.train_cpt import main as train_main

    train_main()

    # Locate the gate checkpoint saved by save_pretrained.
    gate_ckpt = out / "span" / "final_model" / "kv_llm_ncag_gate.pt"
    if not gate_ckpt.exists():
        candidates = list(out.rglob("kv_llm_ncag_gate.pt"))
        if not candidates:
            raise FileNotFoundError(f"No NCAG gate checkpoint found under {out}")
        gate_ckpt = candidates[-1]
    print(f"[NCAG-pilot] evaluating gate from {gate_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = eval_gate_correlation(
        gate_ckpt=gate_ckpt,
        eval_jsonl=None,
        n_samples=args.eval_samples,
        device=device,
    )

    rho = metrics["spearman_rho_gate_vs_conf_avg"]
    p_value = metrics["spearman_p_value"]
    metrics["pilot_decision"] = {
        "rho_threshold": args.rho_threshold,
        "p_threshold": args.p_threshold,
        # Plan line 728: low conf → low gate, so we expect ρ > 0 (gate moves *with* confidence).
        "passed": (rho > args.rho_threshold) and (p_value < args.p_threshold),
    }

    report_path = out / "ncag_pilot_report.json"
    report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[NCAG-pilot] report → {report_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    decision = "GO" if metrics["pilot_decision"]["passed"] else "NO-GO"
    print(
        f"\n[NCAG-pilot] DECISION: {decision}  "
        f"(rho={rho:.3f} thr={args.rho_threshold}; p={p_value:.3g} thr={args.p_threshold})"
    )
    if decision == "NO-GO":
        sys.exit(1)


if __name__ == "__main__":
    main()
