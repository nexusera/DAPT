#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

from run_attention_kv_nsp import (
    _aggregate_attention,
    _build_perfect_noise_tensors,
    _load_dapt_model,
    _load_pair_samples,
    _mean_std,
    _plot_heatmap,
    _prepare_encoding,
    _to_tensor_2d,
)
from noise_feature_processor import NoiseFeatureProcessor


def _choose_mask_indices(strategy: str, key_idx: List[int], value_idx: List[int], span_len: int) -> List[int]:
    span_len = max(1, int(span_len))
    if not key_idx or not value_idx:
        return []
    if strategy == "entity":
        # Practical proxy: mask a compact center span in value segment.
        center = len(value_idx) // 2
        lo = max(0, center - span_len // 2)
        hi = min(len(value_idx), lo + span_len)
        return value_idx[lo:hi]
    if strategy == "boundary":
        # Boundary-focused: first tokens of value segment near [SEP] boundary.
        return value_idx[:span_len]
    raise ValueError(f"Unknown strategy: {strategy}")


def _select_rows(A: np.ndarray, row_idx: Sequence[int], col_idx: Sequence[int]) -> float:
    if not row_idx or not col_idx:
        return 0.0
    return float(A[np.ix_(list(row_idx), list(col_idx))].sum())


def _plot_metric_boxplot(per_sample: Sequence[Dict[str, Any]], metric_key: str, out_file: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    by_strategy: Dict[str, List[float]] = {}
    for r in per_sample:
        by_strategy.setdefault(str(r["strategy"]), []).append(float(r.get(metric_key, 0.0)))
    labels = sorted(by_strategy.keys())
    if not labels:
        return
    data = [by_strategy[x] for x in labels]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.4, 4.2), dpi=160)
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel(metric_key)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="KV-MLM attention analysis (entity/boundary masking).")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--noise_bins_json", type=str, default=None)
    parser.add_argument("--base_model_name", type=str, default="hfl/chinese-macbert-base")
    parser.add_argument("--noise_mode_hint", type=str, default="bucket", choices=["bucket", "linear", "mlp"])

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_samples_per_group", type=int, default=120)
    parser.add_argument("--last_n_layers", type=int, default=4)
    parser.add_argument("--mask_span_len", type=int, default=1)
    parser.add_argument("--mask_strategy", type=str, default="both", choices=["entity", "boundary", "both"])
    parser.add_argument("--inject_perfect_noise", action="store_true")
    parser.add_argument("--exclude_special_tokens", action="store_true")
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = out_dir / "cases"
    figs_dir = out_dir / "figures"
    cases_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_pair_samples(Path(args.input_file))
    if not samples:
        raise RuntimeError(f"No pair samples loaded from: {args.input_file}")

    # Keep balanced volume across groups if possible.
    buckets: Dict[str, List[Any]] = {}
    for s in samples:
        buckets.setdefault(s.group, []).append(s)
    chosen: List[Any] = []
    for _, rows in buckets.items():
        if len(rows) <= args.max_samples_per_group:
            chosen.extend(rows)
        else:
            chosen.extend(random.sample(rows, args.max_samples_per_group))
    samples = chosen

    tok_path = args.tokenizer_path or args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    if tokenizer.mask_token_id is None:
        raise RuntimeError("Tokenizer has no mask_token_id; cannot run MLM masking analysis.")

    noise_processor = None
    noise_bin_edges = None
    if args.noise_bins_json and Path(args.noise_bins_json).is_file():
        noise_processor = NoiseFeatureProcessor.load(args.noise_bins_json)
        noise_bin_edges = noise_processor.bin_edges

    model = _load_dapt_model(
        model_dir=Path(args.model_dir),
        device=args.device,
        base_model_name=args.base_model_name,
        noise_mode_hint=args.noise_mode_hint,
        noise_bin_edges=noise_bin_edges,
    )
    model.eval()
    noise_mode = str(getattr(model.config, "noise_mode", args.noise_mode_hint)).lower()

    strategies = ["entity", "boundary"] if args.mask_strategy == "both" else [args.mask_strategy]
    per_sample: List[Dict[str, Any]] = []
    total = len(samples)
    global_idx = 0

    with torch.no_grad():
        for s in samples:
            enc, key_idx, value_idx, token_texts = _prepare_encoding(
                tokenizer=tokenizer,
                key_text=s.key_text,
                value_text=s.value_text,
                max_length=args.max_length,
            )
            if not key_idx or not value_idx:
                continue
            ids = list(enc["input_ids"])
            mask = enc["attention_mask"]
            special = enc.get("special_tokens_mask", [0] * len(ids))
            ttypes = enc.get("token_type_ids")
            if ttypes is None:
                ttypes = [0] * len(ids)

            valid_idx = [i for i, m in enumerate(mask) if int(m) == 1]
            if args.exclude_special_tokens:
                valid_idx = [i for i in valid_idx if int(special[i]) == 0]
            if not valid_idx:
                continue

            sep_idx = [i for i, v in enumerate(special) if int(v) == 1]
            boundary_idx = sorted(set(sep_idx + [key_idx[-1], value_idx[0]]))

            for strategy in strategies:
                mask_idx = _choose_mask_indices(strategy, key_idx=key_idx, value_idx=value_idx, span_len=args.mask_span_len)
                if not mask_idx:
                    continue
                masked_ids = ids[:]
                for mi in mask_idx:
                    masked_ids[mi] = tokenizer.mask_token_id

                input_ids_t = _to_tensor_2d(masked_ids, dtype=torch.long, device=args.device)
                mask_t = _to_tensor_2d(mask, dtype=torch.long, device=args.device)
                ttype_t = _to_tensor_2d(ttypes, dtype=torch.long, device=args.device)

                noise_ids_t = None
                noise_values_t = None
                if args.inject_perfect_noise:
                    noise_ids_t, noise_values_t = _build_perfect_noise_tensors(
                        seq_len=len(masked_ids),
                        device=args.device,
                        noise_mode=noise_mode,
                        noise_processor=noise_processor,
                    )

                out = model(
                    input_ids=input_ids_t,
                    attention_mask=mask_t,
                    token_type_ids=ttype_t,
                    noise_ids=noise_ids_t,
                    noise_values=noise_values_t,
                    output_attentions=True,
                    return_dict=True,
                )
                attentions = out.attentions
                if attentions is None:
                    continue
                A = _aggregate_attention(attentions, last_n_layers=args.last_n_layers)

                denom = _select_rows(A, mask_idx, valid_idx)
                if denom <= 0:
                    continue
                same_block = _select_rows(A, mask_idx, value_idx) / denom
                to_key = _select_rows(A, mask_idx, key_idx) / denom
                to_boundary = _select_rows(A, mask_idx, boundary_idx) / denom

                rec = {
                    "sample_id": s.sample_id,
                    "group": s.group,
                    "strategy": strategy,
                    "mask_idx": mask_idx,
                    "same_value_block_mass": float(same_block),
                    "to_key_mass": float(to_key),
                    "to_sep_boundary_mass": float(to_boundary),
                    "key_len": len(key_idx),
                    "value_len": len(value_idx),
                    "tokens": token_texts,
                }
                rec["mask_to_value_submatrix"] = A[np.ix_(mask_idx, value_idx)].tolist()
                per_sample.append(rec)

                global_idx += 1
                if args.progress_every > 0 and (global_idx % args.progress_every == 0):
                    print(f"[progress] processed_pairs={global_idx} (raw samples={total})")

    if not per_sample:
        raise RuntimeError("No valid MLM attention records generated.")

    (out_dir / "per_sample_metrics.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in per_sample), encoding="utf-8"
    )

    summary: Dict[str, Any] = {
        "num_records": len(per_sample),
        "strategy_summary": {},
        "group_strategy_summary": {},
        "config": {
            "model_dir": args.model_dir,
            "input_file": args.input_file,
            "max_length": args.max_length,
            "last_n_layers": args.last_n_layers,
            "mask_span_len": args.mask_span_len,
            "mask_strategy": args.mask_strategy,
            "exclude_special_tokens": bool(args.exclude_special_tokens),
        },
    }

    strategies_seen = sorted({x["strategy"] for x in per_sample})
    groups_seen = sorted({x["group"] for x in per_sample})
    for st in strategies_seen:
        rows = [x for x in per_sample if x["strategy"] == st]
        summary["strategy_summary"][st] = {
            "same_value_block_mass": _mean_std([x["same_value_block_mass"] for x in rows]),
            "to_key_mass": _mean_std([x["to_key_mass"] for x in rows]),
            "to_sep_boundary_mass": _mean_std([x["to_sep_boundary_mass"] for x in rows]),
        }
    for g in groups_seen:
        summary["group_strategy_summary"][g] = {}
        for st in strategies_seen:
            rows = [x for x in per_sample if x["group"] == g and x["strategy"] == st]
            summary["group_strategy_summary"][g][st] = {
                "same_value_block_mass": _mean_std([x["same_value_block_mass"] for x in rows]),
                "to_key_mass": _mean_std([x["to_key_mass"] for x in rows]),
                "to_sep_boundary_mass": _mean_std([x["to_sep_boundary_mass"] for x in rows]),
            }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # plots
    _plot_metric_boxplot(
        per_sample, "same_value_block_mass", figs_dir / "same_value_block_mass_boxplot.png", "Same Value Block Mass by Strategy"
    )
    _plot_metric_boxplot(per_sample, "to_key_mass", figs_dir / "to_key_mass_boxplot.png", "To Key Mass by Strategy")
    _plot_metric_boxplot(
        per_sample, "to_sep_boundary_mass", figs_dir / "to_sep_boundary_mass_boxplot.png", "To SEP/Boundary Mass by Strategy"
    )

    # case heatmaps: top records by same_value_block_mass per strategy
    for st in strategies_seen:
        rows = sorted([x for x in per_sample if x["strategy"] == st], key=lambda x: x["same_value_block_mass"], reverse=True)[:6]
        for r in rows:
            sub = np.array(r["mask_to_value_submatrix"], dtype=np.float64)
            fn = cases_dir / f"{st}_{r['sample_id']}_mask2value.png"
            _plot_heatmap(sub, fn, title=f"{st} | same_value={r['same_value_block_mass']:.4f}")

    report_lines = []
    report_lines.append("# KV-MLM Attention Report")
    report_lines.append("")
    report_lines.append(f"- records: {len(per_sample)}")
    report_lines.append(f"- strategies: {', '.join(strategies_seen)}")
    report_lines.append("")
    report_lines.append("## Strategy Summary")
    for st in strategies_seen:
        v = summary["strategy_summary"][st]
        report_lines.append(
            f"- {st}: same_value={v['same_value_block_mass']['mean']:.4f}±{v['same_value_block_mass']['std']:.4f}; "
            f"to_key={v['to_key_mass']['mean']:.4f}±{v['to_key_mass']['std']:.4f}; "
            f"to_sep_boundary={v['to_sep_boundary_mass']['mean']:.4f}±{v['to_sep_boundary_mass']['std']:.4f}"
        )
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[OK] per-sample metrics: {out_dir / 'per_sample_metrics.jsonl'}")
    print(f"[OK] summary: {out_dir / 'summary.json'}")
    print(f"[OK] markdown report: {out_dir / 'report.md'}")
    print(f"[OK] cases dir: {cases_dir}")
    print(f"[OK] figures dir: {figs_dir}")


if __name__ == "__main__":
    main()

