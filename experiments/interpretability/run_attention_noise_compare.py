#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str]) -> None:
    print("[cmd]", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")


def _safe_get(d: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    if isinstance(cur, (int, float)):
        return float(cur)
    return default


def _build_compare(with_summary: Dict[str, Any], without_summary: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "num_samples": {
            "with_noise": int(with_summary.get("num_samples", 0)),
            "without_noise": int(without_summary.get("num_samples", 0)),
        },
        "groups": {},
        "noise_groups": {},
        "tests": {
            "with_noise": with_summary.get("tests", {}),
            "without_noise": without_summary.get("tests", {}),
        },
    }

    group_names = sorted(set((with_summary.get("groups") or {}).keys()) | set((without_summary.get("groups") or {}).keys()))
    for g in group_names:
        wm = _safe_get(with_summary, "groups", g, "csam", "mean", default=0.0)
        wom = _safe_get(without_summary, "groups", g, "csam", "mean", default=0.0)
        out["groups"][g] = {
            "with_noise_csam_mean": wm,
            "without_noise_csam_mean": wom,
            "delta_with_minus_without": wm - wom,
            "with_noise_csam_std": _safe_get(with_summary, "groups", g, "csam", "std", default=0.0),
            "without_noise_csam_std": _safe_get(without_summary, "groups", g, "csam", "std", default=0.0),
        }

    noise_names = sorted(
        set((with_summary.get("noise_groups") or {}).keys()) | set((without_summary.get("noise_groups") or {}).keys())
    )
    for n in noise_names:
        wm = _safe_get(with_summary, "noise_groups", n, "csam", "mean", default=0.0)
        wom = _safe_get(without_summary, "noise_groups", n, "csam", "mean", default=0.0)
        ws = _safe_get(with_summary, "noise_groups", n, "csam", "std", default=0.0)
        wos = _safe_get(without_summary, "noise_groups", n, "csam", "std", default=0.0)
        out["noise_groups"][n] = {
            "with_noise_csam_mean": wm,
            "without_noise_csam_mean": wom,
            "delta_with_minus_without": wm - wom,
            "with_noise_csam_std": ws,
            "without_noise_csam_std": wos,
            "delta_std_with_minus_without": ws - wos,
        }
    return out


def _build_report(compare: Dict[str, Any], with_dir: Path, without_dir: Path) -> str:
    lines: List[str] = []
    lines.append("# Noise-Embedding Attention Compare Report")
    lines.append("")
    lines.append(f"- with-noise run: `{with_dir}`")
    lines.append(f"- without-noise run: `{without_dir}`")
    lines.append("")
    lines.append("## Group CSAM Compare")
    for g, v in (compare.get("groups") or {}).items():
        lines.append(
            "- {g}: with={wm:.4f}, without={wom:.4f}, delta={d:+.4f}, with_std={ws:.4f}, without_std={wos:.4f}".format(
                g=g,
                wm=v.get("with_noise_csam_mean", 0.0),
                wom=v.get("without_noise_csam_mean", 0.0),
                d=v.get("delta_with_minus_without", 0.0),
                ws=v.get("with_noise_csam_std", 0.0),
                wos=v.get("without_noise_csam_std", 0.0),
            )
        )
    lines.append("")
    lines.append("## Noise-bucket CSAM Compare")
    if not (compare.get("noise_groups") or {}):
        lines.append("- No noise buckets found in input metadata.")
    else:
        for n, v in (compare.get("noise_groups") or {}).items():
            lines.append(
                "- {n}: with={wm:.4f}, without={wom:.4f}, delta_mean={d:+.4f}, with_std={ws:.4f}, without_std={wos:.4f}, delta_std={ds:+.4f}".format(
                    n=n,
                    wm=v.get("with_noise_csam_mean", 0.0),
                    wom=v.get("without_noise_csam_mean", 0.0),
                    d=v.get("delta_with_minus_without", 0.0),
                    ws=v.get("with_noise_csam_std", 0.0),
                    wos=v.get("without_noise_csam_std", 0.0),
                    ds=v.get("delta_std_with_minus_without", 0.0),
                )
            )
    lines.append("")
    lines.append("## Quick Interpretation")
    lines.append("- Prefer `delta_with_minus_without > 0` on CSAM means, especially under low-quality/noisy buckets.")
    lines.append("- Prefer `delta_std_with_minus_without < 0` (with-noise has smaller std), indicating better stability.")
    lines.append("")
    lines.append("## Raw Test Outputs")
    lines.append("- with-noise tests: see `with_noise/summary.json`")
    lines.append("- without-noise tests: see `without_noise/summary.json`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KV-NSP attention twice (with/without Noise-Embedding) and compare.")
    parser.add_argument("--with_noise_model_dir", type=str, required=True)
    parser.add_argument("--without_noise_model_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--noise_bins_json", type=str, default=None)
    parser.add_argument("--noise_meta_file", type=str, default=None, help="Optional JSON/JSONL with noise_level/conf_avg.")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_samples_per_group", type=int, default=200)
    parser.add_argument("--last_n_layers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument("--device", type=str, default=("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"))

    parser.add_argument("--inject_perfect_noise", action="store_true")
    parser.add_argument("--auto_generate_negatives", action="store_true")
    parser.add_argument("--run_rollout", action="store_true")
    parser.add_argument("--exclude_special_tokens", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with_dir = out_dir / "with_noise"
    without_dir = out_dir / "without_noise"

    nsp_script = Path(__file__).resolve().parent / "run_attention_kv_nsp.py"
    if not nsp_script.is_file():
        raise FileNotFoundError(f"Cannot find nsp script: {nsp_script}")

    common = [
        "--tokenizer_path",
        args.tokenizer_path,
        "--input_file",
        args.input_file,
        "--max_length",
        str(args.max_length),
        "--max_samples_per_group",
        str(args.max_samples_per_group),
        "--last_n_layers",
        str(args.last_n_layers),
        "--topk",
        str(args.topk),
        "--seed",
        str(args.seed),
        "--progress_every",
        str(args.progress_every),
        "--device",
        args.device,
    ]
    if args.noise_bins_json:
        common += ["--noise_bins_json", args.noise_bins_json]
    if args.noise_meta_file:
        common += ["--noise_meta_file", args.noise_meta_file]
    if args.inject_perfect_noise:
        common.append("--inject_perfect_noise")
    if args.auto_generate_negatives:
        common.append("--auto_generate_negatives")
    if args.run_rollout:
        common.append("--run_rollout")
    if args.exclude_special_tokens:
        common.append("--exclude_special_tokens")

    cmd_with = [
        sys.executable,
        str(nsp_script),
        "--model_dir",
        args.with_noise_model_dir,
        "--output_dir",
        str(with_dir),
    ] + common
    cmd_without = [
        sys.executable,
        str(nsp_script),
        "--model_dir",
        args.without_noise_model_dir,
        "--output_dir",
        str(without_dir),
    ] + common

    print("[stage] run with-noise model")
    _run(cmd_with)
    print("[stage] run without-noise model")
    _run(cmd_without)

    with_summary = json.loads((with_dir / "summary.json").read_text(encoding="utf-8"))
    without_summary = json.loads((without_dir / "summary.json").read_text(encoding="utf-8"))
    compare = _build_compare(with_summary, without_summary)

    compare_json = out_dir / "compare_summary.json"
    compare_json.write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
    compare_md = out_dir / "compare_report.md"
    compare_md.write_text(_build_report(compare, with_dir, without_dir), encoding="utf-8")

    print(f"[OK] with-noise dir: {with_dir}")
    print(f"[OK] without-noise dir: {without_dir}")
    print(f"[OK] compare summary: {compare_json}")
    print(f"[OK] compare report: {compare_md}")


if __name__ == "__main__":
    main()

