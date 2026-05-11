#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from run_attention_kv_nsp import _cohens_d, _mann_whitney_u


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _vals(rows: List[Dict[str, Any]], strategy: str, metric: str) -> List[float]:
    out: List[float] = []
    for r in rows:
        if str(r.get("strategy")) != strategy:
            continue
        v = r.get(metric)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare KV-MLM attention metrics between main and no_kvmlm runs.")
    parser.add_argument("--main_metrics", type=str, required=True, help="main run per_sample_metrics.jsonl")
    parser.add_argument("--abl_metrics", type=str, required=True, help="no_kvmlm run per_sample_metrics.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main_rows = _load_jsonl(Path(args.main_metrics))
    abl_rows = _load_jsonl(Path(args.abl_metrics))
    if not main_rows or not abl_rows:
        raise RuntimeError("Empty metrics rows in main/ablation input.")

    metrics = ("same_value_block_mass", "to_key_mass", "to_sep_boundary_mass")
    strategies = sorted(set(str(r.get("strategy")) for r in main_rows) | set(str(r.get("strategy")) for r in abl_rows))

    out: Dict[str, Any] = {
        "main_metrics": args.main_metrics,
        "abl_metrics": args.abl_metrics,
        "tests": {},
        "means": {},
    }
    for st in strategies:
        out["means"][st] = {}
        for mk in metrics:
            x = _vals(main_rows, st, mk)
            y = _vals(abl_rows, st, mk)
            if not x or not y:
                continue
            out["means"][st][mk] = {
                "main_mean": float(sum(x) / len(x)),
                "abl_mean": float(sum(y) / len(y)),
                "delta_main_minus_abl": float(sum(x) / len(x) - sum(y) / len(y)),
            }
            t = _mann_whitney_u(x, y)
            out["tests"][f"{st}__{mk}"] = {
                **t,
                "cohens_d": _cohens_d(x, y),
                "n_main": len(x),
                "n_abl": len(y),
            }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "kv_mlm_main_vs_no_kvmlm_summary.json"
    summary_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# KV-MLM Main vs w/o KV-MLM Significance")
    lines.append("")
    lines.append(f"- main metrics: `{args.main_metrics}`")
    lines.append(f"- ablation metrics: `{args.abl_metrics}`")
    lines.append("")
    lines.append("## Means")
    for st, info in out["means"].items():
        for mk, v in info.items():
            lines.append(
                f"- {st} / {mk}: main={v['main_mean']:.4f}, ablation={v['abl_mean']:.4f}, delta={v['delta_main_minus_abl']:+.4f}"
            )
    lines.append("")
    lines.append("## Significance Tests")
    for k, v in out["tests"].items():
        lines.append(
            f"- {k}: p={v.get('p_value',1.0):.6g}, d={v.get('cohens_d',0.0):.4f}, method={v.get('method','n/a')}, n_main={v.get('n_main',0)}, n_abl={v.get('n_abl',0)}"
        )
    (out_dir / "kv_mlm_main_vs_no_kvmlm_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] summary: {summary_path}")
    print(f"[OK] report: {out_dir / 'kv_mlm_main_vs_no_kvmlm_report.md'}")


if __name__ == "__main__":
    main()

