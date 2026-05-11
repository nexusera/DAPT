#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _fmt(x: Optional[Any], ndigits: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, (int, float)):
        return f"{x:.{ndigits}f}"
    return str(x)


def parse_task1(report: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    exact_f1 = _get(report, ["tasks", "task1", "metrics", "exact", "f1"])
    approx_f1 = _get(report, ["tasks", "task1", "metrics", "approx", "f1"])
    return (
        float(exact_f1) if exact_f1 is not None else None,
        float(approx_f1) if approx_f1 is not None else None,
    )


def parse_task3(report: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    ee = _get(report, ["tasks", "task3", "metrics", "exact_exact", "f1"])
    ea = _get(report, ["tasks", "task3", "metrics", "exact_approximate", "f1"])
    aa = _get(report, ["tasks", "task3", "metrics", "approximate_approximate", "f1"])
    return (
        float(ee) if ee is not None else None,
        float(ea) if ea is not None else None,
        float(aa) if aa is not None else None,
    )


def parse_task2(report: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    global_approx = _get(report, ["tasks", "task2_global", "metrics", "approx", "f1"])
    pos_approx = _get(report, ["tasks", "task2_pos_only", "metrics", "approx", "f1"])
    return (
        float(global_approx) if global_approx is not None else None,
        float(pos_approx) if pos_approx is not None else None,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize tokenizer-ablation downstream reports (Task1/2/3) into a compact table"
    )
    ap.add_argument(
        "--runs_dir",
        type=str,
        default=os.environ.get("RUNS_DIR")
        or os.path.join(os.environ.get("DAPT_ROOT", "/data/ocean/DAPT"), "runs"),
        help="Directory that contains t{n}_report_task{1,2,3}.json (default: $DAPT_ROOT/runs)",
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["t1", "t2", "t3", "t4"],
        help="Variants to summarize (default: t1 t2 t3 t4)",
    )
    ap.add_argument(
        "--ndigits",
        type=int,
        default=4,
        help="Number of digits after decimal in table (default: 4)",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    rows: List[List[str]] = []

    for v in args.variants:
        t1_path = runs_dir / f"{v}_report_task1.json"
        t2_path = runs_dir / f"{v}_report_task2.json"
        t3_path = runs_dir / f"{v}_report_task3.json"

        t1_exact = t1_approx = None
        t3_ee = t3_ea = t3_aa = None
        t2_global = t2_pos = None

        if t1_path.exists():
            r = _load_json(t1_path)
            t1_exact, t1_approx = parse_task1(r)

        if t3_path.exists():
            r = _load_json(t3_path)
            t3_ee, t3_ea, t3_aa = parse_task3(r)

        if t2_path.exists():
            r = _load_json(t2_path)
            t2_global, t2_pos = parse_task2(r)

        rows.append(
            [
                v,
                _fmt(t1_exact, args.ndigits),
                _fmt(t1_approx, args.ndigits),
                _fmt(t3_ee, args.ndigits),
                _fmt(t3_ea, args.ndigits),
                _fmt(t3_aa, args.ndigits),
                _fmt(t2_global, args.ndigits),
                _fmt(t2_pos, args.ndigits),
            ]
        )

    headers = [
        "variant",
        "task1_f1_exact",
        "task1_f1_approx",
        "task3_f1_ee",
        "task3_f1_ea",
        "task3_f1_aa",
        "task2_global_f1_approx",
        "task2_pos_f1_approx",
    ]

    # Markdown table (easy to paste into docs)
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
