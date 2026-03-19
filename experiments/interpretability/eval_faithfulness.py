#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional


EPS = 1e-12


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _top_scores(rec: Dict[str, Any]) -> List[float]:
    out: List[float] = []
    for it in rec.get("top_tokens", []) or []:
        if not isinstance(it, dict):
            continue
        v = _safe_float(it.get("score"))
        if v is not None:
            out.append(abs(v))
    return out


def _extract_true_scores(rec: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Try to read true perturbation scores if present.
    Supported keys:
      - faithfulness.full_score / deleted_score / kept_score
      - full_score / deleted_score / kept_score at root
    """
    blk = rec.get("faithfulness") if isinstance(rec.get("faithfulness"), dict) else {}
    full_score = _safe_float(blk.get("full_score", rec.get("full_score")))
    deleted_score = _safe_float(blk.get("deleted_score", rec.get("deleted_score")))
    kept_score = _safe_float(blk.get("kept_score", rec.get("kept_score")))
    return {
        "full_score": full_score,
        "deleted_score": deleted_score,
        "kept_score": kept_score,
    }


def _metric_deletion_aopc(rec: Dict[str, Any]) -> Dict[str, Any]:
    s = _extract_true_scores(rec)
    if s["full_score"] is not None and s["deleted_score"] is not None:
        return {
            "value": float(s["full_score"] - s["deleted_score"]),
            "mode": "true",
            "note": "computed from full_score - deleted_score",
        }

    top = _top_scores(rec)
    if not top:
        return {"value": None, "mode": "missing", "note": "no top_tokens scores"}

    # proxy: larger top-token attribution magnitude => stronger expected deletion effect
    return {
        "value": float(mean(top)),
        "mode": "proxy",
        "note": "proxy from mean(|top_token_attribution|)",
    }


def _metric_comprehensiveness(rec: Dict[str, Any]) -> Dict[str, Any]:
    s = _extract_true_scores(rec)
    if s["full_score"] is not None and s["deleted_score"] is not None:
        return {
            "value": float(s["full_score"] - s["deleted_score"]),
            "mode": "true",
            "note": "computed from full_score - deleted_score",
        }

    top = _top_scores(rec)
    if not top:
        return {"value": None, "mode": "missing", "note": "no top_tokens scores"}

    return {
        "value": float(sum(top)),
        "mode": "proxy",
        "note": "proxy from sum(|top_token_attribution|)",
    }


def _metric_sufficiency(rec: Dict[str, Any]) -> Dict[str, Any]:
    s = _extract_true_scores(rec)
    if s["full_score"] is not None and s["kept_score"] is not None:
        return {
            "value": float(s["full_score"] - s["kept_score"]),
            "mode": "true",
            "note": "computed from full_score - kept_score (lower is better)",
        }

    top = _top_scores(rec)
    if not top:
        return {"value": None, "mode": "missing", "note": "no top_tokens scores"}

    total = sum(top)
    top1 = max(top)
    # proxy: if one/few tokens dominate, "top set is sufficient" tends to be stronger
    return {
        "value": float(1.0 - top1 / (total + EPS)),
        "mode": "proxy",
        "note": "proxy from attribution concentration (1 - top1_share)",
    }


METRIC_FN = {
    "deletion_aopc": _metric_deletion_aopc,
    "comprehensiveness": _metric_comprehensiveness,
    "sufficiency": _metric_sufficiency,
}


def _summarize(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(vals),
        "mean": float(mean(vals)),
        "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def run(args: argparse.Namespace) -> None:
    metrics = args.metric or ["deletion_aopc", "comprehensiveness", "sufficiency"]
    for m in metrics:
        if m not in METRIC_FN:
            raise ValueError(f"Unsupported metric: {m}. Supported: {sorted(METRIC_FN.keys())}")

    records = list(_iter_jsonl(args.ig_file))
    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]

    per_metric_values: Dict[str, List[float]] = {m: [] for m in metrics}
    mode_counter: Dict[str, Dict[str, int]] = {m: {"true": 0, "proxy": 0, "missing": 0} for m in metrics}
    per_sample: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        sample_res: Dict[str, Any] = {
            "index": idx,
            "analysis_id": rec.get("analysis_id"),
            "metrics": {},
        }
        for m in metrics:
            item = METRIC_FN[m](rec)
            val = item.get("value")
            mode = item.get("mode", "missing")
            mode_counter[m][mode] = mode_counter[m].get(mode, 0) + 1
            if val is not None:
                per_metric_values[m].append(float(val))
            sample_res["metrics"][m] = item
        per_sample.append(sample_res)

    summary = {
        "ig_file": args.ig_file,
        "num_samples": len(records),
        "metrics": {},
    }

    for m in metrics:
        summary["metrics"][m] = {
            "aggregate": _summarize(per_metric_values[m]),
            "mode_count": mode_counter[m],
            "interpretation": (
                "true mode uses model scores before/after perturbation; "
                "proxy mode uses attribution-only approximation and is for relative comparison only"
            ),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_sample": per_sample}, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved faithfulness report to {out_path}")
    for m in metrics:
        agg = summary["metrics"][m]["aggregate"]
        modes = summary["metrics"][m]["mode_count"]
        print(f"[METRIC] {m}: mean={agg['mean']} count={agg['count']} modes={modes}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate faithfulness metrics from IG output JSONL")
    p.add_argument("--ig_file", type=str, required=True, help="IG output jsonl path")
    p.add_argument(
        "--metric",
        type=str,
        nargs="+",
        default=["deletion_aopc", "comprehensiveness", "sufficiency"],
        help="Metrics to compute",
    )
    p.add_argument("--output", type=str, required=True, help="Output json report path")
    p.add_argument("--max_samples", type=int, default=0, help="Optional cap for debugging")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
