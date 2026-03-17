#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _iter_json_or_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                if isinstance(o, dict):
                    rows.append(o)
            except Exception:
                continue
    return rows


def _extract_text(raw: Dict[str, Any]) -> str:
    for k in ("text", "raw_text", "content", "ocr_text"):
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            return v

    ocr_raw = raw.get("ocr_raw")
    if isinstance(ocr_raw, str) and ocr_raw.strip():
        return ocr_raw
    if isinstance(ocr_raw, dict):
        wr = ocr_raw.get("words_result")
        if isinstance(wr, list):
            words = []
            for it in wr:
                if isinstance(it, dict) and isinstance(it.get("words"), str):
                    words.append(it["words"])
            if words:
                return "".join(words)

    if isinstance(raw.get("spans"), dict):
        return ""
    return ""


def _to_key(idx: int, obj: Dict[str, Any]) -> str:
    for k in ("report_index", "id", "doc_id", "uid"):
        v = obj.get(k)
        if v is not None:
            return str(v)
    return str(idx)


def _normalize_pred(item: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "pred_text": item.get("pred_text", ""),
        "question_key": item.get("question_key") or item.get("key"),
        "task": item.get("task"),
    }
    if "pairs" in item:
        out["pairs"] = item.get("pairs")
    if "structured" in item:
        out["structured"] = item.get("structured")
    if "entities" in item:
        out["entities"] = item.get("entities")
    for k in ("start_char", "end_char", "score"):
        if k in item:
            out[k] = item[k]
    return out


def _normalize_gt(item: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ("spans", "transferred_annotations", "key_value_pairs", "gt_text", "answer_text"):
        if k in item:
            out[k] = item[k]
    return out


def build(args: argparse.Namespace) -> None:
    preds = list(_iter_json_or_jsonl(args.pred_file))
    gts = list(_iter_json_or_jsonl(args.gt_file)) if args.gt_file else []
    raws = list(_iter_json_or_jsonl(args.raw_file)) if args.raw_file else []

    gt_map = {_to_key(i, x): x for i, x in enumerate(gts)}
    raw_map = {_to_key(i, x): x for i, x in enumerate(raws)}

    out_rows: List[Dict[str, Any]] = []
    for i, p in enumerate(preds):
        k = _to_key(i, p)
        g = gt_map.get(k, {})
        r = raw_map.get(k, {})
        text = _extract_text(r)

        row: Dict[str, Any] = {
            "analysis_id": f"{args.task}_{i}",
            "task": args.task,
            "report_index": p.get("report_index", g.get("report_index", r.get("report_index", i))),
            "text": text,
            "pred": _normalize_pred(p),
            "gt": _normalize_gt(g),
        }

        if isinstance(r.get("noise_values"), list):
            row["noise_values"] = r.get("noise_values")

        out_rows.append(row)

    if args.max_samples and args.max_samples > 0:
        out_rows = out_rows[: args.max_samples]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] Saved {len(out_rows)} rows to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build analysis_set.jsonl for attribution experiments")
    p.add_argument("--task", type=str, choices=["task1", "task2", "task3"], required=True)
    p.add_argument("--pred_file", type=str, required=True)
    p.add_argument("--gt_file", type=str, default="")
    p.add_argument("--raw_file", type=str, default="")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
