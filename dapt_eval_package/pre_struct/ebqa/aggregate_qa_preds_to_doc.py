#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Aggregate EBQA QA-level predictions back to document-level predictions.

Why this exists:
- `predict_ebqa.py` outputs QA-level JSONL records (one per question/chunk), e.g.
  {"report_index": 12, "question_key": "姓名", "pred_text": "张三", "score": 3.14, ...}
- Downstream evaluation / teammate alignment scripts expect doc-level records with full text and `pred_pairs`, e.g.
  {"id": "14723", "report_title": "入院记录", "text": "...", "pred_pairs": [{"key":"姓名","value":"张三"}, ...]}

This script groups QA-level preds by `report_index`, attaches raw doc metadata
from the original KV-NER JSON (real_*_with_ocr.json), and writes doc-level JSONL.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json_or_jsonl(path: str) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first in ("{", "["):
            return json.load(f)

    items: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _doc_meta_from_raw(item: Dict[str, Any]) -> Tuple[str, str, str]:
    doc_id = str(item.get("record_id") or item.get("id") or "")
    title = str(item.get("category") or item.get("report_title") or item.get("title") or "通用病历")
    text = str(item.get("ocr_text") or item.get("text") or "")
    return doc_id, title, text


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate EBQA QA-level preds into doc-level pred_pairs.")
    ap.add_argument("--raw_file", required=True, help="Original KV-NER JSON/JSONL used for EBQA conversion (e.g. real_test_with_ocr.json)")
    ap.add_argument("--qa_pred_file", required=True, help="QA-level preds JSONL from predict_ebqa.py (e.g. runs/ebqa_macbert_preds.jsonl)")
    ap.add_argument("--output_file", required=True, help="Output doc-level JSONL (e.g. runs/ebqa_macbert_doc_preds.jsonl)")
    ap.add_argument("--prefer", choices=["score", "last"], default="score", help="When duplicated (doc,key), keep max-score or last")
    ap.add_argument(
        "--report_index_keys",
        type=str,
        default="report_index,report_id,doc_index,doc_id,task_id",
        help="Comma-separated candidate keys for document index/id in QA preds",
    )
    ap.add_argument(
        "--question_key_keys",
        type=str,
        default="question_key,key,field,question",
        help="Comma-separated candidate keys for the schema field name in QA preds",
    )
    ap.add_argument(
        "--pred_text_keys",
        type=str,
        default="pred_text,pred_answer,answer,prediction,pred,text",
        help="Comma-separated candidate keys for predicted value text in QA preds",
    )
    args = ap.parse_args()

    raw = _read_json_or_jsonl(args.raw_file)
    if isinstance(raw, dict):
        for k in ("data", "items", "results"):
            if k in raw and isinstance(raw[k], list):
                raw = raw[k]
                break
    if not isinstance(raw, list):
        raise ValueError(f"raw_file must be a JSON array or JSONL list; got {type(raw)}")

    report_index_keys = [k.strip() for k in str(args.report_index_keys).split(",") if k.strip()]
    question_key_keys = [k.strip() for k in str(args.question_key_keys).split(",") if k.strip()]
    pred_text_keys = [k.strip() for k in str(args.pred_text_keys).split(",") if k.strip()]

    def _pick_first(d: Dict[str, Any], keys: List[str]) -> Any:
        for k in keys:
            if k in d:
                return d.get(k)
        return None

    # group[report_index][question_key] = (pred_text, score)
    group: Dict[int, Dict[str, Tuple[str, Optional[float]]]] = defaultdict(dict)
    seen = 0
    used = 0
    skip_no_ridx = 0
    skip_no_key = 0
    for it in _iter_jsonl(args.qa_pred_file):
        seen += 1
        ridx = _pick_first(it, report_index_keys)
        if ridx is None:
            skip_no_ridx += 1
            continue
        try:
            ridx_i = int(ridx)
        except Exception:
            # Some pipelines use string ids; we can't safely map them to raw-file index.
            skip_no_ridx += 1
            continue

        key = _pick_first(it, question_key_keys)
        if not key:
            skip_no_key += 1
            continue
        key = str(key).strip()
        if not key:
            skip_no_key += 1
            continue

        pred_text = _pick_first(it, pred_text_keys)
        pred_text = str(pred_text or "")

        score_val = it.get("score")
        try:
            score_f = float(score_val) if score_val is not None else None
        except Exception:
            score_f = None

        if args.prefer == "last":
            group[ridx_i][key] = (pred_text, score_f)
            used += 1
        else:
            prev = group[ridx_i].get(key)
            if prev is None:
                group[ridx_i][key] = (pred_text, score_f)
            else:
                _, prev_score = prev
                if prev_score is None and score_f is None:
                    # tie -> keep last
                    group[ridx_i][key] = (pred_text, score_f)
                elif prev_score is None:
                    group[ridx_i][key] = (pred_text, score_f)
                elif score_f is None:
                    pass
                elif score_f >= prev_score:
                    group[ridx_i][key] = (pred_text, score_f)

            used += 1

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_count = 0
    docs_with_pairs = 0
    total_pairs = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx, raw_item in enumerate(raw):
            if not isinstance(raw_item, dict):
                continue
            doc_id, title, text = _doc_meta_from_raw(raw_item)
            kv = group.get(idx, {})
            pred_pairs = [{"key": k, "value": v} for k, (v, _s) in kv.items()]

            if pred_pairs:
                docs_with_pairs += 1
                total_pairs += len(pred_pairs)

            rec = {
                "id": doc_id,
                "report_title": title,
                "text": text,
                "pred_pairs": pred_pairs,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_count += 1

            print(f"[OK] Read QA preds: {seen} lines")
            print(
            f"[OK] Used QA preds: {used} lines | skipped(no report_index)={skip_no_ridx} skipped(no key)={skip_no_key}"
            )
            print(f"[OK] Wrote doc preds: {out_count} lines -> {out_path}")
            avg_pairs = (total_pairs / docs_with_pairs) if docs_with_pairs else 0.0
            print(f"[OK] Coverage: docs_with_pairs={docs_with_pairs}/{out_count} avg_pairs_per_doc={avg_pairs:.2f}")


if __name__ == "__main__":
    main()
