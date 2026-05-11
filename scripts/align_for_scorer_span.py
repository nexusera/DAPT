#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace("：", "").replace(":", "").strip()


def get_text_hash(text: Any) -> str:
    if not text:
        return ""
    clean = "".join(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", str(text)))[:100]
    return hashlib.md5(clean.encode()).hexdigest() if clean else ""


def _read_any_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        # Peek first non-whitespace
        while True:
            ch = f.read(1)
            if not ch:
                return None
            if not ch.isspace():
                break
        f.seek(0)
        if ch == "[" or ch == "{":
            return json.load(f)

    # Fallback: JSONL
    items = []
    with open(path, "r", encoding="utf-8") as f:
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


def _find_span_sequential(ocr_text: str, value: str, last_pos: int) -> Tuple[Optional[List[int]], int]:
    """Find span of value in ocr_text using teammate's 4-step fallback.

    Returns (span_or_none, new_last_pos).
    """
    if not ocr_text or not value:
        return None, last_pos

    val = str(value)

    # 1) Direct search from last_pos
    start = ocr_text.find(val, last_pos)
    v_len = len(val)

    # 2) Strip ends
    if start == -1:
        v_strip = val.strip()
        start = ocr_text.find(v_strip, last_pos)
        v_len = len(v_strip)

    # Helper: build whitespace-tolerant pattern
    def _pattern(s: str) -> str:
        chars = [c for c in s.strip() if not c.isspace()]
        if not chars:
            return ""
        return "".join([re.escape(c) + r"\s*" for c in chars])

    # 3) Regex from last_pos (tolerate internal spaces)
    if start == -1:
        pattern_str = _pattern(val)
        if pattern_str:
            match = re.search(pattern_str, ocr_text[last_pos:])
            if match:
                start = last_pos + match.start()
                v_len = match.end() - match.start()

    # 4) Regex global fallback
    if start == -1:
        pattern_str = _pattern(val)
        if pattern_str:
            match = re.search(pattern_str, ocr_text)
            if match:
                start = match.start()
                v_len = match.end() - match.start()

    if start != -1 and v_len > 0:
        end = start + v_len
        return [int(start), int(end)], int(end)

    return None, last_pos


def process_gt(gt_in: str, gt_out: str) -> Dict[str, str]:
    """Convert raw GT JSON (list) to scorer JSONL with spans.

    Returns: hash_to_id mapping for optional pred id recovery.
    """
    print(f"[align_for_scorer_span] Converting GT: {gt_in}")

    raw = _read_any_json(gt_in)
    if isinstance(raw, dict):
        # Some exports wrap data; try common keys
        for k in ("data", "items", "results"):
            if k in raw and isinstance(raw[k], list):
                raw = raw[k]
                break

    if not isinstance(raw, list):
        raise ValueError(f"GT input must be a JSON array or JSONL list; got {type(raw)}")

    excluded_labels = {"键名", "值", "KEY", "VALUE", "Unknown"}

    out_records: List[dict] = []
    hash_to_id: Dict[str, str] = {}

    for item in raw:
        if not isinstance(item, dict):
            continue

        rid = str(item.get("record_id") or item.get("id") or "N/A")
        ocr_text = str(item.get("ocr_text") or item.get("text") or "")
        title = str(item.get("category") or item.get("title") or item.get("report_title") or "")

        h = get_text_hash(ocr_text)
        if h and rid and rid != "N/A":
            hash_to_id[h] = rid

        raw_annos = item.get("transferred_annotations", []) or item.get("annotations", [])
        if not isinstance(raw_annos, list):
            raw_annos = []

        # Build anno spans by sequential search (teammate strategy)
        anno_spans: Dict[str, Optional[List[int]]] = {}
        last_pos = 0
        for a in raw_annos:
            if not isinstance(a, dict):
                continue
            aid = a.get("original_id") or a.get("id")
            if not aid:
                continue
            val = a.get("text", "")
            span, last_pos = _find_span_sequential(ocr_text, str(val or ""), last_pos)
            anno_spans[str(aid)] = span

        # Build annos map
        annos: Dict[str, dict] = {}
        for a in raw_annos:
            if not isinstance(a, dict):
                continue
            aid = a.get("original_id") or a.get("id")
            if aid:
                annos[str(aid)] = a

        pairs: List[dict] = []
        matched_ids = set()

        # Relations (explicit key->value)
        for rel in item.get("relations", []) or []:
            if not isinstance(rel, dict):
                continue
            fid = rel.get("from_id")
            tid = rel.get("to_id")
            if fid is None or tid is None:
                continue
            fid = str(fid)
            tid = str(tid)
            f_node = annos.get(fid)
            t_node = annos.get(tid)
            if not f_node or not t_node:
                continue

            k = normalize_text(f_node.get("text", ""))
            v = normalize_text(t_node.get("text", ""))
            if k:
                pairs.append(
                    {
                        "key": k,
                        "value": v,
                        "key_span": anno_spans.get(fid),
                        "value_span": anno_spans.get(tid),
                    }
                )

            matched_ids.add(fid)
            matched_ids.add(tid)

        # Leftover annotations: treat label as key (implicit key) and text as value
        for aid, node in annos.items():
            if aid in matched_ids:
                continue
            labels = node.get("labels") or []
            label = labels[0] if isinstance(labels, list) and labels else "Unknown"
            val = normalize_text(node.get("text", ""))
            if label not in excluded_labels and val:
                pairs.append(
                    {
                        "key": str(label),
                        "value": val,
                        "key_span": None,
                        "value_span": anno_spans.get(aid),
                    }
                )

        out_records.append(
            {
                "id": rid,
                "report_title": title,
                "ocr_text": ocr_text,
                "pairs": pairs,
            }
        )

    os.makedirs(os.path.dirname(gt_out) or ".", exist_ok=True)
    with open(gt_out, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[align_for_scorer_span] GT done: {len(out_records)} lines -> {gt_out}")
    return hash_to_id


def process_pred(pred_in: str, pred_out: str, hash_to_id: Optional[Dict[str, str]] = None) -> None:
    """Normalize pred JSONL to scorer format and optionally recover id by ocr_text hash."""
    print(f"[align_for_scorer_span] Converting Pred: {pred_in}")

    out_records: List[dict] = []
    for item in _iter_jsonl(pred_in):
        if not isinstance(item, dict):
            continue
        rid = str(item.get("id") or "N/A")
        ocr_text = str(item.get("ocr_text") or item.get("text") or "")
        title = str(item.get("report_title") or item.get("title") or "")

        if (not rid or rid == "N/A") and hash_to_id:
            h = get_text_hash(ocr_text)
            if h and h in hash_to_id:
                rid = hash_to_id[h]

        processed_pairs: List[dict] = []
        for p in item.get("pairs", []) or []:
            if not isinstance(p, dict):
                continue
            k = normalize_text(p.get("key"))
            v = normalize_text(p.get("value"))
            if not k or not v:
                continue
            processed_pairs.append(
                {
                    "key": k,
                    "value": v,
                    "key_span": p.get("key_span"),
                    # scorer doesn't use value_span, but keep for consistency/debug
                    "value_span": p.get("value_span"),
                }
            )

        out_records.append(
            {
                "id": rid,
                "report_title": title,
                "ocr_text": ocr_text,
                "pairs": processed_pairs,
            }
        )

    os.makedirs(os.path.dirname(pred_out) or ".", exist_ok=True)
    with open(pred_out, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[align_for_scorer_span] Pred done: {len(out_records)} lines -> {pred_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align GT/Pred for scorer with span recovery (teammate-compatible)")
    parser.add_argument("--gt_in", required=True, help="Raw GT JSON (array) path")
    parser.add_argument("--pred_in", required=True, help="Pred JSONL path (e.g., *_preds.jsonl)")
    parser.add_argument("--gt_out", required=True, help="Output aligned GT JSONL")
    parser.add_argument("--pred_out", required=True, help="Output aligned Pred JSONL")
    args = parser.parse_args()

    hash_to_id = process_gt(args.gt_in, args.gt_out)
    process_pred(args.pred_in, args.pred_out, hash_to_id=hash_to_id)

    # Count lines to help user detect mismatches early
    def _count_lines(p: str) -> int:
        if not os.path.exists(p):
            return 0
        with open(p, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    gt_n = _count_lines(args.gt_out)
    pred_n = _count_lines(args.pred_out)
    print("-" * 30)
    print(f"[align_for_scorer_span] Check: GT lines={gt_n}, Pred lines={pred_n}")
    if gt_n != pred_n:
        print(
            "[align_for_scorer_span] WARNING: line count mismatch. "
            "scorer.py will error because it zips by order. "
            "Make sure pred file is generated from the same test set and no samples were skipped."
        )
    print("-" * 30)


if __name__ == "__main__":
    main()
