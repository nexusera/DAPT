#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Mine real case studies for KV-NER from model outputs.

This script is designed for the workflow in DAPT/pipeline_xiaorong.md:
- Run inference via compare_models.py to generate standardized JSONL:
  *_preds.jsonl and *_gt.jsonl
- Optionally keep the original test_data JSON/JSONL that contains richer OCR payload
  such as `ocr_raw`, `noise_values(_per_word)`, and/or confidence fields.

Goal:
- Automatically find samples where *our model* correctly predicts at least one GT pair
  that the *baseline* misses.
- Prefer samples with low OCR confidence / strong noise signals.
- Export a short markdown snippet you can paste into paper as a Case Study draft.

Example:
  python DAPT/scripts/mine_case_study_kvner.py \
    --gt /data/ocean/DAPT/runs/macbert_eval_summary_gt.jsonl \
    --pred_ours /data/ocean/DAPT/runs/macbert_eval_summary_preds.jsonl \
    --pred_base /data/ocean/DAPT/runs/staged_eval_summary_preds.jsonl \
    --source_test /data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \
    --topk 5 \
    --out_md /data/ocean/DAPT/runs/case_study_candidates.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROB_KEYS = ["probability", "prob", "score", "scores", "confidence", "conf"]


def _read_any_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return None
            if not ch.isspace():
                break
        f.seek(0)
        if ch in ("[", "{"):
            return json.load(f)

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


def _normalize_for_match(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    # remove spaces + common punctuation, keep CJK/letters/digits
    return "".join(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", s)).lower()


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    try:
        v = float(str(x))
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _get_text_hash(text: Any) -> str:
    if not text:
        return ""
    clean = "".join(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", str(text)))[:100]
    return hashlib.md5(clean.encode()).hexdigest() if clean else ""


def _extract_ocr_text(item: dict) -> str:
    # standardized outputs
    for k in ("ocr_text", "text", "report"):
        if k in item and item.get(k):
            return str(item.get(k))

    # nested
    data = item.get("data")
    if isinstance(data, dict) and data.get("ocr_text"):
        return str(data.get("ocr_text"))

    # raw baidu ocr payload
    ocr_raw = item.get("ocr_raw")
    if isinstance(ocr_raw, dict):
        wr = ocr_raw.get("words_result")
        if isinstance(wr, list):
            words = []
            for w in wr:
                if isinstance(w, dict) and "words" in w:
                    words.append(str(w["words"]))
            if words:
                return "".join(words)
    return ""


def _index_source_records(source_items: Sequence[dict]) -> Dict[str, dict]:
    """Index source records by (1) id and (2) ocr_text hash for fallback."""
    by_id: Dict[str, dict] = {}
    by_hash: Dict[str, dict] = {}

    for it in source_items:
        if not isinstance(it, dict):
            continue
        rid = it.get("id") or it.get("record_id")
        if rid is not None:
            by_id[str(rid)] = it

        ocr_text = _extract_ocr_text(it)
        h = _get_text_hash(ocr_text)
        if h:
            by_hash[h] = it

    # store hash index under special key? return both via tuple is nicer
    # but keep simple: encode as dict with two levels
    return {"__by_id__": by_id, "__by_hash__": by_hash}  # type: ignore[return-value]


def _find_source_item(source_index: dict, sample_id: str, ocr_text: str) -> Optional[dict]:
    by_id = source_index.get("__by_id__", {})
    if sample_id in by_id:
        return by_id[sample_id]
    h = _get_text_hash(ocr_text)
    by_hash = source_index.get("__by_hash__", {})
    return by_hash.get(h)


def _pairs_from_std_record(rec: dict) -> List[Tuple[str, str]]:
    pairs = []
    for p in rec.get("pairs", []) or []:
        if not isinstance(p, dict):
            continue
        k = p.get("key")
        v = p.get("value")
        if k is None:
            continue
        k = str(k)
        v = "" if v is None else str(v)
        pairs.append((k, v))
    return pairs


def _pairs_set_normalized(rec: dict) -> set[Tuple[str, str]]:
    out: set[Tuple[str, str]] = set()
    for k, v in _pairs_from_std_record(rec):
        nk = _normalize_for_match(k)
        nv = _normalize_for_match(v)
        if nk:
            out.add((nk, nv))
    return out


@dataclass
class ConfSummary:
    mean_conf: Optional[float] = None
    min_conf: Optional[float] = None
    source: str = ""


def _conf_from_noise_values(noise_values: Any) -> Optional[ConfSummary]:
    # noise_values could be:
    # - global 7-d list
    # - per-char list[[7-d], ...]
    if not isinstance(noise_values, list) or not noise_values:
        return None

    # global vector
    if len(noise_values) == 7 and all(not isinstance(v, (list, tuple)) for v in noise_values):
        conf_avg = _safe_float(noise_values[0])
        conf_min = _safe_float(noise_values[1])
        if conf_avg is None and conf_min is None:
            return None
        return ConfSummary(mean_conf=conf_avg, min_conf=conf_min, source="noise_values(global)")

    # per-char vectors
    conf_avgs: List[float] = []
    conf_mins: List[float] = []
    for row in noise_values:
        if not (isinstance(row, (list, tuple)) and len(row) >= 2):
            continue
        a = _safe_float(row[0])
        m = _safe_float(row[1])
        if a is not None:
            conf_avgs.append(a)
        if m is not None:
            conf_mins.append(m)
    if not conf_avgs and not conf_mins:
        return None
    mean_conf = (sum(conf_avgs) / len(conf_avgs)) if conf_avgs else None
    min_conf = min(conf_mins) if conf_mins else None
    return ConfSummary(mean_conf=mean_conf, min_conf=min_conf, source="noise_values(per_char)")


def _conf_from_noise_values_per_word(nv_words: Any) -> Optional[ConfSummary]:
    if not isinstance(nv_words, list) or not nv_words:
        return None
    conf_avgs: List[float] = []
    conf_mins: List[float] = []
    for row in nv_words:
        if not (isinstance(row, (list, tuple)) and len(row) >= 2):
            continue
        a = _safe_float(row[0])
        m = _safe_float(row[1])
        if a is not None:
            conf_avgs.append(a)
        if m is not None:
            conf_mins.append(m)
    if not conf_avgs and not conf_mins:
        return None
    mean_conf = (sum(conf_avgs) / len(conf_avgs)) if conf_avgs else None
    min_conf = min(conf_mins) if conf_mins else None
    return ConfSummary(mean_conf=mean_conf, min_conf=min_conf, source="noise_values_per_word")


def _conf_from_ocr_raw(ocr_raw: Any) -> Optional[ConfSummary]:
    if not isinstance(ocr_raw, dict):
        return None

    # Some OCR dumps may already provide top-level stats
    top_avg = _safe_float(ocr_raw.get("conf_avg"))
    top_min = _safe_float(ocr_raw.get("conf_min"))
    if top_avg is not None or top_min is not None:
        return ConfSummary(mean_conf=top_avg, min_conf=top_min, source="ocr_raw(top)")

    words = ocr_raw.get("words_result")
    if not isinstance(words, list) or not words:
        return None

    probs: List[float] = []
    # word-level
    for w in words:
        if not isinstance(w, dict):
            continue
        for pk in PROB_KEYS:
            v = _safe_float(w.get(pk))
            if v is not None:
                probs.append(v)
                break
        # char-level (if present)
        chars = w.get("chars")
        if isinstance(chars, list):
            for ch in chars:
                if not isinstance(ch, dict):
                    continue
                for pk in PROB_KEYS:
                    v = _safe_float(ch.get(pk))
                    if v is not None:
                        probs.append(v)
                        break

    if not probs:
        return None

    return ConfSummary(mean_conf=sum(probs) / len(probs), min_conf=min(probs), source="ocr_raw(words_result)")


def extract_conf_summary(source_item: Optional[dict]) -> ConfSummary:
    if not source_item:
        return ConfSummary(source="missing")

    # 1) direct top-level
    top_avg = _safe_float(source_item.get("conf_avg"))
    top_min = _safe_float(source_item.get("conf_min"))
    if top_avg is not None or top_min is not None:
        return ConfSummary(mean_conf=top_avg, min_conf=top_min, source="source(top)")

    # 2) noise vectors
    nv_words = source_item.get("noise_values_per_word")
    if nv_words is None and isinstance(source_item.get("data"), dict):
        nv_words = source_item["data"].get("noise_values_per_word")
    cs = _conf_from_noise_values_per_word(nv_words)
    if cs:
        return cs

    nv = source_item.get("noise_values")
    if nv is None and isinstance(source_item.get("data"), dict):
        nv = source_item["data"].get("noise_values")
    cs = _conf_from_noise_values(nv)
    if cs:
        return cs

    # 3) ocr_raw
    ocr_raw = source_item.get("ocr_raw")
    cs = _conf_from_ocr_raw(ocr_raw)
    if cs:
        return cs

    return ConfSummary(source="unknown")


def _shorten(text: str, max_len: int = 260) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len // 2] + " ... " + t[-max_len // 2 :]


def _context_around(text: str, needle: str, window: int = 60) -> Optional[str]:
    if not text or not needle:
        return None
    idx = text.find(needle)
    if idx < 0:
        return None
    s = max(0, idx - window)
    e = min(len(text), idx + len(needle) + window)
    return text[s:e].replace("\n", " ")


def mine_cases(
    gt_by_id: Dict[str, dict],
    ours_by_id: Dict[str, dict],
    base_by_id: Dict[str, dict],
    source_index: Optional[dict],
    topk: int,
    min_delta: int,
) -> List[dict]:
    candidates: List[dict] = []

    for sid, gt in gt_by_id.items():
        ours = ours_by_id.get(sid)
        base = base_by_id.get(sid)
        if not ours or not base:
            continue

        gt_set = _pairs_set_normalized(gt)
        ours_set = _pairs_set_normalized(ours)
        base_set = _pairs_set_normalized(base)

        ours_correct = ours_set & gt_set
        base_correct = base_set & gt_set
        delta = ours_correct - base_correct

        if len(delta) < min_delta:
            continue

        ocr_text = str(gt.get("ocr_text") or ours.get("ocr_text") or base.get("ocr_text") or "")
        report_title = str(gt.get("report_title") or "")

        source_item = None
        if source_index is not None:
            source_item = _find_source_item(source_index, sid, ocr_text)

        conf = extract_conf_summary(source_item)
        mean_conf = conf.mean_conf

        # prefer cases where the corrected value is not literally present in OCR
        delta_list = sorted(list(delta))
        evidence = []
        not_found_cnt = 0
        for nk, nv in delta_list[:5]:
            # nv is normalized; search in normalized OCR
            nocr = _normalize_for_match(ocr_text)
            found = bool(nv and nv in nocr)
            if not found:
                not_found_cnt += 1
            evidence.append({"key": nk, "value": nv, "gt_value_in_ocr": found})

        score = 0.0
        score += 10.0 * len(delta)
        score += 3.0 * not_found_cnt
        if mean_conf is not None:
            score += 6.0 * (1.0 - max(0.0, min(1.0, mean_conf)))

        # a readable snippet: try to use the (non-normalized) GT value for context when possible
        gt_pairs = _pairs_from_std_record(gt)
        chosen_span = None
        chosen_k, chosen_v = None, None
        if gt_pairs:
            # pick the first delta pair and map back to original strings (best-effort)
            d0k, d0v = delta_list[0]
            for k, v in gt_pairs:
                if _normalize_for_match(k) == d0k and _normalize_for_match(v) == d0v:
                    chosen_k, chosen_v = k, v
                    break
        if chosen_v:
            chosen_span = _context_around(ocr_text, chosen_v)

        candidates.append(
            {
                "id": sid,
                "report_title": report_title,
                "score": score,
                "delta_pairs": [
                    {"key": k, "value": v}
                    for (k, v) in sorted(list(delta))
                ],
                "conf": {
                    "mean": conf.mean_conf,
                    "min": conf.min_conf,
                    "source": conf.source,
                },
                "ocr_text_preview": _shorten(ocr_text, 320),
                "context_preview": chosen_span,
                "source_fields": {
                    "has_source": bool(source_item),
                    "has_ocr_raw": bool(isinstance((source_item or {}).get("ocr_raw"), dict)),
                    "has_noise_values": "noise_values" in (source_item or {}),
                    "has_noise_values_per_word": "noise_values_per_word" in (source_item or {}),
                    "has_transferred_annotations": bool((source_item or {}).get("transferred_annotations")),
                },
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:topk]


def _load_std_jsonl_by_id(path: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for rec in _iter_jsonl(path):
        sid = rec.get("id")
        if sid is None:
            continue
        out[str(sid)] = rec
    return out


def render_markdown(cases: List[dict], ours_name: str, base_name: str) -> str:
    lines: List[str] = []
    lines.append("# Case study candidates (KV-NER)\n")
    lines.append(f"Ours: {ours_name}  ")
    lines.append(f"Baseline: {base_name}\n")

    for i, c in enumerate(cases, 1):
        lines.append(f"## Candidate {i}: id={c['id']}\n")
        lines.append(f"- report_title: {c.get('report_title','')}\n")
        conf = c.get("conf", {})
        lines.append(
            "- ocr_conf: mean={mean} min={min} (from {source})\n".format(
                mean=conf.get("mean"), min=conf.get("min"), source=conf.get("source")
            )
        )
        lines.append(f"- win_pairs (ours correct but baseline wrong): {len(c.get('delta_pairs', []))}\n")
        for p in c.get("delta_pairs", [])[:6]:
            lines.append(f"  - {p['key']} => {p['value']}\n")

        if c.get("context_preview"):
            lines.append("\nOCR local context (best-effort):\n\n")
            lines.append("```\n" + str(c["context_preview"]) + "\n```\n")

        lines.append("\nOCR text preview:\n\n")
        lines.append("```\n" + str(c.get("ocr_text_preview", "")) + "\n```\n")

        lines.append(
            "\nPaper-ready paragraph template (edit wording to match your method claims):\n\n"
        )
        lines.append(
            "> Figure [X] shows a representative OCR failure case. "
            "In this sample, the OCR output contains low-confidence fragments, and the baseline model fails to recover the correct medical entity/value. "
            "In contrast, our noise-aware encoder leverages confidence-related signals and surrounding clinical context to correctly predict the key-value field, demonstrating improved robustness under OCR artifacts.\n"
        )
        lines.append("\n---\n")

    return "".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Mine real KV-NER case study samples from preds/gt JSONL")
    ap.add_argument("--gt", required=True, help="Standardized GT jsonl (*_gt.jsonl)")
    ap.add_argument("--pred_ours", required=True, help="Standardized preds jsonl for our model (*_preds.jsonl)")
    ap.add_argument("--pred_base", required=True, help="Standardized preds jsonl for baseline (*_preds.jsonl)")
    ap.add_argument("--source_test", default=None, help="Original test set json/jsonl (optional, for OCR/conf/noise fields)")
    ap.add_argument("--topk", type=int, default=5, help="How many candidates to output")
    ap.add_argument("--min_delta", type=int, default=1, help="Min number of win pairs required")
    ap.add_argument("--ours_name", default="ours", help="Name string for markdown")
    ap.add_argument("--base_name", default="baseline", help="Name string for markdown")
    ap.add_argument("--out_md", default=None, help="Write markdown summary to this path")
    ap.add_argument("--out_json", default=None, help="Write selected cases to this json path")
    args = ap.parse_args()

    gt_by_id = _load_std_jsonl_by_id(args.gt)
    ours_by_id = _load_std_jsonl_by_id(args.pred_ours)
    base_by_id = _load_std_jsonl_by_id(args.pred_base)

    source_index = None
    if args.source_test:
        raw = _read_any_json(args.source_test)
        items: List[dict] = []
        if isinstance(raw, dict):
            for k in ("data", "items", "results"):
                if k in raw and isinstance(raw[k], list):
                    items = raw[k]
                    break
            if not items:
                items = [raw]
        elif isinstance(raw, list):
            items = [x for x in raw if isinstance(x, dict)]
        else:
            items = []
        source_index = _index_source_records(items)

    cases = mine_cases(
        gt_by_id=gt_by_id,
        ours_by_id=ours_by_id,
        base_by_id=base_by_id,
        source_index=source_index,
        topk=args.topk,
        min_delta=args.min_delta,
    )

    if not cases:
        print("No candidates found. Try lowering --min_delta or ensure preds/gt share same ids.")
        return

    md = render_markdown(cases, ours_name=args.ours_name, base_name=args.base_name)

    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Wrote markdown to: {args.out_md}")
    else:
        print(md)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)
        print(f"Wrote json to: {args.out_json}")


if __name__ == "__main__":
    main()
