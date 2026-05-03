#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 OCR 合并结果中按置信度输出低/高样例，并尽量回查原图路径。

建议输入 fetch_and_merge_baidu_ocr.py 的输出（通常保留 source_image/relative_image_path/data.image）。

示例：
  cd /data/ocean/DAPT
  python3 scripts/print_ocr_conf_examples_with_image.py \
    --input biaozhu_with_ocr/real_train_with_ocr.json \
    --k 2 \
    --image_root /data/ocean/semi_struct/all_type_pic_oss_csv \
    --extra_root /data/ocean/medstruct_benchmark/intermediate/annotated_images_for_ocr \
    --extra_root2 /data/ocean/medstruct_benchmark/ocr_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


Candidate = Tuple[float, str, str, str, str]
# (confidence, record_id, word_text, ocr_text_preview, source)


def _load_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层必须是 JSON 数组")
    return data


def _extract_rel_path(image_field: str) -> str:
    if not image_field:
        return ""
    if "?d=" in image_field:
        return image_field.split("?d=", 1)[1].lstrip("/")
    return image_field.lstrip("/")


def _tail_after_u_segment(rel_path: str) -> Optional[str]:
    parts = [p for p in rel_path.split("/") if p]
    for i, p in enumerate(parts):
        if p.startswith("U"):
            return "/".join(parts[i:])
    return None


def _rel_candidates(rel_path: str, root: Path) -> List[str]:
    cands: List[str] = []
    base = rel_path.lstrip("/")
    if base:
        cands.append(base)

    long_prefix = "semi_struct/all_type_pic_oss_csv/"
    if base.startswith(long_prefix):
        cands.append(base[len(long_prefix):])

    root_name = root.name
    if base.startswith(root_name + "/"):
        cands.append(base[len(root_name) + 1 :])

    more: List[str] = []
    for x in list(cands):
        parts = [p for p in x.split("/") if p]
        if parts and parts[0].isdigit():
            more.append("/".join(parts[1:]))
    cands.extend(more)

    seen = set()
    uniq = []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _resolve_image_path(item: Dict[str, Any], roots: List[Path]) -> str:
    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    record_id = str(item.get("record_id") or item.get("id") or "")

    # 与 fetch_and_merge_baidu_ocr.py 一致：优先 relative_image_path -> record_id.jpg -> source_image -> ocr_raw.source_image -> data.image
    image_field: Optional[str] = None
    if item.get("relative_image_path"):
        image_field = str(item["relative_image_path"])
    if not image_field and record_id:
        image_field = f"{record_id}.jpg"
    if not image_field and item.get("source_image"):
        image_field = str(item["source_image"])
    if not image_field and isinstance(item.get("ocr_raw"), dict) and item["ocr_raw"].get("source_image"):
        image_field = str(item["ocr_raw"]["source_image"])
    if not image_field and isinstance(data, dict) and data.get("image"):
        image_field = str(data["image"])

    rel = _extract_rel_path(str(image_field) if image_field else "")
    if not rel:
        return "(无图片线索字段)"

    for root in roots:
        for cand in _rel_candidates(rel, root):
            p = root / cand
            if p.exists():
                return str(p)
            alt = _tail_after_u_segment(cand)
            if alt:
                p2 = root / alt
                if p2.exists():
                    return str(p2)
    return f"(未找到文件) rel={rel}"


def _iter_candidates(item: Dict[str, Any], preview_len: int) -> Iterable[Candidate]:
    data = item.get("data") if isinstance(item.get("data"), dict) else item
    ocr_raw = data.get("ocr_raw") or item.get("ocr_raw") or {}
    if not isinstance(ocr_raw, dict):
        ocr_raw = {}
    words_result = ocr_raw.get("words_result") or []
    if not isinstance(words_result, list):
        words_result = []

    record_id = str(item.get("record_id") or item.get("id") or "")
    ocr_text = str(data.get("ocr_text") or item.get("ocr_text") or "")
    preview = ocr_text[:preview_len]

    for w in words_result:
        if not isinstance(w, dict):
            continue
        word_text = str(w.get("words", ""))
        pr = w.get("probability")
        if isinstance(pr, (int, float)):
            yield (float(pr), record_id, word_text, preview, "word.probability")
        elif isinstance(pr, dict):
            avg = pr.get("average")
            if isinstance(avg, (int, float)):
                yield (float(avg), record_id, word_text, preview, "word.probability.average")

        chars = w.get("chars", [])
        if isinstance(chars, list):
            for ch in chars:
                if isinstance(ch, dict) and isinstance(ch.get("probability"), (int, float)):
                    yield (float(ch["probability"]), record_id, word_text, preview, "char.probability")


def _pick_examples(items: List[Dict[str, Any]], k: int, preview_len: int) -> Tuple[List[Candidate], List[Candidate]]:
    rows: List[Candidate] = []
    for it in items:
        rows.extend(_iter_candidates(it, preview_len))
    if not rows:
        return [], []
    rows.sort(key=lambda x: x[0])
    k = max(1, min(k, len(rows)))
    return rows[:k], list(reversed(rows[-k:]))


def _index_by_record_id(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for it in items:
        rid = str(it.get("record_id") or it.get("id") or "")
        if rid and rid not in idx:
            idx[rid] = it
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description="输出 OCR 置信度低/高样例并回查原图路径")
    ap.add_argument("--input", required=True, help="OCR 合并 JSON（推荐 biaozhu_with_ocr/*.json）")
    ap.add_argument("--k", type=int, default=2, help="低/高各输出条数，默认 2")
    ap.add_argument("--preview_len", type=int, default=160, help="ocr_text 预览长度")
    ap.add_argument("--image_root", default="", help="主图片根目录")
    ap.add_argument("--extra_root", default="", help="备用图片根目录")
    ap.add_argument("--extra_root2", default="", help="备用图片根目录2")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    items = _load_json(input_path)
    lows, highs = _pick_examples(items, k=args.k, preview_len=args.preview_len)
    if not lows and not highs:
        print("未找到任何可解析的置信度字段。")
        return 0

    rid_index = _index_by_record_id(items)
    roots: List[Path] = []
    for x in (args.image_root, args.extra_root, args.extra_root2):
        if x:
            roots.append(Path(x))

    print(f"输入文件: {input_path}")
    print(f"总样本数: {len(items)}")
    print(f"输出条数: 低 {len(lows)} / 高 {len(highs)}")
    print("")

    print("=== LOW CONFIDENCE EXAMPLES ===")
    for i, row in enumerate(lows, 1):
        conf, rid, word, preview, src = row
        item = rid_index.get(rid)
        img = _resolve_image_path(item, roots) if item is not None and roots else "(未提供 image_root)"
        print(
            f"[LOW {i}] conf={conf:.4f} | source={src} | record_id={rid}\n"
            f"  word={word!r}\n"
            f"  image={img}\n"
            f"  ocr_text[:{args.preview_len}]={preview!r}"
        )

    print("")
    print("=== HIGH CONFIDENCE EXAMPLES ===")
    for i, row in enumerate(highs, 1):
        conf, rid, word, preview, src = row
        item = rid_index.get(rid)
        img = _resolve_image_path(item, roots) if item is not None and roots else "(未提供 image_root)"
        print(
            f"[HIGH {i}] conf={conf:.4f} | source={src} | record_id={rid}\n"
            f"  word={word!r}\n"
            f"  image={img}\n"
            f"  ocr_text[:{args.preview_len}]={preview!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

