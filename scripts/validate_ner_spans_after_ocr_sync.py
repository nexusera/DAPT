#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校验 MedStruct 风格 JSON 中，transferred_annotations 与当前 ocr_text 是否对齐。

与 load_labelstudio_export 中 Case C 逻辑一致：
  - 若有 start/end：检查下标与切片文本
  - 若无：用 ocr_text.find(entity_text) 模拟锚定

用法：
  cd /data/ocean/DAPT
  python3 scripts/validate_ner_spans_after_ocr_sync.py \\
      --json biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json

  # 只统计前 200 条（试跑）
  python3 scripts/validate_ner_spans_after_ocr_sync.py --json path/to.json --limit 200

  # 将问题样本 record_id 写入文件
  python3 scripts/validate_ner_spans_after_ocr_sync.py --json path/to.json --bad_ids_out /tmp/bad_ids.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _joined_words(item: Dict[str, Any]) -> str:
    ocr = item.get("ocr_raw") or {}
    wr = ocr.get("words_result") if isinstance(ocr, dict) else None
    if not isinstance(wr, list):
        return ""
    return "".join(str(w.get("words", "")) for w in wr if isinstance(w, dict))


def _normalize_slice(t: str) -> str:
    return (t or "").strip()


def validate_item(item: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """返回 (本条是否通过, 问题说明列表)。"""
    issues: List[str] = []
    text = str(item.get("ocr_text") or "")
    joined = _joined_words(item)
    if joined and text != joined:
        issues.append(f"ocr_text≠join(words): len(text)={len(text)} len(join)={len(joined)}")

    ta = item.get("transferred_annotations")
    if not isinstance(ta, list):
        return (len(issues) == 0, issues)

    for i, ann in enumerate(ta):
        if not isinstance(ann, dict) or not ann.get("labels"):
            continue
        val_text = str(ann.get("text", "") or "")
        s = ann.get("start")
        e = ann.get("end")

        if s is not None and e is not None:
            try:
                si, ei = int(s), int(e)
            except (TypeError, ValueError):
                issues.append(f"ann[{i}] 非法 start/end: {s},{e}")
                continue
            if not (0 <= si < ei <= len(text)):
                issues.append(
                    f"ann[{i}] 下标越界: [{si},{ei}] len(ocr_text)={len(text)} label={ann.get('labels')}"
                )
                continue
            slice_t = _normalize_slice(text[si:ei])
            norm_val = _normalize_slice(val_text)
            if norm_val and slice_t != norm_val:
                issues.append(
                    f"ann[{i}] 切片≠text: slice={slice_t!r} text={norm_val!r} [{si},{ei}]"
                )
        else:
            if not val_text or not text:
                continue
            found = text.find(val_text)
            if found == -1:
                stripped = val_text.strip()
                found = text.find(stripped) if stripped != val_text else -1
            if found == -1:
                issues.append(
                    f"ann[{i}] 无 start/end 且 ocr_text 中找不到实体文本: {val_text[:40]!r}..."
                    if len(val_text) > 40
                    else f"ann[{i}] 无 start/end 且 ocr_text 中找不到实体文本: {val_text!r}"
                )

    return (len(issues) == 0, issues)


def main() -> int:
    ap = argparse.ArgumentParser(description="校验 ocr_text 与 transferred_annotations 对齐")
    ap.add_argument("--json", required=True, type=Path, help="JSON 数组文件路径（相对 cwd 或绝对）")
    ap.add_argument("--dapt_root", type=Path, default=None, help="若 --json 为相对路径且不在 cwd，可指定 DAPT 根目录")
    ap.add_argument("--limit", type=int, default=0, help="最多检查前 N 条，0 表示全部")
    ap.add_argument("--bad_ids_out", type=Path, default=None, help="写出未通过样本的 record_id（每行一个）")
    args = ap.parse_args()

    path = args.json
    if not path.is_file() and args.dapt_root:
        path = args.dapt_root / path
    if not path.is_file():
        print(f"ERROR: 文件不存在: {args.json}", file=sys.stderr)
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print("ERROR: 顶层应为 JSON 数组", file=sys.stderr)
        return 1

    n = len(data)
    end = n if args.limit <= 0 else min(n, args.limit)
    bad = 0
    bad_ids: List[str] = []
    mismatch_join = 0
    examples = 0

    for idx in range(end):
        item = data[idx]
        joined = _joined_words(item)
        text = str(item.get("ocr_text") or "")
        if joined and text != joined:
            mismatch_join += 1
        ok, issues = validate_item(item)
        if not ok:
            bad += 1
            rid = item.get("record_id") or item.get("id") or str(idx)
            bad_ids.append(str(rid))
            if examples < 8:
                print(f"\n--- FAIL idx={idx} record_id={rid} ---")
                for msg in issues[:6]:
                    print(f"  {msg}")
                if len(issues) > 6:
                    print(f"  ... 另有 {len(issues) - 6} 条")
                examples += 1

    print(f"\n{'='*60}")
    print(f"文件: {path}")
    print(f"检查条数: {end} / 总 {n}")
    print(f"transferred_annotations 未通过: {bad} ({100.0 * bad / end:.1f}%)")
    print(f"ocr_text≠join(words)（本脚本单独计数）: {mismatch_join} ({100.0 * mismatch_join / end:.1f}%)")

    if args.bad_ids_out and bad_ids:
        args.bad_ids_out.parent.mkdir(parents=True, exist_ok=True)
        args.bad_ids_out.write_text("\n".join(bad_ids) + "\n", encoding="utf-8")
        print(f"已写入 bad record_id: {args.bad_ids_out}")

    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
