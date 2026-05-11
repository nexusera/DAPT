#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 record_id 在标注 JSON 中定位对应原图线索，并可在本地原图目录中检索文件。

示例：
  cd /data/ocean/DAPT
  python3 scripts/find_original_image_by_record_id.py --record_id 34094

  python3 scripts/find_original_image_by_record_id.py \
    --record_id 34094 \
    --input biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json \
    --image_root /data/oss/hxzh-mr/all_type_pic_oss_csv/20251204_5w
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


IMAGE_KEYS = (
    "image",
    "img",
    "img_path",
    "image_path",
    "img_url",
    "ocr_image",
    "file_upload",
    "source",
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _load_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层必须是 JSON 数组。")
    return data


def _get_data_block(item: Dict[str, Any]) -> Dict[str, Any]:
    data = item.get("data")
    return data if isinstance(data, dict) else {}


def _find_item_by_record_id(items: List[Dict[str, Any]], record_id: str) -> Optional[Dict[str, Any]]:
    for it in items:
        rid = str(it.get("record_id") or it.get("id") or "")
        if rid == record_id:
            return it
    return None


def _collect_clues(item: Dict[str, Any]) -> List[str]:
    d = _get_data_block(item)
    clues: List[str] = []
    seen: Set[str] = set()
    for k in IMAGE_KEYS:
        v = d.get(k, item.get(k))
        if isinstance(v, str):
            s = v.strip()
            if s and s not in seen:
                seen.add(s)
                clues.append(s)
    return clues


def _derive_keywords(clues: Iterable[str]) -> Set[str]:
    kws: Set[str] = set()
    for c in clues:
        p = Path(c)
        name = p.name
        stem = p.stem
        if name:
            kws.add(name)
        if stem:
            kws.add(stem)
        # 提取看起来像 U 开头 ID 或长 hash 片段
        for m in re.findall(r"U\d{8,}", c):
            kws.add(m)
        for m in re.findall(r"[0-9a-fA-F]{16,}", c):
            kws.add(m)
    return {k for k in kws if k}


def _search_images(image_root: Path, keywords: Set[str], max_results: int) -> List[Path]:
    if not image_root.is_dir():
        return []
    hits: List[Path] = []
    lowered = {k.lower() for k in keywords}
    for p in image_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        full = str(p).lower()
        if any(k in full for k in lowered):
            hits.append(p)
            if len(hits) >= max_results:
                break
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="按 record_id 查找原图线索/路径")
    parser.add_argument("--record_id", required=True, help="目标 record_id")
    parser.add_argument(
        "--input",
        default="biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json",
        help="输入 JSON 文件（默认训练集）",
    )
    parser.add_argument(
        "--image_root",
        default="",
        help="可选：原图根目录。提供后会尝试检索实际图片文件。",
    )
    parser.add_argument("--max_results", type=int, default=20, help="最多输出检索命中数")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    items = _load_array(input_path)
    item = _find_item_by_record_id(items, str(args.record_id))
    if item is None:
        print(f"未找到 record_id={args.record_id}")
        return 1

    d = _get_data_block(item)
    print(f"命中 record_id={args.record_id}")
    print(f"input={input_path}")
    print("")
    print("=== 关键字段 ===")
    for k in ("record_id", "id", "uid", "task_id"):
        v = item.get(k, d.get(k))
        if v not in (None, "", []):
            print(f"{k}: {v}")

    clues = _collect_clues(item)
    print("")
    print("=== 图片线索字段 ===")
    if clues:
        for i, c in enumerate(clues, 1):
            print(f"[{i}] {c}")
    else:
        print("未发现明显图片字段。")

    kws = _derive_keywords(clues)
    if kws:
        print("")
        print("=== 检索关键词 ===")
        for k in sorted(kws):
            print(k)

    if args.image_root:
        root = Path(args.image_root)
        print("")
        print(f"=== 在 image_root 检索 ===")
        print(f"image_root={root}")
        hits = _search_images(root, kws, max_results=max(1, args.max_results))
        if hits:
            print(f"命中 {len(hits)} 条：")
            for p in hits:
                print(str(p))
        else:
            print("未命中图片文件。请检查 image_root 是否正确，或增大关键词覆盖范围。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

