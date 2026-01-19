#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check intersection between OCR JSON (e.g., char_ocr_9297.json) and annotation JSONs.
Usage:
  python check_ocr_annotation_intersection.py \
    --ocr ~/semi_label/ocr_rerun/char_ocr_9297.json \
    --anno_dir /data/ocean/DAPT/biaozhu_data

Outputs:
  - Total OCR items
  - Total annotation items
  - Intersection by record_id (if available)
  - Intersection by relative_image_path (normalized basename)
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Union


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_candidates(obj: dict) -> Tuple[Union[str, None], Union[str, None]]:
    """Extract (record_id, relative_image_path) candidates from a single object."""
    record_id = obj.get("record_id") or obj.get("id") or obj.get("doc_id")
    rip = (
        obj.get("relative_image_path")
        or obj.get("image_path")
        or obj.get("img_path")
        or obj.get("img_name")
        or obj.get("file_path")
        or obj.get("path")
    )
    return record_id, rip


def normalize_path(p: str) -> str:
    """Normalize a path string to compare basenames."""
    p = p.strip()
    # remove leading ./ or / if present
    p = p.lstrip("./")
    return os.path.basename(p)


def load_ocr_items(ocr_json: Path) -> Tuple[Set[str], Set[str]]:
    data = load_json(ocr_json)
    # If wrapped in dict, pick first list-like value
    if isinstance(data, dict):
        for k in ["data", "ocr_list", "items", "results"]:
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
    if not isinstance(data, list):
        raise ValueError("Unexpected OCR json structure, expected list or dict containing list")

    record_ids: Set[str] = set()
    relpaths: Set[str] = set()
    for obj in data:
        if not isinstance(obj, dict):
            continue
        rid, rip = extract_candidates(obj)
        if rid is not None:
            record_ids.add(str(rid))
        if rip:
            relpaths.add(normalize_path(str(rip)))
    return record_ids, relpaths


def load_annotation_items(anno_dir: Path) -> Tuple[int, Set[str], Set[str]]:
    record_ids: Set[str] = set()
    relpaths: Set[str] = set()
    total = 0
    for p in sorted(anno_dir.glob("*.json")):
        data = load_json(p)
        if not isinstance(data, list):
            continue
        for obj in data:
            if not isinstance(obj, dict):
                continue
            total += 1
            rid, rip = extract_candidates(obj)
            if rid is not None:
                record_ids.add(str(rid))
            if rip:
                relpaths.add(normalize_path(str(rip)))
    return total, record_ids, relpaths


def main():
    ap = argparse.ArgumentParser(description="Check OCR vs annotation intersection")
    ap.add_argument("--ocr", required=True, type=Path, help="Path to OCR json (char_ocr_9297.json)")
    ap.add_argument("--anno_dir", required=True, type=Path, help="Directory with annotation json files")
    args = ap.parse_args()

    # OCR
    ocr_record_ids, ocr_relpaths = load_ocr_items(args.ocr)
    # Annotations
    anno_total, anno_record_ids, anno_relpaths = load_annotation_items(args.anno_dir)

    # Intersections
    rid_inter = ocr_record_ids & anno_record_ids if ocr_record_ids and anno_record_ids else set()
    path_inter = ocr_relpaths & anno_relpaths if ocr_relpaths and anno_relpaths else set()

    print("==== Summary ====")
    print(f"OCR total items (counted by list length): {len(ocr_record_ids) or len(ocr_relpaths)}")
    print(f"OCR record_ids collected: {len(ocr_record_ids)}")
    print(f"OCR relpaths collected:   {len(ocr_relpaths)}")
    print(f"Annotation total objects: {anno_total}")
    print(f"Annotation record_ids:    {len(anno_record_ids)}")
    print(f"Annotation relpaths:      {len(anno_relpaths)}")
    print("---- Intersections ----")
    print(f"record_id intersection:   {len(rid_inter)}")
    print(f"relpath intersection:     {len(path_inter)}")

    # If both empty, warn
    if not ocr_record_ids and not ocr_relpaths:
        print("[WARN] OCR file had no usable record_id or relative_image_path fields.")
    if not anno_record_ids and not anno_relpaths:
        print("[WARN] Annotation files had no usable record_id or relative_image_path fields.")


if __name__ == "__main__":
    main()
