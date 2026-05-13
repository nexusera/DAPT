#!/usr/bin/env python3
"""
将 OCR JSON 列表导出为纯文本（按顺序，一行一条），用于构建 OCR-only 数据集。

用法示例：
    python scripts/data/export_ocr_texts.py \
        --ocr_json /home/ocean/semi_label/ocr_rerun/char_ocr_9297.json \
        --output /data/ocean/bpe_workspace/train_ocr_9297.txt

支持：
- json 或 jsonl，内部可嵌套在 {"data": [...]} / {"ocr_list": [...]} / {"items": [...]}。
- 仅处理百度 OCR 风格的 words_result（词级文本），按顺序拼成一行。
"""

import argparse
import json
import os
from typing import Any, List


def load_ocr_list(path: str) -> List[Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"OCR json not found: {path}")
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ["data", "ocr_list", "items"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        return [obj]
    raise ValueError(f"Unsupported OCR JSON format: {path}")


def extract_text(ocr_obj: Any) -> str:
    if isinstance(ocr_obj, dict) and "words_result" in ocr_obj:
        words = []
        for item in ocr_obj.get("words_result", []):
            if isinstance(item, dict) and "words" in item:
                s = item["words"].strip()
                if s:
                    words.append(s)
        return " ".join(words)
    return ""


def main():
    parser = argparse.ArgumentParser(description="Export OCR JSON to plain text (one line per doc)")
    parser.add_argument("--ocr_json", type=str, required=True, help="input OCR json/jsonl path")
    parser.add_argument("--output", type=str, required=True, help="output txt path")
    args = parser.parse_args()

    ocr_list = load_ocr_list(args.ocr_json)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as fout:
        kept = 0
        for obj in ocr_list:
            txt = extract_text(obj)
            if txt.strip():
                fout.write(txt.strip() + "\n")
                kept += 1
    print(f"Done. Exported {kept} lines to {args.output}")


if __name__ == "__main__":
    main()
