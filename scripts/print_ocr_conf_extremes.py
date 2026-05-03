#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打印 OCR 置信度最低/最高样本（默认各 1 条）。

支持从以下位置提取置信度：
1) words_result[].probability (float)
2) words_result[].probability.average
3) words_result[].chars[].probability

默认按字符级置信度（char.probability）聚合为候选，再全局排序。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


Candidate = Tuple[float, str, str, str, str]
# (confidence, record_id, word_text, ocr_text_preview, source)


def _iter_word_candidates(item: Dict[str, Any], preview_len: int) -> Iterable[Candidate]:
    data = item.get("data", item)
    ocr_raw = data.get("ocr_raw", {}) or {}
    words_result = ocr_raw.get("words_result", []) or []

    record_id = str(item.get("record_id") or item.get("id") or "")
    ocr_text_preview = str(data.get("ocr_text", ""))[:preview_len]

    for w in words_result:
        if not isinstance(w, dict):
            continue
        word_text = str(w.get("words", ""))

        # 1) word.probability
        pr = w.get("probability")
        if isinstance(pr, (int, float)):
            yield (float(pr), record_id, word_text, ocr_text_preview, "word.probability")
        elif isinstance(pr, dict):
            avg = pr.get("average")
            if isinstance(avg, (int, float)):
                yield (float(avg), record_id, word_text, ocr_text_preview, "word.probability.average")

        # 2) chars[].probability
        chars = w.get("chars", [])
        if isinstance(chars, list):
            for ch in chars:
                if not isinstance(ch, dict):
                    continue
                cpr = ch.get("probability")
                if isinstance(cpr, (int, float)):
                    yield (float(cpr), record_id, word_text, ocr_text_preview, "char.probability")


def _load_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层应为 JSON 数组。")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="打印 OCR 置信度最低/最高样本")
    parser.add_argument(
        "--input",
        type=str,
        default="biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json",
        help="输入 JSON（顶层为数组）",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="最低/最高各输出多少条（默认 1）",
    )
    parser.add_argument(
        "--preview_len",
        type=int,
        default=120,
        help="ocr_text 预览长度",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    items = _load_json_array(input_path)
    candidates: List[Candidate] = []
    for item in items:
        candidates.extend(_iter_word_candidates(item, args.preview_len))

    if not candidates:
        print("未找到任何置信度字段（word/char probability）。")
        print("可先运行: python3 scripts/inspect_kvbert_data_formats.py --index 0 -v")
        return 0

    candidates.sort(key=lambda x: x[0])
    k = max(1, min(args.k, len(candidates)))
    lows = candidates[:k]
    highs = candidates[-k:]

    print(f"输入文件: {input_path}")
    print(f"总候选数: {len(candidates)}")
    print(f"输出条数: 低 {k} / 高 {k}")
    print("")

    print("=== LOWEST ===")
    for i, row in enumerate(lows, 1):
        conf, rid, word, preview, src = row
        print(
            f"[LOW {i}] conf={conf:.4f} | source={src} | record_id={rid} | "
            f"word={word!r} | ocr_text[:{args.preview_len}]={preview!r}"
        )

    print("")
    print("=== HIGHEST ===")
    for i, row in enumerate(reversed(highs), 1):
        conf, rid, word, preview, src = row
        print(
            f"[HIGH {i}] conf={conf:.4f} | source={src} | record_id={rid} | "
            f"word={word!r} | ocr_text[:{args.preview_len}]={preview!r}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

