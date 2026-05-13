#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探查 OCR 原始 JSON 中是否包含置信度/质量相关字段。

检查内容：
- 顶层是否已有 conf_avg / conf_min / conf_var_log / conf_gap
- 是否有 punct_err_ratio / char_break_ratio / align_score
- 词级/字符级是否有 probability / prob / score / confidence 字段

用法示例：
    python scripts/analysis/probe_ocr_conf_keys.py \
        --ocr ~/semi_label/ocr_rerun/char_ocr_9297.json \
        --max 500

输出：
- 记录总数
- 顶层候选 key 的覆盖计数
- words_result / chars 中概率类字段的覆盖计数
"""

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

# 需要检查的顶层候选字段
TOP_KEYS = [
    "conf_avg",
    "conf_min",
    "conf_var_log",
    "conf_gap",
    "punct_err_ratio",
    "char_break_ratio",
    "align_score",
]

# 概率/置信度的常见字段名（词级或字符级）
PROB_KEYS = ["probability", "prob", "score", "scores", "confidence", "conf"]


def load_ocr_list(path: str) -> List[Any]:
    """加载 OCR json/jsonl 并返回列表。

    支持：
    - 纯列表
    - 包装在 {"data"}, {"ocr_list"}, {"items"}, {"results"} 之一
    - jsonl 按行一个对象
    """
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
        for key in ["data", "ocr_list", "items", "results"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        return [obj]
    raise ValueError(f"Unsupported OCR JSON format: {path}")


def count_top_keys(objs: Iterable[Dict[str, Any]]) -> Counter:
    """统计顶层候选 key 的出现次数。"""
    cnt = Counter()
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        for k in TOP_KEYS:
            if k in obj:
                cnt[k] += 1
    return cnt


def count_prob_keys_in_words(objs: Iterable[Dict[str, Any]]) -> Tuple[Counter, Counter]:
    """统计 words_result / chars 中概率类字段的出现次数。

    返回：词级计数，字符级计数。
    """
    word_cnt = Counter()
    char_cnt = Counter()
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        words = obj.get("words_result")
        if not isinstance(words, list):
            continue
        for w in words:
            if not isinstance(w, dict):
                continue
            for pk in PROB_KEYS:
                if pk in w:
                    word_cnt[pk] += 1
            chars = w.get("chars")
            if isinstance(chars, list):
                for ch in chars:
                    if not isinstance(ch, dict):
                        continue
                    for pk in PROB_KEYS:
                        if pk in ch:
                            char_cnt[pk] += 1
    return word_cnt, char_cnt


def main():
    parser = argparse.ArgumentParser(description="探查 OCR JSON 是否含置信度/质量字段")
    parser.add_argument("--ocr", required=True, help="OCR json/jsonl 路径")
    parser.add_argument("--max", type=int, default=0, help="最多抽查前 N 条（0 表示全量）")
    args = parser.parse_args()

    ocr_list = load_ocr_list(args.ocr)
    if args.max > 0:
        ocr_list = ocr_list[: args.max]

    total = len(ocr_list)
    top_counts = count_top_keys(ocr_list)
    word_prob, char_prob = count_prob_keys_in_words(ocr_list)

    print("==== OCR 字段探查 ====")
    print(f"样本数: {total}")
    print("-- 顶层候选 key 覆盖 --")
    for k in TOP_KEYS:
        print(f"{k}: {top_counts.get(k, 0)}")
    print("-- words_result 概率类字段覆盖 --")
    for pk in PROB_KEYS:
        print(f"word.{pk}: {word_prob.get(pk, 0)}")
    print("-- chars 概率类字段覆盖 --")
    for pk in PROB_KEYS:
        print(f"char.{pk}: {char_prob.get(pk, 0)}")

    # 简短结论
    if not any(top_counts.values()) and not any(word_prob.values()) and not any(char_prob.values()):
        print("结论：未发现指定的置信度/质量相关字段，需要从上游 OCR 源输出获取。")


if __name__ == "__main__":
    main()
