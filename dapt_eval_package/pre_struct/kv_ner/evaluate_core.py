"""
evaluate_core.py
----------------
H4: evaluate.py 与 evaluate_with_dapt_noise.py 的公共工具函数，
避免两份文件中约 200 行完全相同的代码导致指标漂移。

用法（在各 evaluate 文件顶部）：
    from pre_struct.kv_ner.evaluate_core import (
        set_seed, _read_jsonl, _normalize_text_for_eval, _extract_ground_truth
    )
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch


def set_seed(seed: Optional[int]) -> None:
    """设置随机种子（seed 为 None 时跳过）。"""
    if seed is None:
        return
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件，每行一个 JSON 对象。"""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    results = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _normalize_text_for_eval(s: str) -> str:
    """
    文本归一化：Unicode NFKC、统一连字符、裁剪边界标点。
    evaluate.py 与 evaluate_with_dapt_noise.py 使用相同逻辑。
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2014", "-").replace("\u2013", "-")
    s = s.replace("\u3000", " ")
    s = re.sub(r"^\s+|\s+$", "", s)
    edge_punct = "。，、；:;,:()[]{}<>"
    i = 0
    while i < len(s) and s[i] in edge_punct:
        i += 1
    j = len(s)
    while j > i and s[j - 1] in edge_punct:
        j -= 1
    return s[i:j]


def _extract_ground_truth(item: Dict[str, Any]) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    """
    从 GT item 提取 keys 和 pairs。

    Returns:
        (gt_keys: set of key texts, gt_pairs: set of (key_text, value_text))
    """
    gt_keys: Set[str] = set()
    gt_pairs: Set[Tuple[str, str]] = set()

    if "spans" in item:
        for k, v in item["spans"].items():
            v_text = v.get("text", "") if isinstance(v, dict) else ""
            gt_keys.add(k)
            if v_text:
                gt_pairs.add((k, v_text))
    elif "key_value_pairs" in item:
        for p in item["key_value_pairs"]:
            k_text = p["key"]["text"]
            v_text = p.get("value_text", "")
            gt_keys.add(k_text)
            if v_text:
                gt_pairs.add((k_text, v_text))

    return gt_keys, gt_pairs
