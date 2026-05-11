#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 ocr_raw 计算 7 维噪声特征并写回标注 JSON。

- 目标字段：conf_avg, conf_min, conf_var_log, conf_gap, punct_err_ratio, char_break_ratio, align_score
- 来源：ocr_raw.words_result / chars 的 probability、location、paragraphs_result
- 输出：
    item["noise_values"] = [7 floats, 文档级聚合]
    item["noise_values_per_word"] = [[7 floats] * num_words] （便于后续对齐 token 时使用）

用法示例：
    python compute_noise_from_ocr.py \
      --inputs biaozhu_with_ocr/merged_*.json \
      --output_dir biaozhu_with_ocr_noise

说明：
- 只读写本地 JSON 列表文件（LabelStudio 导出的结构）。
- 如果某条没有 ocr_raw 或 words_result 为空，则跳过并不写 noise_values。
- align_score 依赖 paragraphs_result 的分段；若缺失则按整页平均 top 计算。
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path
from statistics import mean, variance
from typing import Dict, List, Optional, Tuple

FEATURES = [
    "conf_avg",
    "conf_min",
    "conf_var_log",
    "conf_gap",
    "punct_err_ratio",
    "char_break_ratio",
    "align_score",
]


def safe_mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(mean(vals))


def safe_min(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(min(vals))


def safe_var(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0
    try:
        return float(variance(vals))
    except Exception:
        return 0.0


def is_bad_punct(ch: str) -> bool:
    """非中英数字则视为标点/噪声，支持多字符字符串。"""
    if not ch:
        return False
    if len(ch) > 1:
        return any(is_bad_punct(c) for c in ch)
    if ch.isalnum():
        return False
    code = ord(ch)
    if 0x4E00 <= code <= 0x9FFF:  # 中日韩表意文字
        return False
    return True


def compute_word_features(word_obj: Dict, para_mean_top: Optional[float], page_h: float) -> Optional[Tuple[List[float], List[float]]]:
    """
    计算单个 words_result 的特征。

    返回： (word_level_feats, char_probs)
    word_level_feats: [conf_avg, conf_min, conf_var_log, conf_gap, punct_err_ratio, char_break_ratio, align_score]
    char_probs: 当前行所有 char 的概率列表
    """
    # 概率采集
    probs = []
    if isinstance(word_obj.get("probability"), (int, float)):
        probs.append(float(word_obj["probability"]))
    chars = word_obj.get("chars") or []
    char_probs = []
    for ch in chars:
        if isinstance(ch, dict) and isinstance(ch.get("probability"), (int, float)):
            char_probs.append(float(ch["probability"]))
    if char_probs:
        probs.extend(char_probs)
    if not probs:
        return None

    conf_avg = mean(probs)
    conf_min = min(probs)
    v = safe_var(probs)
    conf_var_log = math.log10(v + 1e-12)
    conf_gap = conf_avg - conf_min

    # punct_err_ratio: 非中英数字的占比（优先用 chars，否则用 words 文本）
    text_fields = [c.get("char", "") for c in chars if isinstance(c, dict) and c.get("char")]
    if not text_fields and isinstance(word_obj.get("words"), str):
        text_fields = [word_obj["words"]]
    flat_chars = list("".join(text_fields))
    bad = sum(1 for ch in flat_chars if is_bad_punct(ch))
    punct_err_ratio = bad / max(len(flat_chars), 1)

    # char_break_ratio: 行字符数 / 行宽
    width = None
    loc = word_obj.get("location") or {}
    if isinstance(loc, dict) and isinstance(loc.get("width"), (int, float)):
        width = float(loc["width"])
    char_cnt = len(flat_chars)
    char_break_ratio = char_cnt / max(width if width else char_cnt, 1.0)

    # align_score: 行 top 与所在段落平均 top 的偏差（归一到页高）
    top = None
    if isinstance(loc, dict) and isinstance(loc.get("top"), (int, float)):
        top = float(loc["top"])
    align_score = 0.0
    if top is not None and page_h > 0:
        base = para_mean_top if para_mean_top is not None else top
        align_score = abs(top - base) / page_h

    return [conf_avg, conf_min, conf_var_log, conf_gap, punct_err_ratio, char_break_ratio, align_score], char_probs


def compute_paragraph_means(words_result: List[Dict], paragraphs_result: Optional[List[Dict]]) -> List[Optional[float]]:
    """为每个 word 计算所在段落的平均 top。"""
    if not paragraphs_result:
        return [None] * len(words_result)
    para_mean = [None] * len(paragraphs_result)
    # 先算每个段落的平均 top
    for idx, para in enumerate(paragraphs_result):
        idxs = para.get("words_result_idx") or []
        tops = []
        for i in idxs:
            if 0 <= i < len(words_result):
                loc = words_result[i].get("location") or {}
                if isinstance(loc, dict) and isinstance(loc.get("top"), (int, float)):
                    tops.append(float(loc["top"]))
        para_mean[idx] = mean(tops) if tops else None
    # 为每个 word 取对应段落平均 top
    word_para_mean = [None] * len(words_result)
    for idx, para in enumerate(paragraphs_result):
        idxs = para.get("words_result_idx") or []
        for i in idxs:
            if 0 <= i < len(words_result):
                word_para_mean[i] = para_mean[idx]
    return word_para_mean


def process_item(item: Dict) -> bool:
    ocr = item.get("ocr_raw") or {}
    words = ocr.get("words_result") or []
    if not isinstance(words, list) or not words:
        return False

    # 页面高度：max(top+height)
    page_h = 0.0
    for w in words:
        loc = w.get("location") or {}
        if isinstance(loc, dict) and isinstance(loc.get("top"), (int, float)) and isinstance(loc.get("height"), (int, float)):
            page_h = max(page_h, float(loc["top"]) + float(loc["height"]))
    para_means = compute_paragraph_means(words, ocr.get("paragraphs_result"))

    per_word_feats: List[List[float]] = []
    all_probs: List[float] = []
    for idx, w in enumerate(words):
        feats = compute_word_features(w, para_means[idx] if idx < len(para_means) else None, page_h)
        if feats is None:
            continue
        feat_vec, char_probs = feats
        per_word_feats.append(feat_vec)
        all_probs.extend(char_probs)

    if not per_word_feats:
        return False

    # 文档级聚合
    def col(i):
        vals = [v[i] for v in per_word_feats if v is not None]
        return vals

    conf_avg = safe_mean([v[0] for v in per_word_feats])
    conf_min = safe_min([v[1] for v in per_word_feats])
    conf_var_log = safe_mean([v[2] for v in per_word_feats])
    conf_gap = None
    if conf_avg is not None and conf_min is not None:
        conf_gap = conf_avg - conf_min
    punct_err_ratio = safe_mean([v[4] for v in per_word_feats])
    char_break_ratio = safe_mean([v[5] for v in per_word_feats])
    align_score = safe_mean([v[6] for v in per_word_feats])

    doc_vec = [
        conf_avg if conf_avg is not None else 0.0,
        conf_min if conf_min is not None else 0.0,
        conf_var_log if conf_var_log is not None else 0.0,
        conf_gap if conf_gap is not None else 0.0,
        punct_err_ratio if punct_err_ratio is not None else 0.0,
        char_break_ratio if char_break_ratio is not None else 0.0,
        align_score if align_score is not None else 0.0,
    ]

    item["noise_values"] = doc_vec
    item["noise_values_per_word"] = per_word_feats
    return True


def process_file(path: Path, output_dir: Path) -> Tuple[int, int]:
    data = json.load(path.open("r", encoding="utf-8"))
    ok = 0
    for item in data:
        if process_item(item):
            ok += 1
    out_path = output_dir / path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(data, out_path.open("w", encoding="utf-8"), ensure_ascii=False)
    return len(data), ok


def main():
    ap = argparse.ArgumentParser(description="从 ocr_raw 计算 7 维噪声特征并写回 JSON")
    ap.add_argument("--inputs", nargs="+", required=True, help="输入 JSON 文件或 glob，如 biaozhu_with_ocr/merged_*.json")
    ap.add_argument("--output_dir", required=True, type=Path, help="输出目录")
    args = ap.parse_args()

    files: List[Path] = []
    for pat in args.inputs:
        if any(ch in pat for ch in "*?["):
            files.extend([Path(p) for p in glob.glob(pat)])
        else:
            files.append(Path(pat))
    files = sorted({p.resolve() for p in files})

    total_items = 0
    total_ok = 0
    for fp in files:
        if not fp.exists():
            print(f"[WARN] 文件不存在，跳过: {fp}")
            continue
        n, ok = process_file(fp, args.output_dir)
        total_items += n
        total_ok += ok
        print(f"[DONE] {fp.name}: {ok}/{n} 条写入 noise_values -> {args.output_dir/fp.name}")
    print(f"汇总: {total_ok}/{total_items} 条写入 noise_values")


if __name__ == "__main__":
    main()
