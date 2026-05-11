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
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

from noise_feature_processor import compute_word_noise_vec


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


def _has_probability_signal(word_obj: Dict) -> bool:
    prob = word_obj.get("probability")
    if isinstance(prob, (int, float)):
        return True
    if isinstance(prob, dict) and any(isinstance(prob.get(k), (int, float)) for k in ("average", "min", "variance")):
        return True
    return any(
        isinstance(ch, dict) and isinstance(ch.get("probability"), (int, float))
        for ch in (word_obj.get("chars") or [])
    )


def process_item(item: Dict) -> bool:
    ocr = item.get("ocr_raw") or {}
    words = ocr.get("words_result") or []
    if not isinstance(words, list) or not words:
        return False

    para_means = compute_paragraph_means(words, ocr.get("paragraphs_result"))

    per_word_feats: List[List[float]] = []
    for idx, w in enumerate(words):
        if not _has_probability_signal(w):
            continue
        default_para_top = para_means[idx] if idx < len(para_means) and para_means[idx] is not None else 0.0
        per_word_feats.append(compute_word_noise_vec(w, default_para_top))

    if not per_word_feats:
        return False

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
