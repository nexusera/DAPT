#!/usr/bin/env python3
"""
NoiseFeatureProcessor
- 计算 7 维 OCR 连续特征的分桶边界（一次性 fit），并将连续值映射为离散 ID。
- 特征顺序: [conf_avg, conf_min, conf_var_log, conf_gap, punct_err_ratio, char_break_ratio, align_score]
- 桶数:      [64,       64,       32,          32,       16,               32,               64]
- 规则:
    * 0.0 一律映射到 ID=0 (Anchor Bin)
    * char_break_ratio 截断到 0.25
    * align_score 截断到 3500
    * punct_err_ratio 可用固定阈值，也可用分位数（此处使用分位数，保留 anchor=0）
"""

import json
import os
from typing import Dict, List, Sequence

import numpy as np


FEATURES = [
    "conf_avg",
    "conf_min",
    "conf_var_log",
    "conf_gap",
    "punct_err_ratio",
    "char_break_ratio",
    "align_score",
]

NUM_BINS = {
    "conf_avg": 64,
    "conf_min": 64,
    "conf_var_log": 32,
    "conf_gap": 32,
    "punct_err_ratio": 16,
    "char_break_ratio": 32,
    "align_score": 64,
}

CLIP = {
    "char_break_ratio": 0.25,
    "align_score": 3500.0,
}


def compute_word_noise_vec(
    word_item: dict,
    default_para_top: float = 0.0,
) -> List[float]:
    """
    M13: 单词级 7 维噪声特征计算的唯一权威实现。
    供 compute_noise_from_ocr.py、serving/core/noise_extractor.py 等共用，
    避免三份代码独立维护时产生对齐漂移。

    参数格式（百度 OCR 标准 words_result 条目）：
        word_item = {
            "words": "文字",
            "probability": {"average": 0.99, "min": 0.95, "variance": 0.001},
            "location": {"top": 100, "left": 20, "width": 80, "height": 24},
            "chars": [{"char": "文", "probability": 0.99}, ...]  # 可选
        }

    返回：[conf_avg, conf_min, conf_var_log, conf_gap,
           punct_err_ratio, char_break_ratio, align_score]
    """
    import math as _math

    # ── 概率特征 ──────────────────────────────────────────────────────────────
    probs: List[float] = []

    prob_raw = word_item.get("probability")
    if isinstance(prob_raw, dict):
        p = prob_raw.get("average")
        if isinstance(p, (int, float)):
            probs.append(float(p))
    elif isinstance(prob_raw, (int, float)):
        probs.append(float(prob_raw))

    for ch in (word_item.get("chars") or []):
        if isinstance(ch, dict):
            cp = ch.get("probability")
            if isinstance(cp, (int, float)):
                probs.append(float(cp))

    if not probs:
        avg = mn = var = 0.0
    else:
        avg = sum(probs) / len(probs)
        mn = min(probs)
        var = sum((p - avg) ** 2 for p in probs) / len(probs) if len(probs) >= 2 else 0.0

    var_log = _math.log10(var + 1e-12)
    gap = avg - mn

    # ── 标点异常率 ────────────────────────────────────────────────────────────
    word = str(word_item.get("words") or "")
    bad = sum(1 for c in word if not (("\u4e00" <= c <= "\u9fff") or c.isdigit()))
    punct_ratio = bad / max(1, len(word))

    # ── 断字率（截断到 CLIP） ─────────────────────────────────────────────────
    loc = word_item.get("location") or {}
    width = float(loc.get("width") or 0.0)
    char_break = len(word) / max(1.0, width)
    char_break = min(char_break, CLIP["char_break_ratio"])

    # ── 版面对齐分数（截断到 CLIP） ───────────────────────────────────────────
    top = float(loc.get("top") or 0.0)
    align = min(abs(top - default_para_top), CLIP["align_score"])

    return [avg, mn, var_log, gap, punct_ratio, char_break, align]


def _extract_feature_arrays(ocr_list: Sequence[dict]) -> Dict[str, List[float]]:
    # M13: 此函数用于 fit 分桶边界，词级特征计算已迁至 compute_word_noise_vec()
    out = {k: [] for k in FEATURES}
    for obj in ocr_list:
        if not isinstance(obj, dict) or "words_result" not in obj:
            continue
        paragraphs_result = obj.get("paragraphs_result", []) or []
        # 段落统计（align）
        para_tops = []
        for para in paragraphs_result:
            idxs = para.get("words_result_idx", []) or []
            pts = []
            for wid in idxs:
                if 0 <= wid < len(obj["words_result"]):
                    loc = obj["words_result"][wid].get("location", {}) or {}
                    if "top" in loc:
                        pts.append(float(loc.get("top", 0)))
            if pts:
                para_tops.append(np.mean(pts))
        default_para_top = np.mean(para_tops) if para_tops else 0.0

        for wid, item in enumerate(obj["words_result"]):
            prob = item.get("probability", {}) or {}
            avg = float(prob.get("average", 0.0)) if isinstance(prob.get("average", 0.0), (int, float)) else 0.0
            mn = float(prob.get("min", 0.0)) if isinstance(prob.get("min", 0.0), (int, float)) else 0.0
            var = float(prob.get("variance", 0.0)) if isinstance(prob.get("variance", 0.0), (int, float)) else 0.0
            var_log = np.log10(var + 1e-12)
            gap = avg - mn

            word = item.get("words", "") or ""
            bad = sum(1 for ch in word if not (("\u4e00" <= ch <= "\u9fff") or ch.isdigit()))
            punct_ratio = bad / max(1, len(word))

            loc = item.get("location", {}) or {}
            width = float(loc.get("width", 0.0)) if isinstance(loc.get("width", 0.0), (int, float)) else 0.0
            char_break = len(word) / max(1.0, width)
            if "char_break_ratio" in CLIP:
                char_break = min(char_break, CLIP["char_break_ratio"])

            top = float(loc.get("top", 0.0)) if isinstance(loc.get("top", 0.0), (int, float)) else 0.0
            align = abs(top - default_para_top)
            if "align_score" in CLIP:
                align = min(align, CLIP["align_score"])

            out["conf_avg"].append(avg)
            out["conf_min"].append(mn)
            out["conf_var_log"].append(var_log)
            out["conf_gap"].append(gap)
            out["punct_err_ratio"].append(punct_ratio)
            out["char_break_ratio"].append(char_break)
            out["align_score"].append(align)
    return out


class NoiseFeatureProcessor:
    def __init__(self, bin_edges: Dict[str, List[float]] = None):
        self.bin_edges = bin_edges or {}

    def fit_bins(self, ocr_list: Sequence[dict], quantile_method: str = "linear"):
        """计算各特征的分桶边界（不含 0 anchor，0 单独为 ID 0）"""
        arrays = _extract_feature_arrays(ocr_list)
        edges = {}
        for feat in FEATURES:
            vals = np.array(arrays.get(feat, []), dtype=np.float64)
            if vals.size == 0:
                edges[feat] = []
                continue
            # anchor 0 单独处理
            non_zero = vals[vals > 0]
            if non_zero.size == 0:
                edges[feat] = []
                continue
            # clip
            if feat in CLIP:
                non_zero = np.minimum(non_zero, CLIP[feat])
            nb = NUM_BINS[feat]
            qs = np.linspace(0, 1, nb + 1)[1:]  # nb 分位点
            bounds = np.quantile(non_zero, qs, method=quantile_method)
            # 去重（digitize 需要严格递增）
            uniq = []
            for b in bounds:
                if not uniq or b > uniq[-1]:
                    uniq.append(float(b))
            edges[feat] = uniq
        self.bin_edges = edges
        return edges

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.bin_edges, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            edges = json.load(f)
        return cls(edges)
    def to_id(self, x: float, feat: str) -> int:
        """
        N3: 将连续特征值映射到离散桶 ID，调用方必须了解以下契约：

        契约
        ────
        - **ID=0 为 Anchor Bin**：0.0 值和 NaN/inf 均映射到 0，代表"完美/无噪声"。
        - **正常范围：[1, len(bin_edges[feat])]**：`+1` 偏移确保非零值不会落入 Anchor。
        - **Embedding 表大小须为 NUM_BINS[feat] + 1**：索引 0 是 anchor，
          1..NUM_BINS[feat] 是正常桶，因此 embedding 行数 = NUM_BINS + 1。
          例如 conf_avg: NUM_BINS=64 → embedding shape (65, embed_dim)。
        - **bin_edges 不含 anchor 边界**：`fit_bins` 只用非零分位数，
          不在 edges 里存 0.0，调用方不应在 edges 中手动添加 0.0。
        """
        # 1. 处理绝对锚点（0.0 或异常值）
        if x == 0 or not np.isfinite(x):
            return 0

        # 2. 应用截断策略
        if feat in CLIP:
            x = min(x, CLIP[feat])

        edges = self.bin_edges.get(feat, [])
        if not edges:
            return 0

        # 3. 映射到 ID；+1 偏移跳过 Anchor Bin（ID=0）
        # digitize 返回值 i 满足 edges[i-1] <= x < edges[i]，结果范围 [1, len(edges)]
        return int(np.digitize([x], edges, right=False)[0]) + 1

    
    def map_batch(self, values: List[List[float]]) -> List[List[int]]:
        """values: [seq_len][7 floats] -> ids: [seq_len][7 ints]"""
        ids = []
        for row in values:
            ids.append([self.to_id(v, feat) for v, feat in zip(row, FEATURES)])
        return ids


if __name__ == "__main__":
    # 简单自测
    proc = NoiseFeatureProcessor({"conf_avg": [0.5, 0.9], "align_score": [100, 1000]})
    assert proc.to_id(0.0, "conf_avg") == 0
    assert proc.to_id(0.6, "conf_avg") == 1
    assert proc.to_id(0.95, "conf_avg") == 2
    assert proc.to_id(5000, "align_score") == 2  # clipped
    print("NoiseFeatureProcessor basic test passed.")

