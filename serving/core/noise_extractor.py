# -*- coding: utf-8 -*-
"""
从 OCR words_result 实时计算 7 维噪声特征，并展开为逐字符向量。

7 维特征（顺序固定，与预训练一致）：
  [conf_avg, conf_min, conf_var_log, conf_gap, punct_err_ratio, char_break_ratio, align_score]

完美值（非 OCR 文本）：
  [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# M13: 词级特征计算的权威实现已迁移至 noise_feature_processor.compute_word_noise_vec()。
# 此处 _compute_word_noise 保留为兼容 wrapper，内部委托给共用函数。
try:
    from noise_feature_processor import compute_word_noise_vec as _compute_word_noise_vec
    _USE_COMMON = True
except ImportError:
    _USE_COMMON = False

PERFECT_VALUES: List[float] = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 截断阈值（与 noise_feature_processor.py 一致）
_CLIP_CHAR_BREAK = 0.25
_CLIP_ALIGN = 3500.0


def _compute_word_noise(
    word_item: Dict[str, Any],
    default_para_top: float = 0.0,
) -> List[float]:
    """
    为一个 words_result 条目计算 7 维噪声特征（词级）。

    与训练预处理脚本 compute_noise_from_ocr.py 对齐：
      - 优先使用 chars 字段的逐字符 probability（精细，含字间方差）
      - 其次使用 probability 字段（word 级）
        · dict: {"average":..., "min":..., "variance":...}  （百度 accurate 标准接口）
        · float: 0.99  （baidu_ocr.py 封装的内部接口）
      - 若均缺失，各维度退化为 0
    """
    # 采集概率样本：word 级 + char 级（compute_noise_from_ocr.py 同策略）
    probs: List[float] = []

    prob_raw = word_item.get("probability")
    if isinstance(prob_raw, dict):
        p = prob_raw.get("average")
        if isinstance(p, (int, float)):
            probs.append(float(p))
    elif isinstance(prob_raw, (int, float)):
        probs.append(float(prob_raw))

    # 字符级 probability（disp_chars=true 时 baidu_ocr.py 返回 chars 列表）
    chars = word_item.get("chars") or []
    for ch in chars:
        if not isinstance(ch, dict):
            continue
        cp = ch.get("probability")
        if isinstance(cp, (int, float)):
            probs.append(float(cp))

    if not probs:
        avg = mn = var = 0.0
    else:
        avg = sum(probs) / len(probs)
        mn = min(probs)
        if len(probs) >= 2:
            var = sum((p - avg) ** 2 for p in probs) / len(probs)
        else:
            var = 0.0

    # 置信度相关
    var_log = math.log10(var + 1e-12)
    gap = avg - mn

    # 标点异常率：非中文/数字字符占比
    word = str(word_item.get("words") or "")
    bad = sum(
        1 for ch in word
        if not (("\u4e00" <= ch <= "\u9fff") or ch.isdigit())
    )
    punct_ratio = bad / max(1, len(word))

    # 断字率
    loc = word_item.get("location") or {}
    width = float(loc.get("width") or 0.0)
    char_break = len(word) / max(1.0, width)
    char_break = min(char_break, _CLIP_CHAR_BREAK)

    # 版面对齐分数
    top = float(loc.get("top") or 0.0)
    align = abs(top - default_para_top)
    align = min(align, _CLIP_ALIGN)

    return [avg, mn, var_log, gap, punct_ratio, char_break, align]


# M13: 如果可以导入共用实现，用 compute_word_noise_vec 覆盖本地定义
if _USE_COMMON:
    _compute_word_noise = lambda w, t=0.0: _compute_word_noise_vec(w, t)  # noqa: E731


def compute_char_noise_from_words_result(
    words_result: List[Dict[str, Any]],
    paragraphs_result: Optional[List[Any]] = None,
) -> List[List[float]]:
    """
    将 words_result 展开为逐字符 7 维噪声向量。

    Args:
        words_result: 百度 OCR / PaddleOCR 的逐行识别结果列表。
        paragraphs_result: 段落结构（用于计算 align_score 基准），可为 None。

    Returns:
        List[List[float]]，长度 = sum(len(w["words"]) for w in words_result)。
        每个元素是 7 维 float 列表。
    """
    # 计算段落平均 top（用于 align_score 基准）
    default_para_top = 0.0
    if paragraphs_result:
        para_tops: List[float] = []
        for para in paragraphs_result:
            if not isinstance(para, dict):
                continue
            idxs = para.get("words_result_idx") or []
            pts: List[float] = []
            for wid in idxs:
                if isinstance(wid, int) and 0 <= wid < len(words_result):
                    loc = (words_result[wid].get("location") or {})
                    if "top" in loc:
                        pts.append(float(loc["top"]))
            if pts:
                para_tops.append(sum(pts) / len(pts))
        if para_tops:
            default_para_top = sum(para_tops) / len(para_tops)

    char_noise: List[List[float]] = []
    for item in words_result:
        if not isinstance(item, dict):
            continue
        word = str(item.get("words") or "")
        n_chars = len(word)
        if n_chars == 0:
            continue
        noise_vec = _compute_word_noise(item, default_para_top)
        # 该词的每个字符共享同一噪声向量
        char_noise.extend([noise_vec] * n_chars)

    return char_noise


def build_char_noise(
    ocr_text: str,
    words_result: Optional[List[Dict[str, Any]]] = None,
    paragraphs_result: Optional[List[Any]] = None,
    noise_values: Optional[List[List[float]]] = None,
) -> List[List[float]]:
    """
    按优先级选择噪声来源，返回与 ocr_text 等长的逐字符 7 维噪声。

    优先级：
      1. noise_values（调用方预计算，长度已在请求层校验）
      2. words_result → 实时计算
      3. 无噪声 → 全部填完美值
    """
    # 优先级 1：调用方已提供
    if noise_values and len(noise_values) == len(ocr_text):
        return [list(v) for v in noise_values]

    # 优先级 2：从 words_result 计算
    if words_result:
        char_noise = compute_char_noise_from_words_result(
            [w.model_dump() if hasattr(w, "model_dump") else w for w in words_result],
            paragraphs_result,
        )
        # 对齐到 ocr_text 长度（words 拼接文本可能与传入 ocr_text 有出入）
        if len(char_noise) >= len(ocr_text):
            return char_noise[: len(ocr_text)]
        # 不足则补完美值
        padding = [PERFECT_VALUES[:] for _ in range(len(ocr_text) - len(char_noise))]
        return char_noise + padding

    # 优先级 3：全部完美值
    return [PERFECT_VALUES[:] for _ in range(len(ocr_text))]


def noise_summary(char_noise: List[List[float]]) -> Dict[str, Optional[float]]:
    """
    计算噪声摘要统计（用于响应体）。
    """
    if not char_noise:
        return {"avg_confidence": None, "min_confidence": None, "low_conf_char_ratio": None}

    conf_avgs = [row[0] for row in char_noise]
    avg_conf = sum(conf_avgs) / len(conf_avgs)
    min_conf = min(conf_avgs)
    low_ratio = sum(1 for v in conf_avgs if v < 0.7) / len(conf_avgs)

    return {
        "avg_confidence": round(avg_conf, 4),
        "min_confidence": round(min_conf, 4),
        "low_conf_char_ratio": round(low_ratio, 4),
    }
