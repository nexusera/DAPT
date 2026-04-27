"""
tests/test_noise_core.py
------------------------
H14: 噪声核心模块最小单元测试集。

覆盖范围（评审要求的 4 类）：
  1. NoiseFeatureProcessor 分桶分位数往返（fit → to_id 一致性）
  2. PrecomputedWWMCollator 输出 shape / mask 数量
  3. build_zero_feats 维度检查（C1 回归保护）
  4. _expand_word_noise_to_chars / _broadcast_global_noise 行为验证（H5 回归保护）

运行：
    cd /data/ocean/DAPT
    python -m pytest tests/test_noise_core.py -v
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import List

# 确保 DAPT 根目录在 sys.path 中（支持从任意 cwd 运行）
_DAPT_ROOT = Path(__file__).resolve().parents[1]
if str(_DAPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DAPT_ROOT))

import pytest
import torch


# ─────────────────────────────────────────────────────────────────────────────
# 1. NoiseFeatureProcessor 分桶分位数往返
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseFeatureProcessor:
    """验证 fit_bins → to_id 的一致性与边界行为。"""

    def _make_processor(self):
        from noise_feature_processor import NoiseFeatureProcessor, FEATURES
        # 构造一个简单的双边界 processor（无需真实 OCR 数据）
        edges = {feat: [0.3, 0.7] for feat in FEATURES}
        return NoiseFeatureProcessor(edges), FEATURES

    def test_anchor_bin_zero(self):
        """0.0 值永远映射到 ID=0（anchor bin）。"""
        proc, FEATURES = self._make_processor()
        for feat in FEATURES:
            assert proc.to_id(0.0, feat) == 0, f"feat={feat}: 0.0 should map to anchor 0"

    def test_nan_inf_map_to_zero(self):
        """NaN / Inf 应安全映射到 anchor 0，不抛异常。"""
        import math
        proc, FEATURES = self._make_processor()
        for val in (float("nan"), float("inf"), float("-inf")):
            for feat in FEATURES:
                result = proc.to_id(val, feat)
                assert result == 0, f"feat={feat} val={val}: expected 0, got {result}"

    def test_bin_monotonicity(self):
        """值越大，to_id 结果不减（桶 ID 单调不减）。"""
        proc, _ = self._make_processor()
        feat = "conf_avg"
        ids = [proc.to_id(v, feat) for v in [0.1, 0.3, 0.5, 0.7, 0.9]]
        for a, b in zip(ids, ids[1:]):
            assert a <= b, f"Non-monotonic: {ids}"

    def test_map_batch_shape(self):
        """map_batch 输出形状与输入 seq_len 一致。"""
        from noise_feature_processor import FEATURES, NoiseFeatureProcessor
        proc = NoiseFeatureProcessor({feat: [0.5] for feat in FEATURES})
        seq_len = 10
        values = [[0.5] * len(FEATURES) for _ in range(seq_len)]
        ids = proc.map_batch(values)
        assert len(ids) == seq_len
        assert all(len(row) == len(FEATURES) for row in ids)

    def test_clip_char_break_ratio(self):
        """char_break_ratio 超过 0.25 时应被截断，不影响 to_id 崩溃。"""
        proc, _ = self._make_processor()
        id_clipped = proc.to_id(999.0, "char_break_ratio")
        id_max = proc.to_id(0.25, "char_break_ratio")
        # 截断后两者应相同
        assert id_clipped == id_max


# ─────────────────────────────────────────────────────────────────────────────
# 2. PrecomputedWWMCollator shape / mask 数量
# ─────────────────────────────────────────────────────────────────────────────

class TestPrecomputedWWMCollator:
    """验证全词掩码 collator 的输出形状与 -100 mask 数量。"""

    def _make_tokenizer(self):
        """使用轻量 HuggingFace tokenizer（bert-base-chinese 已缓存则优先）。"""
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained("bert-base-chinese")
        except Exception:
            pytest.skip("bert-base-chinese tokenizer not available offline")

    def test_output_keys(self):
        """collator 输出必须包含 input_ids / attention_mask / labels。"""
        from pretraining_common import PrecomputedWWMCollator
        tok = self._make_tokenizer()
        collator = PrecomputedWWMCollator(tokenizer=tok)
        features = [
            {"input_ids": [101, 2769, 3221, 1092, 102], "word_ids": [None, 0, 1, 2, None]},
        ]
        batch = collator(features)
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

    def test_labels_shape(self):
        """labels 形状与 input_ids 相同。"""
        from pretraining_common import PrecomputedWWMCollator
        tok = self._make_tokenizer()
        collator = PrecomputedWWMCollator(tokenizer=tok)
        features = [
            {"input_ids": [101, 2769, 3221, 102], "word_ids": [None, 0, 1, None]},
            {"input_ids": [101, 1045, 102], "word_ids": [None, 0, None]},
        ]
        batch = collator(features)
        assert batch["input_ids"].shape == batch["labels"].shape

    def test_non_masked_labels_are_minus100(self):
        """未被掩码的 token，labels 应为 -100。"""
        from pretraining_common import PrecomputedWWMCollator
        tok = self._make_tokenizer()
        collator = PrecomputedWWMCollator(tokenizer=tok, mlm_probability=0.15)
        features = [
            {"input_ids": [101, 2769, 3221, 1092, 102], "word_ids": [None, 0, 1, 2, None]},
        ]
        batch = collator(features)
        labels = batch["labels"][0]
        # 至少有一部分 token 是 -100
        assert (labels == -100).any(), "Expected some -100 labels (non-masked)"


# ─────────────────────────────────────────────────────────────────────────────
# 3. build_zero_feats 维度检查（C1 回归保护）
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildZeroFeats:
    """确保 build_zero_feats 的维度等于 len(FEATURES)=7，不再硬编码为 5。"""

    def test_noise_dim_equals_features(self):
        from noise_feature_processor import FEATURES
        from add_noise_features import build_zero_feats
        seq_len = 8
        noise, mask = build_zero_feats(seq_len)
        assert len(noise) == seq_len, "noise 行数必须等于 seq_len"
        assert len(mask) == seq_len, "mask 行数必须等于 seq_len"
        assert all(len(row) == len(FEATURES) for row in noise), (
            f"每行维度应为 {len(FEATURES)}，实际为 {[len(r) for r in noise]}"
        )
        assert all(len(row) == len(FEATURES) for row in mask), (
            f"mask 每行维度应为 {len(FEATURES)}，实际为 {[len(r) for r in mask]}"
        )

    def test_zero_values_and_false_mask(self):
        """noise 全零，mask 全 False。"""
        from add_noise_features import build_zero_feats
        noise, mask = build_zero_feats(3)
        assert all(v == 0.0 for row in noise for v in row)
        assert all(v is False for row in mask for v in row)


# ─────────────────────────────────────────────────────────────────────────────
# 4. _expand_word_noise_to_chars / _broadcast_global_noise（H5 回归保护）
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseHelpers:
    """验证 data_utils 中噪声工具函数的基本行为。"""

    def _get_funcs(self):
        sys.path.insert(0, str(_DAPT_ROOT / "dapt_eval_package" / "pre_struct"))
        from kv_ner.data_utils import _expand_word_noise_to_chars, _broadcast_global_noise
        return _expand_word_noise_to_chars, _broadcast_global_noise

    def test_broadcast_global_noise_repeats(self):
        """7 维单向量广播到 text_len 个 token。"""
        _, broadcast = self._get_funcs()
        vec = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = broadcast(vec, 5)
        assert len(result) == 5
        assert result[0] == vec

    def test_broadcast_returns_original_if_not_global(self):
        """非全局格式（列表的列表）直接返回原值。"""
        _, broadcast = self._get_funcs()
        per_char = [[0.5] * 7 for _ in range(3)]
        result = broadcast(per_char, 3)
        assert result is per_char

    def test_expand_word_to_chars_length(self):
        """按 words_result 中每个词的字符数展开。"""
        expand, _ = self._get_funcs()
        ocr_raw = {
            "words_result": [
                {"words": "你好"},     # 2 chars
                {"words": "世界！"},   # 3 chars
            ]
        }
        noise_per_word = [[0.5] * 7, [0.8] * 7]
        result = expand(ocr_raw, noise_per_word)
        assert result is not None
        assert len(result) == 5  # 2 + 3

    def test_expand_returns_none_on_bad_input(self):
        """格式错误时返回 None，不抛异常。"""
        expand, _ = self._get_funcs()
        assert expand(None, [[0.5] * 7]) is None
        assert expand({}, None) is None
        assert expand("not_a_dict", [[0.5] * 7]) is None
