# -*- coding: utf-8 -*-
"""
模型推理引擎：加载 BertCrfTokenClassifier，封装滑动窗口推理（单请求 + 批量）。

复用逻辑来自：
  - dapt_eval_package/pre_struct/kv_ner/compare_models.py  (predict_ner_sliding_window, _build_noise_features)
  - dapt_eval_package/pre_struct/kv_ner/modeling.py        (BertCrfTokenClassifier)
  - noise_fusion.py                                         (aggregate_token_noise_values, needs_bucket_ids)
  - noise_feature_processor.py                             (NoiseFeatureProcessor.map_batch)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# ── sys.path 注入 ──────────────────────────────────────────────────────────────
# DAPT 根目录（含 noise_fusion.py, noise_feature_processor.py 等）
_DAPT_ROOT = Path(__file__).resolve().parents[2]
# dapt_eval_package 目录（含 pre_struct/ 包）
_EVAL_PKG = _DAPT_ROOT / "dapt_eval_package"

for _p in [str(_DAPT_ROOT), str(_EVAL_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pre_struct.kv_ner.modeling import BertCrfTokenClassifier
from pre_struct.kv_ner.noise_utils import PERFECT_VALUES as _KV_PERFECT_VALUES
from noise_fusion import aggregate_token_noise_values, needs_bucket_ids, uses_continuous_noise
from noise_feature_processor import NoiseFeatureProcessor

# 完美噪声 tensor（懒初始化）的占位符
_PERFECT = list(_KV_PERFECT_VALUES)


# ─────────────────────────────────────────────────────────────────────────────
# WindowData: 单个 tokenized 滑动窗口的所有信息
# ─────────────────────────────────────────────────────────────────────────────
class _WindowData:
    __slots__ = (
        "input_ids", "attention_mask", "token_type_ids",
        "offset_mapping", "noise_tensor", "noise_key",
        "item_idx",
    )

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        offset_mapping: List[List[int]],
        noise_tensor: Optional[torch.Tensor],
        noise_key: str,   # "noise_ids" | "noise_values" | ""
        item_idx: int,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.offset_mapping = offset_mapping
        self.noise_tensor = noise_tensor
        self.noise_key = noise_key
        self.item_idx = item_idx


class ModelEngine:
    """
    推理引擎，服务启动时通过 load() 初始化。

    提供两种调用路径：
      - run()       : 单条请求，直接 GPU 前向（兼容旧接口）
      - run_batch() : 多条请求合并成一次 GPU 批量前向（被 DynamicBatchEngine 调用）
    """

    def __init__(self) -> None:
        self.model: Optional[BertCrfTokenClassifier] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.noise_processor: Optional[NoiseFeatureProcessor] = None
        self.id2label: Dict[int, str] = {}
        self.o_id: int = 0
        self.device: torch.device = torch.device("cpu")
        self.noise_mode: str = "bucket"
        self.max_seq_length: int = 512
        self._ready: bool = False

    # ── 初始化 ────────────────────────────────────────────────────────────────

    def load(
        self,
        model_dir: str,
        noise_bins_path: str,
        device: str = "cuda",
        noise_mode: str = "bucket",
        max_seq_length: int = 512,
        use_torch_compile: bool = False,
    ) -> None:
        """启动时调用一次，加载模型和分桶处理器。"""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if str(self.device) != device:
            logger.warning("CUDA 不可用，降级到 CPU")

        self.max_seq_length = max_seq_length

        logger.info(f"加载分桶配置: {noise_bins_path}")
        self.noise_processor = NoiseFeatureProcessor.load(noise_bins_path)

        logger.info(f"加载模型: {model_dir}")
        self.model = BertCrfTokenClassifier.from_pretrained(
            model_dir,
            map_location=str(self.device),
        )
        self.model.to(self.device)
        self.model.eval()

        # 从加载好的模型取 id2label / o_id（与 checkpoint 完全对应）
        self.id2label = {int(k): v for k, v in self.model.id2label.items()}
        self.o_id = next(
            (i for i, n in self.id2label.items() if n == "O"), 0
        )
        self.noise_mode = str(
            getattr(self.model, "noise_mode", None) or noise_mode
        ).lower()

        # ── torch.compile 加速（PyTorch >= 2.0，可选）──────────────────────
        if use_torch_compile:
            try:
                logger.info("尝试 torch.compile 编译模型（首次推理会有预热延迟）...")
                self.model = torch.compile(self.model, mode="reduce-overhead")  # type: ignore[assignment]
                logger.info("torch.compile 成功")
            except Exception as exc:
                logger.warning(f"torch.compile 失败，将使用 eager 模式: {exc}")

        # tokenizer 可能保存在 model_dir/tokenizer/ 子目录（训练时的保存习惯）
        _tok_dir = Path(model_dir) / "tokenizer"
        tokenizer_dir = str(_tok_dir) if _tok_dir.is_dir() else model_dir
        logger.info(f"加载 Tokenizer: {tokenizer_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)

        self._ready = True
        logger.info("模型引擎初始化完成")

    @property
    def ready(self) -> bool:
        return self._ready

    # ── 单请求推理（兼容路径，DynamicBatchEngine 关闭时使用） ─────────────────

    def run(
        self,
        text: str,
        char_noise: Optional[List[List[float]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """单条请求同步推理，使用 run_batch() 实现以保证代码复用。"""
        if not self._ready:
            raise RuntimeError("模型尚未加载，请先调用 load()")
        results = self.run_batch([{"text": text, "char_noise": char_noise}])
        return results[0]

    # ── 批量推理（核心）──────────────────────────────────────────────────────

    def run_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Tuple[List[Dict[str, Any]], Dict[str, float]]]:
        """
        批量推理：对多条 (text, char_noise) 同时做 GPU 前向。

        Args:
            items: List of {"text": str, "char_noise": Optional[List[List[float]]]}

        Returns:
            与 items 等长的列表，每项为 (entities, timing_dict)
        """
        if not self._ready:
            raise RuntimeError("模型尚未加载，请先调用 load()")
        if not items:
            return []

        # ── 1. Tokenize 所有请求（CPU，可并行） ──────────────────────────────
        t_tok = time.perf_counter()
        all_windows: List[_WindowData] = []
        # per_item_window_count[i] = 该 item 产生的窗口数
        per_item_window_count: List[int] = []

        for item_idx, item in enumerate(items):
            text: str = item.get("text") or ""
            char_noise: Optional[List[List[float]]] = item.get("char_noise")
            count = self._tokenize_item(text, char_noise, item_idx, all_windows)
            per_item_window_count.append(count)

        tok_ms = (time.perf_counter() - t_tok) * 1000

        if not all_windows:
            empty = ([], {"tokenize": tok_ms, "model_forward": 0.0, "crf_decode": 0.0})
            return [empty] * len(items)

        # ── 2. 拼装 GPU batch ────────────────────────────────────────────────
        t_fwd = time.perf_counter()
        kwargs = self._build_batch_kwargs(all_windows)

        with torch.no_grad():
            decoded_batch: List[List[int]] = self.model.predict(**kwargs)

        fwd_ms = (time.perf_counter() - t_fwd) * 1000

        # ── 3. 解码 BIO → 实体，按 item 归并 ────────────────────────────────
        t_crf = time.perf_counter()
        per_item_merged: List[Dict[Tuple, Dict]] = [{} for _ in items]

        attn_mask_cpu = kwargs["attention_mask"].cpu()
        seq_len = attn_mask_cpu.shape[1]

        for win_idx, win in enumerate(all_windows):
            raw_seq = list(decoded_batch[win_idx])
            if len(raw_seq) < seq_len:
                raw_seq += [self.o_id] * (seq_len - len(raw_seq))
            mask_list = [bool(v) for v in attn_mask_cpu[win_idx].tolist()]
            text = items[win.item_idx].get("text") or ""
            batch_entities = self._entity_records(raw_seq, mask_list, win.offset_mapping, text)
            merged = per_item_merged[win.item_idx]
            for ent in batch_entities:
                key = (ent["type"], ent["start"], ent["end"])
                if key not in merged:
                    merged[key] = ent

        crf_ms = (time.perf_counter() - t_crf) * 1000

        # ── 4. 组装结果 ───────────────────────────────────────────────────────
        timing = {
            "tokenize": round(tok_ms, 2),
            "model_forward": round(fwd_ms, 2),
            "crf_decode": round(crf_ms, 2),
        }
        results: List[Tuple[List[Dict], Dict[str, float]]] = []
        for i, item in enumerate(items):
            text = item.get("text") or ""
            if not text:
                results.append(([], timing))
                continue
            entities = sorted(per_item_merged[i].values(), key=lambda e: (e["start"], e["end"]))
            results.append((list(entities), timing))

        return results

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _tokenize_item(
        self,
        text: str,
        char_noise: Optional[List[List[float]]],
        item_idx: int,
        out_windows: List[_WindowData],
    ) -> int:
        """对单条文本 tokenize，产生若干 _WindowData 追加到 out_windows，返回窗口数。"""
        if not text:
            return 0

        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            stride=128,
            padding="max_length",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        num_windows = len(encoding["input_ids"])

        for w in range(num_windows):
            offset_mapping: List[List[int]] = encoding["offset_mapping"][w].tolist()
            noise_tensor: Optional[torch.Tensor] = None
            noise_key = ""

            if char_noise is not None and self.noise_processor is not None:
                nk = self._build_noise_kwargs(offset_mapping, char_noise)
                if "noise_ids" in nk:
                    noise_tensor = nk["noise_ids"]
                    noise_key = "noise_ids"
                elif "noise_values" in nk:
                    noise_tensor = nk["noise_values"]
                    noise_key = "noise_values"

            out_windows.append(
                _WindowData(
                    input_ids=encoding["input_ids"][w],
                    attention_mask=encoding["attention_mask"][w],
                    token_type_ids=encoding["token_type_ids"][w],
                    offset_mapping=offset_mapping,
                    noise_tensor=noise_tensor,
                    noise_key=noise_key,
                    item_idx=item_idx,
                )
            )
        return num_windows

    def _build_batch_kwargs(self, windows: List[_WindowData]) -> Dict[str, torch.Tensor]:
        """将所有窗口的张量 stack 成 GPU batch 字典。"""
        input_ids = torch.stack([w.input_ids for w in windows]).to(self.device)
        attention_mask = torch.stack([w.attention_mask for w in windows]).to(self.device)
        token_type_ids = torch.stack([w.token_type_ids for w in windows]).to(self.device)

        kwargs: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        # 决定整批使用 noise_ids 还是 noise_values（根据模式统一对齐）
        seq_len = input_ids.shape[1]
        has_noise_ids = any(w.noise_key == "noise_ids" for w in windows)
        has_noise_values = any(w.noise_key == "noise_values" for w in windows)

        if has_noise_ids:
            perfect_ids = torch.zeros(seq_len, 7, dtype=torch.long)
            stacked = torch.stack([
                w.noise_tensor if w.noise_key == "noise_ids" else perfect_ids
                for w in windows
            ])
            kwargs["noise_ids"] = stacked.to(self.device)
        elif has_noise_values:
            perfect_vals = torch.zeros(seq_len, 7, dtype=torch.float32)
            stacked = torch.stack([
                w.noise_tensor if w.noise_key == "noise_values" else perfect_vals
                for w in windows
            ])
            kwargs["noise_values"] = stacked.to(self.device)

        return kwargs

    def _build_noise_kwargs(
        self,
        offset_mapping: List[List[int]],
        char_noise: List[List[float]],
    ) -> Dict[str, torch.Tensor]:
        """字符级噪声 → token 级 → 分桶，返回 {noise_ids | noise_values} tensor。"""
        token_noise = aggregate_token_noise_values(
            offset_mapping,
            char_noise,
            chunk_char_start=0,
            perfect_values=_PERFECT,
        )
        if needs_bucket_ids(self.noise_mode):
            ids = self.noise_processor.map_batch(token_noise)
            return {"noise_ids": torch.tensor(ids, dtype=torch.long)}
        if uses_continuous_noise(self.noise_mode):
            return {"noise_values": torch.tensor(token_noise, dtype=torch.float32)}
        return {}

    def _entity_records(
        self,
        label_ids: List[int],
        mask: List[bool],
        offsets: List[List[int]],
        text: str,
    ) -> List[Dict[str, Any]]:
        """BIO token 标签序列 → 字符级实体列表。"""
        from pre_struct.kv_ner.metrics import char_spans

        spans = char_spans(label_ids, mask, offsets, self.id2label)
        entities: List[Dict[str, Any]] = []
        for ent_type, start, end in spans:
            snippet = text[start:end] if 0 <= start < end <= len(text) else ""
            entities.append(
                {"type": ent_type, "text": snippet.strip(), "start": start, "end": end}
            )
        return entities


# 全局单例
engine = ModelEngine()
