#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

"""EBQA 训练脚本（按 report_index 分组切分，token 级指标 + 可选字符级指标选 best + 动态长度 cap + 软长度正则）。
补充：在训练循环中通过 pre_struct/ebqa/model_ebqa.py 的 EBQAModel 构造底层 HF 模型。
"""

import os
import json
import math
import time
import contextlib
import random
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

# ---- ensure package roots on sys.path for script execution ----
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
_PRE_STRUCT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))  # .../pre_struct
_PKG_ROOT = os.path.abspath(os.path.join(_PRE_STRUCT_ROOT, ".."))  # .../dapt_eval_package
for _p in (_HERE, _PRE_STRUCT_ROOT, _PKG_ROOT, os.getcwd()):
    if _p not in sys.path:
        sys.path.append(_p)

# plotting (headless-safe)
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # allow training to proceed without matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
    IterableDataset,
    get_worker_info,
)

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup,
)

# === 项目数据集与 collator ===
from da_core.dataset import EnhancedQADataset, QACollator
from da_core.utils import _load_jsonl_or_json

# ==== 引入你自定义的模型/解码器（优先绝对，再相对）====
try:
    from pre_struct.ebqa.model_ebqa import EBQAModel, EBQADecoder, NoiseAwareBertForQuestionAnswering  # type: ignore
except Exception:
    from .model_ebqa import EBQAModel, EBQADecoder, NoiseAwareBertForQuestionAnswering  # type: ignore

try:
    from tqdm import tqdm
except Exception:

    def tqdm(x, *args, **kwargs):
        return x


sys.path.append(".")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- 放在 import 后、set_seed 下方 ---
def _is_cuda_device(dev: str) -> bool:
    try:
        return str(dev).lower().startswith("cuda")
    except Exception:
        return False


def _autocast_dtype_for_gpu() -> "torch.dtype":
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if major >= 8:
            return torch.bfloat16
    return torch.float16


import hashlib


def _stable_hash_int(x: str) -> int:
    """稳定哈希 -> 0..2**32-1（跨进程/重启一致）"""
    h = hashlib.md5(x.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(h[:8], 16)


def _is_eval_sample(sample: dict, eval_ratio: float) -> bool:
    """基于 report_index 的稳定哈希切分（不加载全库也能复现）。"""
    ridx = sample.get("report_index")
    if isinstance(ridx, int):
        key = f"ridx:{ridx}"
    else:
        key = str(sample.get("report_id") or sample.get("question_key") or "na")
    v = _stable_hash_int(key) % 10000  # 0..9999
    thr = int(max(0.0, min(1.0, float(eval_ratio))) * 10000)
    return v < thr


def _model_supports_noise(model: Any) -> bool:
    try:
        return "noise_ids" in inspect.signature(model.forward).parameters
    except Exception:
        return False


class JsonlStreamDataset(IterableDataset):
    """预计算 JSONL 的流式数据集（不把样本常驻内存）。"""

    def __init__(self, jsonl_path: str, predicate, description: str = ""):
        super().__init__()
        self.jsonl_path = str(jsonl_path)
        self.predicate = predicate  # callable(sample)->bool
        self.description = description
        self.is_stream = True  # 供外部判断是否流式

    def __iter__(self):
        wi = get_worker_info()
        if wi is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = wi.id, wi.num_workers

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if (idx % num_workers) != worker_id:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except Exception:
                    continue
                try:
                    if self.predicate(sample):
                        yield sample
                except Exception:
                    continue


def _stream_meta(jsonl_path: str) -> dict:
    return {
        "mtime": os.path.getmtime(jsonl_path),
        "size": os.path.getsize(jsonl_path),
    }


def _load_cached_stream_split(jsonl_path: str, eval_ratio: float):
    meta_path = f"{jsonl_path}.split_counts.json"
    if not os.path.isfile(meta_path):
        return None
    try:
        cur_meta = _stream_meta(jsonl_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if abs(float(data.get("eval_ratio", -1.0)) - float(eval_ratio)) > 1e-9:
            return None
        if int(data.get("size", -1)) != int(cur_meta["size"]):
            return None
        if abs(float(data.get("mtime", -1.0)) - float(cur_meta["mtime"])) > 1e-6:
            return None
        return int(data.get("n_train", -1)), int(data.get("n_eval", -1))
    except Exception:
        return None


def _save_cached_stream_split(jsonl_path: str, eval_ratio: float, n_train: int, n_eval: int) -> None:
    meta_path = f"{jsonl_path}.split_counts.json"
    try:
        cur_meta = _stream_meta(jsonl_path)
        payload = {
            "eval_ratio": float(eval_ratio),
            "mtime": float(cur_meta["mtime"]),
            "size": int(cur_meta["size"]),
            "n_train": int(n_train),
            "n_eval": int(n_eval),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass


def _count_stream_split_once(jsonl_path: str, eval_ratio: float) -> tuple[int, int]:
    """单次遍历统计 train/eval 行数（解析 JSON 但不驻留）。"""
    n_train = n_eval = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except Exception:
                continue
            if _is_eval_sample(sample, eval_ratio):
                n_eval += 1
            else:
                n_train += 1
    return n_train, n_eval


def _count_stream_split(jsonl_path: str, eval_ratio: float) -> tuple[int, int, bool]:
    """统计流式数据的 train/eval 数量，带缓存，避免重复长时间扫描。"""
    cached = _load_cached_stream_split(jsonl_path, eval_ratio)
    if cached and cached[0] >= 0 and cached[1] >= 0:
        return cached[0], cached[1], True

    n_train, n_eval = _count_stream_split_once(jsonl_path, eval_ratio)
    _save_cached_stream_split(jsonl_path, eval_ratio, n_train, n_eval)
    return n_train, n_eval, False


def _dataset_has_noise(ds, data_path: Optional[str] = None, sample_limit: int = 50) -> bool:
    """Lightweight probe for noise_ids in dataset or backing jsonl."""
    try:
        if getattr(ds, "is_stream", False):
            path = data_path or getattr(ds, "jsonl_path", None)
            if path and os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        if idx >= sample_limit:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            sample = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(sample, dict) and "noise_ids" in sample:
                            return True
            return False

        if hasattr(ds, "samples") and ds.samples:
            for sample in ds.samples[: min(sample_limit, len(ds.samples))]:
                if isinstance(sample, dict) and "noise_ids" in sample:
                    return True
    except Exception:
        return False
    return False


@dataclass
class TrainConfig:
    # 数据
    data_path: str
    precomputed: bool
    report_struct_path: str
    # 模型/分词器
    model_name_or_path: str
    tokenizer_name_or_path: str
    # 长度（需与样本构建一致）
    max_seq_len: int
    max_tokens_ctx: int
    max_answer_len: int
    # 训练
    output_dir: str
    num_train_epochs: int
    per_device_batch_size: int
    grad_accum_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    # dataloader
    num_workers: int
    pin_memory: bool
    # 评估与保存
    eval_ratio: float
    save_every_epochs: int
    # 早停（按关键指标）
    early_stopping_patience: int
    early_stopping_min_delta: float
    # 可视化/记录
    plot_update_every: int
    metrics_filename: str
    # 其它
    seed: int
    device: str
    allow_tf32: bool
    chunk_mode: str  # 'budget' / 'newline'
    # 训练细节
    label_smoothing: float = 0.0
    null_margin: float = 0.0
    null_margin_weight: float = 0.0
    use_weighted_sampler: bool = False
    #  选 best 指标
    best_metric: str = "token_f1"  # ["token_f1","char_f1"]
    #  负样本下采样
    negative_downsample: float = 1.0  # 保留比例，1.0=全部保留，0.3=保留30%
    char_eval_subset_reports: int = 200
    #  软长度正则（默认很小，打开即可）
    length_penalty_weight: float = 0.05
    length_penalty_cap: int = 256
    length_penalty_margin: float = 4.0
    #  长度合理性加权训练
    length_reasonableness_weight: float = 0.0  # 长度合理性损失权重
    length_reasonableness_scale: float = 2.0   # 不合理样本的损失放大倍数
    #  补丁：短字段训练优化
    short_field_weight: float = 1.2  # 短字段样本的权重倍数（>1.0表示加权）
    short_field_boost: float = 0.2   # 短字段解码时的评分加成
    #  no-answer 判定阈值
    null_threshold: float = 0.0  # null_score - span_score > threshold 时判定为无答案
    #  噪声特征
    use_noise: bool = False
    noise_embed_dim: int = 16


def load_datasets(cfg: TrainConfig):
    """
    - 当 cfg.precomputed==True 且 data_path 以 .jsonl 结尾时，使用流式 IterableDataset：
        * Hash 切分（可复现） -> 分别构建 train/eval 两个流式数据集
    - 否则沿用基于 EnhancedQADataset 的内存数据集。
    """
    data_path = str(cfg.data_path)
    is_streamable = bool(cfg.precomputed and str(data_path).lower().endswith(".jsonl"))

    if is_streamable:
        # 检查前几个样本是否包含智能边界信息
        import json
        ds_train = JsonlStreamDataset(
            jsonl_path=data_path,
            predicate=lambda s: not _is_eval_sample(s, cfg.eval_ratio),
            description="train",
        )
        ds_eval = JsonlStreamDataset(
            jsonl_path=data_path,
            predicate=lambda s: _is_eval_sample(s, cfg.eval_ratio),
            description="eval",
        )
        
        print("[INFO] Stream mode for precomputed JSONL enabled.")
        return ds_train, ds_eval

    # ===== 非流式 =====
    if cfg.precomputed:
        raw_samples = _load_jsonl_or_json(data_path)
        if not raw_samples:
            raise RuntimeError(f"No precomputed samples found in {data_path}")

        ds_full = EnhancedQADataset(
            data_path=data_path,
            tokenizer_name=cfg.tokenizer_name_or_path,
            report_struct_path=cfg.report_struct_path,
            autobuild=False,
            show_progress=True,
            keep_debug_fields=True,  # 保留调试字段
            chunk_mode=cfg.chunk_mode,
            only_title_keys=True,
            use_concurrent_build=True,
            max_workers=None,
            dynamic_answer_length=True,  # 默认启用动态长度
            inference_mode=False,  # 训练模式
        )
        ds_full.samples = raw_samples
    else:
        # 检查是否需要字符级评估（需要调试字段）
        need_char_eval = os.environ.get("EBQA_CHAR_EVAL", "0") == "1"
        
        ds_full = EnhancedQADataset(
            data_path=data_path,
            tokenizer_name=cfg.tokenizer_name_or_path,
            report_struct_path=cfg.report_struct_path,
            max_seq_len=cfg.max_seq_len,
            max_tokens_ctx=cfg.max_tokens_ctx,
            max_answer_len=cfg.max_answer_len,
            autobuild=True,
            show_progress=True,
            keep_debug_fields=need_char_eval,  # 如需字符级评估，必须在构建时就保留调试字段
            chunk_mode=cfg.chunk_mode,
            only_title_keys=True,
            negative_downsample=cfg.negative_downsample,
            use_concurrent_build=True,
            max_workers=None,
            seed=cfg.seed,  # 与训练 seed 对齐，确保可复现
        )

    n_total = len(ds_full)
    if n_total <= 1:
        raise RuntimeError(
            f"No samples loaded from {data_path}. Check precomputed flag and file path."
        )

    # ——按 report_index 分组切分——
    from collections import defaultdict

    grp = defaultdict(list)  # ridx -> [sample_idx...]
    for i, s in enumerate(ds_full.samples):
        ridx = int(s.get("report_index", -1))
        grp[ridx].append(i)

    ridxs = list(grp.keys())
    random.shuffle(ridxs)
    n_eval_reports = max(1, int(len(ridxs) * cfg.eval_ratio))
    eval_ridxs = set(ridxs[-n_eval_reports:])
    train_ridxs = set(ridxs[:-n_eval_reports])

    train_indices = [i for r in train_ridxs for i in grp[r]]
    eval_indices = [i for r in eval_ridxs for i in grp[r]]

    ds_full.samples = [ds_full.samples[i] for i in (train_indices + eval_indices)]

    # 轻量 subset（共享其余属性）
    ds_train = type(ds_full).__new__(type(ds_full))
    ds_eval = type(ds_full).__new__(type(ds_full))
    for d in (ds_train, ds_eval):
        d.__dict__ = dict(ds_full.__dict__)
    ds_train.samples = ds_full.samples[: len(train_indices)]
    ds_eval.samples = ds_full.samples[len(train_indices) :]

    print(
        f"[INFO] Total samples: {n_total} -> train={len(ds_train)}, eval={len(ds_eval)} "
        f"(reports: train={len(train_ridxs)}, eval={len(eval_ridxs)})"
    )
    return ds_train, ds_eval


def _build_weighted_sampler(ds) -> WeightedRandomSampler:
    """按 question_key 反频率加权采样。"""
    from collections import Counter

    keys = [s.get("question_key", "") for s in ds.samples]
    cnt = Counter(keys)
    weights = torch.tensor(
        [1.0 / max(1, cnt.get(k, 1)) for k in keys], dtype=torch.float
    )
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def create_dataloaders(cfg: TrainConfig, ds_train, ds_eval, tokenizer=None):
    # 从 tokenizer 获取真实 pad 值，避免换模型时的 CLS/PAD 混淆
    pad_id = 0
    vocab_size = None
    if tokenizer is not None:
        if hasattr(tokenizer, 'pad_token_id'):
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 兼容“基础 vocab_size + added_tokens”，优先取 len(tokenizer)
        try:
            vocab_size = len(tokenizer)
        except Exception:
            if hasattr(tokenizer, 'vocab_size'):
                vocab_size = tokenizer.vocab_size
    
    collator = QACollator(
        pad_id=pad_id,
        pad_token_type_id=0,
        pad_attention_mask=0,
        keep_debug_fields=True,
        vocab_size=vocab_size,  # 传递词表大小用于兼容性检查
    )
    is_iter_train = isinstance(ds_train, IterableDataset)
    is_iter_eval = isinstance(ds_eval, IterableDataset)

    if is_iter_train and cfg.use_weighted_sampler:
        print("[WARN] Weighted sampler is ignored for streaming dataset.")

    # 仅在 CUDA 下启用 pinned memory
    pin = bool(
        cfg.pin_memory and _is_cuda_device(cfg.device) and torch.cuda.is_available()
    )

    prefetch_kw = {"prefetch_factor": 1} if cfg.num_workers > 0 else {}

    if is_iter_train:
        dl_train = DataLoader(
            ds_train,
            batch_size=cfg.per_device_batch_size,
            num_workers=min(2, cfg.num_workers),
            pin_memory=pin,
            persistent_workers=False,
            collate_fn=collator,
            drop_last=False,
            **prefetch_kw,
        )
    else:
        train_sampler = (
            _build_weighted_sampler(ds_train)
            if cfg.use_weighted_sampler
            else RandomSampler(ds_train)
        )
        dl_train = DataLoader(
            ds_train,
            sampler=train_sampler,
            batch_size=cfg.per_device_batch_size,
            num_workers=cfg.num_workers,
            pin_memory=pin,
            persistent_workers=bool(cfg.num_workers > 0),
            collate_fn=collator,
            drop_last=False,
            **prefetch_kw,
        )

    if is_iter_eval:
        dl_eval = DataLoader(
            ds_eval,
            batch_size=max(1, cfg.per_device_batch_size * 2),
            num_workers=min(2, cfg.num_workers),
            pin_memory=pin,
            persistent_workers=False,
            collate_fn=collator,
            drop_last=False,
            **prefetch_kw,
        )
    else:
        dl_eval = DataLoader(
            ds_eval,
            sampler=SequentialSampler(ds_eval),
            batch_size=max(1, cfg.per_device_batch_size * 2),
            num_workers=cfg.num_workers,
            pin_memory=pin,
            persistent_workers=bool(cfg.num_workers > 0),
            collate_fn=collator,
            drop_last=False,
            **prefetch_kw,
        )
    return dl_train, dl_eval


# =============================
# 用“EBQAModel”构造底层 HF 模型
def build_model(cfg: TrainConfig):
    """通过 EBQAModel 加载底层 BertForQuestionAnswering；训练仍用本脚本的自定义 loop。"""
    if EBQAModel is None:
        print(
            "[WARN] EBQAModel not importable, falling back to AutoConfig+from_pretrained."
        )
        config = AutoConfig.from_pretrained(cfg.model_name_or_path)
        if cfg.use_noise:
            config.use_noise = True
            config.noise_embed_dim = cfg.noise_embed_dim
            try:
                model = NoiseAwareBertForQuestionAnswering.from_pretrained(
                    cfg.model_name_or_path, config=config
                )
            except Exception:
                model = AutoModelForQuestionAnswering.from_pretrained(
                    cfg.model_name_or_path, config=config
                )
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(
                cfg.model_name_or_path, config=config
            )
    else:
        ebqa = EBQAModel(
            model_name_or_path=cfg.model_name_or_path,
            tokenizer_name_or_path=cfg.tokenizer_name_or_path,
            per_device_train_batch_size=cfg.per_device_batch_size,
            per_device_eval_batch_size=max(1, cfg.per_device_batch_size * 2),
            output_dir=str(Path(cfg.output_dir) / "trainer_stub"),
            logging_steps=100,
            save_strategy="no",
            fp16=_is_cuda_device(cfg.device) and torch.cuda.is_available(),
            # ✅ 新增：和数据集保持一致
            max_answer_len=cfg.max_answer_len,
            use_noise=cfg.use_noise,
            noise_embed_dim=cfg.noise_embed_dim,
        )
        model = ebqa.model
        print("[INFO] Model loaded via EBQAModel (BertForQuestionAnswering inside).")

    model.to(cfg.device)

    # 在 CUDA 且允许时开启 TF32
    if _is_cuda_device(cfg.device) and cfg.allow_tf32 and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            print("[INFO] TF32 enabled.")
        except Exception:
            pass
    return model


def _save_history_json(
    history: dict, out_dir: str, filename: str = "metrics_history.json"
):
    try:
        out_path = Path(out_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] failed to write history json: {e!r}")


def _save_plots(history: dict, out_dir: str):
    if plt is None:
        return
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # train loss
        if history.get("train_loss_steps") and history.get("train_loss_values"):
            plt.figure()
            plt.plot(
                history["train_loss_steps"],
                history["train_loss_values"],
                linestyle="-",
                linewidth=1.5,
            )
            plt.xlabel("step")
            plt.ylabel("train_loss")
            plt.title("Train Loss")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(str(Path(out_dir) / "train_loss.png"))
            plt.close()
        # eval loss
        if history.get("eval_steps") and history.get("eval_loss_values"):
            plt.figure()
            plt.plot(
                history["eval_steps"],
                history["eval_loss_values"],
                linestyle="-",
                linewidth=1.5,
            )
            plt.xlabel("step")
            plt.ylabel("eval_loss")
            plt.title("Eval Loss")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(str(Path(out_dir) / "eval_loss.png"))
            plt.close()
        # lr
        if history.get("lr_steps") and history.get("lr_values"):
            plt.figure()
            plt.plot(
                history["lr_steps"],
                history["lr_values"],
                linestyle="-",
                linewidth=1.5,
            )
            plt.xlabel("step")
            plt.ylabel("learning_rate")
            plt.title("Learning Rate")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(str(Path(out_dir) / "lr.png"))
            plt.close()
        # both
        if history.get("train_loss_steps") and history.get("eval_steps"):
            plt.figure()
            plt.plot(
                history["train_loss_steps"],
                history["train_loss_values"],
                label="train_loss",
                linestyle="-",
                linewidth=1.5,
            )
            plt.plot(
                history["eval_steps"],
                history["eval_loss_values"],
                label="eval_loss",
                linestyle="-",
                linewidth=1.5,
            )
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("Loss (train & eval)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(str(Path(out_dir) / "loss_both.png"))
            plt.close()
    except Exception as e:
        print(f"[WARN] failed to save plots: {e!r}")


# =================  损失函数们  =================
def _ce_with_label_smoothing(logits, targets, epsilon=0.0, reduction: str = "mean"):
    """
    Cross-entropy with optional label smoothing.
    reduction: "mean" | "none"
    - "mean": returns a scalar (default)
    - "none": returns per-sample loss (shape [B])
    """
    if epsilon <= 0:
        return F.cross_entropy(logits, targets, reduction=reduction)
    n_class = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)
    onehot = F.one_hot(targets, num_classes=n_class).float()
    smoothed = (1 - epsilon) * onehot + epsilon / n_class
    loss_vec = -(smoothed * logp).sum(dim=-1)
    if reduction == "none":
        return loss_vec
    return loss_vec.mean()


def _null_margin_loss(start_logits, end_logits, s_gold, e_gold, margin=0.5):
    # 仅对 no-answer 样本（label=(0,0)）生效
    mask = (s_gold == 0) & (e_gold == 0)
    if mask.any():
        s = start_logits[mask]  # [N, T]
        e = end_logits[mask]
        s_cls = s[:, 0]
        e_cls = e[:, 0]
        s_max_noncls = s[:, 1:].max(dim=-1).values
        e_max_noncls = e[:, 1:].max(dim=-1).values
        s_loss = (margin - (s_cls - s_max_noncls)).clamp_min(0.0)
        e_loss = (margin - (e_cls - e_max_noncls)).clamp_min(0.0)
        return (s_loss + e_loss).mean()
    return start_logits.new_zeros([])




#  软长度正则：基于“掩码 softmax 的期望跨度”，仅对有答案样本生效
def _length_regularizer(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    s_gold: torch.Tensor,
    e_gold: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    margin: float = 4.0,
    cap: int = 256,
) -> torch.Tensor:
    """
    惩罚“预测的期望长度”超出 gold_len + margin 的部分，并对极端长尾加 cap 上限。
    - 仅对有答案样本（s_gold,e_gold != 0）生效；
    - softmax 前对无效位点做 -inf 屏蔽，只在上下文(token_type_ids==1)上归一化；但允许 CLS 位置0参与。
    """
    s = start_logits.float().clone()
    e = end_logits.float().clone()

    # 有效上下文 token（含 CLS）
    mask_ctx = (attention_mask == 1) & (token_type_ids == 1)
    mask_cls = torch.zeros_like(mask_ctx)
    mask_cls[:, 0] = True
    valid = mask_ctx | mask_cls

    neg_inf = torch.tensor(-1e9, device=s.device, dtype=torch.float32)
    s = s.masked_fill(~valid, neg_inf)
    e = e.masked_fill(~valid, neg_inf)

    p_s = torch.softmax(s, dim=-1)
    p_e = torch.softmax(e, dim=-1)

    T = start_logits.size(-1)
    pos = torch.arange(T, device=s.device, dtype=torch.float32)
    es = (p_s * pos).sum(dim=-1)
    ee = (p_e * pos).sum(dim=-1)
    elen = (ee - es + 1.0).clamp_min(1.0)

    gold_len = (e_gold - s_gold + 1).clamp_min(1).to(elen.dtype)
    has_ans = ~((s_gold == 0) & (e_gold == 0))

    over = (elen - (gold_len + float(margin))).clamp_min(0.0)
    if cap is not None and cap > 0:
        over = torch.minimum(over, torch.tensor(float(cap), device=over.device))

    if has_ans.any():
        return over[has_ans].mean().to(start_logits.dtype)
    return start_logits.new_zeros([])

# ================= 评估 =================
@torch.no_grad()
def _eval_token_metrics(model, dl_eval, device, supports_noise: bool = False) -> Dict[str, float]:
    """
    训练时的评估指标（平衡版：容错±2 token）
    
    返回多个指标：
    - em: 完全匹配率（start和end都完全正确）
    - f1_token: 严格边界F1（边界偏差>2个token则不给分）
    - f1_overlap: 原始overlap F1（仅供参考）
    - boundary_acc: 边界准确率（start和end都在±2 token内）
    
    容错策略：±2 token（约等于±2个中文字符）
    - 既保证边界质量，又不扼杀BERT的语义理解能力
    - 对"胡章红"vs"胡章"这种1字偏差，仍会给分
    """
    model.eval()
    em_n = 0
    f1_strict_sum = 0.0
    f1_overlap_sum = 0.0
    boundary_correct_n = 0
    n = 0
    for batch in dl_eval:
        for k in (
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "start_positions",
            "end_positions",
        ):
            batch[k] = batch[k].to(device, non_blocking=True)
        kwargs = {}
        if supports_noise and ("noise_ids" in batch):
            kwargs["noise_ids"] = batch["noise_ids"].to(device, non_blocking=True)

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            return_dict=True,
            **kwargs,
        )
        start_logits = out.start_logits
        end_logits = out.end_logits
        mask_valid = batch["attention_mask"] == 1
        if "token_type_ids" in batch:
            mask_valid = mask_valid & (batch["token_type_ids"] == 1)
        try:
            mask_valid[:, 0] = True  # 允许 CLS
        except Exception:
            pass
        s_logits = start_logits.clone()
        e_logits = end_logits.clone()
        s_logits[~mask_valid] = -1e9
        e_logits[~mask_valid] = -1e9

        s_pred = s_logits.argmax(dim=-1)
        e_pred = e_logits.argmax(dim=-1)
        
        # 强制 end >= start，避免负跨度导致的 end 位置错误
        e_pred = torch.maximum(e_pred, s_pred)
        
        s_gold = batch["start_positions"]
        e_gold = batch["end_positions"]
        
        # 1. EM: 完全匹配（最严格）
        exact_match = (s_pred == s_gold) & (e_pred == e_gold)
        em_n += torch.sum(exact_match).item()
        
        # 2. 计算边界偏差
        start_diff = torch.abs(s_pred - s_gold)
        end_diff = torch.abs(e_pred - e_gold)
        max_boundary_diff = torch.maximum(start_diff, end_diff)  # 取最大偏差
        
        # 边界准确率：start和end都在±2 token内
        boundary_correct = (start_diff <= 2) & (end_diff <= 2)
        boundary_correct_n += torch.sum(boundary_correct).item()
        
        # 3. Overlap F1（完全基于内容重叠，仅供对比）
        inter = (
            (torch.minimum(e_pred, e_gold) - torch.maximum(s_pred, s_gold) + 1)
            .clamp_min(0)
            .float()
        )
        len_pred = (e_pred - s_pred + 1).clamp_min(1).float()
        len_gold = (e_gold - s_gold + 1).clamp_min(1).float()
        denom = (len_pred + len_gold).clamp_min(1.0)
        f1_overlap = (2.0 * inter / denom).clamp(0.0, 1.0)
        f1_overlap_sum += f1_overlap.sum().item()
        
        # 4. 严格边界F1（主指标）：使用渐进式惩罚
        # 偏差0: 100%分数
        # 偏差1: 95%分数（轻微惩罚）
        # 偏差2: 85%分数（中度惩罚）
        # 偏差>2: 0分（不容忍）
        boundary_penalty = torch.zeros_like(max_boundary_diff, dtype=torch.float)
        boundary_penalty[max_boundary_diff == 0] = 1.0   # 完美
        boundary_penalty[max_boundary_diff == 1] = 0.95  # 轻微偏差
        boundary_penalty[max_boundary_diff == 2] = 0.85  # 中度偏差
        # max_boundary_diff > 2: 保持0
        
        # 应用渐进式惩罚
        f1_strict = f1_overlap * boundary_penalty
        f1_strict_sum += f1_strict.sum().item()
        
        n += s_pred.size(0)

    em = em_n / max(1, n)
    f1_strict = f1_strict_sum / max(1, n)
    f1_overlap = f1_overlap_sum / max(1, n)
    boundary_acc = boundary_correct_n / max(1, n)
    
    return {
        "em": em,                    # 完全匹配率
        "f1_token": f1_strict,       # 严格边界F1（主指标）
        "f1_overlap": f1_overlap,    # Overlap F1（参考）
        "boundary_acc": boundary_acc # 边界准确率
    }


@torch.no_grad()
def _eval_char_metrics_via_decoder(
    hf_model, dl_eval, device, tokenizer,
    subset_lim: int = 200,
    max_answer_len: int = 128,
    short_field_boost: float = 0.2,
    null_threshold: float = 0.0,
    supports_noise: bool = False,
):
    if EBQADecoder is None:
        return None
    decoder = EBQADecoder(tokenizer, max_answer_len=max_answer_len, short_field_boost=short_field_boost)
    em = 0
    f1_sum = 0.0
    n = 0
    seen_reports = set()
    for batch in dl_eval:
        for need in ("offset_mapping","sequence_ids","chunk_char_start","chunk_text"):
            if need not in batch:
                return None
        bs = len(batch["offset_mapping"])
        for k in ("input_ids","attention_mask","token_type_ids"):
            batch[k] = batch[k].to(device, non_blocking=True)
        kwargs = {}
        if supports_noise and ("noise_ids" in batch):
            kwargs["noise_ids"] = batch["noise_ids"].to(device, non_blocking=True)
        out = hf_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            return_dict=True,
            **kwargs,
        )
        s_logits = out.start_logits.detach().cpu().numpy()
        e_logits = out.end_logits.detach().cpu().numpy()

        # 读取可选短字段标记
        is_short_all = batch.get("is_short_field", [False]*bs)

        for i in range(bs):
            ridx = int(batch.get("report_index", [0]*bs)[i]); seen_reports.add(ridx)

            pred = decoder.best_span_in_chunk(
                start_logits=s_logits[i],
                end_logits=e_logits[i],
                offset_mapping=batch["offset_mapping"][i],
                sequence_ids=list(batch["sequence_ids"][i]),
                chunk_text=str(batch["chunk_text"][i]),
                chunk_char_start=int(batch["chunk_char_start"][i]),
                is_short_field=bool(is_short_all[i]),
            )
            
            # 加入标准 no-answer 判定（与推理一致）
            null_score = float(s_logits[i][0] + e_logits[i][0])
            span_score = float(pred["score"])
            if (null_score - span_score) > null_threshold:  # null 分数更高，判定为无答案
                p_s = p_e = -1
            else:
                p_s, p_e = int(pred["start_char"]), int(pred["end_char"])
            
            s_tok = int(batch["start_positions"][i])
            e_tok = int(batch["end_positions"][i])
            if s_tok == 0 and e_tok == 0:
                g_s = g_e = -1
            else:
                off = batch["offset_mapping"][i]
                s_rel = off[s_tok][0]
                e_rel = off[e_tok][1]
                if s_rel is None or e_rel is None:
                    g_s = g_e = -1
                else:
                    g_s = int(batch["chunk_char_start"][i]) + int(s_rel)
                    g_e = int(batch["chunk_char_start"][i]) + int(e_rel)
            if g_s < 0 or g_e < 0:
                em += int(p_s < 0 or p_e < 0)
                f1_sum += 1.0 if (p_s < 0 or p_e < 0) else 0.0
            else:
                em += int(p_s == g_s and p_e == g_e)
                inter = max(0, min(p_e, g_e) - max(p_s, g_s))
                len_p = max(1, p_e - p_s)
                len_g = max(1, g_e - g_s)
                f1_sum += 2.0 * inter / (len_p + len_g)
            n += 1

        if subset_lim and len(seen_reports) >= subset_lim:
            break

    if n == 0:
        return None
    return {"em_char": em / n, "f1_char": f1_sum / n}


def reevaluate_and_select_best(cfg: TrainConfig, dl_eval, tokenizer=None):
    """重新评估所有保存的checkpoint，选择真正的best模型"""
    from transformers import AutoModelForQuestionAnswering
    
    output_path = Path(cfg.output_dir)
    
    # 查找所有checkpoint
    checkpoints = []
    
    # 添加按epoch保存的checkpoints
    for ckpt_dir in output_path.glob("checkpoint-epoch*"):
        if ckpt_dir.is_dir():
            epoch_num = int(ckpt_dir.name.replace("checkpoint-epoch", ""))
            checkpoints.append(("epoch", epoch_num, ckpt_dir))
    
    # 添加final模型
    final_dir = output_path / "final"
    if final_dir.exists() and final_dir.is_dir():
        checkpoints.append(("final", 999, final_dir))
    
    # 添加当前的best模型（作为基准）
    best_dir = output_path / "best"
    if best_dir.exists() and best_dir.is_dir():
        checkpoints.append(("best", -1, best_dir))
    
    if not checkpoints:
        print("[WARN] 未找到任何checkpoint，跳过重新评估")
        return
    
    print(f"\n找到 {len(checkpoints)} 个checkpoints:")
    for ckpt_type, ckpt_id, ckpt_path in checkpoints:
        print(f"  - {ckpt_type} (id={ckpt_id}): {ckpt_path.name}")
    
    # 评估每个checkpoint
    results = []
    
    for ckpt_type, ckpt_id, ckpt_path in checkpoints:
        print(f"\n评估 {ckpt_type} (id={ckpt_id})...")
        
        try:
            # 加载模型
            ckpt_config = AutoConfig.from_pretrained(ckpt_path)
            if getattr(ckpt_config, "use_noise", False) or cfg.use_noise:
                ckpt_config.use_noise = True
                ckpt_config.noise_embed_dim = cfg.noise_embed_dim
                try:
                    ckpt_model = NoiseAwareBertForQuestionAnswering.from_pretrained(
                        ckpt_path, config=ckpt_config
                    )
                except Exception:
                    ckpt_model = AutoModelForQuestionAnswering.from_pretrained(
                        ckpt_path, config=ckpt_config
                    )
            else:
                ckpt_model = AutoModelForQuestionAnswering.from_pretrained(ckpt_path, config=ckpt_config)
            ckpt_model.to(cfg.device)
            ckpt_model.eval()

            ckpt_supports_noise = _model_supports_noise(ckpt_model)
            
            # Token级评估（快速）
            metrics_token = _eval_token_metrics(ckpt_model, dl_eval, cfg.device, supports_noise=ckpt_supports_noise)
            
            # 字符级评估（可选，较慢）
            metrics_char = None
            if os.environ.get("EBQA_CHAR_EVAL", "0") == "1" and EBQADecoder is not None:
                try:
                    metrics_char = _eval_char_metrics_via_decoder(
                        ckpt_model,
                        dl_eval,
                        cfg.device,
                        tokenizer,
                        subset_lim=cfg.char_eval_subset_reports,
                        max_answer_len=cfg.max_answer_len,
                        short_field_boost=cfg.short_field_boost,
                        null_threshold=cfg.null_threshold,
                        supports_noise=ckpt_supports_noise,
                    )
                except Exception as e:
                    print(f"  [WARN] 字符级评估失败: {e!r}")
            
            # 选择关键指标
            if cfg.best_metric.lower() == "char_f1" and metrics_char and "f1_char" in metrics_char:
                key_metric_val = metrics_char["f1_char"]
                key_metric_name = "char_f1"
            else:
                key_metric_val = metrics_token["f1_token"]
                key_metric_name = "f1_token"
            
            results.append({
                "type": ckpt_type,
                "id": ckpt_id,
                "path": ckpt_path,
                "key_metric": key_metric_val,
                "key_metric_name": key_metric_name,
                "metrics_token": metrics_token,
                "metrics_char": metrics_char,
            })
            
            print(f"  F1-token={metrics_token['f1_token']:.4f}, EM={metrics_token['em']:.4f}, boundary-acc={metrics_token['boundary_acc']:.4f}")
            if metrics_char:
                print(f"  F1-char={metrics_char.get('f1_char', 0):.4f}")
            
            # 清理显存
            del ckpt_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  [ERROR] 评估失败: {e!r}")
            continue
    
    if not results:
        print("[WARN] 没有成功评估的checkpoint")
        return
    
    # 选择最佳checkpoint
    best_result = max(results, key=lambda x: x["key_metric"])
    
    print("\n" + "=" * 80)
    print("评估结果汇总:")
    print("=" * 80)
    for res in sorted(results, key=lambda x: x["key_metric"], reverse=True):
        marker = " ⭐ BEST" if res == best_result else ""
        print(f"  {res['type']} (id={res['id']}): {res['key_metric_name']}={res['key_metric']:.4f}{marker}")
    
    # 如果best checkpoint和当前的best目录不同，更新best目录
    current_best_path = output_path / "best"
    if best_result["path"] != current_best_path:
        print(f"\n更新best模型: {best_result['type']} (id={best_result['id']})")
        
        # 删除旧的best目录
        import shutil
        if current_best_path.exists():
            shutil.rmtree(current_best_path)
        
        # 复制新的best
        shutil.copytree(best_result["path"], current_best_path)
        
        print(f"✓ Best模型已更新: {best_result['key_metric_name']}={best_result['key_metric']:.4f}")
    else:
        print(f"\n✓ 当前best模型已是最优: {best_result['key_metric_name']}={best_result['key_metric']:.4f}")
    
    # 保存评估结果
    eval_results_file = output_path / "reevaluation_results.json"
    eval_results_file.write_text(
        json.dumps(
            {
                "best": {
                    "type": best_result["type"],
                    "id": best_result["id"],
                    "key_metric": best_result["key_metric"],
                    "key_metric_name": best_result["key_metric_name"],
                    "metrics_token": best_result["metrics_token"],
                    "metrics_char": best_result["metrics_char"],
                },
                "all_checkpoints": [
                    {
                        "type": r["type"],
                        "id": r["id"],
                        "key_metric": r["key_metric"],
                        "metrics_token": r["metrics_token"],
                        "metrics_char": r["metrics_char"],
                    }
                    for r in sorted(results, key=lambda x: x["key_metric"], reverse=True)
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"✓ 评估结果已保存到: {eval_results_file}")


# ================= 训练循环 =================
def train_loop(cfg: TrainConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    use_amp = _is_cuda_device(cfg.device) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = _autocast_dtype_for_gpu()

    if _is_cuda_device(cfg.device) and torch.cuda.is_available():
        try:
            print("[INFO] CUDA device:", torch.cuda.get_device_name())
        except Exception:
            pass
        print(
            "[INFO] autocast dtype:",
            "bfloat16" if amp_dtype is torch.bfloat16 else "float16",
        )

    # ===== 数据 & DataLoader =====
    ds_train, ds_eval = load_datasets(cfg)

    data_has_noise = _dataset_has_noise(ds_train, cfg.data_path if hasattr(cfg, "data_path") else None)
    if data_has_noise and not cfg.use_noise:
        cfg.use_noise = True
        print("[INFO] Detected noise_ids in dataset; enabling noise-aware model.")
    if cfg.use_noise and not data_has_noise:
        print("[WARN] use_noise=True but dataset lacks noise_ids; continuing without noise inputs.")
    
    # 加载 tokenizer 以获取真实 pad 值（使用 AutoTokenizer 支持各种模型）
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name_or_path, use_fast=True)
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError(
                "Expected a fast tokenizer (is_fast=False). EBQA requires return_offsets_mapping."
            )
        _probe = "肿瘤标志物"
        _pieces = tokenizer.tokenize(_probe)
        if len(_pieces) == 1 and _pieces[0] == tokenizer.unk_token:
            raise RuntimeError(
                "Fast tokenizer appears misconfigured (probe tokenizes to a single [UNK]). "
                "Regenerate tokenizer.json: python DAPT/repair_fast_tokenizer.py --tokenizer_dir <TOKENIZER_DIR>"
            )
        
        # ✅ 兼容性检查：验证数据与模型的词表是否匹配
        if tokenizer:
            try:
                vocab_size = len(tokenizer)
            except Exception:
                vocab_size = getattr(tokenizer, 'vocab_size', None)
            if vocab_size is not None:
                print(f"[INFO] 当前tokenizer词表大小: {vocab_size}")
            
            # 检查数据集中的token ids范围（仅检查非流式数据集）
            if not getattr(ds_train, "is_stream", False) and hasattr(ds_train, 'samples'):
                max_token_id = -1
                check_limit = min(100, len(ds_train.samples))  # 只检查前100个样本
                for i, sample in enumerate(ds_train.samples[:check_limit]):
                    if 'input_ids' in sample:
                        ids = sample['input_ids']
                        if isinstance(ids, (list, tuple)):
                            sample_max = max(ids) if ids else 0
                            max_token_id = max(max_token_id, sample_max)
                
                if max_token_id >= vocab_size:
                    print(f"\n{'='*80}")
                    print(f"[警告] 数据兼容性问题检测！")
                    print(f"  - 数据中的最大token id: {max_token_id}")
                    print(f"  - 当前模型词表大小: {vocab_size}")
                    print(f"  - token id超出范围: 数据可能是用其他tokenizer预处理的")
                    print(f"\n[自动修复] 训练时将自动替换超出范围的token id")
                    print(f"           但建议重新预处理数据以获得最佳效果")
                    print(f"{'='*80}\n")
    except Exception as e:
        print(f"[WARN] 无法加载tokenizer: {e}")
        tokenizer = None
    
    dl_train, dl_eval = create_dataloaders(cfg, ds_train, ds_eval, tokenizer=tokenizer)

    if getattr(ds_train, "is_stream", False):
        n_train, n_eval, cached_counts = _count_stream_split(cfg.data_path, cfg.eval_ratio)
        suffix = " (cached)" if cached_counts else ""
        print(f"[INFO] Stream split counts -> train={n_train}, eval={n_eval}{suffix}")
    else:
        n_train, n_eval = len(ds_train), len(ds_eval)

    steps_per_epoch = math.ceil(
        max(1, n_train)
        / max(1, cfg.per_device_batch_size)
        / max(1, cfg.grad_accum_steps)
    )
    t_total = steps_per_epoch * cfg.num_train_epochs
    warmup_steps = int(t_total * cfg.warmup_ratio)

    # ===== 模型 & 优化器 & 调度器 =====
    model = build_model(cfg)
    supports_noise = _model_supports_noise(model)
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped, lr=cfg.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)

    history = {
        "train_loss_steps": [],
        "train_loss_values": [],
        "eval_steps": [],
        "eval_loss_values": [],
        "eval_em_values": [],           # 新增：完全匹配率
        "eval_f1_strict_values": [],    # 新增：严格边界F1
        "eval_f1_overlap_values": [],   # 新增：Overlap F1
        "eval_boundary_acc_values": [], # 新增：边界准确率
        "lr_steps": [],
        "lr_values": [],
    }
    global_step, accum_loss = 0, 0.0

    best_key_metric = -1.0
    epochs_no_improve = 0

    for epoch in range(1, cfg.num_train_epochs + 1):
        model.train()
        running = 0.0
        micro_steps_since_update = 0
        num_batches = 0

        pbar_train = tqdm(
            total=n_train,
            desc=f"Epoch {epoch}/{cfg.num_train_epochs} [train]",
            dynamic_ncols=True,
            unit="sample",
        )

        optimizer.zero_grad(set_to_none=True)

        for batch in dl_train:
            num_batches += 1
            bs = (
                int(batch["input_ids"].size(0))
                if "input_ids" in batch
                else cfg.per_device_batch_size
            )
            pbar_train.update(bs)

            for k in (
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "start_positions",
                "end_positions",
            ):
                batch[k] = batch[k].to(cfg.device, non_blocking=True)
            # 可选训练权重张量也搬到设备
            for k_opt in ("length_reasonableness", "is_short_field"):
                if k_opt in batch and torch.is_tensor(batch[k_opt]):
                    batch[k_opt] = batch[k_opt].to(cfg.device, non_blocking=True)

            noise_on_device = None
            if supports_noise and ("noise_ids" in batch):
                noise_on_device = batch["noise_ids"].to(cfg.device, non_blocking=True)
            model_kwargs = {"noise_ids": noise_on_device} if noise_on_device is not None else {}

            with (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp
                else contextlib.nullcontext()
            ):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    return_dict=True,
                    **model_kwargs,
                )

                if (
                    (cfg.label_smoothing > 0.0)
                    or (cfg.null_margin > 0.0 and cfg.null_margin_weight > 0.0)
                    or (cfg.length_penalty_weight > 0.0)
                    or (cfg.length_reasonableness_weight > 0.0)
                    or (cfg.short_field_weight != 1.0)  #  补丁：短字段权重
                ):
                    s_logits, e_logits = out.start_logits, out.end_logits
                    # 使用逐样本损失，便于按样本加权
                    s_loss_vec = _ce_with_label_smoothing(
                        s_logits, batch["start_positions"], epsilon=cfg.label_smoothing, reduction="none"
                    )
                    e_loss_vec = _ce_with_label_smoothing(
                        e_logits, batch["end_positions"], epsilon=cfg.label_smoothing, reduction="none"
                    )
                    base_loss_per_sample = s_loss_vec + e_loss_vec  # [B]
                    loss = base_loss_per_sample.mean()
                    base_ce_mean = loss  # 记录基础 CE 均值，便于后续替换为加权版本

                    if cfg.null_margin > 0.0 and cfg.null_margin_weight > 0.0:
                        nm = _null_margin_loss(
                            s_logits,
                            e_logits,
                            batch["start_positions"],
                            batch["end_positions"],
                            margin=cfg.null_margin,
                        )
                        loss = loss + cfg.null_margin_weight * nm

                    #  软长度正则（极小的系数，抑制无意义变长粘连）
                    if cfg.length_penalty_weight > 0.0:
                        lp = _length_regularizer(
                            s_logits,
                            e_logits,
                            batch["start_positions"],
                            batch["end_positions"],
                            attention_mask=batch["attention_mask"],
                            token_type_ids=batch["token_type_ids"],
                            margin=float(cfg.length_penalty_margin),
                            cap=int(cfg.length_penalty_cap),
                        )
                        loss = loss + float(cfg.length_penalty_weight) * lp
                    
                    #  联合权重处理：支持同时使用长度合理性权重和短字段权重
                    need_weighting = (
                        (cfg.length_reasonableness_weight > 0.0 and "length_reasonableness" in batch) or
                        (cfg.short_field_weight != 1.0 and "is_short_field" in batch)
                    )

                    if need_weighting:
                        final_weights = torch.ones_like(base_loss_per_sample)

                        # 应用长度合理性权重
                        if cfg.length_reasonableness_weight > 0.0 and "length_reasonableness" in batch:
                            length_reasonableness = batch["length_reasonableness"].float()
                            length_weights = torch.clamp(
                                cfg.length_reasonableness_scale - (cfg.length_reasonableness_scale - 1.0) * length_reasonableness,
                                1.0, cfg.length_reasonableness_scale
                            )
                            final_weights = final_weights * length_weights

                        # 应用短字段权重
                        if cfg.short_field_weight != 1.0 and "is_short_field" in batch:
                            is_short_field = batch["is_short_field"].float()
                            short_weights = is_short_field * cfg.short_field_weight + (1.0 - is_short_field) * 1.0
                            final_weights = final_weights * short_weights

                        # 应用综合权重（仅作用于基础 CE 部分）
                        weighted_ce = (base_loss_per_sample * final_weights).mean()

                        # 仅替换基础 CE，保留已加的正则/边界/长度项
                        loss = (loss - base_ce_mean) + weighted_ce
                else:
                    loss = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch["token_type_ids"],
                        start_positions=batch["start_positions"],
                        end_positions=batch["end_positions"],
                        return_dict=True,
                        **model_kwargs,
                    ).loss

                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            running += loss.item()
            accum_loss += loss.item()
            micro_steps_since_update += 1

            if micro_steps_since_update == cfg.grad_accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1
                step_loss = accum_loss
                accum_loss = 0.0
                micro_steps_since_update = 0

                history["train_loss_steps"].append(global_step)
                history["train_loss_values"].append(float(step_loss))

                cur_lr = (
                    scheduler.get_last_lr()[0]
                    if scheduler is not None
                    else optimizer.param_groups[0]["lr"]
                )
                history["lr_steps"].append(global_step)
                history["lr_values"].append(float(cur_lr))

                pbar_train.set_postfix(
                    {
                        "avg_loss": f"{running/max(1,num_batches):.4f}",
                        "lr": f"{cur_lr:.2e}",
                    }
                )
                if (cfg.plot_update_every > 0) and (
                    (global_step % cfg.plot_update_every) == 0
                ):
                    _save_plots(history, cfg.output_dir)
                    _save_history_json(history, cfg.output_dir, cfg.metrics_filename)

        # 尾批 flush
        if micro_steps_since_update != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            step_loss = accum_loss
            accum_loss = 0.0
            history["train_loss_steps"].append(global_step)
            history["train_loss_values"].append(float(step_loss))
            cur_lr = (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            )
            history["lr_steps"].append(global_step)
            history["lr_values"].append(float(cur_lr))
            pbar_train.set_postfix(
                {"avg_loss": f"{running/max(1,num_batches):.4f}", "lr": f"{cur_lr:.2e}"}
            )
            if cfg.plot_update_every > 0 and (global_step % cfg.plot_update_every) == 0:
                _save_plots(history, cfg.output_dir)
                _save_history_json(history, cfg.output_dir, cfg.metrics_filename)

        pbar_train.close()

        # ===== eval =====
        model.eval()
        eval_losses = []
        pbar_eval = tqdm(
            total=n_eval,
            desc=f"Epoch {epoch} [eval]",
            dynamic_ncols=True,
            unit="sample",
        )
        with torch.no_grad():
            for batch in dl_eval:
                bs = (
                    int(batch["input_ids"].size(0))
                    if "input_ids" in batch
                    else max(1, cfg.per_device_batch_size * 2)
                )
                pbar_eval.update(bs)
                for k in (
                    "input_ids",
                    "attention_mask",
                    "token_type_ids",
                    "start_positions",
                    "end_positions",
                ):
                    batch[k] = batch[k].to(cfg.device, non_blocking=True)
                noise_on_device = None
                if supports_noise and ("noise_ids" in batch):
                    noise_on_device = batch["noise_ids"].to(cfg.device, non_blocking=True)
                model_kwargs = {"noise_ids": noise_on_device} if noise_on_device is not None else {}
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    start_positions=batch["start_positions"],
                    end_positions=batch["end_positions"],
                    return_dict=True,
                    **model_kwargs,
                )
                eval_losses.append(out.loss.item())
        pbar_eval.close()
        eval_loss = sum(eval_losses) / max(1, len(eval_losses))

        metrics = _eval_token_metrics(model, dl_eval, cfg.device, supports_noise=supports_noise)
        log_msg = (
            f"[INFO] epoch={epoch} eval_loss={eval_loss:.4f} "
            f"EM={metrics['em']:.4f} "
            f"F1-strict={metrics['f1_token']:.4f} "
            f"F1-overlap={metrics['f1_overlap']:.4f} "
            f"boundary-acc={metrics['boundary_acc']:.4f} "
            f"train_avg_loss={running/max(1,num_batches):.4f}"
        )

        key_metric_name = "F1-strict"  # 使用严格边界F1作为主指标
        key_metric_val = metrics["f1_token"]

        # 可选字符级评估
        if (os.environ.get("EBQA_CHAR_EVAL", "0") == "1") and (EBQADecoder is not None):
            try:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name_or_path, use_fast=True)
                if not getattr(tok, "is_fast", False):
                    raise RuntimeError(
                        "Expected a fast tokenizer (is_fast=False). EBQA requires return_offsets_mapping."
                    )
                m_char = _eval_char_metrics_via_decoder(
                    model,
                    dl_eval,
                    cfg.device,
                    tok,
                    subset_lim=cfg.char_eval_subset_reports,
                    max_answer_len=cfg.max_answer_len,
                    short_field_boost=cfg.short_field_boost,
                    null_threshold=cfg.null_threshold,  # 使用配置的 no-answer 判定阈值
                    supports_noise=supports_noise,
                )
                if isinstance(m_char, dict) and ("f1_char" in m_char):
                    if cfg.best_metric.lower() == "char_f1":
                        key_metric_name = "char-F1"
                        key_metric_val = float(m_char["f1_char"])
                    log_msg += f" | char-F1={float(m_char['f1_char']):.4f}"
            except Exception as e:
                log_msg += f" | char-F1=N/A ({e!r})"

        print(log_msg)

        history["eval_steps"].append(global_step)
        history["eval_loss_values"].append(float(eval_loss))
        history["eval_em_values"].append(float(metrics.get('em', 0)))
        history["eval_f1_strict_values"].append(float(metrics.get('f1_token', 0)))
        history["eval_f1_overlap_values"].append(float(metrics.get('f1_overlap', 0)))
        history["eval_boundary_acc_values"].append(float(metrics.get('boundary_acc', 0)))
        _save_plots(history, cfg.output_dir)
        _save_history_json(history, cfg.output_dir, cfg.metrics_filename)

        improved = key_metric_val > (
            best_key_metric + max(0.0, cfg.early_stopping_min_delta)
        )
        if improved:
            best_key_metric = key_metric_val
            epochs_no_improve = 0
            best_dir = Path(cfg.output_dir) / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            (best_dir / "train_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_name_or_path": cfg.tokenizer_name_or_path,
                        "model_name_or_path": cfg.model_name_or_path,
                        "max_answer_len": cfg.max_answer_len,      # 新增
                        "chunk_mode": cfg.chunk_mode,              # 新增
                        "use_noise": cfg.use_noise,
                        "noise_embed_dim": cfg.noise_embed_dim,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            epochs_no_improve += 1

        if (epoch % cfg.save_every_epochs) == 0:
            ckpt_dir = Path(cfg.output_dir) / f"checkpoint-epoch{epoch}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            (ckpt_dir / "train_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_name_or_path": cfg.tokenizer_name_or_path,
                        "model_name_or_path": cfg.model_name_or_path,
                        "max_answer_len": cfg.max_answer_len,      # 新增
                        "chunk_mode": cfg.chunk_mode,              # 新增
                        "use_noise": cfg.use_noise,
                        "noise_embed_dim": cfg.noise_embed_dim,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        if (cfg.early_stopping_patience > 0) and (
            epochs_no_improve >= cfg.early_stopping_patience
        ):
            print(
                f"[EARLY STOP] No {key_metric_name} improvement for {epochs_no_improve} epoch(s). "
                f"Best {key_metric_name}={best_key_metric:.4f}. Stopping at epoch {epoch}."
            )
            break

    final_dir = Path(cfg.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    (final_dir / "train_config.json").write_text(
        json.dumps(
            {
                "tokenizer_name_or_path": cfg.tokenizer_name_or_path,
                "model_name_or_path": cfg.model_name_or_path,
                "max_answer_len": cfg.max_answer_len,      # 新增
                "chunk_mode": cfg.chunk_mode,              # 新增
                "use_noise": cfg.use_noise,
                "noise_embed_dim": cfg.noise_embed_dim,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _save_plots(history, cfg.output_dir)
    _save_history_json(history, cfg.output_dir, cfg.metrics_filename)
    
    # ===== 重新评估所有checkpoint，选择真正的best =====
    print("\n" + "=" * 80)
    print("重新评估所有checkpoints，选择真正的best模型")
    print("=" * 80)
    
    reevaluate_and_select_best(cfg, dl_eval, tokenizer)
    
    print(
        f"\n[OK] Training finished. Artifacts in: {cfg.output_dir} "
        f"(initial best {'char-F1' if cfg.best_metric.lower()=='char_f1' else 'token-F1'}={best_key_metric:.4f})"
    )


# ======把训练好的 HF 模型“注入” EBQAModel，便于后续 predict ======
def build_ebqa_predictor(cfg: TrainConfig, trained_model) -> Optional[Any]:
    if EBQAModel is None:
        return None
    try:
        predictor = EBQAModel(
            model_name_or_path=cfg.model_name_or_path,
            tokenizer_name_or_path=cfg.tokenizer_name_or_path,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            output_dir=str(Path(cfg.output_dir) / "predictor"),
            logging_steps=100,
            save_strategy="no",
            fp16=False,
            max_answer_len=cfg.max_answer_len,  # 使用训练时的答案长度上限，避免长答案被截断
            use_noise=cfg.use_noise,
            noise_embed_dim=cfg.noise_embed_dim,
        )
        predictor.model = trained_model
        return predictor
    except Exception:
        return None


if __name__ == "__main__":
    try:
        from .config_io import (
            load_config as load_ebqa_cfg,
            train_block as ebqa_train_block,
        )
    except Exception:
        from pre_struct.ebqa.config_io import (
            load_config as load_ebqa_cfg,
            train_block as ebqa_train_block,
        )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Path to EBQA config file")
    args, _ = parser.parse_known_args()
    
    cfg_path = args.config_file if args.config_file else os.environ.get(
        "EBQA_CONFIG_PATH",
        "pre_struct/ebqa/ebqa_config.json",
    )
    cfgd = load_ebqa_cfg(cfg_path)
    tb = ebqa_train_block(cfgd)

    cfg = TrainConfig(
        data_path=str(tb["data_path"]),
        precomputed=bool(tb["precomputed"]),
        report_struct_path=str(cfgd["report_struct_path"]),
        model_name_or_path=str(cfgd["model_name_or_path"]),
        tokenizer_name_or_path=str(cfgd["tokenizer_name_or_path"]),
        max_seq_len=int(cfgd["max_seq_len"]),
        max_tokens_ctx=int(cfgd["max_tokens_ctx"]),
        max_answer_len=int(cfgd.get("max_answer_len", 1000)),  # 默认大值，由动态机制控制
        output_dir=str(cfgd["output_dir"]),
        num_train_epochs=int(tb["num_train_epochs"]),
        per_device_batch_size=int(tb["per_device_batch_size"]),
        grad_accum_steps=int(tb["grad_accum_steps"]),
        learning_rate=float(tb["learning_rate"]),
        weight_decay=float(tb["weight_decay"]),
        warmup_ratio=float(tb["warmup_ratio"]),
        max_grad_norm=float(tb["max_grad_norm"]),
        num_workers=int(tb["num_workers"]),
        pin_memory=bool(tb["pin_memory"]),
        eval_ratio=float(tb["eval_ratio"]),
        save_every_epochs=int(tb["save_every_epochs"]),
        early_stopping_patience=int(tb["early_stopping_patience"]),
        early_stopping_min_delta=float(tb["early_stopping_min_delta"]),
        plot_update_every=int(tb["plot_update_every"]),
        metrics_filename=str(tb["metrics_filename"]),
        seed=int(tb["seed"]),
        device=str(tb["device"]),
        allow_tf32=bool(tb["allow_tf32"]),
        chunk_mode=str(cfgd["chunk_mode"]),
        label_smoothing=float(tb["label_smoothing"]),
        null_margin=float(tb["null_margin"]),
        null_margin_weight=float(tb["null_margin_weight"]),
        use_weighted_sampler=bool(tb["use_weighted_sampler"]),
        # 从配置读取四项（没有则用默认）
        length_penalty_weight=float(tb.get("length_penalty_weight", 0.05)),
        length_penalty_cap=int(tb.get("length_penalty_cap", 256)),
        length_penalty_margin=float(tb.get("length_penalty_margin", 4.0)),
        negative_downsample=float(tb.get("negative_downsample", 1.0)),
        # 传递更多训练细节开关
        best_metric=str(tb.get("best_metric", "token_f1")),
        length_reasonableness_weight=float(tb.get("length_reasonableness_weight", 0.0)),
        length_reasonableness_scale=float(tb.get("length_reasonableness_scale", 2.0)),
        short_field_weight=float(tb.get("short_field_weight", 1.0)),
        short_field_boost=float(tb.get("short_field_boost", 0.0)),
        use_noise=bool(tb.get("use_noise", False)),
        noise_embed_dim=int(tb.get("noise_embed_dim", 16)),
    )

    print(
        f"[INFO] Config: {cfg_path} | Device: {cfg.device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}"
    )
    print(f"[INFO] Data: {cfg.data_path} | precomputed={cfg.precomputed}")
    print(f"[INFO] Model: {cfg.model_name_or_path}")
    print(f"[INFO] Tokenizer: {cfg.tokenizer_name_or_path}")
    print(f"[INFO] use_noise={cfg.use_noise}, noise_embed_dim={cfg.noise_embed_dim}")
    print(f"[INFO] chunk_mode={cfg.chunk_mode}, max_answer_len={cfg.max_answer_len}")
    print(
        f"[INFO] length_penalty: w={cfg.length_penalty_weight}, cap={cfg.length_penalty_cap}, margin={cfg.length_penalty_margin}"
    )

    t0 = time.time()
    train_loop(cfg)
    print(f"[DONE] Total time: {time.time()-t0:.1f}s")
