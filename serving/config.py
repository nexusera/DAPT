# -*- coding: utf-8 -*-
"""
服务配置，通过环境变量或 .env 文件覆盖默认值。

使用示例：
    export MODEL_DIR=/data/ocean/DAPT/runs/kv_ner_finetuned_noise_bucket/best
    export NOISE_BINS_PATH=/data/ocean/DAPT/workspace/noise_bins.json
    uvicorn serving.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── 模型路径 ──────────────────────────────────────────────────────────────
    # 微调后的 KV-NER checkpoint 目录（含 pytorch_model.bin / config.json / tokenizer）
    model_dir: str = "/data/ocean/DAPT/runs/kv_ner_finetuned_noise_bucket/best"

    # 噪声分桶边界文件（由 noise_feature_processor.py 预计算）
    noise_bins_path: str = "/data/ocean/DAPT/workspace/noise_bins.json"

    # ── 推理超参 ──────────────────────────────────────────────────────────────
    max_seq_length: int = 512
    sliding_window_stride: int = 128
    # "bucket" | "linear" | "mlp" | "concat_linear"（需与训练时一致）
    noise_mode: str = "bucket"

    # ── 后处理配置 ────────────────────────────────────────────────────────────
    value_attach_window: int = 50
    value_same_line_only: bool = True
    value_crossline_fallback_len: int = 0
    value_len_cap: int = 64
    value_stop_punct: str = "。；;，,\n"

    # ── 服务行为 ──────────────────────────────────────────────────────────────
    # 请求体最大字节数
    max_request_body_bytes: int = 5 * 1024 * 1024  # 5 MB
    # ocr_text 最大字符数
    max_ocr_text_len: int = 10_000
    # 是否在响应中包含 entities / unmatched_keys / latency_ms 等调试字段
    include_entities: bool = True
    include_unmatched: bool = True
    include_latency: bool = True
    include_noise_summary: bool = True

    # ── Dynamic Batching ──────────────────────────────────────────────────────
    # 是否启用动态批处理（生产环境推荐开启以提升 GPU 利用率）
    enable_dynamic_batching: bool = False
    # 最大聚合批大小
    batch_max_size: int = 16
    # 聚合等待窗口（毫秒）：在此时间内到达的请求合并为一个 GPU batch
    batch_max_wait_ms: float = 10.0

    # ── 模型加速 ──────────────────────────────────────────────────────────────
    # 是否使用 torch.compile（PyTorch >= 2.0）对模型进行编译优化
    # 注意：首次推理会有额外编译延迟（数秒），之后延迟显著降低
    use_torch_compile: bool = False

    # ── 设备 ──────────────────────────────────────────────────────────────────
    device: str = "cuda"  # "cuda" | "cpu"

    # ── DAPT 代码库根路径（用于 sys.path 注入） ─────────────────────────────
    dapt_root: str = str(Path(__file__).resolve().parents[1])


# 单例，全局共享
settings = Settings()
