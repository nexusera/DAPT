# -*- coding: utf-8 -*-
"""
响应体 Pydantic 模型。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class KVPair(BaseModel):
    key: str
    value: str
    key_span: List[int]    # [start, end)
    value_span: List[int]  # [start, end)


class EntityItem(BaseModel):
    type: str   # KEY | VALUE | HOSPITAL
    text: str
    start: int
    end: int


class NoiseSummary(BaseModel):
    avg_confidence: Optional[float] = None
    min_confidence: Optional[float] = None
    low_conf_char_ratio: Optional[float] = None


class LatencyMs(BaseModel):
    noise_extract: Optional[float] = None
    tokenize: Optional[float] = None
    model_forward: Optional[float] = None
    crf_decode: Optional[float] = None
    postprocess: Optional[float] = None
    total: float


class ExtractResponse(BaseModel):
    request_id: str
    status: str  # "success" | "error"

    ocr_text: str
    report_title: str

    kv_pairs: List[KVPair]
    structured: Dict[str, Union[str, List[str]]]
    hospital: Optional[str] = None

    entities: Optional[List[EntityItem]] = None
    unmatched_keys: Optional[List[str]] = None
    unmatched_values: Optional[List[str]] = None

    noise_summary: Optional[NoiseSummary] = None
    latency_ms: Optional[LatencyMs] = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    request_id: str
    status: str = "error"
    error: ErrorDetail
