# -*- coding: utf-8 -*-
"""
请求体 Pydantic 模型，含完整入参校验。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class WordProbability(BaseModel):
    average: Optional[float] = None
    min: Optional[float] = None
    variance: Optional[float] = None


class WordLocation(BaseModel):
    top: Optional[float] = None
    left: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


class WordResult(BaseModel):
    words: str = Field(..., description="该行 OCR 识别文本")
    probability: Optional[WordProbability] = None
    location: Optional[WordLocation] = None

    @field_validator("words")
    @classmethod
    def words_not_empty(cls, v: str) -> str:
        # words 可以是空字符串（OCR 可能识别到空行）
        return v


class ExtractRequest(BaseModel):
    ocr_text: str = Field(
        ...,
        description="OCR 识别的页面全文，保留原始换行。长度 1~10000 字符。",
        min_length=1,
        max_length=10_000,
    )
    report_title: Optional[str] = Field(
        default=None,
        description="报告类型标题（如"凝血功能""病理报告"），最大 64 字符。",
        max_length=64,
    )
    words_result: Optional[List[WordResult]] = Field(
        default=None,
        description=(
            "百度 OCR / PaddleOCR 的逐行识别结果，用于实时计算 7 维噪声特征。"
            "缺失时退化为不使用噪声（填充完美值）。"
        ),
    )
    paragraphs_result: Optional[List[Any]] = Field(
        default=None,
        description="百度 OCR 段落结构，用于计算 align_score。",
    )
    noise_values: Optional[List[List[float]]] = Field(
        default=None,
        description=(
            "调用方已预计算的逐字符 7 维噪声向量，二维数组 [N][7]，"
            "N 必须等于 len(ocr_text)。优先级高于 words_result。"
        ),
    )

    @model_validator(mode="after")
    def validate_noise_values(self) -> "ExtractRequest":
        if self.noise_values is not None:
            n = len(self.noise_values)
            expected = len(self.ocr_text)
            if n != expected:
                raise ValueError(
                    f"noise_values length ({n}) != ocr_text length ({expected})"
                )
            for i, row in enumerate(self.noise_values):
                if not isinstance(row, list) or len(row) != 7:
                    raise ValueError(
                        f"each noise_values element must be a 7-dim float array, "
                        f"but noise_values[{i}] has length {len(row) if isinstance(row, list) else 'N/A'}"
                    )
        return self

    @property
    def effective_report_title(self) -> str:
        t = (self.report_title or "").strip()
        return t if t else "通用病历"
