# -*- coding: utf-8 -*-
"""
POST /api/v1/extract —— 主推理接口。

推理路径自动选择：
  - ENABLE_DYNAMIC_BATCHING=true → 走 DynamicBatchEngine.infer()（异步聚合批）
  - ENABLE_DYNAMIC_BATCHING=false（默认）→ 走 ModelEngine.run()（同步单请求）
"""
from __future__ import annotations

import time
import uuid
import logging
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from serving.config import settings
from serving.core.model_engine import engine
import serving.core.batch_engine as _be_module
from serving.core.noise_extractor import build_char_noise, noise_summary
from serving.core.postprocessor import assemble_kv
from serving.schemas.request import ExtractRequest
from serving.schemas.response import (
    EntityItem,
    ErrorDetail,
    ErrorResponse,
    ExtractResponse,
    KVPair,
    LatencyMs,
    NoiseSummary,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["extract"])

_POST_CFG_KEYS = (
    "value_attach_window",
    "value_same_line_only",
    "value_crossline_fallback_len",
    "value_len_cap",
    "value_stop_punct",
)


def _make_error(
    request_id: str,
    code: str,
    message: str,
    detail: str | None = None,
    status_code: int = 500,
) -> JSONResponse:
    err = ErrorResponse(
        request_id=request_id,
        error=ErrorDetail(code=code, message=message, detail=detail),
    )
    return JSONResponse(err.model_dump(), status_code=status_code)


@router.post("/extract", response_model=ExtractResponse)
async def extract(request: Request, body: ExtractRequest) -> JSONResponse:
    request_id = str(uuid.uuid4())
    t_total_start = time.perf_counter()

    # ── 模型就绪检查 ──────────────────────────────────────────────────────────
    if not engine.ready:
        return _make_error(
            request_id, "MODEL_NOT_READY", "模型尚未加载完成，请稍后重试", status_code=503
        )

    # ── 请求体大小检查 ────────────────────────────────────────────────────────
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_request_body_bytes:
        return _make_error(
            request_id, "PAYLOAD_TOO_LARGE",
            f"请求体超过限制（最大 {settings.max_request_body_bytes // (1024 * 1024)} MB）",
            status_code=413,
        )

    ocr_text = body.ocr_text
    report_title = body.effective_report_title

    try:
        # ── 1. 噪声特征提取（CPU，非阻塞） ───────────────────────────────────
        t_noise_start = time.perf_counter()
        char_noise = build_char_noise(
            ocr_text=ocr_text,
            words_result=(
                [w.model_dump() for w in body.words_result]
                if body.words_result else None
            ),
            paragraphs_result=body.paragraphs_result,
            noise_values=body.noise_values,
        )
        t_noise_ms = (time.perf_counter() - t_noise_start) * 1000

        has_real_noise = bool(body.words_result or body.noise_values)
        n_summary = noise_summary(char_noise) if has_real_noise else None

        # ── 2. 推理（Dynamic Batching 或同步单请求） ─────────────────────────
        post_cfg: Dict[str, Any] = {k: getattr(settings, k, None) for k in _POST_CFG_KEYS}

        if settings.enable_dynamic_batching and _be_module.batch_engine is not None:
            # 异步路径：将请求放入聚合队列，await 结果
            entities, timing = await _be_module.batch_engine.infer(
                text=ocr_text, char_noise=char_noise
            )
        else:
            # 同步路径：直接调用（run() 内部调用 run_batch([item])）
            entities, timing = engine.run(text=ocr_text, char_noise=char_noise)

        # ── 3. 后处理 ─────────────────────────────────────────────────────────
        t_pp_start = time.perf_counter()
        result = assemble_kv(entities, ocr_text, cfg=post_cfg)
        t_pp_ms = (time.perf_counter() - t_pp_start) * 1000

        t_total_ms = (time.perf_counter() - t_total_start) * 1000

        # ── 4. 构建响应 ───────────────────────────────────────────────────────
        resp = ExtractResponse(
            request_id=request_id,
            status="success",
            ocr_text=ocr_text,
            report_title=report_title,
            kv_pairs=[KVPair(**p) for p in result["kv_pairs"]],
            structured=result["structured"],
            hospital=result.get("hospital"),
            entities=(
                [
                    EntityItem(
                        type=e["type"],
                        text=e["text"],
                        start=e["start"],
                        end=e["end"],
                    )
                    for e in entities
                ]
                if settings.include_entities else None
            ),
            unmatched_keys=(
                result.get("key_without_value") if settings.include_unmatched else None
            ),
            unmatched_values=(
                result.get("value_without_key") if settings.include_unmatched else None
            ),
            noise_summary=(
                NoiseSummary(**n_summary)
                if settings.include_noise_summary and n_summary else None
            ),
            latency_ms=(
                LatencyMs(
                    noise_extract=round(t_noise_ms, 2),
                    tokenize=round(timing.get("tokenize", 0), 2),
                    model_forward=round(timing.get("model_forward", 0), 2),
                    crf_decode=round(timing.get("crf_decode", 0), 2),
                    postprocess=round(t_pp_ms, 2),
                    total=round(t_total_ms, 2),
                )
                if settings.include_latency else None
            ),
        )
        return JSONResponse(resp.model_dump(exclude_none=False))

    except Exception as exc:
        logger.exception(f"[{request_id}] 推理异常: {exc}")
        return _make_error(
            request_id,
            "MODEL_ERROR",
            "模型推理内部异常",
            detail=str(exc),
            status_code=500,
        )
