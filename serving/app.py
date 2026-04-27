# -*- coding: utf-8 -*-
"""
FastAPI 应用入口。

启动示例：
    # 开发模式（自动重载）
    uvicorn serving.app:app --reload --port 8000

    # 生产模式（多 worker，每 worker 独立加载模型）
    uvicorn serving.app:app --host 0.0.0.0 --port 8000 --workers 2

环境变量（或 .env 文件）：
    MODEL_DIR                微调后的 KV-NER checkpoint 路径
    NOISE_BINS_PATH          noise_bins.json 路径
    DEVICE                   cuda | cpu（默认 cuda）
    NOISE_MODE               bucket | linear | mlp（默认 bucket，须与训练时一致）
    ENABLE_DYNAMIC_BATCHING  true | false（默认 false）
    BATCH_MAX_SIZE           最大聚合批大小（默认 16）
    BATCH_MAX_WAIT_MS        聚合等待窗口毫秒数（默认 10.0）
    USE_TORCH_COMPILE        true | false（默认 false）
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── sys.path（确保可从 serving/ 目录直接 import dapt 包） ─────────────────────
_DAPT_ROOT = Path(__file__).resolve().parent.parent
_EVAL_PKG = _DAPT_ROOT / "dapt_eval_package"
for _p in [str(_DAPT_ROOT), str(_EVAL_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from serving.config import settings
from serving.core.model_engine import engine
from serving.core.auth import APIKeyMiddleware, RateLimitMiddleware  # H10
import serving.core.batch_engine as _be_module
from serving.routers import extract as extract_router
from serving.routers import health as health_router

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── 生命周期：启动时预加载模型 + 启动 Dynamic Batch Worker ────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("服务启动，开始预加载模型...")
    try:
        engine.load(
            model_dir=settings.model_dir,
            noise_bins_path=settings.noise_bins_path,
            device=settings.device,
            noise_mode=settings.noise_mode,
            max_seq_length=settings.max_seq_length,
            use_torch_compile=settings.use_torch_compile,
        )
        logger.info("模型加载成功。")
    except Exception as exc:
        # M11: 改用 logger.exception 保留完整 traceback，便于排查 OOM / 路径错误等
        logger.exception("模型加载失败，服务以未就绪状态启动（/ready 返回 503）: %s", exc)

    # ── Dynamic Batching ──────────────────────────────────────────────────────
    if settings.enable_dynamic_batching and engine.ready:
        from serving.core.batch_engine import DynamicBatchEngine
        _be_module.batch_engine = DynamicBatchEngine(
            model_engine=engine,
            max_batch_size=settings.batch_max_size,
            max_wait_ms=settings.batch_max_wait_ms,
        )
        await _be_module.batch_engine.start()
        logger.info(
            f"Dynamic Batching 已启用 "
            f"(max_batch={settings.batch_max_size}, max_wait={settings.batch_max_wait_ms}ms)"
        )
    else:
        if settings.enable_dynamic_batching:
            logger.warning("模型未就绪，Dynamic Batching 未启动")
        else:
            logger.info("Dynamic Batching 未启用（同步单请求模式）")

    yield

    # ── 关闭 ─────────────────────────────────────────────────────────────────
    if _be_module.batch_engine is not None:
        await _be_module.batch_engine.stop()
    logger.info("服务关闭。")


# ── FastAPI 应用 ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="KV-BERT 病历半结构化抽取服务",
    description=(
        "输入 OCR 结果（纯文本或含元信息的完整 OCR JSON），"
        "输出半结构化 KV 键值对。"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# H10: 限流中间件（先注册先执行，放在最外层）
app.add_middleware(RateLimitMiddleware, rate_limit_rps=settings.rate_limit_rps)

# H10: API Key 鉴权中间件
app.add_middleware(APIKeyMiddleware, api_key=settings.api_key)

# H9: CORS — 从环境变量 CORS_ORIGINS 读取允许来源，空字符串表示拒绝所有跨域
_cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,       # 不再硬编码 ["*"]
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
    allow_credentials=False,
)


# ── M10: 显式处理 422 Validation 错误，防止被下方通用 handler 吞掉 ─────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning("请求校验失败 [path=%s]: %s", request.url.path, exc.errors())
    return JSONResponse(
        {
            "request_id": "unknown",
            "status": "error",
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "请求参数校验失败",
                "detail": exc.errors(),
            },
        },
        status_code=422,
    )


# ── 全局 JSON 解析异常处理 ────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # H11: 内部细节只写日志，不透传给调用方，避免暴露模型路径/CUDA 错误等内部信息
    logger.exception("未捕获异常 [path=%s]: %s", request.url.path, exc)
    return JSONResponse(
        {
            "request_id": "unknown",
            "status": "error",
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "服务内部错误，请联系管理员",
            },
        },
        status_code=500,
    )


# ── 路由注册 ──────────────────────────────────────────────────────────────────
app.include_router(health_router.router)
app.include_router(extract_router.router)
