# -*- coding: utf-8 -*-
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from serving.core.model_engine import engine

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> JSONResponse:
    """进程存活检查。"""
    return JSONResponse({"status": "ok"})


@router.get("/ready")
async def ready() -> JSONResponse:
    """模型加载就绪检查。"""
    if engine.ready:
        return JSONResponse({"status": "ready"})
    return JSONResponse({"status": "not_ready"}, status_code=503)
