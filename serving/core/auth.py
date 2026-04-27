# -*- coding: utf-8 -*-
"""
serving/core/auth.py
--------------------
H10: API Key 鉴权中间件 + 令牌桶限流。

配置方式（.env 或环境变量）：
    API_KEY=your-secret-key         # 非空时启用鉴权
    RATE_LIMIT_RPS=50               # 每秒最大请求数（0 = 不限流）

调用方须在请求头中携带：
    X-API-Key: your-secret-key
"""
from __future__ import annotations

import asyncio
import time
import logging
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """H10: 若 settings.api_key 非空，校验请求头 X-API-Key，不匹配返回 401。

    健康检查路由 /health 不受鉴权保护，方便负载均衡器探活。
    """

    def __init__(self, app, api_key: str) -> None:
        super().__init__(app)
        self._api_key: Optional[str] = api_key or None

    async def dispatch(self, request: Request, call_next):
        if self._api_key is None:
            return await call_next(request)

        # /health 路由不需要鉴权
        if request.url.path.startswith("/health"):
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if provided != self._api_key:
            logger.warning("Unauthorized request from %s (path=%s)", request.client, request.url.path)
            return JSONResponse(
                {"status": "error", "error": {"code": "UNAUTHORIZED", "message": "Invalid or missing API key"}},
                status_code=401,
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """H10: 简单令牌桶限流。rate_limit_rps=0 表示不限流。

    此实现为单进程版本；多进程部署时建议改用 Redis 等共享存储。
    """

    def __init__(self, app, rate_limit_rps: int) -> None:
        super().__init__(app)
        self._rps = rate_limit_rps
        if self._rps > 0:
            self._tokens = float(self._rps)
            self._last_refill = time.monotonic()
            self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):
        if self._rps <= 0:
            return await call_next(request)

        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(float(self._rps), self._tokens + elapsed * self._rps)
            self._last_refill = now

            if self._tokens < 1.0:
                logger.warning("Rate limit exceeded for %s", request.client)
                return JSONResponse(
                    {"status": "error", "error": {"code": "RATE_LIMITED", "message": "Too many requests, please retry later"}},
                    status_code=429,
                )
            self._tokens -= 1.0

        return await call_next(request)
