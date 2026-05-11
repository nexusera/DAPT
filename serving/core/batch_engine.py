# -*- coding: utf-8 -*-
"""
Dynamic Batching Engine

原理：
  1. 每个 POST /api/v1/extract 请求调用 DynamicBatchEngine.infer()，
     将自身的推理任务放入 asyncio.Queue，然后 await 对应的 Future。
  2. 后台 asyncio Task (_worker_loop) 持续监听队列：
       - 收到第一个 item 后，再等待最多 max_wait_ms 毫秒，
         或直到队列中已有 max_batch_size 个 item。
       - 将这一批 item 交给线程池（run_in_executor），
         调用 ModelEngine.run_batch()（同步 CPU/GPU 操作）。
       - 结果返回后，逐一 set_result 唤醒等待中的协程。
  3. 若 Dynamic Batching 关闭，调用方可直接使用 ModelEngine.run()。

线程安全说明：
  - asyncio.Queue 本身是协程安全的。
  - ModelEngine.run_batch() 是纯 PyTorch 同步调用，在线程池中运行，
    不会阻塞事件循环。
  - Future.set_result() 从线程池回调时使用 loop.call_soon_threadsafe()
    以保证协程安全。
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _InferItem:
    """队列中的一个推理任务。"""
    text: str
    char_noise: Optional[List[List[float]]]
    future: asyncio.Future            # 结果：(entities, timing)
    loop: asyncio.AbstractEventLoop   # 所属事件循环（用于线程安全 set_result）
    enqueued_at: float                # time.perf_counter()，用于计算等待窗口


class DynamicBatchEngine:
    """
    Dynamic Batching 引擎，封装 ModelEngine.run_batch()。

    使用方式：
        # 服务启动时（lifespan）
        await batch_engine.start()

        # 请求处理
        entities, timing = await batch_engine.infer(text, char_noise)

        # 服务关闭时
        await batch_engine.stop()
    """

    def __init__(
        self,
        model_engine,           # ModelEngine 实例（已 load）
        max_batch_size: int = 16,
        max_wait_ms: float = 10.0,
    ) -> None:
        self._engine = model_engine
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """启动后台 worker Task。必须在 async 上下文（lifespan）中调用。"""
        self._queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._worker_loop(), name="dynamic-batch-worker")
        logger.info(
            f"Dynamic Batch Worker 已启动 "
            f"(max_batch={self.max_batch_size}, max_wait={self.max_wait_ms}ms)"
        )

    async def stop(self) -> None:
        """停止后台 worker Task。"""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Dynamic Batch Worker 已停止")

    # ── 请求入口 ──────────────────────────────────────────────────────────────

    async def infer(
        self,
        text: str,
        char_noise: Optional[List[List[float]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        异步提交推理请求，等待结果。

        Returns:
            (entities, timing) 同 ModelEngine.run()
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        item = _InferItem(
            text=text,
            char_noise=char_noise,
            future=future,
            loop=loop,
            enqueued_at=time.perf_counter(),
        )
        await self._queue.put(item)
        return await future

    # ── 后台 Worker ───────────────────────────────────────────────────────────

    async def _worker_loop(self) -> None:
        """持续从队列拉取请求，聚合后提交批量推理。"""
        while True:
            # 等待至少一个请求
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                return

            batch: List[_InferItem] = [first]
            # H13: deadline 以"首个请求入队时刻"为基准，每收到一个新请求便向后滚动
            # 一个窗口长度（但不超过 2× max_wait_ms），避免缓慢到来的请求因
            # 剩余时间耗尽而被漏批。
            _wait_s = self.max_wait_ms / 1000.0
            _max_deadline_s = first.enqueued_at + 2.0 * _wait_s
            deadline = first.enqueued_at + _wait_s

            # 在窗口期内尽量多收集请求
            while len(batch) < self.max_batch_size:
                remaining_s = deadline - time.perf_counter()
                if remaining_s <= 0:
                    break
                try:
                    extra = await asyncio.wait_for(self._queue.get(), timeout=remaining_s)
                    batch.append(extra)
                    # H13: 收到新请求，把 deadline 向后滚动一个窗口，但不超过上限
                    deadline = min(_max_deadline_s, time.perf_counter() + _wait_s)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    # 取消时先把当前 batch 处理完，再退出
                    break

            logger.debug(f"Dispatch batch size={len(batch)}")

            # 在线程池中同步执行 GPU 推理（避免阻塞事件循环）
            loop = asyncio.get_event_loop()
            try:
                results: List[Tuple] = await loop.run_in_executor(
                    None, self._run_batch_sync, batch
                )
                for item, result in zip(batch, results):
                    _safe_set_result(item.future, item.loop, result)
            except Exception as exc:
                logger.exception(f"批量推理异常: {exc}")
                for item in batch:
                    _safe_set_exception(item.future, item.loop, exc)

    def _run_batch_sync(
        self, batch: List[_InferItem]
    ) -> List[Tuple[List[Dict[str, Any]], Dict[str, float]]]:
        """同步调用 ModelEngine.run_batch()，在线程池中执行。"""
        items = [{"text": it.text, "char_noise": it.char_noise} for it in batch]
        return self._engine.run_batch(items)


# ── 线程安全的 Future 操作 ─────────────────────────────────────────────────────

def _safe_set_result(
    future: asyncio.Future,
    loop: asyncio.AbstractEventLoop,
    result: Any,
) -> None:
    """从任意线程安全地设置 Future 结果。"""
    if not future.done():
        loop.call_soon_threadsafe(future.set_result, result)


def _safe_set_exception(
    future: asyncio.Future,
    loop: asyncio.AbstractEventLoop,
    exc: Exception,
) -> None:
    """从任意线程安全地设置 Future 异常。"""
    if not future.done():
        loop.call_soon_threadsafe(future.set_exception, exc)


# 全局单例（仅在 batch_mode=True 时启用）
batch_engine: Optional[DynamicBatchEngine] = None
