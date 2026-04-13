#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速 E2E 冒烟测试（服务启动后运行）。

使用方式：
    python serving/test_api.py [--base-url http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import requests


def check(resp: requests.Response, label: str) -> dict:
    if resp.status_code != 200:
        print(f"[FAIL] {label}  HTTP {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)
    data = resp.json()
    print(f"[OK]   {label}  status={data.get('status')}  "
          f"kv_pairs={len(data.get('kv_pairs', []))}  "
          f"latency={data.get('latency_ms', {}).get('total')}ms")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    session = requests.Session()

    # ── 健康检查 ──────────────────────────────────────────────────────────────
    r = session.get(f"{base}/health")
    assert r.status_code == 200 and r.json()["status"] == "ok", f"health 失败: {r.text}"
    print("[OK]   GET /health")

    # 等待模型就绪（最多 120 秒）
    for _ in range(24):
        r = session.get(f"{base}/ready")
        if r.status_code == 200:
            break
        print("       等待模型加载...")
        time.sleep(5)
    else:
        print("[FAIL] 模型 120 秒内未就绪")
        sys.exit(1)
    print("[OK]   GET /ready")

    url = f"{base}/api/v1/extract"

    # ── 模式 A：纯文本 ────────────────────────────────────────────────────────
    payload_a = {
        "ocr_text": "姓名：张三\n性别：男\n年龄：45岁\n临床诊断：肺腺癌\n送检标本：肺穿刺组织"
    }
    check(session.post(url, json=payload_a), "模式A 纯文本")

    # ── 模式 B：带 words_result ───────────────────────────────────────────────
    payload_b = {
        "ocr_text": "姓名：张三\n性别：男\n年龄：45岁",
        "report_title": "病理报告",
        "words_result": [
            {
                "words": "姓名：张三",
                "probability": {"average": 0.98, "min": 0.95, "variance": 0.001},
                "location": {"top": 50, "left": 30, "width": 200, "height": 30},
            },
            {
                "words": "性别：男",
                "probability": {"average": 0.99, "min": 0.98, "variance": 0.0001},
                "location": {"top": 85, "left": 30, "width": 150, "height": 30},
            },
            {
                "words": "年龄：45岁",
                "probability": {"average": 0.97, "min": 0.90, "variance": 0.003},
                "location": {"top": 120, "left": 30, "width": 180, "height": 30},
            },
        ],
    }
    data_b = check(session.post(url, json=payload_b), "模式B words_result")
    assert data_b.get("noise_summary") is not None, "模式B 应含 noise_summary"

    # ── 模式 C：预计算 noise_values ───────────────────────────────────────────
    text_c = "姓名：张三"
    n = len(text_c)
    payload_c = {
        "ocr_text": text_c,
        "noise_values": [[0.98, 0.95, -6.9, 0.03, 0.0, 0.0, 0.1]] * n,
    }
    check(session.post(url, json=payload_c), "模式C noise_values")

    # ── 错误：ocr_text 为空 ───────────────────────────────────────────────────
    r = session.post(url, json={"ocr_text": ""})
    assert r.status_code == 422, f"空 ocr_text 应返回 422，实际 {r.status_code}"
    print("[OK]   空 ocr_text → 422")

    # ── 错误：noise_values 长度不匹配 ────────────────────────────────────────
    r = session.post(url, json={"ocr_text": "ab", "noise_values": [[1.0]*7]})
    assert r.status_code == 422, f"noise_values 长度错误应返回 422，实际 {r.status_code}"
    print("[OK]   noise_values 长度不匹配 → 422")

    # ── 并发压测（验证 Dynamic Batching 路径） ───────────────────────────────
    import concurrent.futures, statistics

    N_CONCURRENT = 20
    payload_perf = {"ocr_text": "姓名：张三\n性别：男\n年龄：45岁\n临床诊断：肺腺癌"}

    def _single_call():
        t0 = time.time()
        r = requests.post(url, json=payload_perf, timeout=30)
        return (time.time() - t0) * 1000, r.status_code

    print(f"\n并发压测 ({N_CONCURRENT} 个并发请求)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_CONCURRENT) as pool:
        futs = [pool.submit(_single_call) for _ in range(N_CONCURRENT)]
        times, codes = zip(*[f.result() for f in futs])

    all_ok = all(c == 200 for c in codes)
    avg_ms = statistics.mean(times)
    p99_ms = sorted(times)[int(0.99 * len(times))]
    print(f"[{'OK' if all_ok else 'FAIL'}]   并发 {N_CONCURRENT} 请求 "
          f"avg={avg_ms:.0f}ms  p99={p99_ms:.0f}ms  "
          f"success={sum(c == 200 for c in codes)}/{N_CONCURRENT}")

    print("\n所有冒烟测试通过 ✓")


if __name__ == "__main__":
    main()
