# -*- coding: utf-8 -*-
"""
集成测试：从磁盘读取 JSON 请求体，调用已启动的 HTTP 服务 POST /api/v1/extract。

适用场景：
  - 远端 H200 上已启动 uvicorn / Docker
  - 使用仓库内 fixtures，或通过环境变量指定机器上的任意 JSON 文件

环境变量：
  SERVING_BASE_URL   默认 http://127.0.0.1:8000
  EXTRACT_TEST_JSON  请求体 JSON 路径；未设置时使用 tests/fixtures/sample_ocr_request.json

运行示例：
  cd DAPT && export PYTHONPATH="$PWD:$PWD/dapt_eval_package"
  pip install pytest requests
  pytest serving/tests/test_extract_from_file.py -v -m integration
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("requests")
import requests  # noqa: E402


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
DEFAULT_PAYLOAD = FIXTURE_DIR / "sample_ocr_request.json"


def _base_url() -> str:
    return os.environ.get("SERVING_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _payload_path() -> Path:
    p = os.environ.get("EXTRACT_TEST_JSON")
    if p:
        return Path(p).expanduser().resolve()
    return DEFAULT_PAYLOAD


def _service_ready(base: str, timeout: float = 2.0) -> bool:
    try:
        r = requests.get(f"{base}/ready", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


@pytest.fixture(scope="module")
def base_url():
    return _base_url()


@pytest.mark.integration
def test_extract_payload_from_json_file(base_url: str):
    """
    从 JSON 文件读取请求体，调用 /api/v1/extract，校验成功响应结构。
    """
    if not _service_ready(base_url):
        pytest.skip(
            f"服务未就绪: {base_url}/ready 不可用，请先启动 serving 并设置 SERVING_BASE_URL"
        )

    path = _payload_path()
    assert path.is_file(), f"请求体文件不存在: {path}"

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    assert isinstance(payload, dict), "JSON 根对象须为 object"
    assert payload.get("ocr_text"), "ocr_text 必填且非空"

    url = f"{base_url}/api/v1/extract"
    resp = requests.post(url, json=payload, timeout=120)

    if resp.status_code != 200:
        pytest.fail(f"HTTP {resp.status_code}: {resp.text[:2000]}")

    data = resp.json()
    assert data.get("status") == "success"
    assert "request_id" in data
    assert data.get("ocr_text") == payload["ocr_text"]
    assert isinstance(data.get("kv_pairs"), list)
    assert isinstance(data.get("structured"), dict)
    assert "report_title" in data


@pytest.mark.integration
def test_health_endpoints(base_url: str):
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
    except requests.RequestException as exc:
        pytest.skip(f"无法连接 {base_url}: {exc}")

    assert r.status_code == 200
    assert r.json().get("status") == "ok"
