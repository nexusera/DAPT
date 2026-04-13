#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全流程端到端测试：图片 → (已有 OCR txt / 实时 Baidu OCR) → KV-BERT 接口 → 结构化结果

目录结构假设：
    <data_dir>/
        U<id>/
            <hash>.jpg   ← 原始病历图片
            <hash>.txt   ← 已有 OCR 结果（Baidu JSON 格式）或纯文本

用法示例：
    # 使用已有 .txt（不调百度 OCR）
    python serving/tools/e2e_pipeline_test.py \\
        --data_dir /data/oss/hxzh-mr/all_type_pic_oss_csv/20251204_5w \\
        --api_url http://127.0.0.1:8000 \\
        --num_samples 3

    # 强制用图片实时调百度 OCR
    python serving/tools/e2e_pipeline_test.py \\
        --data_dir /data/oss/hxzh-mr/all_type_pic_oss_csv/20251204_5w \\
        --api_url http://127.0.0.1:8000 \\
        --force_ocr \\
        --num_samples 3

    # 只测一张图（指定子目录）
    python serving/tools/e2e_pipeline_test.py \\
        --single_dir /data/oss/hxzh-mr/all_type_pic_oss_csv/20251204_5w/U2025120311363573500008273 \\
        --api_url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# 确保 DAPT 根目录在 sys.path（以便 import baidu_ocr）
_DAPT_ROOT = Path(__file__).resolve().parents[2]
if str(_DAPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DAPT_ROOT))


# ─── OCR 相关 ─────────────────────────────────────────────────────────────────

def _load_txt_as_ocr(txt_path: Path) -> Optional[Dict]:
    """尝试读取 .txt 文件并解析为百度 OCR JSON。
    返回 None 表示不是有效的 OCR JSON。
    """
    try:
        raw = txt_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        data = json.loads(raw)
        # 合法的百度 OCR JSON 必须含 words_result 列表
        if isinstance(data, dict) and isinstance(data.get("words_result"), list):
            return data
    except Exception:
        pass
    return None


def _call_baidu_ocr(jpg_path: Path) -> Dict:
    """调用本地 baidu_ocr.ocr()，返回标准 words_result 格式。"""
    from baidu_ocr import ocr as baidu_ocr_call
    b64 = base64.b64encode(jpg_path.read_bytes()).decode("utf-8")
    return baidu_ocr_call(b64, mode="accurate")


def get_ocr_result(
    subdir: Path,
    force_ocr: bool = False,
) -> Tuple[Optional[Dict], str]:
    """
    获取一个 U.../hash.jpg + hash.txt 子目录的 OCR 结果。

    Returns:
        (ocr_dict, source)
        source: "txt_cache" | "baidu_api" | "none"
    """
    jpgs = list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg")) + list(subdir.glob("*.png"))
    txts = list(subdir.glob("*.txt"))

    if not jpgs and not txts:
        return None, "none"

    # 1. 优先读 .txt 缓存（除非 force_ocr）
    if not force_ocr and txts:
        for txt in txts:
            ocr = _load_txt_as_ocr(txt)
            if ocr:
                return ocr, "txt_cache"

    # 2. 调百度 OCR
    if jpgs:
        jpg = jpgs[0]
        try:
            ocr = _call_baidu_ocr(jpg)
            if ocr and "words_result" in ocr:
                return ocr, "baidu_api"
            else:
                print(f"  [WARN] 百度 OCR 返回异常: {str(ocr)[:200]}")
        except Exception as e:
            print(f"  [WARN] 百度 OCR 调用失败: {e}")

    # 3. .txt 是纯文本兜底
    if txts:
        text = txts[0].read_text(encoding="utf-8").strip()
        if text:
            return {"words_result": [{"words": line} for line in text.splitlines() if line.strip()],
                    "_fallback": "plain_txt"}, "txt_cache"

    return None, "none"


# ─── OCR 文本重建 ─────────────────────────────────────────────────────────────

def _has_location(words_result: List[Dict]) -> bool:
    return any("location" in w for w in words_result)


def _reconstruct_text_spatial(words_result: List[Dict]) -> str:
    """
    有 location 信息时：按空间位置重建文本，同一行的 OCR 块用空格合并，不同行用换行。
    判断"同行"：两个 block 的 top 差值 < min(height1, height2) * 0.6
    """
    blocks = []
    for w in words_result:
        text = w.get("words", "").strip()
        if not text:
            continue
        loc = w.get("location", {})
        top = float(loc.get("top", 0))
        height = float(loc.get("height", 20))
        left = float(loc.get("left", 0))
        blocks.append({"text": text, "top": top, "height": height, "left": left})
    if not blocks:
        return ""
    blocks.sort(key=lambda b: (b["top"], b["left"]))
    lines_out: List[str] = []
    cur_line: List[str] = [blocks[0]["text"]]
    cur_top = blocks[0]["top"]
    cur_h = blocks[0]["height"]
    for b in blocks[1:]:
        if abs(b["top"] - cur_top) < min(cur_h, b["height"]) * 0.6:
            cur_line.append(b["text"])
        else:
            lines_out.append(" ".join(cur_line))
            cur_line = [b["text"]]
            cur_top = b["top"]
            cur_h = b["height"]
    lines_out.append(" ".join(cur_line))
    return "\n".join(lines_out)


def _reconstruct_text_no_location(words_result: List[Dict]) -> str:
    """
    无 location 信息时：与预训练 export_ocr_texts.py 保持一致，用空格拼接。
    预训练文本格式：'姓名 王柏青 性别 男 年龄 72岁'（空格分隔，所有 words 连为一行）
    这是模型在预训练和微调时实际见过的文本格式。
    """
    words = [w.get("words", "").strip() for w in words_result if w.get("words", "").strip()]
    return " ".join(words)


def reconstruct_ocr_text(words_result: List[Dict]) -> str:
    """
    选择合适策略重建 OCR 文本。
    - 有 location：空间重建（同行合并后换行），还原版面结构
    - 无 location：空格拼接，与预训练文本格式一致
    """
    if _has_location(words_result):
        return _reconstruct_text_spatial(words_result)
    return _reconstruct_text_no_location(words_result)


# ─── 构造 API 请求体 ───────────────────────────────────────────────────────────

def build_payload(ocr: Dict, report_title: Optional[str] = None) -> Optional[Dict]:
    """从百度 OCR JSON 构造 ExtractRequest payload。"""
    words_result: List[Dict] = ocr.get("words_result", [])
    if not words_result:
        return None

    ocr_text = reconstruct_ocr_text(words_result)
    if not ocr_text:
        return None

    clean_words = []
    for item in words_result:
        entry: Dict[str, Any] = {"words": item.get("words", "")}
        if isinstance(item.get("probability"), dict):
            entry["probability"] = {k: item["probability"].get(k) for k in ["average", "min", "variance"]}
        if isinstance(item.get("location"), dict):
            entry["location"] = {k: item["location"].get(k) for k in ["top", "left", "width", "height"]}
        clean_words.append(entry)

    payload: Dict[str, Any] = {
        "ocr_text": ocr_text,
        "words_result": clean_words,
    }
    if ocr.get("paragraphs_result"):
        payload["paragraphs_result"] = ocr["paragraphs_result"]
    if report_title:
        payload["report_title"] = report_title

    return payload


# ─── 调用接口 ─────────────────────────────────────────────────────────────────

def call_extract_api(api_url: str, payload: Dict, timeout: int = 60) -> Tuple[Optional[Dict], float]:
    """POST /api/v1/extract，返回 (response_dict, elapsed_ms)。"""
    url = f"{api_url.rstrip('/')}/api/v1/extract"
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=timeout)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    return resp.json(), elapsed_ms


def check_service_ready(api_url: str, max_wait_s: int = 60) -> bool:
    """等待服务就绪，最多等 max_wait_s 秒。"""
    url = f"{api_url.rstrip('/')}/ready"
    for i in range(max_wait_s):
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ready":
                return True
        except Exception:
            pass
        if i == 0:
            print(f"等待服务就绪 ({url})...", end="", flush=True)
        else:
            print(".", end="", flush=True)
        time.sleep(1)
    print()
    return False


# ─── 结果展示 ─────────────────────────────────────────────────────────────────

def print_result(subdir_name: str, ocr_source: str, payload: Dict, result: Dict, elapsed_ms: float):
    """格式化打印单次测试结果。"""
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"样本:     {subdir_name}")
    print(f"OCR来源:  {ocr_source}")
    print(f"文本长度: {len(payload['ocr_text'])} 字符  |  words行数: {len(payload.get('words_result', []))}")
    print(f"端到端延迟: {elapsed_ms:.1f} ms  (接口返回 latency_ms: {result.get('latency_ms', 'N/A')})")
    print()

    kv = result.get("kv_pairs", [])
    structured = result.get("structured", {})
    hospital = result.get("hospital")
    kwv = result.get("key_without_value", [])
    vwk = result.get("value_without_key", [])

    if hospital:
        print(f"  医院: {hospital}")

    if structured:
        print(f"  结构化字段 ({len(structured)} 项):")
        for k, v in list(structured.items())[:20]:
            print(f"    {k}: {v}")
        if len(structured) > 20:
            print(f"    ... (共 {len(structured)} 项)")
    elif kv:
        print(f"  KV 配对 ({len(kv)} 对):")
        for pair in kv[:20]:
            print(f"    [{pair.get('key', '')}] → {pair.get('value', '')}")
        if len(kv) > 20:
            print(f"    ... (共 {len(kv)} 对)")
    else:
        print("  [无 KV 输出]")

    if kwv:
        print(f"  无值的 KEY: {kwv[:5]}")
    if vwk:
        print(f"  无 KEY 的 VALUE: {vwk[:5]}")

    timing = result.get("timing", {})
    if timing:
        parts = [f"{k}={v}ms" for k, v in timing.items()]
        print(f"  内部耗时: {', '.join(parts)}")

    noise = result.get("noise_summary", {})
    if noise:
        print(f"  噪声模式: {noise.get('mode', '?')}  |  来源: {noise.get('source', '?')}")

    print(f"  OCR文本前120字: {payload['ocr_text'][:120]!r}")
    print(f"{bar}")


# ─── 目录扫描 ─────────────────────────────────────────────────────────────────

def iter_sample_dirs(data_dir: str, num_samples: int):
    """遍历 data_dir 下的 U... 子目录，yield Path。"""
    count = 0
    for entry in sorted(Path(data_dir).iterdir()):
        if count >= num_samples:
            break
        if entry.is_dir() and entry.name.startswith("U"):
            yield entry
            count += 1


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KV-BERT 接口全流程端到端测试")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data_dir", help="含 U.../hash.jpg+txt 结构的数据目录")
    src.add_argument("--single_dir", help="单个 U.../hash.jpg+txt 子目录路径")

    parser.add_argument("--api_url", default="http://127.0.0.1:8000", help="KV-BERT 服务地址")
    parser.add_argument("--num_samples", type=int, default=3, help="测试样本数（--data_dir 模式）")
    parser.add_argument("--force_ocr", action="store_true", help="忽略 .txt 缓存，强制调百度 OCR")
    parser.add_argument("--save_dir", default=None, help="保存请求体和响应 JSON 的目录（可选）")
    parser.add_argument("--no_wait", action="store_true", help="不等待服务就绪，直接发请求")
    parser.add_argument("--report_title", default=None, help="手动指定 report_title 字段")
    args = parser.parse_args()

    # 等待服务就绪
    if not args.no_wait:
        ready = check_service_ready(args.api_url, max_wait_s=30)
        if not ready:
            print(f"\n[ERROR] 服务未就绪: {args.api_url}/ready")
            print("请先启动服务：")
            print("  CUDA_VISIBLE_DEVICES=5 uvicorn serving.app:app --host 0.0.0.0 --port 8000 --workers 1")
            sys.exit(1)
        print(" ready!\n")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # 构造样本列表
    if args.single_dir:
        sample_dirs = [Path(args.single_dir)]
    else:
        sample_dirs = list(iter_sample_dirs(args.data_dir, args.num_samples))

    if not sample_dirs:
        print("[ERROR] 未找到任何 U... 子目录，请检查 --data_dir 路径")
        sys.exit(1)

    print(f"共找到 {len(sample_dirs)} 个样本，开始测试...\n")
    success, fail = 0, 0

    for idx, subdir in enumerate(sample_dirs):
        print(f"[{idx+1}/{len(sample_dirs)}] 处理: {subdir.name}")

        # Step 1: 获取 OCR 结果
        ocr, ocr_source = get_ocr_result(subdir, force_ocr=args.force_ocr)
        if not ocr:
            print(f"  [SKIP] 无法获取 OCR 结果")
            fail += 1
            continue
        print(f"  OCR来源: {ocr_source}  |  words行数: {len(ocr.get('words_result', []))}")

        # Step 2: 构造请求体
        payload = build_payload(ocr, report_title=args.report_title)
        if not payload:
            print(f"  [SKIP] OCR 内容为空")
            fail += 1
            continue

        # Step 3: 调用接口
        try:
            result, elapsed_ms = call_extract_api(args.api_url, payload)
        except Exception as e:
            print(f"  [ERROR] 接口调用失败: {e}")
            fail += 1
            continue

        # Step 4: 展示结果
        print_result(subdir.name, ocr_source, payload, result, elapsed_ms)

        # Step 5: 可选保存
        if save_dir:
            stem = subdir.name
            (save_dir / f"{stem}_request.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (save_dir / f"{stem}_response.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        success += 1

    print(f"\n{'='*60}")
    print(f"测试完成: 成功 {success} / 总计 {len(sample_dirs)}，失败 {fail}")
    if save_dir:
        print(f"请求/响应已保存至: {save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
