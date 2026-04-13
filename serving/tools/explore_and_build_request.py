#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具：探查 all_type_pic_oss_csv 目录下的原始 OCR 数据格式，
并从中构造符合 /api/v1/extract 接口规范的请求体 JSON 文件。

用法：
    python serving/tools/explore_and_build_request.py \
        --data_dir /data/oss/hxzh-mr/all_type_pic_oss_csv/20251204_5w \
        --num_samples 3 \
        --output_dir /tmp/api_test_payloads

输出：
    - 在 output_dir 下生成 sample_0.json ~ sample_N.json，每个文件可直接投递接口：
        curl -X POST http://127.0.0.1:8000/api/v1/extract \\
             -H "Content-Type: application/json" \\
             -d @/tmp/api_test_payloads/sample_0.json | jq .
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─── 格式探查 ─────────────────────────────────────────────────────────────────

def detect_format(data: Any) -> str:
    """识别原始数据格式类型。"""
    if isinstance(data, dict) and "words_result" in data:
        return "baidu_ocr"
    if isinstance(data, list) and data and isinstance(data[0], dict) and "data" in data[0]:
        return "report_list"
    if isinstance(data, dict):
        for k in ["text", "content"]:
            if k in data:
                return f"plain_text({k})"
    return "unknown"


def show_structure(data: Any, indent: int = 0) -> None:
    """递归打印 JSON 结构（类型 + 长度，不打印具体值）。"""
    prefix = "  " * indent
    if isinstance(data, dict):
        print(f"{prefix}dict({len(data)} keys): {list(data.keys())}")
        for k, v in list(data.items())[:5]:
            print(f"{prefix}  [{k!r}]:")
            show_structure(v, indent + 2)
        if len(data) > 5:
            print(f"{prefix}  ... ({len(data) - 5} more keys)")
    elif isinstance(data, list):
        print(f"{prefix}list({len(data)} items)")
        if data:
            print(f"{prefix}  [0]:")
            show_structure(data[0], indent + 2)
    elif isinstance(data, str):
        print(f"{prefix}str(len={len(data)}): {data[:60]!r}{'...' if len(data) > 60 else ''}")
    elif isinstance(data, (int, float, bool)):
        print(f"{prefix}{type(data).__name__}: {data}")
    elif data is None:
        print(f"{prefix}null")
    else:
        print(f"{prefix}{type(data).__name__}: {str(data)[:60]}")


# ─── 构造接口请求体 ────────────────────────────────────────────────────────────

def build_request_from_baidu_ocr(data: Dict) -> Optional[Dict]:
    """
    从百度 OCR 格式数据构造 ExtractRequest payload。

    百度 OCR 格式：
    {
        "words_result": [
            {
                "words": "...",
                "probability": {"average": 0.98, "min": 0.91, "variance": 0.003},
                "location": {"top": 120, "left": 45, "width": 200, "height": 30}
            },
            ...
        ],
        "paragraphs_result": [...],   # 可选
        "words_result_num": 42        # 可选
    }
    """
    words_result: List[Dict] = data.get("words_result", [])
    if not words_result:
        return None

    # ocr_text：直接拼接，与 compare_models.py _extract_ocr_text 完全一致
    # NER 微调推理格式：'姓名：王柏青性别：男年龄：72岁'（无分隔符）
    words = [item.get("words", "") for item in words_result if item.get("words", "")]
    ocr_text = "".join(words)
    if not ocr_text.strip():
        return None

    # words_result：保留 words / probability / location，剔除其他冗余字段
    clean_words = []
    for item in words_result:
        entry: Dict[str, Any] = {"words": item.get("words", "")}
        if "probability" in item and isinstance(item["probability"], dict):
            entry["probability"] = {
                k: item["probability"].get(k)
                for k in ["average", "min", "variance"]
            }
        if "location" in item and isinstance(item["location"], dict):
            entry["location"] = {
                k: item["location"].get(k)
                for k in ["top", "left", "width", "height"]
            }
        clean_words.append(entry)

    payload: Dict[str, Any] = {
        "ocr_text": ocr_text,
        "words_result": clean_words,
    }

    # paragraphs_result 直接透传（可选）
    if "paragraphs_result" in data and data["paragraphs_result"]:
        payload["paragraphs_result"] = data["paragraphs_result"]

    return payload


def build_request_fallback(text: str) -> Dict:
    """仅有纯文本时，构造最简请求体（不含噪声元信息）。"""
    return {"ocr_text": text[:10000]}


# ─── 文件扫描 ─────────────────────────────────────────────────────────────────

def iter_raw_records(data_dir: str, max_records: int):
    """
    遍历目录，逐条 yield (file_path, raw_data)，最多 max_records 条。
    支持 .json / .jsonl / .jsonl.gz 格式。
    """
    count = 0
    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if count >= max_records:
                return
            fpath = os.path.join(root, fname)

            if fname.endswith(".json"):
                try:
                    with open(fpath, encoding="utf-8") as f:
                        data = json.load(f)
                    yield fpath, data
                    count += 1
                except Exception as e:
                    print(f"  [WARN] 读取 {fpath} 失败: {e}")

            elif fname.endswith(".jsonl"):
                try:
                    with open(fpath, encoding="utf-8") as f:
                        for line in f:
                            if count >= max_records:
                                return
                            line = line.strip()
                            if not line:
                                continue
                            yield fpath, json.loads(line)
                            count += 1
                except Exception as e:
                    print(f"  [WARN] 读取 {fpath} 失败: {e}")

            elif fname.endswith(".jsonl.gz"):
                import gzip
                try:
                    with gzip.open(fpath, "rt", encoding="utf-8") as f:
                        for line in f:
                            if count >= max_records:
                                return
                            line = line.strip()
                            if not line:
                                continue
                            yield fpath, json.loads(line)
                            count += 1
                except Exception as e:
                    print(f"  [WARN] 读取 {fpath} 失败: {e}")


# ─── 主逻辑 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="探查 OCR 数据并生成 API 测试请求体")
    parser.add_argument("--data_dir", required=True, help="原始 OCR 数据目录")
    parser.add_argument("--num_samples", type=int, default=3, help="生成的样本数")
    parser.add_argument("--output_dir", default="/tmp/api_test_payloads", help="输出目录")
    parser.add_argument("--explore_only", action="store_true", help="只探查格式，不生成请求体")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"探查目录: {args.data_dir}")
    print(f"{'='*60}\n")

    generated = 0
    explored = 0

    for fpath, raw_data in iter_raw_records(args.data_dir, max_records=args.num_samples * 3):
        fmt = detect_format(raw_data)

        # 打印前几条的结构
        if explored < 2:
            print(f"─── 样本来源: {fpath}")
            print(f"    格式识别: {fmt}")
            show_structure(raw_data)
            print()
            explored += 1

        if args.explore_only:
            continue

        # 构造请求体
        payload = None
        if fmt == "baidu_ocr":
            payload = build_request_from_baidu_ocr(raw_data)
        elif fmt.startswith("plain_text"):
            key = fmt.split("(")[1].rstrip(")")
            text = raw_data.get(key, "")
            if text and len(text) >= 10:
                payload = build_request_fallback(text)

        if payload is None:
            continue

        out_path = output_dir / f"sample_{generated}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[{generated}] 已生成: {out_path}")
        print(f"    格式: {fmt}")
        print(f"    ocr_text 长度: {len(payload['ocr_text'])} 字符")
        print(f"    words_result 行数: {len(payload.get('words_result', []))}")
        has_prob = any("probability" in w for w in payload.get("words_result", []))
        has_loc = any("location" in w for w in payload.get("words_result", []))
        print(f"    含 probability: {has_prob}  |  含 location: {has_loc}")
        print(f"    ocr_text 前100字: {payload['ocr_text'][:100]!r}")
        print()

        generated += 1
        if generated >= args.num_samples:
            break

    print(f"\n{'='*60}")
    if not args.explore_only:
        print(f"共生成 {generated} 个请求体文件，保存至: {output_dir}")
        print()
        print("发送测试（服务启动后执行）：")
        for i in range(generated):
            print(f"  curl -s -X POST http://127.0.0.1:8000/api/v1/extract \\")
            print(f"       -H 'Content-Type: application/json' \\")
            print(f"       -d @{output_dir}/sample_{i}.json | python3 -m json.tool")
            print()
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
