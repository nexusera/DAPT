#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从标注 JSON 中找到图片路径，调用百度 OCR 接口获取原始结果，并与标注数据合并保存。

示例：
    python fetch_and_merge_baidu_ocr.py \
        --anno_json biaozhu_data/huizhenbingli1119.json \
        --image_root /data/ocean \
        --output merged_huizhenbingli_with_ocr.json \
        --api_key <your_baidu_api_key> \
        --secret_key <your_baidu_secret_key> \
        --limit 10 --offset 0 --sleep 0.5

说明：
- 标注条目的 data.image 形如 "/data/local-files/?d=semi_pic/huizhenbingli/U.../file.jpeg"，
  会自动截取 ?d= 之后的相对路径，并与 --image_root 拼成实际文件路径。
- 输出会把每条标注对象附加字段 "ocr_raw"，内容为百度返回的完整 JSON。
- 为避免压测接口，提供 --limit / --offset / --sleep 控制调用数量与间隔。
- 默认走官方 openapi（需 api_key/secret_key）；如需改用本地 baidu_ocr.py 封装的 .com 域名接口，
    可加 --use_local_baidu，并可通过 --local_mode 传入 baidu_ocr.ocr 的 mode。
- 需要 requests 库；如未安装： pip install requests
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from baidu_ocr import ocr as local_baidu_ocr

TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
# 默认使用通用高精度接口，可根据需求替换为其他 OCR 端点
DEFAULT_OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate"


def load_annotations(path: Path) -> List[Dict[str, Any]]:
    """Load annotations from JSON or JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == '.jsonl':
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def save_annotations(path: Path, items: List[Dict[str, Any]]):
    """Save items to JSON or JSONL based on extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if path.suffix.lower() == '.jsonl':
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            json.dump(items, f, ensure_ascii=False, indent=2)


def extract_rel_path(image_field: str) -> str:
    """从 data.image 字符串中提取相对路径。

    规则：若包含 '?d='，取其后子串；否则返回去掉开头 '/' 的路径。
    """
    if not image_field:
        return ""
    if "?d=" in image_field:
        return image_field.split("?d=", 1)[1].lstrip("/")
    return image_field.lstrip("/")


def _tail_after_u_segment(rel_path: str) -> Optional[str]:
    """针对入院记录数据的兜底：找到第一个以 'U' 开头的目录，截掉此前所有前缀。

    示例：
        原始: semi_struct/all_type_pic_oss_csv/0923/Uxxxx/abc.jpg
        兜底: Uxxxx/abc.jpg
    """
    parts = [p for p in rel_path.split("/") if p]
    for i, p in enumerate(parts):
        if p.startswith("U"):
            return "/".join(parts[i:])
    return None


def _rel_candidates(rel_path: str, root: Path) -> List[str]:
    """生成多个相对路径候选，兼容重复前缀和可选的日期目录（如 0923）。"""
    cands = []
    base = rel_path.lstrip("/")
    # 1) 原始
    cands.append(base)

    # 2) 去掉重复的长前缀（semi_struct/all_type_pic_oss_csv/）
    long_prefix = "semi_struct/all_type_pic_oss_csv/"
    if base.startswith(long_prefix):
        cands.append(base[len(long_prefix):])

    # 3) 去掉与 root 同名的前缀
    root_name = root.name
    if base.startswith(root_name + "/"):
        cands.append(base[len(root_name) + 1 :])

    # 4) 去掉前缀后的首个全数字目录（如 0923）
    more = []
    for x in list(cands):
        parts = [p for p in x.split("/") if p]
        if parts and parts[0].isdigit():
            more.append("/".join(parts[1:]))
    cands.extend(more)

    # 去重保持顺序
    seen = set()
    uniq = []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def get_access_token(api_key: str, secret_key: str) -> str:
    """获取百度 OCR 的 access_token。"""
    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key,
    }
    resp = requests.post(TOKEN_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "access_token" not in data:
        raise RuntimeError(f"Failed to get access_token: {data}")
    return data["access_token"]


def call_baidu_ocr(ocr_url: str, token: str, image_path: Path) -> Dict[str, Any]:
    """调用百度 OCR，返回 JSON。"""
    with image_path.open("rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    params = {"image": img_b64}
    url = f"{ocr_url}?access_token={token}"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(url, data=params, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def process_items(
    items: List[Dict[str, Any]],
    roots: List[Path],
    ocr_caller,
    limit: Optional[int],
    offset: int,
    sleep_seconds: float,
    missing_log: Optional[Path],
) -> None:
    """就地为每条标注附加 ocr_raw。ocr_caller 接收 Path 返回 OCR JSON。"""
    total = len(items)
    start = offset
    end = total if limit is None else min(total, offset + limit)
    missing_records: List[Dict[str, Any]] = []

    for idx in range(start, end):
        item = items[idx]
        image_field = item.get("data", {}).get("image")
        rel = extract_rel_path(str(image_field) if image_field else "")

        def _try_paths(root: Path, rel_path: str) -> Optional[Path]:
            for cand in _rel_candidates(rel_path, root):
                p = root.joinpath(cand)
                if p.exists():
                    return p
                alt_rel = _tail_after_u_segment(cand)
                if alt_rel:
                    p2 = root.joinpath(alt_rel)
                    if p2.exists():
                        return p2
            return None

        img_path = None
        for r in roots:
            img_path = _try_paths(r, rel)
            if img_path is not None:
                break

        if img_path is None:
            msg = f"[WARN] 图像不存在，已尝试主/备root及U前缀 idx={idx} rel={rel}"
            print(msg, file=sys.stderr)
            missing_records.append(
                {
                    "idx": idx,
                    "rel": rel,
                    "image_field": image_field,
                }
            )
            continue
        try:
            ocr_result = ocr_caller(img_path)
            item["ocr_raw"] = ocr_result
            print(f"[{idx+1}/{end}] ok -> {rel}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] 调用 OCR 失败 idx={idx} path={img_path} err={exc}", file=sys.stderr)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if missing_log and missing_records:
        missing_log.parent.mkdir(parents=True, exist_ok=True)
        with missing_log.open("w", encoding="utf-8") as f:
            json.dump(missing_records, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Missing images logged to: {missing_log} ({len(missing_records)} records)")


def main():
    ap = argparse.ArgumentParser(description="从标注数据调用百度 OCR 并合并原始结果")
    ap.add_argument("--anno_json", required=True, type=Path, help="标注 JSON 路径")
    ap.add_argument("--image_root", required=True, type=Path, help="图片根目录，如 /data/ocean")
    ap.add_argument("--extra_root", type=Path, default=None, help="可选的第二图片根目录，主目录找不到时回退")
    ap.add_argument("--extra_root2", type=Path, default=None, help="可选的第三图片根目录，再次回退")
    ap.add_argument("--output", required=True, type=Path, help="输出合并后的 JSON 路径")
    ap.add_argument("--missing_log", type=Path, default=None, help="记录未找到图片的样本 JSON 路径；未指定则使用 output 同目录的 .missing.json")
    ap.add_argument("--api_key", help="百度 OCR API Key（官方 openapi 模式需要）")
    ap.add_argument("--secret_key", help="百度 OCR Secret Key（官方 openapi 模式需要）")
    ap.add_argument("--ocr_url", default=DEFAULT_OCR_URL, help="OCR 接口 URL，可换其他模型（官方模式）")
    ap.add_argument("--use_local_baidu", action="store_true", help="使用本地 baidu_ocr.py 封装的 .com 域名接口，不需要 token")
    ap.add_argument("--local_mode", default="accurate", help="传给 baidu_ocr.ocr 的 mode")
    ap.add_argument("--limit", type=int, default=0, help="最多处理条数，0 表示全量")
    ap.add_argument("--offset", type=int, default=0, help="起始索引（0-based）")
    ap.add_argument("--sleep", type=float, default=0.5, help="每次调用后的休眠秒数")
    args = ap.parse_args()

    items = load_annotations(args.anno_json)
    if args.use_local_baidu:
        def _caller_local(path: Path):
            b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
            return local_baidu_ocr(b64, mode=args.local_mode)
        ocr_caller = _caller_local
    else:
        if not args.api_key or not args.secret_key:
            raise SystemExit("必须提供 --api_key 与 --secret_key，或加 --use_local_baidu")
        token = get_access_token(args.api_key, args.secret_key)
        def _caller_openapi(path: Path):
            return call_baidu_ocr(args.ocr_url, token, path)
        ocr_caller = _caller_openapi
    limit = None if args.limit <= 0 else args.limit

    missing_log = args.missing_log
    if missing_log is None:
        missing_log = args.output.with_suffix(".missing.json")

    # 构建根目录列表，自动附加 ~/data/semi_struct/all_type_pic_oss_csv 若存在
    roots: List[Path] = [args.image_root]
    for r in (args.extra_root, args.extra_root2):
        if r is not None:
            roots.append(r)
    home_fallback = Path.home() / "data" / "semi_struct" / "all_type_pic_oss_csv"
    if home_fallback.exists():
        roots.append(home_fallback)

    process_items(
        items,
        roots=roots,
        ocr_caller=ocr_caller,
        limit=limit,
        offset=args.offset,
        sleep_seconds=args.sleep,
        missing_log=missing_log,
    )

    save_annotations(args.output, items)
    print(f"Done. Saved: {args.output}")


if __name__ == "__main__":
    main()
