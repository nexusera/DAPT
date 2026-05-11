#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
远端 / 本地统一探查：微调标注样本、评测 JSON、接口推理请求体格式。

避免在 SSH 里粘贴 20+ 行 shell/heredoc 导致截断或引号错误。

用法（远端）：
  cd /data/ocean/DAPT
  python3 scripts/inspect_kvbert_data_formats.py

  # 指定条目索引、训练集、E2E 保存的请求 JSON
  python3 scripts/inspect_kvbert_data_formats.py --index 0 --train_json biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json
  python3 scripts/inspect_kvbert_data_formats.py --e2e_request /tmp/kvbert_debug/Uxxx_request.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _trunc(s: str, n: int = 200) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "..."


def _jprint(obj: Any, max_chars: int = 2000) -> None:
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    print(s if len(s) <= max_chars else s[:max_chars] + "\n... [truncated]")


def _load_json_array(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层应为 JSON 数组")
    return data


def _data_block(item: Dict[str, Any]) -> Dict[str, Any]:
    d = item.get("data")
    return d if isinstance(d, dict) else {}


def summarize_finetune_item(item: Dict[str, Any], label: str, index: int, verbose: bool) -> None:
    """一条 Label Studio / 扁平导出的微调或评测样本。"""
    d = _data_block(item)
    text = str(d.get("ocr_text") or item.get("ocr_text") or "")
    ocr = d.get("ocr_raw") or item.get("ocr_raw") or {}
    if not isinstance(ocr, dict):
        ocr = {}
    wr = ocr.get("words_result") or []
    if not isinstance(wr, list):
        wr = []

    joined = "".join(str(w.get("words", "")) for w in wr if isinstance(w, dict))

    print(f"\n{'='*60}")
    print(f"{label}  [index={index}]")
    print(f"{'='*60}")
    print("顶层 keys:", sorted(item.keys()))
    print("data 子对象 keys:", sorted(d.keys()) if d else "(无 data 块，字段在顶层)")
    print("ocr_text 长度:", len(text), "| preview:", _trunc(text, 180))
    print("words_result 条数:", len(wr))
    print("''.join(words) 长度:", len(joined), "| 与 ocr_text 一致:", joined == text)

    if wr and isinstance(wr[0], dict):
        w0 = wr[0]
        print("首条 word keys:", sorted(w0.keys()))
        ch0 = (w0.get("chars") or [{}])[0] if isinstance(w0.get("chars"), list) else {}
        if isinstance(ch0, dict):
            print("首字 char keys:", sorted(ch0.keys()))
            print("首字含 probability:", "probability" in ch0)
        print("首条 word 含 probability:", "probability" in w0)

    nw = d.get("noise_values_per_word") or item.get("noise_values_per_word")
    if isinstance(nw, list):
        print("noise_values_per_word: 长度", len(nw), "| 首条 7 维:", nw[0] if nw else None)
    else:
        print("noise_values_per_word: 无")

    nv = d.get("noise_values") or item.get("noise_values")
    print("noise_values (文档级):", "有" if nv is not None else "无")

    for anno_key in ("transferred_annotations", "annotations"):
        block = item.get(anno_key)
        if block:
            print(f"{anno_key}: 长度 {len(block) if isinstance(block, list) else type(block)}")
            if verbose and isinstance(block, list) and block:
                print(f"  [0] 预览:")
                _jprint(block[0], max_chars=1200)

    if verbose and wr:
        print("\n--- words_result[0] 全文 ---")
        _jprint(wr[0], max_chars=2500)


def summarize_api_request(req: Dict[str, Any], label: str, verbose: bool) -> None:
    """POST /api/v1/extract 请求体。"""
    print(f"\n{'='*60}")
    print(label)
    print(f"{'='*60}")
    print("顶层 keys:", sorted(req.keys()))
    ot = req.get("ocr_text") or ""
    print("ocr_text 长度:", len(ot), "| preview:", _trunc(str(ot), 180))
    wr = req.get("words_result") or []
    if not isinstance(wr, list):
        wr = []
    print("words_result 条数:", len(wr))
    if wr and isinstance(wr[0], dict):
        w0 = wr[0]
        print("首条 keys:", sorted(w0.keys()))
        ch = w0.get("chars")
        print("首条含 chars:", isinstance(ch, list) and len(ch) > 0)
        if isinstance(ch, list) and ch and isinstance(ch[0], dict):
            print("chars[0] keys:", sorted(ch[0].keys()))
    joined = "".join(str(w.get("words", "")) for w in wr if isinstance(w, dict))
    print("''.join(words) 与 ocr_text 一致:", joined == ot)
    if verbose:
        print("\n--- 请求体预览 ---")
        _jprint(req, max_chars=3500)


def audit_finetune_consistency(path: Path, label: str, max_examples: int = 8) -> None:
    """
    扫描微调 JSON：ocr_text 必须与 ''.join(words_result[].words) 一致，
    否则字符级噪声 noise_values（由 words 展开）与 BERT 输入 ocr_text 错位。
    """
    if not path.is_file():
        print(f"WARN: audit 跳过（文件不存在）: {path}", file=sys.stderr)
        return
    data = _load_json_array(path)
    mismatches: List[tuple] = []
    for i, item in enumerate(data):
        d = _data_block(item)
        text = str(d.get("ocr_text") or item.get("ocr_text") or "")
        ocr = d.get("ocr_raw") or item.get("ocr_raw") or {}
        wr = ocr.get("words_result") if isinstance(ocr, dict) else None
        if not isinstance(wr, list):
            continue
        joined = "".join(str(w.get("words", "")) for w in wr if isinstance(w, dict))
        if text != joined:
            rid = item.get("record_id") or item.get("id")
            mismatches.append((i, len(text), len(joined), rid))

    print(f"\n{'='*60}\nAUDIT: {label}\n文件: {path}\n总条数: {len(data)}  |  ocr_text≠join(words): {len(mismatches)}")
    if mismatches:
        print("说明: 训练时 Sample.text 取 ocr_text，noise 由 words 展开；不一致会导致噪声与字符对不齐。")
        for row in mismatches[:max_examples]:
            print(f"  index={row[0]}  len(ocr_text)={row[1]}  len(join)={row[2]}  record_id={row[3]}")
        if len(mismatches) > max_examples:
            print(f"  ... 另有 {len(mismatches) - max_examples} 条未列出")


def main() -> int:
    parser = argparse.ArgumentParser(description="探查 KV-BERT 微调 JSON 与接口请求格式")
    parser.add_argument(
        "--dapt_root",
        type=Path,
        default=None,
        help="DAPT 仓库根目录，默认为本脚本上级目录的父级（.../DAPT）",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json",
        help="评测/微调测试集 JSON（相对 dapt_root）",
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default="",
        help="可选：训练集 JSON（相对 dapt_root）",
    )
    parser.add_argument(
        "--fixture",
        type=str,
        default="serving/tests/fixtures/sample_ocr_request.json",
        help="接口示例请求（相对 dapt_root）",
    )
    parser.add_argument(
        "--e2e_request",
        type=Path,
        default=None,
        help="可选：e2e_pipeline_test.py --save_dir 保存的 *_request.json 绝对或相对路径",
    )
    parser.add_argument("--index", type=int, default=0, help="微调 JSON 中查看的样本下标")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印 annotations / 首条 word 全文")
    parser.add_argument(
        "--audit",
        action="store_true",
        help="扫描 test_json（及 train_json 若指定）中 ocr_text 与 join(words) 是否一致",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    dapt_root = args.dapt_root or script_dir.parent
    if not dapt_root.is_dir():
        print(f"ERROR: dapt_root 不存在: {dapt_root}", file=sys.stderr)
        return 1

    test_path = dapt_root / args.test_json
    if args.audit and test_path.is_file():
        audit_finetune_consistency(test_path, "评测/测试集")

    if not test_path.is_file():
        print(f"WARN: 未找到测试集: {test_path}", file=sys.stderr)
    else:
        data = _load_json_array(test_path)
        if not data:
            print(f"WARN: {test_path} 为空数组", file=sys.stderr)
        else:
            idx = max(0, min(args.index, len(data) - 1))
            if args.index >= len(data):
                print(f"WARN: index={args.index} 超出范围，改用 {idx}", file=sys.stderr)
            summarize_finetune_item(data[idx], f"微调/评测样本: {test_path.name}", idx, args.verbose)

    if args.train_json.strip():
        train_path = dapt_root / args.train_json
        if args.audit and train_path.is_file():
            audit_finetune_consistency(train_path, "训练集")
        if train_path.is_file():
            data = _load_json_array(train_path)
            idx = max(0, min(args.index, len(data) - 1))
            summarize_finetune_item(data[idx], f"训练样本: {train_path.name}", idx, args.verbose)
        else:
            print(f"WARN: 未找到训练集: {train_path}", file=sys.stderr)

    fix_path = dapt_root / args.fixture
    if fix_path.is_file():
        req = json.loads(fix_path.read_text(encoding="utf-8"))
        summarize_api_request(req, f"接口示例请求: {fix_path}", args.verbose)
    else:
        print(f"WARN: 未找到 fixture: {fix_path}", file=sys.stderr)

    if args.e2e_request:
        p = args.e2e_request
        if not p.is_file():
            p = dapt_root / p
        if p.is_file():
            req = json.loads(p.read_text(encoding="utf-8"))
            summarize_api_request(req, f"E2E 保存请求: {p}", args.verbose)
        else:
            print(f"WARN: 未找到 e2e_request: {args.e2e_request}", file=sys.stderr)

    print(f"\n完成。DAPT_ROOT={dapt_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
