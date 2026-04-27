#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M15: CI 级别的 OCR↔数据集对齐不变量检查。

把"对齐不变量"写进可自动化执行的检查，防止 ocr_text 与 ''.join(words) 静默漂移。
最近 4 次 fix（78476fa、f01f729、fcf1d61、54ac0a3）均源于此类漂移，本脚本是守护线。

不变量列表
──────────────────────────────────────────────────────────────────────────────
INV-1: ocr_text == ''.join(w['words'] for w in words_result)
INV-2: len(noise_values) == len(ocr_text)（若字段存在）
INV-3: len(noise_values[i]) == 7                （若字段存在）
INV-4: noise_values 无 NaN/inf
INV-5: 每条 words_result 条目必须含 'words' 字段

用法（远端）：
  cd /data/ocean/DAPT
  python3 scripts/ci_check_ocr_alignment.py \\
      --files biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json \\
              biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json

  # CI 模式：任何违例以非 0 退出码退出
  python3 scripts/ci_check_ocr_alignment.py --files <json> --strict
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return [data]


def _check_item(
    item: Dict[str, Any],
    idx: int,
    file_tag: str,
) -> List[str]:
    """返回该条目的所有违例描述列表（空表示通过）。"""
    violations: List[str] = []
    prefix = f"[{file_tag}][#{idx}]"

    ocr_text: Optional[str] = item.get("ocr_text") or item.get("text")
    words_result: Optional[List[Any]] = item.get("words_result") or (
        item.get("ocr_raw", {}) or {}
    ).get("words_result")
    noise_values: Optional[List[Any]] = item.get("noise_values")

    # ── INV-1: ocr_text == ''.join(words) ────────────────────────────────────
    if ocr_text is not None and words_result is not None:
        joined = "".join(
            str(w.get("words", ""))
            for w in words_result
            if isinstance(w, dict)
        )
        if joined != ocr_text:
            # 仅报告前 80 字符差异
            ot_s = ocr_text[:80].replace("\n", "↵")
            jn_s = joined[:80].replace("\n", "↵")
            violations.append(
                f"{prefix} INV-1: ocr_text[:{len(ocr_text)}] != join(words)[:{len(joined)}]\n"
                f"  ocr_text[:80]={ot_s!r}\n"
                f"  join[:80]    ={jn_s!r}"
            )

    # ── INV-5: words 字段存在性 ───────────────────────────────────────────────
    if words_result is not None:
        for wi, w in enumerate(words_result):
            if not isinstance(w, dict) or "words" not in w:
                violations.append(
                    f"{prefix} INV-5: words_result[{wi}] 缺少 'words' 字段，值={w!r:.120}"
                )
                break  # 只报第一处

    # ── INV-2/3/4: noise_values 检查 ─────────────────────────────────────────
    if noise_values is not None and ocr_text is not None:
        n = len(noise_values)
        expected = len(ocr_text)
        if n != expected:
            violations.append(
                f"{prefix} INV-2: len(noise_values)={n} != len(ocr_text)={expected}"
            )
        for i, row in enumerate(noise_values):
            if not isinstance(row, (list, tuple)):
                violations.append(f"{prefix} INV-3: noise_values[{i}] 不是列表: {row!r}")
                break
            if len(row) != 7:
                violations.append(
                    f"{prefix} INV-3: noise_values[{i}] 长度={len(row)} != 7"
                )
                break
            for j, v in enumerate(row):
                if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                    violations.append(
                        f"{prefix} INV-4: noise_values[{i}][{j}]={v!r} 包含 NaN/inf"
                    )
                    break
            else:
                continue
            break

    return violations


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="M15: OCR↔数据集对齐不变量 CI 检查"
    )
    parser.add_argument(
        "--files", nargs="+", required=True,
        help="待检查的标注 JSON 文件路径（支持多个）",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="任何违例以退出码 1 退出（CI 模式）",
    )
    parser.add_argument(
        "--max_violations", type=int, default=20,
        help="最多打印多少条违例（默认 20，避免刷屏）",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="只检查前 N 条（0 = 全部）",
    )
    args = parser.parse_args()

    total_items = 0
    total_violations: List[str] = []

    for file_path in args.files:
        p = Path(file_path)
        if not p.exists():
            print(f"[WARN] 文件不存在，跳过: {p}", file=sys.stderr)
            continue

        try:
            items = _load_json_list(p)
        except Exception as exc:
            print(f"[ERROR] 读取 {p} 失败: {exc}", file=sys.stderr)
            if args.strict:
                return 1
            continue

        tag = p.name
        check_items = items[: args.sample] if args.sample > 0 else items
        print(f"[INFO] 检查 {tag}：共 {len(check_items)} 条 / {len(items)} 条（total）")

        for idx, item in enumerate(check_items):
            viols = _check_item(item, idx, tag)
            total_violations.extend(viols)
        total_items += len(check_items)

    # ── 输出结果 ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"检查总数: {total_items} 条  |  违例总数: {len(total_violations)}")
    if total_violations:
        shown = total_violations[: args.max_violations]
        for v in shown:
            print(f"  ✗ {v}")
        if len(total_violations) > args.max_violations:
            print(f"  ... 还有 {len(total_violations) - args.max_violations} 条违例未显示")
        print(f"\n{'─'*60}")
        if args.strict:
            print("[FAIL] 存在对齐违例，CI 检查未通过。")
            return 1
        print("[WARN] 存在对齐违例，请修复后再提交。")
        return 0
    else:
        print("[PASS] 所有对齐不变量检查通过。")
        return 0


if __name__ == "__main__":
    sys.exit(main())
