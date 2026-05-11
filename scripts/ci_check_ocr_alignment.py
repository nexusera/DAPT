#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M15: CI 级别的 OCR↔数据集对齐不变量检查。

不变量说明（两种 noise_values 格式自动区分）
───────────────────────────────────────────────────────────
训练数据格式（compute_noise_from_ocr.py 输出）:
  noise_values       = [float]*7  ← 文档级摘要，7 个浮点数
  noise_values_per_word = [[float]*7]*num_words ← 词级，可选

服务层请求格式（serving/schemas/request.py）:
  noise_values = [[float]*7]*N  ← 逐字符，N=len(ocr_text)

检查的不变量
  INV-1 : ocr_text == ''.join(w['words'] for w in words_result)
           （发现不等时同时报告两者长度差和前 80 字符对比，便于定位）
  INV-2a: 训练格式 → len(noise_values) == 7（文档级 7 维）
  INV-2b: 服务格式 → len(noise_values) == len(ocr_text)（逐字符）
  INV-3 : 服务格式 → 每行长度 == 7
  INV-4 : noise_values 所有数值无 NaN/inf
  INV-5 : words_result 每条必须含 'words' 字段
  INV-6 : noise_values_per_word（如存在）行数 == len(words_result)

用法（远端）：
  cd /data/ocean/DAPT
  # 检查训练数据
  python3 scripts/ci_check_ocr_alignment.py \\
      --files biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json \\
              biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \\
      --strict

  # 仅检查 INV-1 对齐（忽略噪声格式差异）
  python3 scripts/ci_check_ocr_alignment.py --files <json> --inv1_only

  # 跳过 INV-1（只检查噪声维度）
  python3 scripts/ci_check_ocr_alignment.py --files <json> --skip_inv1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── 格式检测 ──────────────────────────────────────────────────────────────────

def _detect_noise_format(noise_values: Any) -> str:
    """
    返回 'doc'（训练数据 7 维文档级）、'char'（服务层逐字符）或 'unknown'。

    区分逻辑：
      - 若第一个元素是 float/int → doc（[float]*7）
      - 若第一个元素是 list/tuple → char（[[float]*7]*N）
    """
    if not isinstance(noise_values, (list, tuple)) or len(noise_values) == 0:
        return "unknown"
    first = noise_values[0]
    if isinstance(first, (int, float)):
        return "doc"
    if isinstance(first, (list, tuple)):
        return "char"
    return "unknown"


# ── 核心检查 ──────────────────────────────────────────────────────────────────

def _check_item(
    item: Dict[str, Any],
    idx: int,
    file_tag: str,
    skip_inv1: bool = False,
    inv1_only: bool = False,
) -> List[Tuple[str, str]]:
    """
    返回该条目的所有违例列表，每个元素为 (inv_code, message)。
    空列表表示全部通过。
    """
    violations: List[Tuple[str, str]] = []
    prefix = f"[{file_tag}][#{idx}]"

    ocr_text: Optional[str] = item.get("ocr_text") or item.get("text")
    words_result: Optional[List[Any]] = item.get("words_result") or (
        (item.get("ocr_raw") or {}).get("words_result")
    )
    noise_values: Optional[Any] = item.get("noise_values")
    noise_per_word: Optional[Any] = item.get("noise_values_per_word")

    # ── INV-5: words 字段存在性 ───────────────────────────────────────────────
    if words_result is not None and not inv1_only:
        for wi, w in enumerate(words_result):
            if not isinstance(w, dict) or "words" not in w:
                violations.append((
                    "INV-5",
                    f"{prefix} INV-5: words_result[{wi}] 缺少 'words' 字段，值={str(w)[:80]!r}",
                ))
                break

    # ── INV-1: ocr_text == join(words) ───────────────────────────────────────
    if not skip_inv1 and ocr_text is not None and words_result is not None:
        joined = "".join(
            str(w.get("words", ""))
            for w in words_result
            if isinstance(w, dict)
        )
        if joined != ocr_text:
            # 计算不一致首位
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(ocr_text, joined)) if a != b),
                min(len(ocr_text), len(joined)),
            )
            violations.append((
                "INV-1",
                (
                    f"{prefix} INV-1: ocr_text({len(ocr_text)}字) ≠ join(words)({len(joined)}字)，"
                    f"首个差异位 idx={first_diff}\n"
                    f"  ocr_text[:80] ={ocr_text[:80].replace(chr(10),'↵')!r}\n"
                    f"  join(words)[:80]={joined[:80].replace(chr(10),'↵')!r}"
                ),
            ))

    if inv1_only:
        return violations

    # ── noise_values 检查 ─────────────────────────────────────────────────────
    if noise_values is not None:
        fmt = _detect_noise_format(noise_values)

        if fmt == "doc":
            # ── INV-2a: 文档级格式 → 必须正好 7 个浮点数 ────────────────────
            if len(noise_values) != 7:
                violations.append((
                    "INV-2a",
                    f"{prefix} INV-2a: 文档级 noise_values 长度={len(noise_values)}，应为 7",
                ))
            # INV-4: 无 NaN/inf
            for j, v in enumerate(noise_values):
                if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                    violations.append((
                        "INV-4",
                        f"{prefix} INV-4: noise_values[{j}]={v!r} 包含 NaN/inf",
                    ))

        elif fmt == "char":
            # ── INV-2b: 逐字符格式 → 长度 == len(ocr_text) ──────────────────
            if ocr_text is not None and len(noise_values) != len(ocr_text):
                violations.append((
                    "INV-2b",
                    f"{prefix} INV-2b: 逐字符 noise_values 长度={len(noise_values)} "
                    f"!= len(ocr_text)={len(ocr_text) if ocr_text else 'N/A'}",
                ))
            # ── INV-3: 每行长度 == 7 ─────────────────────────────────────────
            for i, row in enumerate(noise_values):
                if not isinstance(row, (list, tuple)):
                    violations.append((
                        "INV-3",
                        f"{prefix} INV-3: noise_values[{i}] 不是列表: {str(row)[:40]!r}",
                    ))
                    break
                if len(row) != 7:
                    violations.append((
                        "INV-3",
                        f"{prefix} INV-3: noise_values[{i}] 长度={len(row)} != 7",
                    ))
                    break
                # INV-4: 无 NaN/inf
                for j, v in enumerate(row):
                    if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                        violations.append((
                            "INV-4",
                            f"{prefix} INV-4: noise_values[{i}][{j}]={v!r} 包含 NaN/inf",
                        ))
                        break

        else:
            violations.append((
                "INV-2?",
                f"{prefix} INV-2?: noise_values 格式无法识别，首元素类型={type(noise_values[0]).__name__}",
            ))

    # ── INV-6: noise_values_per_word 行数 == len(words_result) ───────────────
    if noise_per_word is not None and words_result is not None:
        if len(noise_per_word) != len(words_result):
            violations.append((
                "INV-6",
                f"{prefix} INV-6: noise_values_per_word 行数={len(noise_per_word)} "
                f"!= len(words_result)={len(words_result)}",
            ))

    return violations


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="M15: OCR↔数据集对齐不变量 CI 检查（自动识别训练/服务层 noise 格式）"
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
        "--max_violations", type=int, default=30,
        help="最多打印多少条违例（默认 30）",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="只检查前 N 条（0 = 全部）",
    )
    parser.add_argument(
        "--skip_inv1", action="store_true",
        help="跳过 INV-1 ocr_text 对齐检查（仅检查 noise 格式）",
    )
    parser.add_argument(
        "--inv1_only", action="store_true",
        help="只检查 INV-1 ocr_text 对齐，跳过所有 noise 检查",
    )
    args = parser.parse_args()

    total_items = 0
    inv_counters: Dict[str, int] = {}
    all_violations: List[Tuple[str, str]] = []

    for file_path in args.files:
        p = Path(file_path)
        if not p.exists():
            print(f"[WARN] 文件不存在，跳过: {p}", file=sys.stderr)
            continue

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            items: List[Dict[str, Any]] = (
                data if isinstance(data, list)
                else data.get("data", [data])
            )
        except Exception as exc:
            print(f"[ERROR] 读取 {p} 失败: {exc}", file=sys.stderr)
            if args.strict:
                return 1
            continue

        check_items = items[: args.sample] if args.sample > 0 else items
        print(f"[INFO] 检查 {p.name}：{len(check_items)} 条（共 {len(items)} 条）")

        # 检测整个文件的 noise 格式（取第一个有 noise_values 的样本）
        sample_fmt = "unknown"
        for it in check_items[:10]:
            nv = it.get("noise_values")
            if nv is not None:
                sample_fmt = _detect_noise_format(nv)
                break
        print(f"  → noise_values 格式检测: {sample_fmt}")

        for idx, item in enumerate(check_items):
            viols = _check_item(
                item, idx, p.name,
                skip_inv1=args.skip_inv1,
                inv1_only=args.inv1_only,
            )
            for code, msg in viols:
                inv_counters[code] = inv_counters.get(code, 0) + 1
            all_violations.extend(viols)

        total_items += len(check_items)

    # ── 输出结果 ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"检查总数: {total_items} 条  |  违例总数: {len(all_violations)}")
    if inv_counters:
        print("  各类违例条数：")
        for code, cnt in sorted(inv_counters.items()):
            print(f"    {code}: {cnt}")

    if all_violations:
        print(f"\n── 违例详情（最多显示 {args.max_violations} 条）──")
        for code, msg in all_violations[: args.max_violations]:
            print(f"  ✗ {msg}")
        if len(all_violations) > args.max_violations:
            print(f"  ... 还有 {len(all_violations) - args.max_violations} 条违例未显示")
        print(f"\n{'─'*60}")

        # INV-1 违例说明
        if inv_counters.get("INV-1", 0) > 0:
            print(
                "\n[提示] INV-1 违例：ocr_text 与 join(words) 不一致。\n"
                "  根因：旧数据未同步 OCR（fetch_and_merge_baidu_ocr.py 修复前的存量）。\n"
                "  修复方案：重新运行 fetch_and_merge_baidu_ocr.py（不带 --no_sync_ocr_text），\n"
                "  或用 add_noise_features.py 重建数据集。\n"
                "  参考文档：docs/OCR_TEXT_AND_NOISE_ALIGNMENT.md"
            )

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
