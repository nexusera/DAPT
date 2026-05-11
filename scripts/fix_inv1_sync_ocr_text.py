#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INV-1 修复：将 JSON 文件中的 ocr_text 同步为 ''.join(words_result[].words)。

为什么不需要重新调 OCR API：
  words_result 已包含正确的 OCR 结果。INV-1 根因是 ocr_text 在历史管线中
  未被同步更新（旧值来自标注平台导出或更早的 OCR 运行）。

为什么不影响噪声特征：
  noise_values_per_word 按词索引，_expand_word_noise_to_chars 用
  len(words_result[i].words) 展开到字符，与 join(words) 对齐。
  同步 ocr_text 后 len(text) == len(expanded_noise)，错位消除。

注意：
  - 标注 span（start/end）基于旧 ocr_text，中文 span 不受影响；
    英文单词内的空格差异可能导致纯英文实体 span 略微偏移（概率极低）。
  - 旧值自动备份到 ocr_text_original 字段，可随时回滚。

用法（远端）：
  cd /data/ocean/DAPT

  # 预览：只统计会修改多少条，不写文件
  python3 scripts/fix_inv1_sync_ocr_text.py \\
      --files biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json \\
              biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \\
      --dry_run

  # 实际修复（原地覆盖，旧值备份到 ocr_text_original）
  python3 scripts/fix_inv1_sync_ocr_text.py \\
      --files biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json \\
              biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json

  # 修复后验证
  python3 scripts/ci_check_ocr_alignment.py \\
      --files biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json \\
              biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \\
      --strict
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ── 核心同步逻辑 ──────────────────────────────────────────────────────────────

def _get_words_result(item: Dict[str, Any]) -> List[Dict[str, Any]] | None:
    """从 item 中找到 words_result，兼容顶层和 ocr_raw 嵌套两种格式。"""
    wr = item.get("words_result")
    if isinstance(wr, list):
        return wr
    ocr_raw = item.get("ocr_raw")
    if isinstance(ocr_raw, dict):
        wr = ocr_raw.get("words_result")
        if isinstance(wr, list):
            return wr
    return None


def _sync_item(item: Dict[str, Any]) -> Tuple[bool, str]:
    """
    同步单条记录的 ocr_text。

    返回 (changed, reason)：
      - (True, "") 表示已修改
      - (False, reason) 表示未修改，reason 说明原因
    """
    ocr_text = item.get("ocr_text")
    words_result = _get_words_result(item)

    if words_result is None:
        return False, "no words_result"

    joined = "".join(
        str(w.get("words", "")) for w in words_result if isinstance(w, dict)
    )

    if not joined:
        return False, "empty join(words)"

    if ocr_text == joined:
        return False, "already equal"

    # 备份旧值（只在第一次修复时备份，避免覆盖已有备份）
    if "ocr_text_original" not in item:
        item["ocr_text_original"] = ocr_text
    item["ocr_text"] = joined
    return True, ""


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="INV-1 修复：将 ocr_text 同步为 join(words_result[].words)"
    )
    parser.add_argument(
        "--files", nargs="+", required=True,
        help="待修复的 JSON 文件路径（支持多个）",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="只统计需要修改的条数，不写文件",
    )
    parser.add_argument(
        "--no_backup", action="store_true",
        help="覆盖前不创建 .bak 备份文件（默认会备份）",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_fixed = 0
    total_skipped = 0

    for file_path in args.files:
        p = Path(file_path)
        if not p.exists():
            print(f"[WARN] 文件不存在，跳过: {p}", file=sys.stderr)
            continue

        print(f"\n[INFO] 处理: {p.name}")
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[ERROR] 读取失败: {exc}", file=sys.stderr)
            continue

        items: List[Dict[str, Any]] = data if isinstance(data, list) else [data]

        fixed = 0
        skip_reasons: Dict[str, int] = {}

        for item in items:
            changed, reason = _sync_item(item)
            if changed:
                fixed += 1
            else:
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

        total_fixed += fixed
        total_skipped += sum(skip_reasons.values())

        print(f"  需修改: {fixed} 条")
        for reason, cnt in sorted(skip_reasons.items()):
            print(f"  跳过（{reason}）: {cnt} 条")

        if args.dry_run:
            print("  [dry_run] 不写文件。")
            continue

        if fixed == 0:
            print("  无需修改，跳过写入。")
            continue

        # 备份原文件
        if not args.no_backup:
            bak_path = p.with_suffix(f".{ts}.bak.json")
            shutil.copy2(p, bak_path)
            print(f"  已备份至: {bak_path.name}")

        # 写回
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  已写入: {p.name}（修复 {fixed} 条）")

    print(f"\n{'─'*60}")
    if args.dry_run:
        print(f"[dry_run] 合计需修复: {total_fixed} 条，无需修改: {total_skipped} 条")
    else:
        print(f"[完成] 合计修复: {total_fixed} 条，无需修改: {total_skipped} 条")
        if total_fixed > 0:
            print(
                "\n[下一步] 运行 CI 验证：\n"
                "  python3 scripts/ci_check_ocr_alignment.py \\\n"
                f"      --files {' '.join(args.files)} \\\n"
                "      --strict"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
