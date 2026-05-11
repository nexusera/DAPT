#!/usr/bin/env python3
"""
切分并滑窗长文本，降低 512 截断的影响。

用法示例：
  python chunk_long_lines.py \
    --input /data/ocean/bpe_workspace/train.txt \
    --output /data/ocean/bpe_workspace/train_chunked.txt \
    --window 1000 \
    --stride 500

逻辑：
- 按行读取输入（假设每行一条样本）。
- 去掉首尾空白；空行跳过。
- 若行长 <= window，直接输出。
- 若行长 > window，按 window/stride 做字符级滑窗，生成多条行。
  （重叠保证长程信息覆盖，字符近似 token，简单落地）
"""

import argparse
import os


def process_line(line: str, window: int, stride: int):
    line = line.strip()
    if not line:
        return []
    n = len(line)
    if n <= window:
        return [line]
    chunks = []
    start = 0
    while start < n:
        end = start + window
        chunk = line[start:end]
        chunks.append(chunk)
        if end >= n:
            break
        start += stride
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="原始 txt（每行一条样本）")
    ap.add_argument("--output", required=True, help="输出切分/滑窗后的 txt")
    ap.add_argument("--window", type=int, default=1000, help="窗口长度（字符）")
    ap.add_argument("--stride", type=int, default=500, help="滑窗步长（字符）")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    total_in, total_out = 0, 0
    max_len_in, max_len_out = 0, 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            total_in += 1
            max_len_in = max(max_len_in, len(line.rstrip("\n")))
            chunks = process_line(line, args.window, args.stride)
            for c in chunks:
                fout.write(c + "\n")
                total_out += 1
                max_len_out = max(max_len_out, len(c))

    print(f"Done. input lines={total_in}, output lines={total_out}")
    print(f"max_len_in={max_len_in}, max_len_out={max_len_out}")
    print(f"window={args.window}, stride={args.stride}")


if __name__ == "__main__":
    main()

