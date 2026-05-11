#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

bash 00_check_env.sh
# shellcheck disable=SC1091
source config.env

py="python"
script="$(pwd)/build_jieba_dict.py"

shared_dict="$OUT_ROOT/jieba/shared_kept_keys_min5.txt"

# 共享 Jieba 词典（一次生成，多处复用）：
# - VIP 基础词表（build_jieba_dict.py 内置）
# - keys_min5
# - OCR kept vocab（筛选后的 wordpiece 结果）
$py "$script" \
  --output "$shared_dict" \
  --ocr_vocab "$OCR_VOCAB_KEPT" \
  --keys_vocab "$KEYS_MIN5_FILE"

echo "[done] shared jieba dict saved: $shared_dict"
