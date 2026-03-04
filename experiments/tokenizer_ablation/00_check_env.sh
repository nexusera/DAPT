#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f config.env ]]; then
  echo "config.env 不存在：请先 cp config.env.example config.env 并修改路径" >&2
  exit 1
fi

# shellcheck disable=SC1091
source config.env

req_files=("$KEYS_ONLY_FILE" "$KEYS_MIN5_FILE" "$TRAIN_FILE" "$NOISE_BINS_JSON")

for f in "${req_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "缺少必需文件：$f" >&2
    exit 1
  fi
done

if [[ ! -f "$OCR_JSON" ]]; then
  echo "缺少必需文件：OCR_JSON=$OCR_JSON" >&2
  exit 1
fi

# OCR_TEXT_FILE 不强制要求预先存在：30_build_datasets.sh 会在缺失时自动从 OCR_JSON 导出

if [[ ! -f "$OCR_VOCAB_RAW" ]]; then
  echo "提示：OCR_VOCAB_RAW 不存在：$OCR_VOCAB_RAW" >&2
fi
if [[ ! -f "$OCR_VOCAB_KEPT" ]]; then
  echo "提示：OCR_VOCAB_KEPT 不存在：$OCR_VOCAB_KEPT" >&2
fi

mkdir -p \
  "$OUT_ROOT" \
  "$OUT_ROOT/tokenizers" \
  "$OUT_ROOT/jieba" \
  "$OUT_ROOT/datasets" \
  "$OUT_ROOT/datasets/nonocr" \
  "$OUT_ROOT/datasets/ocr" \
  "$OUT_ROOT/datasets_quick" \
  "$OUT_ROOT/logs" \
  "$OUT_ROOT/quick"

python -V

echo "OK"
