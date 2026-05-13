#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

bash 00_check_env.sh
# shellcheck disable=SC1091
source config.env

py="python"

quick_file="$OUT_ROOT/quick/train_quick_${QUICK_LINES}.txt"
quick_ocr_json="$OUT_ROOT/quick/ocr_quick_${QUICK_OCR_DOCS}.json"
quick_ocr_text="$OUT_ROOT/quick/train_ocr_quick_${QUICK_OCR_DOCS}.txt"
mkdir -p "$OUT_ROOT/quick"

$py "$(pwd)/make_subset_corpus.py" \
  --input "$TRAIN_FILE" \
  --output "$quick_file" \
  --lines "$QUICK_LINES"

echo "[done] quick nonocr corpus: $quick_file"

# OCR quick subset (must preserve order)
$py "$(pwd)/make_subset_ocr_json.py" \
  --input "$OCR_JSON" \
  --output "$quick_ocr_json" \
  --n "$QUICK_OCR_DOCS"

$py "${PWD}/../../scripts/data/export_ocr_texts.py" \
  --ocr_json "$quick_ocr_json" \
  --output "$quick_ocr_text"

echo "[done] quick ocr json:  $quick_ocr_json"
echo "[done] quick ocr text:  $quick_ocr_text"
