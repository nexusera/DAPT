#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

bash 00_check_env.sh
# shellcheck disable=SC1091
source config.env

py="python"
script="$(pwd)/build_tokenizer_variant.py"

# T1: snapshot base
$py "$script" \
  --base_tokenizer "$BASE_TOKENIZER" \
  --output_dir "$OUT_ROOT/tokenizers/t1_base"

# T2: + keys_only
$py "$script" \
  --base_tokenizer "$BASE_TOKENIZER" \
  --keys_vocab "$KEYS_ONLY_FILE" \
  --output_dir "$OUT_ROOT/tokenizers/t2_keys"

# T3: + ocr_raw
if [[ -f "$OCR_VOCAB_RAW" ]]; then
  $py "$script" \
    --base_tokenizer "$BASE_TOKENIZER" \
    --ocr_vocab "$OCR_VOCAB_RAW" \
    --output_dir "$OUT_ROOT/tokenizers/t3_ocr_raw"
else
  echo "跳过 T3：找不到 OCR_VOCAB_RAW=$OCR_VOCAB_RAW" >&2
fi

# T4: + ocr_kept + keys_only
if [[ -f "$OCR_VOCAB_KEPT" ]]; then
  $py "$script" \
    --base_tokenizer "$BASE_TOKENIZER" \
    --ocr_vocab "$OCR_VOCAB_KEPT" \
    --keys_vocab "$KEYS_ONLY_FILE" \
    --output_dir "$OUT_ROOT/tokenizers/t4_ocr_llm_keys"
else
  echo "跳过 T4：找不到 OCR_VOCAB_KEPT=$OCR_VOCAB_KEPT" >&2
  exit 1
fi

echo "[done] tokenizers saved under $OUT_ROOT/tokenizers"
