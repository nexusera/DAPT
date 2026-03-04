#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

bash 00_check_env.sh
# shellcheck disable=SC1091
source config.env

py="python"
script="$(pwd)/build_jieba_dict.py"

# T1：不注入 OCR 词表时，建议只保留 keys_min5（以及 VIP），避免 confound
$py "$script" \
  --output "$OUT_ROOT/jieba/t1_base.txt" \
  --keys_vocab "$KEYS_MIN5_FILE"

# T2：keys-only
$py "$script" \
  --output "$OUT_ROOT/jieba/t2_keys.txt" \
  --keys_vocab "$KEYS_MIN5_FILE"

# T3：ocr_raw + keys
if [[ -f "$OCR_VOCAB_RAW" ]]; then
  $py "$script" \
    --output "$OUT_ROOT/jieba/t3_ocr_raw.txt" \
    --ocr_vocab "$OCR_VOCAB_RAW" \
    --keys_vocab "$KEYS_MIN5_FILE"
fi

# T4：ocr_kept + keys
$py "$script" \
  --output "$OUT_ROOT/jieba/t4_ocr_llm_keys.txt" \
  --ocr_vocab "$OCR_VOCAB_KEPT" \
  --keys_vocab "$KEYS_MIN5_FILE"

echo "[done] jieba dicts saved under $OUT_ROOT/jieba"
