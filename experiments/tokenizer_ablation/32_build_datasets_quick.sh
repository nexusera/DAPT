#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

bash 00_check_env.sh
# shellcheck disable=SC1091
source config.env

root="$(cd ../.. && pwd)" # DAPT/
py="python"

quick_file="$OUT_ROOT/quick/train_quick_${QUICK_LINES}.txt"
if [[ ! -f "$quick_file" ]]; then
  echo "quick 语料不存在：$quick_file，请先跑 31_make_quick_corpus.sh" >&2
  exit 1
fi

shuffle_flag="--shuffle_split"
if [[ "${SHUFFLE_SPLIT}" == "0" ]]; then
  shuffle_flag="--no_shuffle_split"
fi

build_one () {
  local exp="$1"
  local tok_dir="$2"
  local jieba_dict="$3"
  local out_ds="$4"

  echo "[build-quick] $exp -> $out_ds"
  $py "$root/build_dataset_final_slim.py" \
    --train_file "$quick_file" \
    --output_path "$out_ds" \
    --tokenizer_path "$tok_dir" \
    --keys_file "$KEYS_MIN5_FILE" \
    --vocab_for_jieba "$jieba_dict" \
    --max_len "$MAX_LEN" \
    --batch_size "$BATCH_SIZE" \
    --num_proc "$NUM_PROC" \
    $shuffle_flag
}

build_one "T1" "$OUT_ROOT/tokenizers/t1_base" "$OUT_ROOT/jieba/t1_base.txt" "$OUT_ROOT/datasets_quick/processed_dataset_t1"
build_one "T2" "$OUT_ROOT/tokenizers/t2_keys" "$OUT_ROOT/jieba/t2_keys.txt" "$OUT_ROOT/datasets_quick/processed_dataset_t2"

if [[ -d "$OUT_ROOT/tokenizers/t3_ocr_raw" && -f "$OUT_ROOT/jieba/t3_ocr_raw.txt" ]]; then
  build_one "T3" "$OUT_ROOT/tokenizers/t3_ocr_raw" "$OUT_ROOT/jieba/t3_ocr_raw.txt" "$OUT_ROOT/datasets_quick/processed_dataset_t3"
fi

build_one "T4" "$OUT_ROOT/tokenizers/t4_ocr_llm_keys" "$OUT_ROOT/jieba/t4_ocr_llm_keys.txt" "$OUT_ROOT/datasets_quick/processed_dataset_t4"

echo "[done] quick datasets saved under $OUT_ROOT/datasets_quick"
