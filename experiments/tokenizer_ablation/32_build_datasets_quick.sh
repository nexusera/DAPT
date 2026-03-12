#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

bash 00_check_env.sh
# shellcheck disable=SC1091
source config.env

root="$(cd ../.. && pwd)" # DAPT/
py=(python -u)

build_words_py="$(pwd)/build_words_dataset_from_jieba.py"
tokenize_words_py="$(pwd)/tokenize_words_dataset.py"

shared_jieba_dict="$OUT_ROOT/jieba/shared_kept_keys_min5.txt"

quick_file="$OUT_ROOT/quick/train_quick_${QUICK_LINES}.txt"
quick_ocr_json="$OUT_ROOT/quick/ocr_quick_${QUICK_OCR_DOCS}.json"
quick_ocr_text="$OUT_ROOT/quick/train_ocr_quick_${QUICK_OCR_DOCS}.txt"
if [[ ! -f "$quick_file" ]]; then
  echo "quick 语料不存在：$quick_file，请先跑 31_make_quick_corpus.sh" >&2
  exit 1
fi
if [[ ! -f "$quick_ocr_json" || ! -f "$quick_ocr_text" ]]; then
  echo "quick OCR 子集不存在：$quick_ocr_json / $quick_ocr_text，请先跑 31_make_quick_corpus.sh" >&2
  exit 1
fi

shuffle_flag="--shuffle_split"
if [[ "${SHUFFLE_SPLIT}" == "0" ]]; then
  shuffle_flag="--no_shuffle_split"
fi

merge_shuffle_flag=""
if [[ "${MERGE_SHUFFLE}" == "1" ]]; then
  merge_shuffle_flag="--shuffle"
fi

build_nonocr_one () {
  local exp="$1"
  local tok_dir="$2"
  local words_ds="$3"
  local out_ds="$4"

  echo "[build-quick-nonocr] $exp -> $out_ds"
  "${py[@]}" "$tokenize_words_py" \
    --words_dataset "$words_ds" \
    --output_path "$out_ds" \
    --tokenizer_path "$tok_dir" \
    --max_len "$MAX_LEN" \
    --batch_size "$BATCH_SIZE" \
    --num_proc "$NUM_PROC"
}

build_ocr_one () {
  local exp="$1"
  local tok_dir="$2"
  local words_ds="$3"
  local out_ds_plain="$4"
  local out_ds_noise="$5"

  echo "[build-quick-ocr] $exp -> $out_ds_plain (no shuffle)"
  "${py[@]}" "$tokenize_words_py" \
    --words_dataset "$words_ds" \
    --output_path "$out_ds_plain" \
    --tokenizer_path "$tok_dir" \
    --max_len "$MAX_LEN" \
    --batch_size "$BATCH_SIZE" \
    --num_proc "$NUM_PROC"

  echo "[noise-quick] $exp -> $out_ds_noise"
  "${py[@]}" "$root/add_noise_features.py" \
    --dataset "$out_ds_plain" \
    --output "$out_ds_noise" \
    --ocr_json "$quick_ocr_json" \
    --bins_json "$NOISE_BINS_JSON" \
    --num_proc "$NUM_PROC"

  echo "[verify-quick] $exp alignment check"
  "${py[@]}" "$root/verify_noise_alignment.py" \
    --dataset "$out_ds_noise" \
    --ocr_json "$quick_ocr_json" \
    --check_samples "$ALIGN_CHECK_SAMPLES" \
    --tokenizer "$tok_dir" || true
}

merge_one () {
  local exp="$1"
  local ocr_ds_noise="$2"
  local nonocr_ds="$3"
  local out_merged="$4"

  echo "[merge-quick] $exp -> $out_merged"
  "${py[@]}" "$root/merge_datasets.py" \
    --ocr_dataset "$ocr_ds_noise" \
    --non_ocr_dataset "$nonocr_ds" \
    --output_path "$out_merged" \
    --ocr_repeat "$OCR_REPEAT" \
    --non_ocr_repeat "$NONOCR_REPEAT" \
    --seed "$MERGE_SEED" \
    $merge_shuffle_flag
}

build_all () {
  local exp="$1"
  local tok_dir="$2"
  local nonocr_words_ds="$3"
  local ocr_words_ds="$4"
  local exp_lc
  exp_lc="$(echo "$exp" | tr '[:upper:]' '[:lower:]')"
  local out_nonocr="$OUT_ROOT/datasets_quick/nonocr/processed_dataset_${exp_lc}"
  local out_ocr_plain="$OUT_ROOT/datasets_quick/ocr/processed_dataset_${exp_lc}_plain"
  local out_ocr_noise="$OUT_ROOT/datasets_quick/ocr/processed_dataset_${exp_lc}_with_noise"
  local out_merged="$OUT_ROOT/datasets_quick/processed_dataset_${exp_lc}"

  mkdir -p "$OUT_ROOT/datasets_quick/nonocr" "$OUT_ROOT/datasets_quick/ocr"
  build_nonocr_one "$exp" "$tok_dir" "$nonocr_words_ds" "$out_nonocr"
  build_ocr_one "$exp" "$tok_dir" "$ocr_words_ds" "$out_ocr_plain" "$out_ocr_noise"
  merge_one "$exp" "$out_ocr_noise" "$out_nonocr" "$out_merged"
}

if [[ ! -f "$shared_jieba_dict" ]]; then
  echo "共享 Jieba 词典不存在：$shared_jieba_dict，请先跑 20_make_jieba_dicts.sh" >&2
  exit 1
fi

nonocr_words_ds="$OUT_ROOT/datasets_words_quick/nonocr_words"
ocr_words_ds="$OUT_ROOT/datasets_words_quick/ocr_words"

echo "[words-quick] build shared words dataset (non-OCR) -> $nonocr_words_ds (shuffle_split=${SHUFFLE_SPLIT})"
"${py[@]}" "$build_words_py" \
  --train_file "$quick_file" \
  --output_path "$nonocr_words_ds" \
  --jieba_userdict "$shared_jieba_dict" \
  --batch_size "$BATCH_SIZE" \
  --num_proc "$NUM_PROC" \
  $shuffle_flag

echo "[words-quick] build shared words dataset (OCR) -> $ocr_words_ds (no shuffle)"
"${py[@]}" "$build_words_py" \
  --train_file "$quick_ocr_text" \
  --output_path "$ocr_words_ds" \
  --jieba_userdict "$shared_jieba_dict" \
  --batch_size "$BATCH_SIZE" \
  --num_proc "$NUM_PROC" \
  --no_shuffle_split

build_all "T1" "$OUT_ROOT/tokenizers/t1_base" "$nonocr_words_ds" "$ocr_words_ds"
build_all "T2" "$OUT_ROOT/tokenizers/t2_keys" "$nonocr_words_ds" "$ocr_words_ds"

if [[ -d "$OUT_ROOT/tokenizers/t3_ocr_raw" ]]; then
  build_all "T3" "$OUT_ROOT/tokenizers/t3_ocr_raw" "$nonocr_words_ds" "$ocr_words_ds"
fi

build_all "T4" "$OUT_ROOT/tokenizers/t4_ocr_llm_keys" "$nonocr_words_ds" "$ocr_words_ds"

echo "[done] quick merged datasets saved under $OUT_ROOT/datasets_quick (processed_dataset_t{1,2,3,4})"
