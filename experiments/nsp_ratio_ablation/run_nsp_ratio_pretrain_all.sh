#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

require_dir "$DATASET_PATH"
require_file_or_dir "$NSP_DATA_DIR"
require_dir "$TOKENIZER_PATH"
require_file "$NOISE_BINS"
require_file "${DAPT_ROOT}/train_dapt_macbert_staged.py"

print_python_runtime
check_dataset_load_compat "$DATASET_PATH"

if ! grep -q -- 'nsp_reverse_negative_ratio' "${DAPT_ROOT}/train_dapt_macbert_staged.py"; then
  echo "[ERR] 远端 train_dapt_macbert_staged.py 仍是旧版本，缺少 nsp ratio 参数。" >&2
  echo "[ERR] 请先同步代码并确认: grep -n 'nsp_reverse_negative_ratio' ${DAPT_ROOT}/train_dapt_macbert_staged.py" >&2
  exit 1
fi

run_variant_pretrain() {
  local variant="$1"
  local gpu="$2"
  export CUDA_VISIBLE_DEVICES="$gpu"

  local ratio_name
  ratio_name="$(variant_ratio_name "$variant")"
  local reverse_ratio
  reverse_ratio="$(variant_reverse_ratio "$variant")"
  local random_ratio
  random_ratio="$(variant_random_ratio "$variant")"

  local pretrain_out
  pretrain_out="$(pretrain_output_dir "$variant")"
  local model_dir
  model_dir="$(pretrain_model_dir "$variant")"
  local pretrain_log="${LOG_DIR}/${variant}_pretrain.gpu${gpu}.log"

  echo "============================================================"
  echo "[${variant}] PRETRAIN START (ratio=${ratio_name}, reverse=${reverse_ratio}, random=${random_ratio}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
  echo "============================================================"

  if [[ "$RESUME" == "1" && -d "$model_dir" ]]; then
    echo "[${variant}] [SKIP] pretrain (found: $model_dir)"
    return 0
  fi

  local cmd=(
    "$PYTHON_BIN" "${DAPT_ROOT}/train_dapt_macbert_staged.py"
    --output_dir "$pretrain_out"
    --dataset_path "$DATASET_PATH"
    --nsp_data_dir "$NSP_DATA_DIR"
    --tokenizer_path "$TOKENIZER_PATH"
    --noise_bins_json "$NOISE_BINS"
    --learning_rate "$LEARNING_RATE"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --num_rounds "$NUM_ROUNDS"
    --mlm_epochs_per_round "$MLM_EPOCHS_PER_ROUND"
    --nsp_epochs_per_round "$NSP_EPOCHS_PER_ROUND"
    --mlm_probability "$MLM_PROBABILITY"
    --max_length "$MAX_LENGTH"
    --mlm_masking "$MLM_MASKING"
    --nsp_negative_prob "$NSP_NEGATIVE_PROB"
    --nsp_reverse_negative_ratio "$reverse_ratio"
    --nsp_random_negative_ratio "$random_ratio"
    --export_fast_tokenizer
  )

  if [[ "$PRETRAIN_USE_FAST_TOKENIZER" == "1" ]]; then
    cmd+=(--pretrain_use_fast_tokenizer)
  fi

  run_logged "$variant" "pretrain" "$pretrain_log" "${cmd[@]}"
  require_path "$variant" "pretrain" "$model_dir" dir

  echo "[${variant}] PRETRAIN DONE -> $model_dir"
}

print_common_header "pretrain"
run_variants_parallel_or_serial run_variant_pretrain

echo "[OK] KV-NSP 比例消融预训练完成。"
for variant in "${VARIANTS[@]}"; do
  ratio_name="$(variant_ratio_name "$variant")"
  echo "  - ratio ${ratio_name}: $(pretrain_model_dir "$variant")"
done
