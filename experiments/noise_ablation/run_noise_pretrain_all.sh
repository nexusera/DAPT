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

if ! grep -q -- '"--noise_mode"' "${DAPT_ROOT}/train_dapt_macbert_staged.py"; then
  echo "[ERR] 远端的 train_dapt_macbert_staged.py 仍是旧版本，未包含 --noise_mode 参数。" >&2
  echo "[ERR] 请先在远端执行 git pull，或直接检查文件: ${DAPT_ROOT}/train_dapt_macbert_staged.py" >&2
  echo "[ERR] 可用命令: grep -n 'noise_mode\\|noise_mlp_hidden_dim' ${DAPT_ROOT}/train_dapt_macbert_staged.py" >&2
  exit 1
fi

run_variant_pretrain() {
  local variant="$1"
  local gpu="$2"
  export CUDA_VISIBLE_DEVICES="$gpu"

  local pretrain_out="$(pretrain_output_dir "$variant")"
  local model_dir="$(pretrain_model_dir "$variant")"
  local pretrain_log="${LOG_DIR}/${variant}_pretrain.gpu${gpu}.log"

  echo "============================================================"
  echo "[${variant}] PRETRAIN START (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
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
    --num_rounds "$NUM_ROUNDS"
    --mlm_epochs_per_round "$MLM_EPOCHS_PER_ROUND"
    --nsp_epochs_per_round "$NSP_EPOCHS_PER_ROUND"
    --mlm_probability "$MLM_PROBABILITY"
    --max_length "$MAX_LENGTH"
    --mlm_masking "$MLM_MASKING"
    --noise_mode "$variant"
    --export_fast_tokenizer
  )
  if [[ "$variant" == "mlp" ]]; then
    cmd+=(--noise_mlp_hidden_dim "$MLP_HIDDEN_DIM")
  fi
  if [[ "$PRETRAIN_USE_FAST_TOKENIZER" == "1" ]]; then
    cmd+=(--pretrain_use_fast_tokenizer)
  fi

  run_logged "$variant" "pretrain" "$pretrain_log" "${cmd[@]}"
  require_path "$variant" "pretrain" "$model_dir" dir

  echo "[${variant}] PRETRAIN DONE -> $model_dir"
}

print_common_header "pretrain"
run_variants_parallel_or_serial run_variant_pretrain

echo "[OK] 预训练阶段完成。"
for variant in "${VARIANTS[@]}"; do
  echo "  - ${variant}: $(pretrain_model_dir "$variant")"
done
