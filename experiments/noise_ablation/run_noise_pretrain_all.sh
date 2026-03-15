#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

PRETRAIN_LAUNCHER="${PRETRAIN_LAUNCHER:-python}"   # python | torchrun
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29521}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
TORCHRUN_CUDA_VISIBLE_DEVICES="${TORCHRUN_CUDA_VISIBLE_DEVICES:-$GPU_LIST}"

use_torchrun=0
if [[ "${PRETRAIN_LAUNCHER}" == "torchrun" || "${NPROC_PER_NODE}" -gt 1 ]]; then
  use_torchrun=1
fi

if [[ "$use_torchrun" == "1" ]]; then
  if ! command -v "$TORCHRUN_BIN" >/dev/null 2>&1; then
    echo "[ERR] PRETRAIN_LAUNCHER=torchrun 但未找到命令: $TORCHRUN_BIN" >&2
    exit 1
  fi
  if [[ "$PARALLEL" == "1" ]]; then
    echo "[ERR] 使用 torchrun 多进程预训练时，当前脚本仅支持 PARALLEL=0（串行逐个 variant 运行）。" >&2
    echo "[ERR] 建议: PARALLEL=0 GPU_LIST=2,3 NPROC_PER_NODE=2 PRETRAIN_LAUNCHER=torchrun" >&2
    exit 1
  fi
fi

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
  local cuda_visible_devices="$gpu"
  if [[ "$use_torchrun" == "1" ]]; then
    cuda_visible_devices="$TORCHRUN_CUDA_VISIBLE_DEVICES"
  fi
  export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"

  local pretrain_out="$(pretrain_output_dir "$variant")"
  local model_dir="$(pretrain_model_dir "$variant")"
  local done_mark="${pretrain_out}/.pretrain_done"
  local pretrain_log="${LOG_DIR}/${variant}_pretrain.gpu${gpu}.log"

  echo "============================================================"
  echo "[${variant}] PRETRAIN START (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
  echo "============================================================"

  if [[ "$RESUME" == "0" && "$CLEAN_BEFORE_RUN" == "1" ]]; then
    if [[ -d "$pretrain_out" ]]; then
      echo "[${variant}] [CLEAN] remove stale output dir: $pretrain_out"
      rm -rf "$pretrain_out"
    fi
  fi

  local has_model_bin="0"
  local has_tokenizer="0"
  if [[ -s "${model_dir}/pytorch_model.bin" || -s "${model_dir}/model.safetensors" ]]; then
    has_model_bin="1"
  fi
  if [[ -s "${model_dir}/tokenizer.json" || -s "${model_dir}/tokenizer_config.json" || -s "${model_dir}/vocab.txt" ]]; then
    has_tokenizer="1"
  fi
  local model_complete="0"
  if [[ -s "${model_dir}/config.json" && "$has_model_bin" == "1" && "$has_tokenizer" == "1" ]]; then
    model_complete="1"
  fi

  if [[ "$RESUME" == "1" && "$model_complete" == "1" ]]; then
    echo "[${variant}] [SKIP] pretrain (valid model found: $model_dir)"
    return 0
  fi
  if [[ "$RESUME" == "1" && -d "$model_dir" && "$model_complete" != "1" ]]; then
    echo "[${variant}] [WARN] 发现不完整模型目录，将重新训练: $model_dir"
  fi

  local port_offset=0
  case "$variant" in
    bucket) port_offset=0 ;;
    linear) port_offset=1 ;;
    mlp)    port_offset=2 ;;
    *)      port_offset=0 ;;
  esac
  local master_port=$((MASTER_PORT_BASE + port_offset))

  local cmd=(
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
    --noise_mode "$variant"
    --export_fast_tokenizer
  )
  if [[ "$use_torchrun" == "1" ]]; then
    cmd=(
      "$TORCHRUN_BIN"
      --nproc_per_node "$NPROC_PER_NODE"
      --master_port "$master_port"
      "${DAPT_ROOT}/train_dapt_macbert_staged.py"
      "${cmd[@]}"
    )
  else
    cmd=(
      "$PYTHON_BIN" "${DAPT_ROOT}/train_dapt_macbert_staged.py"
      "${cmd[@]}"
    )
  fi
  if [[ "$variant" == "mlp" ]]; then
    cmd+=(--noise_mlp_hidden_dim "$MLP_HIDDEN_DIM")
  fi
  if [[ "$PRETRAIN_USE_FAST_TOKENIZER" == "1" ]]; then
    cmd+=(--pretrain_use_fast_tokenizer)
  fi

  run_logged "$variant" "pretrain" "$pretrain_log" "${cmd[@]}"
  require_path "$variant" "pretrain" "$model_dir" dir
  if [[ ! -s "${model_dir}/config.json" || ( ! -s "${model_dir}/pytorch_model.bin" && ! -s "${model_dir}/model.safetensors" ) ]]; then
    echo "[ERR] [${variant}] pretrain 结束后模型文件不完整: ${model_dir}" >&2
    return 1
  fi
  date '+%F %T %z' > "$done_mark"

  echo "[${variant}] PRETRAIN DONE -> $model_dir"
}

print_common_header "pretrain"
run_variants_parallel_or_serial run_variant_pretrain

echo "[OK] 预训练阶段完成。"
for variant in "${VARIANTS[@]}"; do
  echo "  - ${variant}: $(pretrain_model_dir "$variant")"
done
