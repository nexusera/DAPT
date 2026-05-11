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
NSP_LOSS_GUARD_ENABLE="${NSP_LOSS_GUARD_ENABLE:-1}"
NSP_LOSS_GUARD_ROUND="${NSP_LOSS_GUARD_ROUND:-2}"
NSP_LOSS_GUARD_TARGET="${NSP_LOSS_GUARD_TARGET:-0.6931}"
NSP_LOSS_GUARD_TOL="${NSP_LOSS_GUARD_TOL:-0.005}"

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
    IFS=',' read -r -a _torchrun_gpus <<< "$GPU_LIST"
    _torchrun_gpu_count=0
    for idx in "${!_torchrun_gpus[@]}"; do
      _torchrun_gpus[$idx]="${_torchrun_gpus[$idx]//[[:space:]]/}"
      if [[ -n "${_torchrun_gpus[$idx]}" ]]; then
        _torchrun_gpu_count=$((_torchrun_gpu_count + 1))
      fi
    done
    _required_gpu_count=$(( ${#VARIANTS[@]} * NPROC_PER_NODE ))
    if [[ "$_torchrun_gpu_count" -lt "$_required_gpu_count" ]]; then
      echo "[ERR] torchrun+PARALLEL=1 需要至少 ${_required_gpu_count} 张卡（${#VARIANTS[@]} variants × NPROC_PER_NODE=${NPROC_PER_NODE}），当前 GPU_LIST 仅 ${_torchrun_gpu_count} 张。" >&2
      echo "[ERR] 建议: GPU_LIST=0,1,2,3,4,5 PARALLEL=1 PRETRAIN_LAUNCHER=torchrun NPROC_PER_NODE=2" >&2
      exit 1
    fi
  fi
fi

require_dir "$DATASET_PATH"
require_file_or_dir "$NSP_DATA_DIR"
require_dir "$TOKENIZER_PATH"
require_file "$NOISE_BINS"
require_file "${DAPT_ROOT}/train_dapt_macbert_staged.py"

print_python_runtime
check_dataset_load_compat "$DATASET_PATH"

check_nsp_loss_guard() {
  local variant="$1"
  local pretrain_out="$2"

  if [[ "$NSP_LOSS_GUARD_ENABLE" != "1" ]]; then
    return 0
  fi

  local round_dir="${pretrain_out}/round_${NSP_LOSS_GUARD_ROUND}_nsp"
  if [[ ! -d "$round_dir" ]]; then
    echo "[WARN] [${variant}] NSP guard skipped: round dir missing: ${round_dir}"
    return 0
  fi

  if ! "$PYTHON_BIN" - "$variant" "$round_dir" "$NSP_LOSS_GUARD_TARGET" "$NSP_LOSS_GUARD_TOL" <<'PY'
import glob
import json
import math
import os
import sys

variant = sys.argv[1]
round_dir = sys.argv[2]
target = float(sys.argv[3])
tol = float(sys.argv[4])

paths = sorted(glob.glob(os.path.join(round_dir, "checkpoint-*", "trainer_state.json")))
if not paths:
    root_state = os.path.join(round_dir, "trainer_state.json")
    if os.path.exists(root_state):
        paths = [root_state]

if not paths:
    print(f"[WARN] [{variant}] NSP guard skipped: trainer_state.json not found under {round_dir}")
    sys.exit(0)

with open(paths[-1], "r", encoding="utf-8") as f:
    state = json.load(f)

losses = [x.get("loss") for x in state.get("log_history", []) if isinstance(x, dict) and "loss" in x]
losses = [float(x) for x in losses if x is not None and math.isfinite(float(x))]
if not losses:
    print(f"[WARN] [{variant}] NSP guard skipped: no finite loss found in trainer_state")
    sys.exit(0)

last_loss = losses[-1]
if abs(last_loss - target) <= tol:
    print(
        f"[ERR] [{variant}] NSP guard triggered: round2 NSP last_loss={last_loss:.6f} "
        f"is within [{target - tol:.6f}, {target + tol:.6f}] (target={target:.6f}, tol={tol:.6f})."
    )
    print(f"[ERR] [{variant}] Recent losses: " + ", ".join(f"{x:.6f}" for x in losses[-10:]))
    sys.exit(2)

print(
    f"[OK] [{variant}] NSP guard pass: round2 NSP last_loss={last_loss:.6f}, "
    f"target={target:.6f}, tol={tol:.6f}"
)
PY
  then
    return 1
  fi

  return 0
}

if ! grep -q -- 'nsp_reverse_negative_ratio' "${DAPT_ROOT}/train_dapt_macbert_staged.py"; then
  echo "[ERR] 远端 train_dapt_macbert_staged.py 仍是旧版本，缺少 nsp ratio 参数。" >&2
  echo "[ERR] 请先同步代码并确认: grep -n 'nsp_reverse_negative_ratio' ${DAPT_ROOT}/train_dapt_macbert_staged.py" >&2
  exit 1
fi

run_variant_pretrain() {
  local variant="$1"
  local gpu="$2"
  local cuda_visible_devices="$gpu"
  if [[ "$use_torchrun" == "1" ]]; then
    if [[ "$PARALLEL" == "1" ]]; then
      IFS=',' read -r -a _gpus <<< "$GPU_LIST"
      local variant_idx=-1
      for i in "${!VARIANTS[@]}"; do
        if [[ "${VARIANTS[$i]}" == "$variant" ]]; then
          variant_idx="$i"
          break
        fi
      done
      if [[ "$variant_idx" -lt 0 ]]; then
        echo "[ERR] 无法为 variant=${variant} 计算 GPU 分组" >&2
        return 1
      fi
      local start=$((variant_idx * NPROC_PER_NODE))
      local end=$((start + NPROC_PER_NODE - 1))
      local group=()
      for gi in $(seq "$start" "$end"); do
        local g="${_gpus[$gi]//[[:space:]]/}"
        if [[ -z "$g" ]]; then
          echo "[ERR] variant=${variant} 的 torchrun GPU 分组不足（idx=${gi}）" >&2
          return 1
        fi
        group+=("$g")
      done
      local joined=""
      for g in "${group[@]}"; do
        if [[ -z "$joined" ]]; then
          joined="$g"
        else
          joined+=",$g"
        fi
      done
      cuda_visible_devices="$joined"
    else
      cuda_visible_devices="$TORCHRUN_CUDA_VISIBLE_DEVICES"
    fi
  fi
  export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"

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
  local done_mark="${pretrain_out}/.pretrain_done"

  echo "============================================================"
  echo "[${variant}] PRETRAIN START (ratio=${ratio_name}, reverse=${reverse_ratio}, random=${random_ratio}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
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
    r11) port_offset=0 ;;
    r31) port_offset=1 ;;
    r13) port_offset=2 ;;
    *)   port_offset=0 ;;
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
    --nsp_negative_prob "$NSP_NEGATIVE_PROB"
    --nsp_reverse_negative_ratio "$reverse_ratio"
    --nsp_random_negative_ratio "$random_ratio"
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

  if [[ "$PRETRAIN_USE_FAST_TOKENIZER" == "1" ]]; then
    cmd+=(--pretrain_use_fast_tokenizer)
  fi

  run_logged "$variant" "pretrain" "$pretrain_log" "${cmd[@]}"
  check_nsp_loss_guard "$variant" "$pretrain_out"
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

echo "[OK] KV-NSP 比例消融预训练完成。"
for variant in "${VARIANTS[@]}"; do
  ratio_name="$(variant_ratio_name "$variant")"
  echo "  - ratio ${ratio_name}: $(pretrain_model_dir "$variant")"
done
