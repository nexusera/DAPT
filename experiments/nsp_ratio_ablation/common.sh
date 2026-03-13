#!/usr/bin/env bash

if [[ -n "${NSP_RATIO_ABLATION_COMMON_SH_LOADED:-}" ]]; then
  return 0
fi
NSP_RATIO_ABLATION_COMMON_SH_LOADED=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DAPT_ROOT="/data/ocean/DAPT"
if [[ ! -d "$DEFAULT_DAPT_ROOT" ]]; then
  DEFAULT_DAPT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

DAPT_ROOT="${DAPT_ROOT:-$DEFAULT_DAPT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1

if [[ -n "${VARIANTS:-}" ]]; then
  IFS=',' read -r -a VARIANTS <<< "$VARIANTS"
else
  VARIANTS=(r11 r31 r13)
fi

PARALLEL="${PARALLEL:-0}"
RESUME="${RESUME:-1}"
GPU_LIST="${GPU_LIST:-0,1,2}"

DATASET_PATH="${DATASET_PATH:-${DAPT_ROOT}/workspace/processed_dataset}"
NSP_DATA_DIR="${NSP_DATA_DIR:-${DAPT_ROOT}/data/pseudo_kv_labels_filtered.json}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${DAPT_ROOT}/my-medical-tokenizer}"
NOISE_BINS="${NOISE_BINS:-${DAPT_ROOT}/workspace/noise_bins.json}"
QUERY_SET="${QUERY_SET:-${DAPT_ROOT}/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/keys_merged_1027_cleaned.json}"
REAL_TRAIN_JSON="${REAL_TRAIN_JSON:-${DAPT_ROOT}/biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json}"
REAL_TEST_JSON="${REAL_TEST_JSON:-${DAPT_ROOT}/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json}"

LEARNING_RATE="${LEARNING_RATE:-5e-5}"
NUM_ROUNDS="${NUM_ROUNDS:-3}"
MLM_EPOCHS_PER_ROUND="${MLM_EPOCHS_PER_ROUND:-1}"
NSP_EPOCHS_PER_ROUND="${NSP_EPOCHS_PER_ROUND:-3}"
MLM_PROBABILITY="${MLM_PROBABILITY:-0.15}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MLM_MASKING="${MLM_MASKING:-kv_wwm}"
NSP_NEGATIVE_PROB="${NSP_NEGATIVE_PROB:-0.5}"
PRETRAIN_USE_FAST_TOKENIZER="${PRETRAIN_USE_FAST_TOKENIZER:-0}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"

LOG_DIR="${LOG_DIR:-${DAPT_ROOT}/runs/nsp_ratio_ablation_logs}"
GEN_DIR="${GEN_DIR:-${DAPT_ROOT}/experiments/nsp_ratio_ablation/generated_configs}"
mkdir -p "$LOG_DIR" "$GEN_DIR" "${DAPT_ROOT}/runs" "${DAPT_ROOT}/data/kv_ner_prepared_comparison"

variant_ratio_name() {
  local variant="$1"
  case "$variant" in
    r11) echo "1_1" ;;
    r31) echo "3_1" ;;
    r13) echo "1_3" ;;
    *) echo "[ERR] 未知 VARIANT: $variant" >&2; exit 1 ;;
  esac
}

variant_reverse_ratio() {
  local variant="$1"
  case "$variant" in
    r11) echo "1" ;;
    r31) echo "3" ;;
    r13) echo "1" ;;
    *) echo "[ERR] 未知 VARIANT: $variant" >&2; exit 1 ;;
  esac
}

variant_random_ratio() {
  local variant="$1"
  case "$variant" in
    r11) echo "1" ;;
    r31) echo "1" ;;
    r13) echo "3" ;;
    *) echo "[ERR] 未知 VARIANT: $variant" >&2; exit 1 ;;
  esac
}

require_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "[ERR] 缺少文件: $p" >&2
    exit 1
  fi
}

require_dir() {
  local p="$1"
  if [[ ! -d "$p" ]]; then
    echo "[ERR] 缺少目录: $p" >&2
    exit 1
  fi
}

require_file_or_dir() {
  local p="$1"
  if [[ ! -f "$p" && ! -d "$p" ]]; then
    echo "[ERR] 路径不存在: $p" >&2
    exit 1
  fi
}

report_failure() {
  local variant="$1"
  local stage="$2"
  local logfile="$3"
  local rc="$4"
  echo "[ERR] [${variant}] 阶段失败: ${stage} (exit=${rc})" >&2
  echo "[ERR] [${variant}] 日志定位: ${logfile}" >&2
  if [[ -f "$logfile" ]]; then
    echo "[ERR] [${variant}] 最近日志片段:------------------------------" >&2
    tail -n 80 "$logfile" >&2 || true
    echo "[ERR] [${variant}] --------------------------------------------" >&2
  fi
}

run_logged() {
  local variant="$1"
  local stage="$2"
  local logfile="$3"
  shift 3

  echo "[RUN] [${variant}] ${stage}"
  echo "[LOG] ${logfile}"
  printf '[CMD] '
  printf '%q ' "$@"
  printf '\n'

  set +e
  "$@" 2>&1 | tee "$logfile"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -ne 0 ]]; then
    report_failure "$variant" "$stage" "$logfile" "$rc"
    return "$rc"
  fi
}

run_logged_in_dir() {
  local work_dir="$1"
  local variant="$2"
  local stage="$3"
  local logfile="$4"
  shift 4

  echo "[RUN] [${variant}] ${stage}"
  echo "[CWD] ${work_dir}"
  echo "[LOG] ${logfile}"
  printf '[CMD] '
  printf '%q ' "$@"
  printf '\n'

  set +e
  (
    cd "$work_dir"
    "$@"
  ) 2>&1 | tee "$logfile"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -ne 0 ]]; then
    report_failure "$variant" "$stage" "$logfile" "$rc"
    return "$rc"
  fi
}

require_path() {
  local variant="$1"
  local stage="$2"
  local path="$3"
  local kind="${4:-path}"
  if [[ "$kind" == "file" ]]; then
    [[ -s "$path" ]] && return 0
  elif [[ "$kind" == "dir" ]]; then
    [[ -d "$path" ]] && return 0
  else
    [[ -e "$path" ]] && return 0
  fi
  echo "[ERR] [${variant}] ${stage} 未生成预期输出: ${path}" >&2
  return 1
}

pretrain_output_dir() {
  local variant="$1"
  local ratio_name
  ratio_name="$(variant_ratio_name "$variant")"
  echo "${DAPT_ROOT}/workspace/output_ablation_nsp_ratio_${ratio_name}"
}

pretrain_model_dir() {
  local variant="$1"
  echo "$(pretrain_output_dir "$variant")/final_staged_model"
}

kv_summary_path() {
  local variant="$1"
  local ratio_name
  ratio_name="$(variant_ratio_name "$variant")"
  echo "${DAPT_ROOT}/runs/nsp_ratio_${ratio_name}_eval_summary.json"
}

kv_aligned_gt() {
  local variant="$1"
  local ratio_name
  ratio_name="$(variant_ratio_name "$variant")"
  echo "${DAPT_ROOT}/runs/nsp_ratio_${ratio_name}_task13_aligned_gt.jsonl"
}

kv_aligned_pred() {
  local variant="$1"
  local ratio_name
  ratio_name="$(variant_ratio_name "$variant")"
  echo "${DAPT_ROOT}/runs/nsp_ratio_${ratio_name}_task13_aligned_preds.jsonl"
}

ebqa_train_jsonl() {
  local variant="$1"
  local ratio_name
  ratio_name="$(variant_ratio_name "$variant")"
  echo "${DAPT_ROOT}/data/kv_ner_prepared_comparison/ebqa_train_real_ratio_${ratio_name}.jsonl"
}

ebqa_eval_jsonl() {
  local variant="$1"
  local ratio_name
  ratio_name="$(variant_ratio_name "$variant")"
  echo "${DAPT_ROOT}/data/kv_ner_prepared_comparison/ebqa_eval_real_ratio_${ratio_name}.jsonl"
}

gen_kv_config() {
  local variant="$1"
  local model_dir="$2"
  local ratio_name
  ratio_name="$(variant_ratio_name "$variant")"
  local out_cfg="${GEN_DIR}/kv_ner_config_nsp_ratio_${ratio_name}.json"

  "$PYTHON_BIN" - "$DAPT_ROOT" "$ratio_name" "$REAL_TRAIN_JSON" "$REAL_TEST_JSON" "$model_dir" "$out_cfg" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
ratio_name = sys.argv[2]
real_train = sys.argv[3]
real_test = sys.argv[4]
model_dir = sys.argv[5]
out_cfg = Path(sys.argv[6])

tpl = root / "dapt_eval_package" / "pre_struct" / "kv_ner" / "kv_ner_config_macbert.json"
if not tpl.is_file():
    raise FileNotFoundError(f"Missing KV template: {tpl}")

with tpl.open("r", encoding="utf-8") as f:
    kv = json.load(f)

kv["model_name_or_path"] = model_dir
kv.setdefault("train", {})["data_path"] = real_train
kv["train"]["test_data_path"] = real_test
kv["train"]["output_dir"] = f"runs/kv_ner_finetuned_nsp_ratio_{ratio_name}"

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    json.dump(kv, f, ensure_ascii=False, indent=2)
PY
}

gen_ebqa_config() {
  local variant="$1"
  local model_dir="$2"
  local ebqa_train="$3"
  local ebqa_eval="$4"
  local ratio_name
  ratio_name="$(variant_ratio_name "$variant")"
  local out_cfg="${GEN_DIR}/ebqa_config_nsp_ratio_${ratio_name}.json"

  "$PYTHON_BIN" - "$DAPT_ROOT" "$ratio_name" "$QUERY_SET" "$model_dir" "$ebqa_train" "$ebqa_eval" "$out_cfg" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
ratio_name = sys.argv[2]
query_set = sys.argv[3]
model_dir = sys.argv[4]
ebqa_train = sys.argv[5]
ebqa_eval = sys.argv[6]
out_cfg = Path(sys.argv[7])

tpl = root / "dapt_eval_package" / "pre_struct" / "ebqa" / "ebqa_config_macbert.json"
if not tpl.is_file():
    raise FileNotFoundError(f"Missing EBQA template: {tpl}")

with tpl.open("r", encoding="utf-8") as f:
    ebqa = json.load(f)

output_dir = root / "runs" / f"ebqa_nsp_ratio_{ratio_name}"
ebqa["report_struct_path"] = query_set
ebqa["model_name_or_path"] = model_dir
ebqa["tokenizer_name_or_path"] = model_dir
ebqa["output_dir"] = str(output_dir)
ebqa["model_dir"] = str(output_dir / "best")
ebqa.setdefault("train", {})["data_path"] = ebqa_train
ebqa.setdefault("predict", {})["input_file"] = ebqa_eval

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    json.dump(ebqa, f, ensure_ascii=False, indent=2)
PY
}

parse_gpu_list() {
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
  if [[ "${#GPUS[@]}" -eq 0 ]]; then
    echo "[ERR] GPU_LIST 不能为空" >&2
    exit 1
  fi
  for idx in "${!GPUS[@]}"; do
    GPUS[$idx]="${GPUS[$idx]//[[:space:]]/}"
  done
}

print_common_header() {
  local stage="$1"
  echo "[INFO] STAGE=$stage"
  echo "[INFO] DAPT_ROOT=$DAPT_ROOT"
  echo "[INFO] VARIANTS=${VARIANTS[*]}"
  echo "[INFO] NSP_NEGATIVE_PROB=$NSP_NEGATIVE_PROB"
  echo "[INFO] PARALLEL=$PARALLEL RESUME=$RESUME"
  echo "[INFO] LOG_DIR=$LOG_DIR"
}

run_variants_parallel_or_serial() {
  local runner="$1"
  parse_gpu_list

  if [[ "$PARALLEL" == "1" ]]; then
    if [[ "${#GPUS[@]}" -lt "${#VARIANTS[@]}" ]]; then
      echo "[ERR] PARALLEL=1 时 GPU_LIST 至少需要 ${#VARIANTS[@]} 张卡，当前仅有 ${#GPUS[@]} 张。" >&2
      exit 1
    fi
    echo "[INFO] 并行启动三组实验"
    pids=()
    for i in "${!VARIANTS[@]}"; do
      variant="${VARIANTS[$i]}"
      gpu="${GPUS[$i]}"
      echo "[LAUNCH] ${variant} -> GPU ${gpu}"
      "$runner" "$variant" "$gpu" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        echo "[ERR] 检测到子任务失败 (pid=${pid})，正在停止其它实验..." >&2
        for other_pid in "${pids[@]}"; do
          if [[ "$other_pid" != "$pid" ]]; then
            kill "$other_pid" 2>/dev/null || true
          fi
        done
        wait || true
        exit 1
      fi
    done
  else
    local run_gpu="${GPUS[0]}"
    echo "[INFO] 串行执行，统一使用 GPU ${run_gpu}"
    for variant in "${VARIANTS[@]}"; do
      "$runner" "$variant" "$run_gpu"
    done
  fi
}
