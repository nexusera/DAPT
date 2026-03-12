#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DAPT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$DAPT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1

VARIANTS=(bucket linear mlp)
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
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-128}"
PRETRAIN_USE_FAST_TOKENIZER="${PRETRAIN_USE_FAST_TOKENIZER:-0}"

RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
RUN_KVNER="${RUN_KVNER:-1}"
RUN_EBQA="${RUN_EBQA:-1}"

LOG_DIR="${LOG_DIR:-${DAPT_ROOT}/runs/noise_ablation_logs}"
GEN_DIR="${GEN_DIR:-${DAPT_ROOT}/experiments/noise_ablation/generated_configs}"
mkdir -p "$LOG_DIR" "$GEN_DIR" "${DAPT_ROOT}/runs" "${DAPT_ROOT}/data/kv_ner_prepared_comparison"

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

  set +e
  "$@" 2>&1 | tee "$logfile"
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
  echo "${DAPT_ROOT}/workspace/output_ablation_noise_${variant}"
}

pretrain_model_dir() {
  local variant="$1"
  echo "$(pretrain_output_dir "$variant")/final_staged_model"
}

kv_summary_path() {
  local variant="$1"
  echo "${DAPT_ROOT}/runs/noise_${variant}_eval_summary.json"
}

kv_aligned_gt() {
  local variant="$1"
  echo "${DAPT_ROOT}/runs/noise_${variant}_task13_aligned_gt.jsonl"
}

kv_aligned_pred() {
  local variant="$1"
  echo "${DAPT_ROOT}/runs/noise_${variant}_task13_aligned_preds.jsonl"
}

ebqa_train_jsonl() {
  local variant="$1"
  echo "${DAPT_ROOT}/data/kv_ner_prepared_comparison/ebqa_train_real_${variant}.jsonl"
}

ebqa_eval_jsonl() {
  local variant="$1"
  echo "${DAPT_ROOT}/data/kv_ner_prepared_comparison/ebqa_eval_real_${variant}.jsonl"
}

gen_runtime_configs() {
  local variant="$1"
  local model_dir="$2"
  local ebqa_train="$3"
  local ebqa_eval="$4"

  "$PYTHON_BIN" - "$DAPT_ROOT" "$variant" "$REAL_TRAIN_JSON" "$REAL_TEST_JSON" "$QUERY_SET" "$model_dir" "$ebqa_train" "$ebqa_eval" "$GEN_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
variant = sys.argv[2]
real_train = sys.argv[3]
real_test = sys.argv[4]
query_set = sys.argv[5]
model_dir = sys.argv[6]
ebqa_train = sys.argv[7]
ebqa_eval = sys.argv[8]
gen_dir = Path(sys.argv[9])
gen_dir.mkdir(parents=True, exist_ok=True)

kv_tpl = root / "dapt_eval_package" / "pre_struct" / "kv_ner" / f"kv_ner_config_noise_{variant}.json"
ebqa_tpl = root / "dapt_eval_package" / "pre_struct" / "ebqa" / f"ebqa_config_noise_{variant}.json"
if not kv_tpl.is_file():
    raise FileNotFoundError(f"Missing KV template: {kv_tpl}")
if not ebqa_tpl.is_file():
    raise FileNotFoundError(f"Missing EBQA template: {ebqa_tpl}")

with kv_tpl.open("r", encoding="utf-8") as f:
    kv = json.load(f)
with ebqa_tpl.open("r", encoding="utf-8") as f:
    ebqa = json.load(f)

kv["model_name_or_path"] = model_dir
kv.setdefault("train", {})["data_path"] = real_train
kv["train"]["test_data_path"] = real_test
kv["train"]["output_dir"] = f"runs/kv_ner_finetuned_noise_{variant}"

kv_out = gen_dir / f"kv_ner_config_noise_{variant}.json"
with kv_out.open("w", encoding="utf-8") as f:
    json.dump(kv, f, ensure_ascii=False, indent=2)

output_dir = root / "runs" / f"ebqa_noise_{variant}"
ebqa["report_struct_path"] = query_set
ebqa["model_name_or_path"] = model_dir
ebqa["tokenizer_name_or_path"] = model_dir
ebqa["output_dir"] = str(output_dir)
ebqa["model_dir"] = str(output_dir / "best")
ebqa.setdefault("train", {})["data_path"] = ebqa_train
ebqa.setdefault("predict", {})["input_file"] = ebqa_eval

ebqa_out = gen_dir / f"ebqa_config_noise_{variant}.json"
with ebqa_out.open("w", encoding="utf-8") as f:
    json.dump(ebqa, f, ensure_ascii=False, indent=2)
PY
}

run_variant() {
  local variant="$1"
  local gpu="$2"
  export CUDA_VISIBLE_DEVICES="$gpu"

  local pretrain_out
  local model_dir
  local kv_cfg
  local ebqa_cfg
  local ebqa_train
  local ebqa_eval
  local summary
  local pred_jsonl
  local aligned_gt
  local aligned_pred
  local report_t1
  local report_t3
  local ebqa_best_dir
  local ebqa_pred_qa
  local ebqa_pred_doc
  local ebqa_aligned_dir
  local ebqa_aligned_gt
  local ebqa_aligned_pred
  local report_t2

  pretrain_out="$(pretrain_output_dir "$variant")"
  model_dir="$(pretrain_model_dir "$variant")"
  kv_cfg="${GEN_DIR}/kv_ner_config_noise_${variant}.json"
  ebqa_cfg="${GEN_DIR}/ebqa_config_noise_${variant}.json"
  ebqa_train="$(ebqa_train_jsonl "$variant")"
  ebqa_eval="$(ebqa_eval_jsonl "$variant")"
  summary="$(kv_summary_path "$variant")"
  pred_jsonl="${summary%.json}_preds.jsonl"
  aligned_gt="$(kv_aligned_gt "$variant")"
  aligned_pred="$(kv_aligned_pred "$variant")"
  report_t1="${DAPT_ROOT}/runs/noise_${variant}_report_task1.json"
  report_t3="${DAPT_ROOT}/runs/noise_${variant}_report_task3.json"
  ebqa_best_dir="${DAPT_ROOT}/runs/ebqa_noise_${variant}/best"
  ebqa_pred_qa="${DAPT_ROOT}/runs/ebqa_noise_${variant}_preds.jsonl"
  ebqa_pred_doc="${DAPT_ROOT}/runs/ebqa_noise_${variant}_doc_preds.jsonl"
  ebqa_aligned_dir="${DAPT_ROOT}/runs/ebqa_noise_${variant}_aligned"
  ebqa_aligned_gt="${ebqa_aligned_dir}/gt_ebqa_aligned.jsonl"
  ebqa_aligned_pred="${ebqa_aligned_dir}/aligned_$(basename "$ebqa_pred_doc")"
  report_t2="${DAPT_ROOT}/runs/noise_${variant}_report_task2.json"

  echo "============================================================"
  echo "[${variant}] START (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
  echo "============================================================"

  if [[ "$RUN_PRETRAIN" == "1" ]]; then
    if [[ "$RESUME" == "1" && -d "$model_dir" ]]; then
      echo "[${variant}] [SKIP] pretrain (found: $model_dir)"
    else
      local pretrain_log="${LOG_DIR}/${variant}_pretrain.gpu${gpu}.log"
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
    fi
  else
    require_path "$variant" "pretrain-skip" "$model_dir" dir
  fi

  gen_runtime_configs "$variant" "$model_dir" "$ebqa_train" "$ebqa_eval"
  require_path "$variant" "generate-config" "$kv_cfg" file
  require_path "$variant" "generate-config" "$ebqa_cfg" file

  if [[ "$RUN_KVNER" == "1" ]]; then
    echo "[${variant}] Task1/3 (KV-NER)"

    local kv_best_dir="${DAPT_ROOT}/runs/kv_ner_finetuned_noise_${variant}/best"
    if [[ "$RESUME" == "1" && -d "$kv_best_dir" ]]; then
      echo "[${variant}] [SKIP] KV-NER train (found: $kv_best_dir)"
    else
      run_logged "$variant" "kvner-train" "${LOG_DIR}/${variant}_kvner_train.gpu${gpu}.log" \
        "$PYTHON_BIN" "${DAPT_ROOT}/dapt_eval_package/pre_struct/kv_ner/train_with_noise.py" \
        --config "$kv_cfg" \
        --noise_bins "$NOISE_BINS"
      require_path "$variant" "kvner-train" "$kv_best_dir" dir
    fi

    if [[ "$RESUME" == "1" && -s "$summary" ]]; then
      echo "[${variant}] [SKIP] KV-NER predict (found: $summary)"
    else
      run_logged "$variant" "kvner-predict" "${LOG_DIR}/${variant}_kvner_predict.gpu${gpu}.log" \
        "$PYTHON_BIN" "${DAPT_ROOT}/dapt_eval_package/pre_struct/kv_ner/compare_models.py" \
        --ner_config "$kv_cfg" \
        --keys_file "$QUERY_SET" \
        --test_data "$REAL_TEST_JSON" \
        --noise_bins "$NOISE_BINS" \
        --output_summary "$summary"
      require_path "$variant" "kvner-predict" "$summary" file
      require_path "$variant" "kvner-predict" "$pred_jsonl" file
    fi

    if [[ "$RESUME" == "1" && -s "$aligned_gt" && -s "$aligned_pred" ]]; then
      echo "[${variant}] [SKIP] Task1/3 align (found: $aligned_pred)"
    else
      run_logged "$variant" "task13-align" "${LOG_DIR}/${variant}_task13_align.gpu${gpu}.log" \
        "$PYTHON_BIN" "${DAPT_ROOT}/scripts/align_for_scorer_span.py" \
        --gt_in "$REAL_TEST_JSON" \
        --pred_in "$pred_jsonl" \
        --gt_out "$aligned_gt" \
        --pred_out "$aligned_pred"
      require_path "$variant" "task13-align" "$aligned_gt" file
      require_path "$variant" "task13-align" "$aligned_pred" file
    fi

    if [[ "$RESUME" == "1" && -s "$report_t1" ]]; then
      echo "[${variant}] [SKIP] Task1 score (found: $report_t1)"
    else
      run_logged "$variant" "task1-score" "${LOG_DIR}/${variant}_task1_score.gpu${gpu}.log" \
        "$PYTHON_BIN" "${DAPT_ROOT}/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py" \
        --pred_file "$aligned_pred" \
        --gt_file "$aligned_gt" \
        --schema_file "$QUERY_SET" \
        --task_type task1 \
        --overlap_threshold -1 \
        --output_file "$report_t1"
      require_path "$variant" "task1-score" "$report_t1" file
    fi

    if [[ "$RESUME" == "1" && -s "$report_t3" ]]; then
      echo "[${variant}] [SKIP] Task3 score (found: $report_t3)"
    else
      run_logged "$variant" "task3-score" "${LOG_DIR}/${variant}_task3_score.gpu${gpu}.log" \
        "$PYTHON_BIN" "${DAPT_ROOT}/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py" \
        --pred_file "$aligned_pred" \
        --gt_file "$aligned_gt" \
        --schema_file "$QUERY_SET" \
        --task_type task3 \
        --overlap_threshold -1 \
        --output_file "$report_t3"
      require_path "$variant" "task3-score" "$report_t3" file
    fi
  fi

  if [[ "$RUN_EBQA" == "1" ]]; then
    echo "[${variant}] Task2 (EBQA)"

    if [[ "$RESUME" == "1" && -s "$ebqa_train" ]]; then
      echo "[${variant}] [SKIP] EBQA convert train (found: $ebqa_train)"
    else
      run_logged "$variant" "ebqa-convert-train" "${LOG_DIR}/${variant}_ebqa_convert_train.gpu${gpu}.log" \
        "$PYTHON_BIN" "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py" \
        --input_file "$REAL_TRAIN_JSON" \
        --output_file "$ebqa_train" \
        --struct_path "$QUERY_SET" \
        --tokenizer_name "$model_dir" \
        --noise_bins "$NOISE_BINS"
      require_path "$variant" "ebqa-convert-train" "$ebqa_train" file
    fi

    if [[ "$RESUME" == "1" && -s "$ebqa_eval" ]]; then
      echo "[${variant}] [SKIP] EBQA convert eval (found: $ebqa_eval)"
    else
      run_logged "$variant" "ebqa-convert-eval" "${LOG_DIR}/${variant}_ebqa_convert_eval.gpu${gpu}.log" \
        "$PYTHON_BIN" "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py" \
        --input_file "$REAL_TEST_JSON" \
        --output_file "$ebqa_eval" \
        --struct_path "$QUERY_SET" \
        --tokenizer_name "$model_dir" \
        --noise_bins "$NOISE_BINS"
      require_path "$variant" "ebqa-convert-eval" "$ebqa_eval" file
    fi

    if [[ "$RESUME" == "1" && -d "$ebqa_best_dir" ]]; then
      echo "[${variant}] [SKIP] EBQA train (found: $ebqa_best_dir)"
    else
      run_logged "$variant" "ebqa-train" "${LOG_DIR}/${variant}_ebqa_train.gpu${gpu}.log" \
        "$PYTHON_BIN" "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/train_ebqa.py" \
        --config "$ebqa_cfg"
      require_path "$variant" "ebqa-train" "$ebqa_best_dir" dir
    fi

    if [[ "$RESUME" == "1" && -s "$ebqa_pred_qa" ]]; then
      echo "[${variant}] [SKIP] EBQA predict (found: $ebqa_pred_qa)"
    else
      run_logged "$variant" "ebqa-predict" "${LOG_DIR}/${variant}_ebqa_predict.gpu${gpu}.log" \
        "$PYTHON_BIN" "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/predict_ebqa.py" \
        --model_dir "$ebqa_best_dir" \
        --tokenizer "$model_dir" \
        --data_path "$ebqa_eval" \
        --output_preds "$ebqa_pred_qa"
      require_path "$variant" "ebqa-predict" "$ebqa_pred_qa" file
    fi

    if [[ "$RESUME" == "1" && -s "$ebqa_pred_doc" ]]; then
      echo "[${variant}] [SKIP] EBQA aggregate (found: $ebqa_pred_doc)"
    else
      run_logged "$variant" "ebqa-aggregate" "${LOG_DIR}/${variant}_ebqa_aggregate.gpu${gpu}.log" \
        "$PYTHON_BIN" "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/aggregate_qa_preds_to_doc.py" \
        --raw_file "$REAL_TEST_JSON" \
        --qa_pred_file "$ebqa_pred_qa" \
        --output_file "$ebqa_pred_doc" \
        --prefer score
      require_path "$variant" "ebqa-aggregate" "$ebqa_pred_doc" file
    fi

    if [[ "$RESUME" == "1" && -s "$ebqa_aligned_gt" && -s "$ebqa_aligned_pred" ]]; then
      echo "[${variant}] [SKIP] EBQA align (found: $ebqa_aligned_pred)"
    else
      run_logged "$variant" "ebqa-align" "${LOG_DIR}/${variant}_ebqa_align.gpu${gpu}.log" \
        "$PYTHON_BIN" "${DAPT_ROOT}/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/preprocess_ebqa_real_h200.py" \
        --gt_file "$REAL_TEST_JSON" \
        --pred_file "$ebqa_pred_doc" \
        --output_dir "$ebqa_aligned_dir"
      require_path "$variant" "ebqa-align" "$ebqa_aligned_gt" file
      require_path "$variant" "ebqa-align" "$ebqa_aligned_pred" file
    fi

    if [[ "$RESUME" == "1" && -s "$report_t2" ]]; then
      echo "[${variant}] [SKIP] Task2 score (found: $report_t2)"
    else
      pushd "${DAPT_ROOT}/dapt_eval_package/MedStruct-S-master" >/dev/null
      run_logged "$variant" "task2-score" "${LOG_DIR}/${variant}_task2_score.gpu${gpu}.log" \
        "$PYTHON_BIN" scorer.py \
        --pred_file "$ebqa_aligned_pred" \
        --gt_file "$ebqa_aligned_gt" \
        --query_set "$QUERY_SET" \
        --task_type task2 \
        --output_file "$report_t2"
      popd >/dev/null
      require_path "$variant" "task2-score" "$report_t2" file
    fi
  fi

  echo "============================================================"
  echo "[${variant}] DONE"
  echo "============================================================"
}

require_dir "$DATASET_PATH"
if [[ -d "$NSP_DATA_DIR" ]]; then
  :
elif [[ -f "$NSP_DATA_DIR" ]]; then
  :
else
  echo "[ERR] 缺少 NSP 数据: $NSP_DATA_DIR" >&2
  exit 1
fi
require_dir "$TOKENIZER_PATH"
require_file "$NOISE_BINS"
require_file "$QUERY_SET"
require_file "$REAL_TRAIN_JSON"
require_file "$REAL_TEST_JSON"
require_file "${DAPT_ROOT}/train_dapt_macbert_staged.py"
require_file "${DAPT_ROOT}/dapt_eval_package/pre_struct/kv_ner/train_with_noise.py"
require_file "${DAPT_ROOT}/dapt_eval_package/pre_struct/kv_ner/compare_models.py"
require_file "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py"
require_file "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/train_ebqa.py"
require_file "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/predict_ebqa.py"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "[ERR] GPU_LIST 不能为空" >&2
  exit 1
fi

for idx in "${!GPUS[@]}"; do
  GPUS[$idx]="${GPUS[$idx]//[[:space:]]/}"
done

if [[ "$PARALLEL" == "1" && "${#GPUS[@]}" -lt "${#VARIANTS[@]}" ]]; then
  echo "[ERR] PARALLEL=1 时 GPU_LIST 至少需要 ${#VARIANTS[@]} 张卡，当前仅有 ${#GPUS[@]} 张。" >&2
  exit 1
fi

echo "[INFO] DAPT_ROOT=$DAPT_ROOT"
echo "[INFO] VARIANTS=${VARIANTS[*]}"
echo "[INFO] PARALLEL=$PARALLEL RESUME=$RESUME"
echo "[INFO] RUN_PRETRAIN=$RUN_PRETRAIN RUN_KVNER=$RUN_KVNER RUN_EBQA=$RUN_EBQA"
echo "[INFO] LOG_DIR=$LOG_DIR"

if [[ "$PARALLEL" == "1" ]]; then
  echo "[INFO] 并行启动三组实验"
  pids=()
  for i in "${!VARIANTS[@]}"; do
    variant="${VARIANTS[$i]}"
    gpu="${GPUS[$i]}"
    echo "[LAUNCH] ${variant} -> GPU ${gpu}"
    run_variant "$variant" "$gpu" &
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
  run_gpu="${GPUS[0]}"
  echo "[INFO] 串行执行，统一使用 GPU ${run_gpu}"
  for variant in "${VARIANTS[@]}"; do
    run_variant "$variant" "$run_gpu"
  done
fi

echo "[OK] 全部噪声消融实验执行完成。"
echo "[OK] 结果文件："
for variant in "${VARIANTS[@]}"; do
  echo "  - Task1: ${DAPT_ROOT}/runs/noise_${variant}_report_task1.json"
  echo "  - Task3: ${DAPT_ROOT}/runs/noise_${variant}_report_task3.json"
  echo "  - Task2: ${DAPT_ROOT}/runs/noise_${variant}_report_task2.json"
done
