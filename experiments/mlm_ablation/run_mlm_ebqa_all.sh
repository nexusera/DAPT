#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

require_file "$NOISE_BINS"
require_file "$QUERY_SET"
require_file "$REAL_TRAIN_JSON"
require_file "$REAL_TEST_JSON"
require_file "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py"
require_file "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/train_ebqa.py"
require_file "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/predict_ebqa.py"
require_file "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/aggregate_qa_preds_to_doc.py"
require_file "${DAPT_ROOT}/dapt_eval_package/MedStruct-S-master/utils/preprocess_ebqa_real_h200.py"
require_file "${DAPT_ROOT}/dapt_eval_package/MedStruct-S-master/scorer.py"

run_variant_ebqa() {
  local variant="$1"
  local gpu="$2"
  export CUDA_VISIBLE_DEVICES="$gpu"

  local model_dir="$(pretrain_model_dir "$variant")"
  local ebqa_train="$(ebqa_train_jsonl "$variant")"
  local ebqa_eval="$(ebqa_eval_jsonl "$variant")"
  local ebqa_cfg="${GEN_DIR}/ebqa_config_mlm_${variant}.json"
  local ebqa_best_dir="${DAPT_ROOT}/runs/ebqa_mlm_${variant}/best"
  local ebqa_pred_qa="${DAPT_ROOT}/runs/ebqa_mlm_${variant}_preds.jsonl"
  local ebqa_pred_doc="${DAPT_ROOT}/runs/ebqa_mlm_${variant}_doc_preds.jsonl"
  local ebqa_aligned_dir="${DAPT_ROOT}/runs/ebqa_mlm_${variant}_aligned"
  local ebqa_aligned_gt="${ebqa_aligned_dir}/gt_ebqa_aligned.jsonl"
  local ebqa_aligned_pred="${ebqa_aligned_dir}/aligned_$(basename "$ebqa_pred_doc")"
  local report_t2="${DAPT_ROOT}/runs/mlm_${variant}_report_task2.json"

  require_path "$variant" "pretrained-model" "$model_dir" dir
  gen_ebqa_config "$variant" "$model_dir" "$ebqa_train" "$ebqa_eval"
  require_path "$variant" "generate-config" "$ebqa_cfg" file

  echo "============================================================"
  echo "[${variant}] EBQA START (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
  echo "============================================================"

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
      "$PYTHON_BIN" "${DAPT_ROOT}/dapt_eval_package/MedStruct-S-master/utils/preprocess_ebqa_real_h200.py" \
      --gt_file "$REAL_TEST_JSON" \
      --pred_file "$ebqa_pred_doc" \
      --output_dir "$ebqa_aligned_dir"
    require_path "$variant" "ebqa-align" "$ebqa_aligned_gt" file
    require_path "$variant" "ebqa-align" "$ebqa_aligned_pred" file
  fi

  if [[ "$RESUME" == "1" && -s "$report_t2" ]]; then
    echo "[${variant}] [SKIP] Task2 score (found: $report_t2)"
  else
    run_logged "$variant" "task2-score" "${LOG_DIR}/${variant}_task2_score.gpu${gpu}.log" \
      "$PYTHON_BIN" "${DAPT_ROOT}/scripts/run_medstruct_scorer.py" \
      --pred_file "$ebqa_aligned_pred" \
      --gt_file "$ebqa_aligned_gt" \
      --query_set "$QUERY_SET" \
      --task_type task2 \
      --output_file "$report_t2"
    require_path "$variant" "task2-score" "$report_t2" file
  fi

  echo "[${variant}] EBQA DONE"
}

print_common_header "ebqa"
run_variants_parallel_or_serial run_variant_ebqa

echo "[OK] MLM ablation EBQA 阶段完成。"
for variant in "${VARIANTS[@]}"; do
  echo "  - ${variant} Task2: ${DAPT_ROOT}/runs/mlm_${variant}_report_task2.json"
done
