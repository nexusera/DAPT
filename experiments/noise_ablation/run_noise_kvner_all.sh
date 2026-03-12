#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

require_file "$NOISE_BINS"
require_file "$QUERY_SET"
require_file "$REAL_TEST_JSON"
require_file "$REAL_TRAIN_JSON"
require_file "${DAPT_ROOT}/dapt_eval_package/pre_struct/kv_ner/train_with_noise.py"
require_file "${DAPT_ROOT}/dapt_eval_package/pre_struct/kv_ner/compare_models.py"
require_file "${DAPT_ROOT}/scripts/align_for_scorer_span.py"
require_file "${DAPT_ROOT}/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py"

run_variant_kvner() {
  local variant="$1"
  local gpu="$2"
  export CUDA_VISIBLE_DEVICES="$gpu"

  local model_dir="$(pretrain_model_dir "$variant")"
  local kv_cfg="${GEN_DIR}/kv_ner_config_noise_${variant}.json"
  local kv_best_dir="${DAPT_ROOT}/runs/kv_ner_finetuned_noise_${variant}/best"
  local summary="$(kv_summary_path "$variant")"
  local pred_jsonl="${summary%.json}_preds.jsonl"
  local aligned_gt="$(kv_aligned_gt "$variant")"
  local aligned_pred="$(kv_aligned_pred "$variant")"
  local report_t1="${DAPT_ROOT}/runs/noise_${variant}_report_task1.json"
  local report_t3="${DAPT_ROOT}/runs/noise_${variant}_report_task3.json"

  require_path "$variant" "pretrained-model" "$model_dir" dir
  gen_kv_config "$variant" "$model_dir"
  require_path "$variant" "generate-config" "$kv_cfg" file

  echo "============================================================"
  echo "[${variant}] KV-NER START (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
  echo "============================================================"

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

  echo "[${variant}] KV-NER DONE"
}

print_common_header "kv-ner"
run_variants_parallel_or_serial run_variant_kvner

echo "[OK] KV-NER 阶段完成。"
for variant in "${VARIANTS[@]}"; do
  echo "  - ${variant} Task1: ${DAPT_ROOT}/runs/noise_${variant}_report_task1.json"
  echo "  - ${variant} Task3: ${DAPT_ROOT}/runs/noise_${variant}_report_task3.json"
done
