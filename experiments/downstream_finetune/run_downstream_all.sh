#!/usr/bin/env bash
set -euo pipefail

# One-click downstream finetune + eval for 4 tokenizer variants (T1~T4)
# - Task1/3: KV-NER pipeline (train_with_noise.py -> compare_models.py -> align_for_scorer_span.py -> scorer.py)
# - Task2: EBQA pipeline (convert_ebqa.py -> train_ebqa.py -> predict_ebqa.py -> aggregate -> preprocess_ebqa_real_h200.py -> MedStruct-S-master/scorer.py)

DAPT_ROOT="${DAPT_ROOT:-/data/ocean/DAPT}"
OUT_ROOT="${OUT_ROOT:-/data/ocean/DAPT/ablation/tokenizer}"
SEED="${SEED:-42}"

# Run 4 variants in parallel, one GPU per variant.
# Override example: export GPU_LIST=0,1,4,5
GPU_LIST="${GPU_LIST:-0,1,4,5}"

NOISE_BINS="${NOISE_BINS:-${DAPT_ROOT}/workspace/noise_bins.json}"
QUERY_SET="${QUERY_SET:-${DAPT_ROOT}/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/keys_merged_1027_cleaned.json}"

REAL_TRAIN_JSON="${REAL_TRAIN_JSON:-${DAPT_ROOT}/biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json}"
REAL_TEST_JSON="${REAL_TEST_JSON:-${DAPT_ROOT}/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json}"

GEN_DIR="${GEN_DIR:-${DAPT_ROOT}/experiments/downstream_finetune/generated_configs}"
LOG_DIR="${LOG_DIR:-${DAPT_ROOT}/runs/downstream_logs}"
mkdir -p "$LOG_DIR"

mkdir -p "${DAPT_ROOT}/runs"
mkdir -p "${DAPT_ROOT}/data/kv_ner_prepared_comparison"

cd "$DAPT_ROOT"

python "${DAPT_ROOT}/experiments/downstream_finetune/gen_downstream_configs.py" \
  --dapt_root "$DAPT_ROOT" \
  --out_root "$OUT_ROOT" \
  --seed "$SEED" \
  --query_set "$QUERY_SET" \
  --output_dir "$GEN_DIR"

variants=(t1 t2 t3 t4)

IFS=',' read -r -a gpus <<< "$GPU_LIST"
if [[ "${#gpus[@]}" -lt "${#variants[@]}" ]]; then
  echo "[ERR] GPU_LIST has ${#gpus[@]} GPUs but need ${#variants[@]} (variants: ${variants[*]})."
  echo "      Set: export GPU_LIST=0,1,4,5"
  exit 1
fi

_variant_model_dir() {
  local v="$1"
  echo "${OUT_ROOT}/runs/${v}_full_seed${SEED}/final_staged_model"
}

run_variant() {
  local v="$1"
  local gpu="$2"

  export CUDA_VISIBLE_DEVICES="$gpu"

  echo "============================================================"
  echo "[${v}] START (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
  echo "============================================================"

  local MODEL_DIR
  MODEL_DIR="$(_variant_model_dir "$v")"
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "[ERR] [${v}] Missing model dir: $MODEL_DIR" >&2
    return 2
  fi

  echo "============================================================"
  echo "[${v}] Task1/3 (KV-NER) finetune + eval"
  echo "============================================================"

  local KV_CFG
  local KV_OUT_SUMMARY
  KV_CFG="${GEN_DIR}/kv_ner_config_${v}.json"
  KV_OUT_SUMMARY="${DAPT_ROOT}/runs/${v}_kvner_eval_summary.json"

  python "${DAPT_ROOT}/dapt_eval_package/pre_struct/kv_ner/train_with_noise.py" \
    --config "$KV_CFG" \
    --noise_bins "$NOISE_BINS" \
    2>&1 | tee "${LOG_DIR}/${v}_kvner_train.gpu${gpu}.log"

  python "${DAPT_ROOT}/dapt_eval_package/pre_struct/kv_ner/compare_models.py" \
    --ner_config "$KV_CFG" \
    --keys_file "$QUERY_SET" \
    --test_data "$REAL_TEST_JSON" \
    --noise_bins "$NOISE_BINS" \
    --output_summary "$KV_OUT_SUMMARY" \
    2>&1 | tee "${LOG_DIR}/${v}_kvner_predict.gpu${gpu}.log"

  # compare_models.py emits:
  #   ${KV_OUT_SUMMARY%.json}_preds.jsonl
  #   ${KV_OUT_SUMMARY%.json}_gt.jsonl
  local KV_PREDS_JSONL
  KV_PREDS_JSONL="${KV_OUT_SUMMARY%.json}_preds.jsonl"

  local ALIGNED_GT
  local ALIGNED_PREDS
  ALIGNED_GT="${DAPT_ROOT}/runs/${v}_task13_aligned_gt.jsonl"
  ALIGNED_PREDS="${DAPT_ROOT}/runs/${v}_task13_aligned_preds.jsonl"

  python "${DAPT_ROOT}/scripts/align_for_scorer_span.py" \
    --gt_in "$REAL_TEST_JSON" \
    --pred_in "$KV_PREDS_JSONL" \
    --gt_out "$ALIGNED_GT" \
    --pred_out "$ALIGNED_PREDS" \
    2>&1 | tee "${LOG_DIR}/${v}_task13_align.gpu${gpu}.log"

  python "${DAPT_ROOT}/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py" \
    --pred_file "$ALIGNED_PREDS" \
    --gt_file "$ALIGNED_GT" \
    --schema_file "$QUERY_SET" \
    --task_type task1 \
    --overlap_threshold -1 \
    --output_file "${DAPT_ROOT}/runs/${v}_report_task1.json" \
    2>&1 | tee "${LOG_DIR}/${v}_task1_score.gpu${gpu}.log"

  python "${DAPT_ROOT}/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py" \
    --pred_file "$ALIGNED_PREDS" \
    --gt_file "$ALIGNED_GT" \
    --schema_file "$QUERY_SET" \
    --task_type task3 \
    --overlap_threshold -1 \
    --output_file "${DAPT_ROOT}/runs/${v}_report_task3.json" \
    2>&1 | tee "${LOG_DIR}/${v}_task3_score.gpu${gpu}.log"

  echo "============================================================"
  echo "[${v}] Task2 (EBQA) finetune + eval"
  echo "============================================================"

  local EBQA_TRAIN_JSONL
  local EBQA_EVAL_JSONL
  EBQA_TRAIN_JSONL="${DAPT_ROOT}/data/kv_ner_prepared_comparison/ebqa_train_real_${v}.jsonl"
  EBQA_EVAL_JSONL="${DAPT_ROOT}/data/kv_ner_prepared_comparison/ebqa_eval_real_${v}.jsonl"

  # 1) Convert KV-NER format -> EBQA JSONL (tokenizer-dependent, so per variant)
  python "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py" \
    --input_file "$REAL_TRAIN_JSON" \
    --output_file "$EBQA_TRAIN_JSONL" \
    --struct_path "$QUERY_SET" \
    --tokenizer_name "$MODEL_DIR" \
    --noise_bins "$NOISE_BINS" \
    2>&1 | tee "${LOG_DIR}/${v}_ebqa_convert_train.gpu${gpu}.log"

  python "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py" \
    --input_file "$REAL_TEST_JSON" \
    --output_file "$EBQA_EVAL_JSONL" \
    --struct_path "$QUERY_SET" \
    --tokenizer_name "$MODEL_DIR" \
    --noise_bins "$NOISE_BINS" \
    2>&1 | tee "${LOG_DIR}/${v}_ebqa_convert_eval.gpu${gpu}.log"

  # 2) Train EBQA
  local EBQA_CFG
  EBQA_CFG="${GEN_DIR}/ebqa_config_${v}.json"
  python "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/train_ebqa.py" \
    --config "$EBQA_CFG" \
    2>&1 | tee "${LOG_DIR}/${v}_ebqa_train.gpu${gpu}.log"

  # 3) Predict
  local EBQA_PREDS_QA
  EBQA_PREDS_QA="${DAPT_ROOT}/runs/ebqa_${v}_preds.jsonl"
  python "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/predict_ebqa.py" \
    --model_dir "${DAPT_ROOT}/runs/ebqa_${v}/best" \
    --tokenizer "$MODEL_DIR" \
    --data_path "$EBQA_EVAL_JSONL" \
    --output_preds "$EBQA_PREDS_QA" \
    2>&1 | tee "${LOG_DIR}/${v}_ebqa_predict.gpu${gpu}.log"

  # 4) Aggregate QA preds -> doc preds
  local EBQA_PREDS_DOC
  EBQA_PREDS_DOC="${DAPT_ROOT}/runs/ebqa_${v}_doc_preds.jsonl"
  python "${DAPT_ROOT}/dapt_eval_package/pre_struct/ebqa/aggregate_qa_preds_to_doc.py" \
    --raw_file "$REAL_TEST_JSON" \
    --qa_pred_file "$EBQA_PREDS_QA" \
    --output_file "$EBQA_PREDS_DOC" \
    --prefer score \
    2>&1 | tee "${LOG_DIR}/${v}_ebqa_aggregate.gpu${gpu}.log"

  # 5) Align for task2 scorer
  local EBQA_ALIGNED_DIR
  EBQA_ALIGNED_DIR="${DAPT_ROOT}/runs/ebqa_${v}_aligned"
  python "${DAPT_ROOT}/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/preprocess_ebqa_real_h200.py" \
    --gt_file "$REAL_TEST_JSON" \
    --pred_file "$EBQA_PREDS_DOC" \
    --output_dir "$EBQA_ALIGNED_DIR" \
    2>&1 | tee "${LOG_DIR}/${v}_ebqa_align.gpu${gpu}.log"

  local ALIGNED_GT_T2
  local ALIGNED_PRED_T2
  ALIGNED_GT_T2="${EBQA_ALIGNED_DIR}/gt_ebqa_aligned.jsonl"
  ALIGNED_PRED_T2="${EBQA_ALIGNED_DIR}/aligned_$(basename "$EBQA_PREDS_DOC")"

  # 6) Score task2 (teammate scorer requires cwd)
  pushd "${DAPT_ROOT}/dapt_eval_package/MedStruct-S-master" >/dev/null
  python scorer.py \
    --pred_file "$ALIGNED_PRED_T2" \
    --gt_file "$ALIGNED_GT_T2" \
    --query_set "$QUERY_SET" \
    --task_type task2 \
    --output_file "${DAPT_ROOT}/runs/${v}_report_task2.json" \
    2>&1 | tee "${LOG_DIR}/${v}_task2_score.gpu${gpu}.log"
  popd >/dev/null

  echo "============================================================"
  echo "[${v}] DONE"
  echo "============================================================"
}

echo "[INFO] Launching variants in parallel"
pids=()
for i in "${!variants[@]}"; do
  v="${variants[$i]}"
  gpu="${gpus[$i]}"
  echo "[LAUNCH] ${v} -> GPU ${gpu}"
  run_variant "$v" "$gpu" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "[ERR] A variant job failed (pid=$pid). Killing remaining jobs..." >&2
    for other_pid in "${pids[@]}"; do
      if [[ "$other_pid" != "$pid" ]]; then
        kill "$other_pid" 2>/dev/null || true
      fi
    done
    wait || true
    exit 1
  fi
done

echo "[OK] All variants finished. Reports under: ${DAPT_ROOT}/runs"
echo "[OK] Finetuned model dirs:"
for v in "${variants[@]}"; do
  echo "  - KV-NER (${v}): ${DAPT_ROOT}/runs/kv_ner_finetuned_${v}/best"
  echo "  - EBQA  (${v}): ${DAPT_ROOT}/runs/ebqa_${v}/best"
done
