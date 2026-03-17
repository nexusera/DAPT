#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

KV_VAL_F1_GUARD_ENABLE="${KV_VAL_F1_GUARD_ENABLE:-1}"
KV_VAL_F1_GUARD_PATIENCE="${KV_VAL_F1_GUARD_PATIENCE:-5}"
KV_VAL_F1_GUARD_TOL="${KV_VAL_F1_GUARD_TOL:-0.001}"
KV_VAL_F1_GUARD_MIN_EPOCHS="${KV_VAL_F1_GUARD_MIN_EPOCHS:-5}"

require_file "$NOISE_BINS"
require_file "$QUERY_SET"
require_file "$REAL_TEST_JSON"
require_file "$REAL_TRAIN_JSON"
require_file "${DAPT_ROOT}/dapt_eval_package/pre_struct/kv_ner/train_with_noise.py"
require_file "${DAPT_ROOT}/dapt_eval_package/pre_struct/kv_ner/compare_models.py"
require_file "${DAPT_ROOT}/scripts/align_for_scorer_span.py"
require_file "${DAPT_ROOT}/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py"

check_pred_nonempty() {
  local variant="$1"
  local stage="$2"
  local pred_file="$3"
  local min_pairs="${4:-1}"
  local stats
  stats="$($PYTHON_BIN - "$pred_file" <<'PY'
import json, sys
pred = sys.argv[1]
n = 0
pairs = 0
with open(pred, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        n += 1
        try:
            obj = json.loads(line)
        except Exception:
            continue
        pairs += len(obj.get('pairs', []) or [])
print(f"{n}\t{pairs}")
PY
)"
  local rows
  local pairs
  rows="${stats%%$'\t'*}"
  pairs="${stats##*$'\t'}"
  echo "[${variant}] [CHECK] ${stage}: rows=${rows}, total_pairs=${pairs}"
  if [[ "${pairs}" -lt "${min_pairs}" ]]; then
    echo "[ERR] [${variant}] ${stage} 预测结果为空或几乎为空：${pred_file}" >&2
    echo "[ERR] [${variant}] 请检查对应日志：${LOG_DIR}/${variant}_kvner_predict.gpu${CUDA_VISIBLE_DEVICES}.log" >&2
    return 1
  fi
}

check_kv_train_quality() {
  local variant="$1"
  local summary_file="$2"
  local min_f1="${3:-0.0001}"
  local val
  val="$($PYTHON_BIN - "$summary_file" <<'PY'
import json, sys
f = sys.argv[1]
obj = json.load(open(f, 'r', encoding='utf-8'))
v = obj.get('best_val_f1', 0.0)
try:
    v = float(v)
except Exception:
    v = 0.0
print(v)
PY
)"
  echo "[${variant}] [CHECK] kvner-train: best_val_f1=${val}"
  if ! $PYTHON_BIN - "$val" "$min_f1" <<'PY'
import sys
v = float(sys.argv[1])
t = float(sys.argv[2])
raise SystemExit(0 if v >= t else 1)
PY
  then
    echo "[ERR] [${variant}] kvner-train 疑似塌缩（best_val_f1=${val} < ${min_f1}）。" >&2
    echo "[ERR] [${variant}] 请检查训练日志和 tokenizer 配置。" >&2
    return 1
  fi
}

check_kv_valf1_guard() {
  local variant="$1"
  local train_log="$2"

  if [[ "$KV_VAL_F1_GUARD_ENABLE" != "1" ]]; then
    return 0
  fi

  if [[ ! -s "$train_log" ]]; then
    echo "[WARN] [${variant}] KV val_f1 guard skipped: missing log ${train_log}"
    return 0
  fi

  if ! "$PYTHON_BIN" - "$variant" "$train_log" "$KV_VAL_F1_GUARD_PATIENCE" "$KV_VAL_F1_GUARD_TOL" "$KV_VAL_F1_GUARD_MIN_EPOCHS" <<'PY'
import re
import sys

variant = sys.argv[1]
log_file = sys.argv[2]
patience = int(float(sys.argv[3]))
tol = float(sys.argv[4])
min_epochs = int(float(sys.argv[5]))

pat = re.compile(r"Validation F1:\s*([0-9]*\.?[0-9]+)")
vals = []
with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            try:
                vals.append(float(m.group(1)))
            except Exception:
                pass

if len(vals) < min_epochs:
    print(f"[WARN] [{variant}] KV val_f1 guard skipped: only {len(vals)} validation points (< {min_epochs})")
    raise SystemExit(0)

max_zero_streak = 0
cur = 0
for v in vals:
    if abs(v) <= tol:
        cur += 1
        if cur > max_zero_streak:
            max_zero_streak = cur
    else:
        cur = 0

if max_zero_streak >= patience:
    tail = ", ".join(f"{x:.4f}" for x in vals[-min(12, len(vals)):])
    print(
        f"[ERR] [{variant}] KV val_f1 guard triggered: max_zero_streak={max_zero_streak} >= patience={patience}, "
        f"tol={tol}."
    )
    print(f"[ERR] [{variant}] Recent val_f1: {tail}")
    raise SystemExit(2)

print(
    f"[OK] [{variant}] KV val_f1 guard pass: max_zero_streak={max_zero_streak}, "
    f"patience={patience}, tol={tol}"
)
PY
  then
    return 1
  fi

  return 0
}

run_variant_kvner() {
  local variant="$1"
  local gpu="$2"
  export CUDA_VISIBLE_DEVICES="$gpu"

  local ratio_name
  ratio_name="$(variant_ratio_name "$variant")"

  local model_dir
  model_dir="$(pretrain_model_dir "$variant")"
  local kv_cfg="${GEN_DIR}/kv_ner_config_nsp_ratio_${ratio_name}.json"
  local kv_best_dir="${DAPT_ROOT}/runs/kv_ner_finetuned_nsp_ratio_${ratio_name}/best"
  local summary
  summary="$(kv_summary_path "$variant")"
  local pred_jsonl="${summary%.json}_preds.jsonl"
  local aligned_gt
  aligned_gt="$(kv_aligned_gt "$variant")"
  local aligned_pred
  aligned_pred="$(kv_aligned_pred "$variant")"
  local report_t1="${DAPT_ROOT}/runs/nsp_ratio_${ratio_name}_report_task1.json"
  local report_t3="${DAPT_ROOT}/runs/nsp_ratio_${ratio_name}_report_task3.json"

  require_path "$variant" "pretrained-model" "$model_dir" dir
  gen_kv_config "$variant" "$model_dir" "$TOKENIZER_PATH"
  require_path "$variant" "generate-config" "$kv_cfg" file

  echo "============================================================"
  echo "[${variant}] KV-NER START (ratio=${ratio_name}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
  echo "============================================================"

  if [[ "$RESUME" == "1" && -d "$kv_best_dir" ]]; then
    echo "[${variant}] [SKIP] KV-NER train (found: $kv_best_dir)"
  else
    local train_log="${LOG_DIR}/${variant}_kvner_train.gpu${gpu}.log"
    run_logged "$variant" "kvner-train" "${LOG_DIR}/${variant}_kvner_train.gpu${gpu}.log" \
      "$PYTHON_BIN" "${DAPT_ROOT}/dapt_eval_package/pre_struct/kv_ner/train_with_noise.py" \
      --config "$kv_cfg" \
      --noise_bins "$NOISE_BINS"
    require_path "$variant" "kvner-train" "$kv_best_dir" dir
    check_kv_valf1_guard "$variant" "$train_log"
    check_kv_train_quality "$variant" "${DAPT_ROOT}/runs/kv_ner_finetuned_nsp_ratio_${ratio_name}/training_summary.json" 0.0001
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
    check_pred_nonempty "$variant" "kvner-predict" "$pred_jsonl" 1
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
    check_pred_nonempty "$variant" "task13-align" "$aligned_pred" 1
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
  ratio_name="$(variant_ratio_name "$variant")"
  echo "  - ratio ${ratio_name} Task1: ${DAPT_ROOT}/runs/nsp_ratio_${ratio_name}_report_task1.json"
  echo "  - ratio ${ratio_name} Task3: ${DAPT_ROOT}/runs/nsp_ratio_${ratio_name}_report_task3.json"
done
