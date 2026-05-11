#!/usr/bin/env bash
set -euo pipefail

# One-click KV-NER true faithfulness comparison for:
#   full / no_noise / no_nsp / no_mlm
#
# Expected existing files (from your pipeline_xiaorong.md alignment stage):
#   runs/macbert_eval_aligned_preds.jsonl, runs/macbert_eval_aligned_gt.jsonl
#   runs/no_noise_eval_aligned_preds.jsonl, runs/no_noise_eval_aligned_gt.jsonl
#   runs/no_nsp_eval_aligned_preds.jsonl, runs/no_nsp_eval_aligned_gt.jsonl
#   runs/no_mlm_eval_aligned_preds.jsonl, runs/no_mlm_eval_aligned_gt.jsonl

ROOT_DIR="${ROOT_DIR:-/data/ocean/DAPT}"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export CUDA_VISIBLE_DEVICES="${GPU_LIST:-1,2}"

TASK="${TASK:-task3}"
RAW_FILE="${RAW_FILE:-biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json}"

IG_STEPS="${IG_STEPS:-64}"
INTERNAL_BATCH_SIZE="${INTERNAL_BATCH_SIZE:-4}"
BASELINE="${BASELINE:-pad}"
DEVICE="${DEVICE:-cuda}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TOP_K="${TOP_K:-10}"
FAITH_TOPK_FRAC="${FAITH_TOPK_FRAC:-0.2}"
FAITH_TOPK_MAX="${FAITH_TOPK_MAX:-32}"

OUT_DIR="${OUT_DIR:-runs/ig}"
mkdir -p "$OUT_DIR"

# Variant registry: name|config|aligned_prefix
# aligned_prefix should map to:
#   runs/${aligned_prefix}_aligned_preds.jsonl and runs/${aligned_prefix}_aligned_gt.jsonl
VARIANTS=()
VARIANTS+=("full|dapt_eval_package/pre_struct/kv_ner/kv_ner_config_macbert.json|macbert_eval")
VARIANTS+=("no_noise|dapt_eval_package/pre_struct/kv_ner/kv_ner_config_no_noise.json|no_noise_eval")
VARIANTS+=("no_nsp|dapt_eval_package/pre_struct/kv_ner/kv_ner_config_no_nsp.json|no_nsp_eval")
VARIANTS+=("no_mlm|dapt_eval_package/pre_struct/kv_ner/kv_ner_config_no_mlm.json|no_mlm_eval")

# Optional filtering by env, e.g. ONLY_VARIANTS=full,no_noise
if [[ -n "${ONLY_VARIANTS:-}" ]]; then
  IFS=',' read -r -a keep_arr <<< "$ONLY_VARIANTS"
  declare -A KEEP=()
  for k in "${keep_arr[@]}"; do KEEP["$k"]=1; done
  filtered=()
  for row in "${VARIANTS[@]}"; do
    name="${row%%|*}"
    if [[ -n "${KEEP[$name]:-}" ]]; then
      filtered+=("$row")
    fi
  done
  VARIANTS=("${filtered[@]}")
fi

if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
  echo "[ERR] No variants to run. Check ONLY_VARIANTS setting." >&2
  exit 1
fi

resolve_model_dir() {
  local cfg="$1"
  python - <<'PY' "$cfg"
import json, os, sys
cfg_path = sys.argv[1]
obj = json.load(open(cfg_path, encoding='utf-8'))
train = obj.get('train', {}) if isinstance(obj, dict) else {}
out_dir = train.get('output_dir') if isinstance(train, dict) else None
if isinstance(out_dir, str) and out_dir.strip():
    cand = os.path.join(out_dir.strip(), 'best')
    print(cand)
else:
    m = obj.get('model_name_or_path', '') if isinstance(obj, dict) else ''
    print(m)
PY
}

run_variant() {
  local name="$1"
  local cfg="$2"
  local prefix="$3"

  local pred_file="runs/${prefix}_aligned_preds.jsonl"
  local gt_file="runs/${prefix}_aligned_gt.jsonl"

  if [[ ! -f "$pred_file" ]]; then
    echo "[ERR] Missing pred file: $pred_file" >&2
    exit 1
  fi
  if [[ ! -f "$gt_file" ]]; then
    echo "[ERR] Missing gt file: $gt_file" >&2
    exit 1
  fi
  if [[ ! -f "$cfg" ]]; then
    echo "[ERR] Missing config file: $cfg" >&2
    exit 1
  fi
  if [[ ! -f "$RAW_FILE" ]]; then
    echo "[ERR] Missing raw file: $RAW_FILE" >&2
    exit 1
  fi

  local model_dir
  model_dir="$(resolve_model_dir "$cfg")"

  local analysis_set="${OUT_DIR}/${name}_analysis_set_${TASK}.jsonl"
  local ig_file="${OUT_DIR}/${name}_kvner_${TASK}_ig_true.jsonl"
  local faith_file="${OUT_DIR}/${name}_kvner_${TASK}_faithfulness_true.json"

  echo "[RUN] variant=$name"
  echo "      cfg=$cfg"
  echo "      model_dir=$model_dir"
  echo "      pred=$pred_file"
  echo "      gt=$gt_file"

  python experiments/interpretability/build_analysis_set.py \
    --task "$TASK" \
    --pred_file "$pred_file" \
    --gt_file "$gt_file" \
    --raw_file "$RAW_FILE" \
    --output "$analysis_set"

  python experiments/interpretability/run_ig_kvner.py \
    --config "$cfg" \
    --model_dir "$model_dir" \
    --analysis_set "$analysis_set" \
    --ig_steps "$IG_STEPS" \
    --internal_batch_size "$INTERNAL_BATCH_SIZE" \
    --baseline "$BASELINE" \
    --device "$DEVICE" \
    --max_samples "$MAX_SAMPLES" \
    --top_k "$TOP_K" \
    --compute_faithfulness \
    --faithfulness_topk_frac "$FAITH_TOPK_FRAC" \
    --faithfulness_topk_max "$FAITH_TOPK_MAX" \
    --output "$ig_file"

  python experiments/interpretability/eval_faithfulness.py \
    --ig_file "$ig_file" \
    --metric deletion_aopc comprehensiveness sufficiency \
    --output "$faith_file"
}

for row in "${VARIANTS[@]}"; do
  IFS='|' read -r name cfg prefix <<< "$row"
  run_variant "$name" "$cfg" "$prefix"
done

# Aggregate comparison table
COMPARE_JSON="${OUT_DIR}/kvner_${TASK}_faithfulness_compare_true.json"
COMPARE_CSV="${OUT_DIR}/kvner_${TASK}_faithfulness_compare_true.csv"

python - <<'PY' "$OUT_DIR" "$TASK" "$COMPARE_JSON" "$COMPARE_CSV" "${VARIANTS[@]}"
import csv, json, os, sys
out_dir, task, out_json, out_csv, *rows = sys.argv[1:]
variants = [r.split('|')[0] for r in rows]

summary = {}
for name in variants:
    fp = os.path.join(out_dir, f"{name}_kvner_{task}_faithfulness_true.json")
    if not os.path.isfile(fp):
        continue
    obj = json.load(open(fp, encoding='utf-8'))
    m = obj.get('summary', {}).get('metrics', {})
    summary[name] = {
        'deletion_aopc_mean': m.get('deletion_aopc', {}).get('aggregate', {}).get('mean'),
        'deletion_aopc_std': m.get('deletion_aopc', {}).get('aggregate', {}).get('std'),
        'deletion_aopc_mode': m.get('deletion_aopc', {}).get('mode_count', {}),
        'comprehensiveness_mean': m.get('comprehensiveness', {}).get('aggregate', {}).get('mean'),
        'comprehensiveness_std': m.get('comprehensiveness', {}).get('aggregate', {}).get('std'),
        'comprehensiveness_mode': m.get('comprehensiveness', {}).get('mode_count', {}),
        'sufficiency_mean': m.get('sufficiency', {}).get('aggregate', {}).get('mean'),
        'sufficiency_std': m.get('sufficiency', {}).get('aggregate', {}).get('std'),
        'sufficiency_mode': m.get('sufficiency', {}).get('mode_count', {}),
        'num_samples': obj.get('summary', {}).get('num_samples'),
    }

json.dump(summary, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

with open(out_csv, 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow([
        'variant','num_samples',
        'deletion_aopc_mean','deletion_aopc_std','deletion_aopc_true_count',
        'comprehensiveness_mean','comprehensiveness_std','comprehensiveness_true_count',
        'sufficiency_mean','sufficiency_std','sufficiency_true_count',
    ])
    for name in variants:
        s = summary.get(name, {})
        dm = s.get('deletion_aopc_mode', {})
        cm = s.get('comprehensiveness_mode', {})
        sm = s.get('sufficiency_mode', {})
        w.writerow([
            name,
            s.get('num_samples'),
            s.get('deletion_aopc_mean'), s.get('deletion_aopc_std'), dm.get('true'),
            s.get('comprehensiveness_mean'), s.get('comprehensiveness_std'), cm.get('true'),
            s.get('sufficiency_mean'), s.get('sufficiency_std'), sm.get('true'),
        ])

print(f"[OK] comparison json: {out_json}")
print(f"[OK] comparison csv : {out_csv}")
PY

echo "[DONE] All variants completed."
echo "       See: ${COMPARE_JSON}"
echo "       See: ${COMPARE_CSV}"
