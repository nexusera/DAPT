#!/usr/bin/env bash
set -euo pipefail

# Example: KV-NSP attention visualization
# Adjust paths before running.

MODEL_DIR="/data/ocean/DAPT/workspace/output_macbert_kvmlm_staged/final_staged_model"
TOKENIZER_PATH="/data/ocean/DAPT/my-medical-tokenizer"
INPUT_FILE="/data/ocean/DAPT/data/pseudo_kv_labels_filtered.json"
NOISE_BINS_JSON="/data/ocean/DAPT/workspace/noise_bins.json"
OUTPUT_DIR="/data/ocean/DAPT/runs/attention_kv_nsp"

python /data/ocean/DAPT/experiments/interpretability/run_attention_kv_nsp.py \
  --model_dir "${MODEL_DIR}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --input_file "${INPUT_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --noise_bins_json "${NOISE_BINS_JSON}" \
  --inject_perfect_noise \
  --auto_generate_negatives \
  --max_length 256 \
  --max_samples_per_group 200 \
  --last_n_layers 4 \
  --topk 5 \
  --run_rollout

# Outputs:
# - ${OUTPUT_DIR}/per_sample_metrics.jsonl
# - ${OUTPUT_DIR}/summary.json
# - ${OUTPUT_DIR}/report.md
# - ${OUTPUT_DIR}/cases/*.png
