#!/usr/bin/env bash
set -euo pipefail

# One-click runner: Noise-Embedding attention explainability compare
# It runs two KV-NSP attention analyses (with-noise vs without-noise) and writes a compare report.

ROOT_DIR="${ROOT_DIR:-/data/ocean/DAPT}"
PY_SCRIPT="${PY_SCRIPT:-$ROOT_DIR/experiments/interpretability/run_attention_noise_compare.py}"

WITH_NOISE_MODEL_DIR="${WITH_NOISE_MODEL_DIR:-/data/ocean/DAPT/workspace/output_ablation_noise_bucket/final_staged_model}"
WITHOUT_NOISE_MODEL_DIR="${WITHOUT_NOISE_MODEL_DIR:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/data/ocean/DAPT/my-medical-tokenizer}"
INPUT_FILE="${INPUT_FILE:-/data/ocean/DAPT/data/pseudo_kv_labels_filtered.json}"
NOISE_BINS_JSON="${NOISE_BINS_JSON:-/data/ocean/DAPT/workspace/noise_bins.json}"

GPU_ID="${GPU_ID:-0}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_SAMPLES_PER_GROUP="${MAX_SAMPLES_PER_GROUP:-200}"
LAST_N_LAYERS="${LAST_N_LAYERS:-4}"
TOPK="${TOPK:-5}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"

INJECT_PERFECT_NOISE="${INJECT_PERFECT_NOISE:-1}"       # 1/0
AUTO_GENERATE_NEGATIVES="${AUTO_GENERATE_NEGATIVES:-1}" # 1/0
RUN_ROLLOUT="${RUN_ROLLOUT:-1}"                         # 1/0
EXCLUDE_SPECIAL_TOKENS="${EXCLUDE_SPECIAL_TOKENS:-1}"   # 1/0

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-$ROOT_DIR/runs}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/attention_noise_compare_${RUN_TAG}}"
LOG_FILE="$OUTPUT_DIR/run.log"
mkdir -p "$OUTPUT_DIR"

# Auto-resolve no-noise model path if not explicitly set.
if [[ -z "${WITHOUT_NOISE_MODEL_DIR}" ]]; then
  CANDIDATES=(
    "/data/ocean/DAPT/workspace/output_ablation_no_noise/final_no_noise_model"
    "/data/ocean/DAPT/workspace/output_no_noise_baseline/final_no_noise_model"
    "/data/ocean/DAPT/workspace/output_ablation_no_noise/final_staged_model"
  )
  for p in "${CANDIDATES[@]}"; do
    if [[ -d "$p" ]]; then
      WITHOUT_NOISE_MODEL_DIR="$p"
      break
    fi
  done
fi

if [[ -z "${WITHOUT_NOISE_MODEL_DIR}" ]]; then
  echo "[error] no valid WITHOUT_NOISE_MODEL_DIR found."
  echo "Please set it manually, e.g.:"
  echo "  WITHOUT_NOISE_MODEL_DIR=/data/ocean/DAPT/workspace/output_ablation_no_noise/final_no_noise_model"
  exit 1
fi

CMD=(python "$PY_SCRIPT"
  --with_noise_model_dir "$WITH_NOISE_MODEL_DIR"
  --without_noise_model_dir "$WITHOUT_NOISE_MODEL_DIR"
  --tokenizer_path "$TOKENIZER_PATH"
  --input_file "$INPUT_FILE"
  --output_dir "$OUTPUT_DIR"
  --noise_bins_json "$NOISE_BINS_JSON"
  --max_length "$MAX_LENGTH"
  --max_samples_per_group "$MAX_SAMPLES_PER_GROUP"
  --last_n_layers "$LAST_N_LAYERS"
  --topk "$TOPK"
  --progress_every "$PROGRESS_EVERY"
  --device "cuda:0"
)

if [[ "$INJECT_PERFECT_NOISE" == "1" ]]; then CMD+=(--inject_perfect_noise); fi
if [[ "$AUTO_GENERATE_NEGATIVES" == "1" ]]; then CMD+=(--auto_generate_negatives); fi
if [[ "$RUN_ROLLOUT" == "1" ]]; then CMD+=(--run_rollout); fi
if [[ "$EXCLUDE_SPECIAL_TOKENS" == "1" ]]; then CMD+=(--exclude_special_tokens); fi

{
  echo "========== Noise-Embedding Attention Compare =========="
  echo "start_time: $(date '+%F %T')"
  echo "gpu_id: $GPU_ID"
  echo "output_dir: $OUTPUT_DIR"
  echo "log_file: $LOG_FILE"
  echo "with_noise_model: $WITH_NOISE_MODEL_DIR"
  echo "without_noise_model: $WITHOUT_NOISE_MODEL_DIR"
  echo "input_file: $INPUT_FILE"
  printf "command: "; printf "%q " "${CMD[@]}"; echo
  echo "======================================================="
} | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$GPU_ID" stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

{
  echo "========== Run Finished =========="
  echo "end_time: $(date '+%F %T')"
  echo "output_dir: $OUTPUT_DIR"
  echo "with_noise: $OUTPUT_DIR/with_noise"
  echo "without_noise: $OUTPUT_DIR/without_noise"
  echo "compare_summary: $OUTPUT_DIR/compare_summary.json"
  echo "compare_report: $OUTPUT_DIR/compare_report.md"
  echo "=================================="
} | tee -a "$LOG_FILE"

