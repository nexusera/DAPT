#!/usr/bin/env bash
set -euo pipefail

# One-click runner: KV-MLM attention explainability

ROOT_DIR="${ROOT_DIR:-/data/ocean/DAPT}"
PY_SCRIPT="${PY_SCRIPT:-$ROOT_DIR/experiments/interpretability/run_attention_kv_mlm.py}"

MODEL_DIR="${MODEL_DIR:-/data/ocean/DAPT/workspace/output_ablation_noise_bucket/final_staged_model}"
NO_KVMLM_MODEL_DIR="${NO_KVMLM_MODEL_DIR:-/data/ocean/DAPT/workspace/output_ablation_no_mlm/final_no_mlm_model}"
# Prefer model-local tokenizer by default to avoid vocab mismatch across ablation checkpoints.
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_DIR}"
INPUT_FILE="${INPUT_FILE:-/data/ocean/DAPT/data/pseudo_kv_labels_filtered.json}"
NOISE_BINS_JSON="${NOISE_BINS_JSON:-/data/ocean/DAPT/workspace/noise_bins.json}"

GPU_ID="${GPU_ID:-0}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_SAMPLES_PER_GROUP="${MAX_SAMPLES_PER_GROUP:-120}"
LAST_N_LAYERS="${LAST_N_LAYERS:-4}"
MASK_SPAN_LEN="${MASK_SPAN_LEN:-1}"
MASK_STRATEGY="${MASK_STRATEGY:-both}"  # entity / boundary / both
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"

INJECT_PERFECT_NOISE="${INJECT_PERFECT_NOISE:-1}"     # 1/0
EXCLUDE_SPECIAL_TOKENS="${EXCLUDE_SPECIAL_TOKENS:-1}" # 1/0

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-$ROOT_DIR/runs}"
MODEL_TAG="${MODEL_TAG:-main}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/attention_kv_mlm_${MODEL_TAG}_${RUN_TAG}}"
LOG_FILE="$OUTPUT_DIR/run.log"
mkdir -p "$OUTPUT_DIR"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[error] model dir not found: $MODEL_DIR"
  echo "Try one of:"
  echo "  MODEL_DIR=/data/ocean/DAPT/workspace/output_ablation_noise_bucket/final_staged_model"
  echo "  MODEL_DIR=$NO_KVMLM_MODEL_DIR"
  exit 1
fi

CMD=(python "$PY_SCRIPT"
  --model_dir "$MODEL_DIR"
  --tokenizer_path "$TOKENIZER_PATH"
  --input_file "$INPUT_FILE"
  --output_dir "$OUTPUT_DIR"
  --noise_bins_json "$NOISE_BINS_JSON"
  --max_length "$MAX_LENGTH"
  --max_samples_per_group "$MAX_SAMPLES_PER_GROUP"
  --last_n_layers "$LAST_N_LAYERS"
  --mask_span_len "$MASK_SPAN_LEN"
  --mask_strategy "$MASK_STRATEGY"
  --progress_every "$PROGRESS_EVERY"
  --device "cuda:0"
)

if [[ "$INJECT_PERFECT_NOISE" == "1" ]]; then CMD+=(--inject_perfect_noise); fi
if [[ "$EXCLUDE_SPECIAL_TOKENS" == "1" ]]; then CMD+=(--exclude_special_tokens); fi

{
  echo "========== KV-MLM Attention Run =========="
  echo "start_time: $(date '+%F %T')"
  echo "gpu_id: $GPU_ID"
  echo "output_dir: $OUTPUT_DIR"
  echo "log_file: $LOG_FILE"
  echo "model_dir: $MODEL_DIR"
  echo "mask_strategy: $MASK_STRATEGY"
  echo "input_file: $INPUT_FILE"
  printf "command: "; printf "%q " "${CMD[@]}"; echo
  echo "=========================================="
} | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$GPU_ID" stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

{
  echo "========== Run Finished =========="
  echo "end_time: $(date '+%F %T')"
  echo "output_dir: $OUTPUT_DIR"
  echo "summary: $OUTPUT_DIR/summary.json"
  echo "report: $OUTPUT_DIR/report.md"
  echo "cases: $OUTPUT_DIR/cases"
  echo "figures: $OUTPUT_DIR/figures"
  echo "=================================="
} | tee -a "$LOG_FILE"

