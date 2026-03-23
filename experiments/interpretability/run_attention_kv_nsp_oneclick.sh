#!/usr/bin/env bash
set -euo pipefail

# One-click runner for KV-NSP attention visualization.
# - Writes full logs to run.log
# - Prints real-time progress in terminal (via python progress prints + tee)
# - Creates timestamped output directory automatically

ROOT_DIR="${ROOT_DIR:-/data/ocean/DAPT}"
PY_SCRIPT="${PY_SCRIPT:-$ROOT_DIR/experiments/interpretability/run_attention_kv_nsp.py}"

MODEL_DIR="${MODEL_DIR:-/data/ocean/DAPT/workspace/output_macbert_kvmlm_staged/final_staged_model}"
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
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/attention_kv_nsp_${RUN_TAG}}"
LOG_FILE="$OUTPUT_DIR/run.log"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[error] script not found: $PY_SCRIPT"
  exit 1
fi
if [[ ! -e "$INPUT_FILE" ]]; then
  echo "[error] input not found: $INPUT_FILE"
  exit 1
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[error] model dir not found: $MODEL_DIR"
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
  --topk "$TOPK"
  --progress_every "$PROGRESS_EVERY"
  --device "cuda:0"
)

if [[ "$INJECT_PERFECT_NOISE" == "1" ]]; then
  CMD+=(--inject_perfect_noise)
fi
if [[ "$AUTO_GENERATE_NEGATIVES" == "1" ]]; then
  CMD+=(--auto_generate_negatives)
fi
if [[ "$RUN_ROLLOUT" == "1" ]]; then
  CMD+=(--run_rollout)
fi
if [[ "$EXCLUDE_SPECIAL_TOKENS" == "1" ]]; then
  CMD+=(--exclude_special_tokens)
fi

{
  echo "========== KV-NSP Attention Run =========="
  echo "start_time: $(date '+%F %T')"
  echo "gpu_id: $GPU_ID"
  echo "output_dir: $OUTPUT_DIR"
  echo "log_file: $LOG_FILE"
  echo "model_dir: $MODEL_DIR"
  echo "tokenizer_path: $TOKENIZER_PATH"
  echo "input_file: $INPUT_FILE"
  printf "command: "
  printf "%q " "${CMD[@]}"
  echo
  echo "=========================================="
} | tee -a "$LOG_FILE"

# stdbuf forces line-buffered output so progress appears immediately in terminal and log.
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

