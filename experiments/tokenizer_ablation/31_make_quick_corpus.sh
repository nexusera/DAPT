#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

bash 00_check_env.sh
# shellcheck disable=SC1091
source config.env

py="python"

quick_file="$OUT_ROOT/quick/train_quick_${QUICK_LINES}.txt"
mkdir -p "$OUT_ROOT/quick"

$py "$(pwd)/make_subset_corpus.py" \
  --input "$TRAIN_FILE" \
  --output "$quick_file" \
  --lines "$QUICK_LINES"

echo "[done] quick corpus: $quick_file"
