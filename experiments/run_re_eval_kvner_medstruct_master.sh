#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_DAPT_ROOT="/data/ocean/DAPT"
if [[ ! -d "$DEFAULT_DAPT_ROOT" ]]; then
  DEFAULT_DAPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

DAPT_ROOT="${DAPT_ROOT:-$DEFAULT_DAPT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1

echo "[INFO] DAPT_ROOT=${DAPT_ROOT}"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"

"${PYTHON_BIN}" "${DAPT_ROOT}/experiments/re_eval_kvner_medstruct_master.py" \
  --dapt_root "${DAPT_ROOT}" \
  "$@"
