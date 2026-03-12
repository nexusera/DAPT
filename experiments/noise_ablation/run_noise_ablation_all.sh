#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
RUN_KVNER="${RUN_KVNER:-1}"
RUN_EBQA="${RUN_EBQA:-1}"

if [[ "$RUN_PRETRAIN" != "1" && "$RUN_KVNER" != "1" && "$RUN_EBQA" != "1" ]]; then
  echo "[ERR] 至少需要启用一个阶段：RUN_PRETRAIN / RUN_KVNER / RUN_EBQA" >&2
  exit 1
fi

echo "[INFO] 总入口脚本仅负责按阶段串联调用。"
echo "[INFO] 远端执行推荐直接使用分阶段脚本："
echo "       - experiments/noise_ablation/run_noise_pretrain_all.sh"
echo "       - experiments/noise_ablation/run_noise_kvner_all.sh"
echo "       - experiments/noise_ablation/run_noise_ebqa_all.sh"

if [[ "$RUN_PRETRAIN" == "1" ]]; then
  bash "${SCRIPT_DIR}/run_noise_pretrain_all.sh"
fi

if [[ "$RUN_KVNER" == "1" ]]; then
  bash "${SCRIPT_DIR}/run_noise_kvner_all.sh"
fi

if [[ "$RUN_EBQA" == "1" ]]; then
  bash "${SCRIPT_DIR}/run_noise_ebqa_all.sh"
fi

echo "[OK] 已完成所选阶段。"
