#!/usr/bin/env bash
# Plan §11 Day 2 night / Day 3 early — KV-BERT 重跑包 R1-R6 wrapper.
#
# Re-runs KV-BERT under the SAME protocol as KV-LLM (matched seeds, same
# H200, same DAPT eval pipeline) so the cross-architecture table is clean.
# All trainers + interpretability scripts already exist in the repo root /
# experiments/interpretability/; this wrapper just stitches the right
# argument set per R-task and routes the log into logs/ so the dashboard
# + harvester pick it up by name.
#
# Usage:
#   bash experiments/rerun_kvbert/run_rerun.sh <task_id> [<gpu_id>]
#
# Supported task_ids (match dashboard catalogue):
#   rerun_kv_bert_full_seed1       (R1 — KV-MLM + KV-NSP + Noise, seed=1)
#   rerun_kv_bert_full_seed2       (R1)
#   rerun_kv_bert_full_seed3       (R1)
#   rerun_kv_bert_no_kvmlm         (R2 — train_dapt_macbert_no_mlm.py)
#   rerun_kv_bert_no_kvnsp         (R2 — train_dapt_macbert_no_nsp.py)
#   rerun_kv_bert_no_noise         (R2 — train_dapt_macbert_no_noise.py)
#   rerun_kv_bert_noise_linear     (R3 — noise_mode=linear)
#   rerun_kv_bert_noise_mlp        (R3 — noise_mode=mlp)
#   rerun_ft_<enc>_medstruct       (R4 — encoder baselines FT, where <enc>
#                                   ∈ {macbert, roberta_wwm, bert_base_chinese, mbert})
#   rerun_kv_bert_attention        (R5 — attention re-gen)
#   rerun_kv_bert_ig               (R5 — IG re-gen)
#   rerun_kv_bert_ft_{cmeie|cblue} (R6 — blocked on data agreement)

set -euo pipefail

TASK="${1:?usage: $0 <task_id> [<gpu_id>]}"
GPU="${2:-3}"

REPO=/data/ocean/code/dapt
LOG_DIR="${REPO}/logs"
MODEL_DIR="${REPO}/model/rerun_kvbert"
mkdir -p "${LOG_DIR}" "${MODEL_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate medical_bert

LOG="${LOG_DIR}/${TASK}.log"
OUT_BASE="${MODEL_DIR}/${TASK}"

case "${TASK}" in
  rerun_kv_bert_full_seed1|rerun_kv_bert_full_seed2|rerun_kv_bert_full_seed3)
    SEED="${TASK##*seed}"
    # R1: KV-MLM full CPT, seed varies. Output dir per seed to keep checkpoints separate.
    # Hyper-params mirror existing train_dapt_kvmlm.py defaults but override seed.
    cd "${REPO}"
    python train_dapt_kvmlm.py \
      --output_dir "${OUT_BASE}" \
      --seed "${SEED}" 2>&1 | tee "${LOG}"
    ;;

  rerun_kv_bert_no_kvmlm)
    cd "${REPO}"; python train_dapt_macbert_no_mlm.py --output_dir "${OUT_BASE}" 2>&1 | tee "${LOG}"
    ;;
  rerun_kv_bert_no_kvnsp)
    cd "${REPO}"; python train_dapt_macbert_no_nsp.py --output_dir "${OUT_BASE}" 2>&1 | tee "${LOG}"
    ;;
  rerun_kv_bert_no_noise)
    cd "${REPO}"; python train_dapt_macbert_no_noise.py --output_dir "${OUT_BASE}" 2>&1 | tee "${LOG}"
    ;;

  rerun_kv_bert_noise_linear)
    cd "${REPO}"; python train_dapt_kvmlm.py --output_dir "${OUT_BASE}" --noise_mode linear 2>&1 | tee "${LOG}"
    ;;
  rerun_kv_bert_noise_mlp)
    cd "${REPO}"; python train_dapt_kvmlm.py --output_dir "${OUT_BASE}" --noise_mode mlp 2>&1 | tee "${LOG}"
    ;;

  rerun_ft_macbert_medstruct|rerun_ft_roberta_wwm_medstruct|rerun_ft_bert_base_chinese_medstruct|rerun_ft_mbert_medstruct)
    # R4 — encoder baselines on MedStruct-S, FT only (no CPT rerun)
    ENC="${TASK#rerun_ft_}"; ENC="${ENC%_medstruct}"
    case "${ENC}" in
      macbert)            CKPT=/data/ocean/model/hfl-chinese-macbert-base ;;
      roberta_wwm)        CKPT=/data/ocean/model/hfl-chinese-roberta-wwm-ext ;;  # may need download
      bert_base_chinese)  CKPT=/data/ocean/model/google-bert/bert-base-chinese ;;  # adjust path
      mbert)              CKPT=/data/ocean/model/bert-base-multilingual-cased ;;
      *) echo "unknown encoder ${ENC}"; exit 2 ;;
    esac
    cd "${REPO}"
    bash experiments/downstream_finetune/run_downstream_all.sh \
      --model "${CKPT}" --tag "${TASK}" --output "${OUT_BASE}" 2>&1 | tee "${LOG}"
    ;;

  rerun_kv_bert_attention)
    # R5(a) — Attention re-gen on the rerun checkpoint (defaults to seed1)
    CKPT="${MODEL_DIR}/rerun_kv_bert_full_seed1"
    cd "${REPO}"
    python experiments/interpretability/run_attention_kv_mlm.py \
      --model_path "${CKPT}" --output_dir "${OUT_BASE}_attention" 2>&1 | tee "${LOG}"
    ;;
  rerun_kv_bert_ig)
    # R5(b) — IG re-gen on rerun checkpoint
    CKPT="${MODEL_DIR}/rerun_kv_bert_full_seed1"
    cd "${REPO}"
    python experiments/interpretability/run_ig_kvner.py \
      --model_path "${CKPT}" --output_dir "${OUT_BASE}_ig" 2>&1 | tee "${LOG}"
    ;;

  rerun_kv_bert_ft_cmeie|rerun_kv_bert_ft_cblue)
    echo "[BLOCKED] ${TASK} pending CMeIE-V2 / CBLUE-CMeEE data agreement"
    exit 3
    ;;

  *)
    echo "unknown task ${TASK}"; exit 2 ;;
esac

echo "[OK] ${TASK} finished — log at ${LOG}"
