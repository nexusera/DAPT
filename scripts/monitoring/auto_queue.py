#!/usr/bin/env python3
"""Auto-launch the next pending CPT / SC task whenever target GPUs free up.

Constraint (updated 2026-05-14): NEW runs can land on GPUs 2-7.
GPU 0/1 are user's other-project workloads, off limits. GPU 2 was
previously off-limits because of another user's vLLM, but they freed it,
so 2-7 are all available for our queue.

Queue is a hard-coded list below, in priority order. Each entry says
which GPUs it needs and the tmux window + command to launch it in. The
watcher loops every poll_secs and:

  1. checks `nvidia-smi` memory used for GPUs {2,3,4,5,6,7}
  2. for each pending entry, sees if its required GPU subset is "free"
     (all listed GPUs have memory.used < free_threshold_mb)
  3. if free, launches via `tmux send-keys` and marks entry as launched

Manual launch via `tmux send-keys -t cot:<window> '<cmd>' Enter`. Logs
to /data/ocean/code/dapt/logs/auto_queue.log.

A queue entry is also "blocked" until its `depends_on` (other entry
names) are launched — useful for FT tasks that need CPT output.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


REPO = "/data/ocean/code/dapt"
NSP = "/data/ocean/DAPT/data/pseudo_kv_labels_filtered.json"
BASE_06B = "/data/ocean/model/Qwen/Qwen3-0.6B-Base"
BASE_17B = "/data/ocean/model/Qwen/Qwen3-1.7B-Base"
ENV_PREFIX = "source /opt/anaconda3/etc/profile.d/conda.sh && conda activate medical_bert"


@dataclass
class QueueEntry:
    name: str
    required_gpus: list[int]
    tmux_window: str
    cmd: str
    depends_on: list[str] = field(default_factory=list)
    require_path: Optional[str] = None  # only launch if this filesystem path exists
    launched_at: Optional[float] = None
    notes: str = ""


def make_cpt_cmd(
    *,
    name: str,
    model: str,
    gpus: list[int],
    schedule: str,
    noise_mode: str = "bucket",
    per_device: int = 64,
    ga: int = 2,
    span_epochs: float = 1.0,
    nsp_epochs: float = 3.0,
    num_rounds: int = 1,
    ddp: bool = False,
) -> str:
    gpu_csv = ",".join(str(g) for g in gpus)
    out_dir = f"{REPO}/model/{name}"
    log = f"{REPO}/logs/{name}.log"
    launcher = "python" if not ddp else f"torchrun --nproc_per_node={len(gpus)} --master_port=2952{gpus[0]}"
    base_args = (
        f"-m kv_llm.train_cpt --model_name_or_path {model} --nsp_data {NSP} "
        f"--output_dir {out_dir} --schedule {schedule} --bf16 --gradient_checkpointing "
        f"--per_device_train_batch_size {per_device} --gradient_accumulation_steps {ga} "
        f"--noise_mode {noise_mode} --logging_steps 25 --save_steps 1000"
    )
    if schedule in {"full", "span", "plain_clm"}:
        base_args += f" --span_epochs_per_round {span_epochs}"
    if schedule in {"full", "nsp"}:
        base_args += f" --nsp_epochs_per_round {nsp_epochs}"
    if schedule == "full":
        base_args += f" --num_rounds {num_rounds}"
    return (
        f"clear; {ENV_PREFIX} && export CUDA_VISIBLE_DEVICES={gpu_csv} && "
        f"{launcher} {base_args} 2>&1 | tee {log}"
    )


# ============================ FT helper ======================================

# Map of CPT-variant key -> (CPT artifact path on disk, FT label, plan_id)
# When new CPT variants finish, add them here.
CPT_OUTPUTS = {
    "06b_full":      (f"{REPO}/model/kv_llm_qwen3_0.6b_full/final_model",                  "0.6B full",        "D2.3"),
    "06b_no_noise":  (f"{REPO}/model/kv_llm_qwen3_0.6b_no_noise/final_model",              "0.6B no_noise",    "D2.3"),
    "06b_no_kvnsp":  (f"{REPO}/model/kv_llm_qwen3_0.6b_no_kvnsp/span/final_model",         "0.6B no_kvnsp",    "D2.3"),
    "06b_plain_clm": (f"{REPO}/model/kv_llm_qwen3_0.6b_plain_clm/plain_clm/final_model",   "0.6B plain_clm",   "D2.4"),
    "06b_no_span":   (f"{REPO}/model/kv_llm_qwen3_0.6b_no_span/kv_nsp/final_model",        "0.6B no_span",     "D2.3"),
    "17b_full":      (f"{REPO}/model/kv_llm_qwen3_1.7b_full/final_model",                  "1.7B full",        "D2.5"),
    "17b_no_noise":  (f"{REPO}/model/kv_llm_qwen3_1.7b_no_noise/final_model",              "1.7B no_noise",    "D3.1"),
    "17b_plain_clm": (f"{REPO}/model/kv_llm_qwen3_1.7b_plain_clm/plain_clm/final_model",   "1.7B plain_clm",   "D3.2"),
}

FT_TRAIN_DATA = f"{REPO}/data_full/medstruct_train_pairs.jsonl"
FT_TEST_DATA  = f"{REPO}/data_full/medstruct_test_pairs.jsonl"


def make_ft_cmd(*, name: str, cpt_dir: str, gpus: list[int], use_lora: bool = True,
                lora_rank: int = 8, epochs: float = 3.0,
                per_device: int = 1, ga: int = 8) -> str:
    gpu_csv = ",".join(str(g) for g in gpus)
    out_dir = f"{REPO}/model/ft/{name}"
    log = f"{REPO}/logs/{name}.log"
    pred_out = f"{REPO}/results/preds/{name}.jsonl"
    lora_flag = "--use_lora" if use_lora else ""
    ft_cmd = (
        f"python -m kv_llm.fine_tune_sft "
        f"--model_name_or_path {cpt_dir} "
        f"--train_data {FT_TRAIN_DATA} "
        f"--output_dir {out_dir} "
        f"{lora_flag} --lora_rank {lora_rank} "
        f"--num_train_epochs {epochs} "
        f"--per_device_train_batch_size {per_device} "
        f"--gradient_accumulation_steps {ga} "
        f"--bf16"
    )
    base_flag = f"--base_model {cpt_dir}" if use_lora else ""
    predict_cmd = (
        f"python -m kv_llm.predict "
        f"--model_dir {out_dir} {base_flag} "
        f"--test_data {FT_TEST_DATA} "
        f"--output {pred_out} "
        f"--bf16 --max_new_tokens 1024"
    )
    return (
        f"clear; {ENV_PREFIX} && export CUDA_VISIBLE_DEVICES={gpu_csv} && "
        f"{ft_cmd} 2>&1 | tee {log} && {predict_cmd} 2>&1 | tee -a {log}"
    )


# ============================ Queue (priority order) =========================

QUEUE: list[QueueEntry] = [
    # P0 — 1.7B w/o NoiseEmb (Day 2 D2.1) — needs 2 GPUs DDP, target 6+7 after
    # the current 1.7B full DDP finishes (auto-fires after that completes).
    QueueEntry(
        name="kv_llm_qwen3_1.7b_no_noise",
        required_gpus=[6, 7],
        tmux_window="kv17b_no_noise",
        cmd=make_cpt_cmd(
            name="kv_llm_qwen3_1.7b_no_noise",
            model=BASE_17B, gpus=[6, 7],
            schedule="full", noise_mode="none",
            per_device=32, ga=2, ddp=True,
        ),
        notes="D2.1 — same shape as 1.7B full but noise_mode=none",
    ),

    # P0 — D3.19 3 seed main results (plan 2026-05-14). seed=42 = seed1 already done.
    # Add seed=2 and seed=3 for KV-LLM 0.6B full + 1.7B full. KV-BERT seed=2/3 is a
    # separate codebase, not queued here.
    # seed=2 was manually launched on GPU 3 at 10:54 (queue state pre-populated).
    QueueEntry(
        name="kv_llm_qwen3_0.6b_full_seed2",
        required_gpus=[3],
        tmux_window="kv06b_full_s2",
        cmd=make_cpt_cmd(
            name="kv_llm_qwen3_0.6b_full_seed2",
            model=BASE_06B, gpus=[3],
            schedule="full", noise_mode="bucket",
            per_device=64, ga=2, ddp=False,
        ).replace("--save_steps 1000", "--save_steps 1000 --seed 2"),
        notes="D3.19 — seed=2 main result (launched manually 10:54 on GPU 3)",
    ),
    QueueEntry(
        name="kv_llm_qwen3_0.6b_full_seed3",
        required_gpus=[3],  # same GPU as seed2; depends_on chains them serially
        tmux_window="kv06b_full_s3",
        cmd=make_cpt_cmd(
            name="kv_llm_qwen3_0.6b_full_seed3",
            model=BASE_06B, gpus=[3],
            schedule="full", noise_mode="bucket",
            per_device=64, ga=2, ddp=False,
        ).replace("--save_steps 1000", "--save_steps 1000 --seed 3"),
        depends_on=["kv_llm_qwen3_0.6b_full_seed2"],
        notes="D3.19 — seed=3 main result, follows seed2 on GPU 3",
    ),
    QueueEntry(
        name="kv_llm_qwen3_1.7b_full_seed2",
        required_gpus=[6, 7],
        tmux_window="kv17b_full_s2",
        cmd=make_cpt_cmd(
            name="kv_llm_qwen3_1.7b_full_seed2",
            model=BASE_17B, gpus=[6, 7],
            schedule="full", noise_mode="bucket",
            per_device=32, ga=2, ddp=True,
        ).replace("--save_steps 1000", "--save_steps 1000 --seed 2"),
        depends_on=["kv_llm_qwen3_1.7b_no_noise"],  # 6+7 occupied by D2.1 until it finishes
        notes="D3.19 — seed=2 main result for variance (1.7B)",
    ),
    QueueEntry(
        name="kv_llm_qwen3_1.7b_full_seed3",
        required_gpus=[6, 7],
        tmux_window="kv17b_full_s3",
        cmd=make_cpt_cmd(
            name="kv_llm_qwen3_1.7b_full_seed3",
            model=BASE_17B, gpus=[6, 7],
            schedule="full", noise_mode="bucket",
            per_device=32, ga=2, ddp=True,
        ).replace("--save_steps 1000", "--save_steps 1000 --seed 3"),
        depends_on=["kv_llm_qwen3_1.7b_full_seed2"],
        notes="D3.19 — seed=3 main result (1.7B)",
    ),

    # === SC2-A random-mask CPT (plan D1.4) ===
    # Same data, same model, only mask strategy differs. Tests whether
    # entity-aware masking provides any gain over random masking in CPT.
    QueueEntry(
        name="kv_llm_qwen3_0.6b_random_mask",
        required_gpus=[3],
        tmux_window="sc2a_random_mask",
        cmd=make_cpt_cmd(
            name="kv_llm_qwen3_0.6b_random_mask",
            model=BASE_06B, gpus=[3],
            schedule="random_mask", noise_mode="bucket",
            per_device=64, ga=2, ddp=False,
        ),
        depends_on=["kv_llm_qwen3_0.6b_full_seed3"],  # serial after seed3 on GPU 3
        notes="D1.4 SC2-A — random span mask, same data, same arch",
    ),

    # === SC3-B Qwen3-Instruct + KV-NSP SFT (plan D1.8) ===
    QueueEntry(
        name="sc3b_qwen3_0.6b_instruct_kvnsp_sft",
        required_gpus=[2],
        tmux_window="sc3b_kvnsp_sft",
        cmd=(
            f"clear; {ENV_PREFIX} && export CUDA_VISIBLE_DEVICES=2 && "
            f"python -m kv_llm.sft_kvnsp "
            f"--model_name_or_path /data/ocean/model/Qwen/Qwen3-0.6B "
            f"--nsp_data {NSP} "
            f"--output_dir {REPO}/model/sc3b_qwen3_0.6b_instruct_kvnsp_sft "
            f"--use_lora --lora_rank 8 --bf16 "
            f"--num_train_epochs 3.0 --per_device_train_batch_size 8 "
            f"--gradient_accumulation_steps 4 "
            f"2>&1 | tee {REPO}/logs/sc3b_qwen3_0.6b_instruct_kvnsp_sft.log"
        ),
        depends_on=["kv_llm_qwen3_1.7b_plain_clm"],  # wait for GPU 2 to free
        notes="D1.8 SC3-B — Instruct + KV-NSP SFT LoRA on GPU 2",
    ),

    # === SC3-C Qwen3-Instruct zero-shot baseline (plan D1.9) ===
    # Eval only, very fast (~10 min); piggyback on any single GPU.
    QueueEntry(
        name="sc3c_qwen3_0.6b_instruct_base",
        required_gpus=[2],
        tmux_window="sc3c_zero_shot",
        cmd=(
            f"clear; {ENV_PREFIX} && export CUDA_VISIBLE_DEVICES=2 && "
            f"python {REPO}/scripts/eval/zero_shot_kv_extract.py "
            f"--model /data/ocean/model/Qwen/Qwen3-0.6B "
            f"--test {REPO}/data_full/medstruct_test_pairs.jsonl "
            f"--output {REPO}/results/eval/sc3c_zero_shot.csv "
            f"--bf16 --use-chat-template "
            f"2>&1 | tee {REPO}/logs/sc3c_qwen3_0.6b_instruct_base.log"
        ),
        depends_on=["sc3b_qwen3_0.6b_instruct_kvnsp_sft"],  # serial on GPU 2
        notes="D1.9 SC3-C — Qwen3-Instruct zero-shot eval",
    ),

    # === D3.21(c) Noise-dim causal ablation (plan §10) ===
    # Tiny eval (~15 min) — only needs the CPT 0.6B full model (done) +
    # medstruct test set. Can run on any free GPU.
    QueueEntry(
        name="d3_21c_noise_dim_perturbation",
        required_gpus=[2],
        tmux_window="d3_21c_noise_pert",
        cmd=(
            f"clear; {ENV_PREFIX} && export CUDA_VISIBLE_DEVICES=2 && "
            f"python {REPO}/scripts/analysis/noise_dim_perturbation.py "
            f"--cpt-dir {REPO}/model/kv_llm_qwen3_0.6b_full/final_model "
            f"--test {REPO}/data_full/medstruct_test_pairs.jsonl "
            f"--output {REPO}/results/eval/d3_21c_noise_dim_06b_full.csv "
            f"--bf16 "
            f"2>&1 | tee {REPO}/logs/d3_21c_noise_dim_perturbation.log"
        ),
        depends_on=["sc3c_qwen3_0.6b_instruct_base"],
        notes="D3.21(c) — zero each of 7 noise dims, score on MedStruct-S",
    ),

    # ====================================================================
    # ===== GPU 4 long chain (fires after current FT predict batch done) =
    # ====================================================================
    # Quick mechanism evals first (10min–1h each), then KV-BERT R1 seeds.

    # D3.21(a) span-mask attention pattern (~10 min)
    QueueEntry(
        name="d3_21a_span_mask_attn",
        required_gpus=[4],
        tmux_window="d3_21a_span_attn",
        cmd=(
            f"clear; {ENV_PREFIX} && export CUDA_VISIBLE_DEVICES=4 && "
            f"python {REPO}/scripts/analysis/attention_extract.py --mode span_mask "
            f"--cpt-dir {REPO}/model/kv_llm_qwen3_0.6b_full/final_model "
            f"--test-data /data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json "
            f"--num-samples 20 --bf16 "
            f"--output {REPO}/results/eval/d3_21a_span_attn.jsonl "
            f"2>&1 | tee {REPO}/logs/d3_21a_span_mask_attn.log"
        ),
        depends_on=["ft_kv_llm_06b_plain_clm_medstruct"],  # last FT on GPU 4
        require_path=f"{REPO}/model/kv_llm_qwen3_0.6b_full/final_model",
        notes="D3.21(a) — span mask attn viz",
    ),
    # D3.21(b) KV-NSP last-token attention (~10 min)
    QueueEntry(
        name="d3_21b_kvnsp_lasttoken_attn",
        required_gpus=[4],
        tmux_window="d3_21b_kvnsp_attn",
        cmd=(
            f"clear; {ENV_PREFIX} && export CUDA_VISIBLE_DEVICES=4 && "
            f"python {REPO}/scripts/analysis/attention_extract.py --mode kvnsp_lasttoken "
            f"--cpt-dir {REPO}/model/kv_llm_qwen3_0.6b_full/final_model "
            f"--kv-pairs {NSP} --num-samples 50 --bf16 "
            f"--output {REPO}/results/eval/d3_21b_kvnsp_attn.jsonl "
            f"2>&1 | tee {REPO}/logs/d3_21b_kvnsp_lasttoken_attn.log"
        ),
        depends_on=["d3_21a_span_mask_attn"],
        notes="D3.21(b) — KV-NSP last-token segment attn",
    ),
    # D3.7 probing classifier (~1h)
    QueueEntry(
        name="probing_layerwise",
        required_gpus=[4],
        tmux_window="d3_7_probing",
        cmd=(
            f"clear; {ENV_PREFIX} && export CUDA_VISIBLE_DEVICES=4 && "
            f"python {REPO}/scripts/analysis/probing_classifier.py "
            f"--model-dir {REPO}/model/kv_llm_qwen3_0.6b_full/final_model "
            f"--base-model /data/ocean/model/Qwen/Qwen3-0.6B-Base "
            f"--probe-data {REPO}/data_full/medstruct_test_pairs.jsonl "
            f"--output {REPO}/results/eval/d3_7_probing_06b.csv "
            f"--bf16 --max-samples 200 "
            f"2>&1 | tee {REPO}/logs/probing_layerwise.log"
        ),
        depends_on=["d3_21b_kvnsp_lasttoken_attn"],
        notes="D3.7 — layer-wise probing (3 tasks × CPT-vs-base)",
    ),
    # D3.8 CKA similarity (~30 min)
    QueueEntry(
        name="cka_similarity",
        required_gpus=[4],
        tmux_window="d3_8_cka",
        cmd=(
            f"clear; {ENV_PREFIX} && export CUDA_VISIBLE_DEVICES=4 && "
            f"python {REPO}/scripts/analysis/cka_similarity.py "
            f"--model-a {REPO}/model/kv_llm_qwen3_0.6b_full/final_model "
            f"--model-b /data/ocean/model/Qwen/Qwen3-0.6B-Base "
            f"--probe-data {REPO}/data_full/medstruct_test_pairs.jsonl "
            f"--output {REPO}/results/eval/d3_8_cka_06b_vs_base.csv "
            f"--bf16 --max-samples 200 "
            f"2>&1 | tee {REPO}/logs/cka_similarity.log"
        ),
        depends_on=["probing_layerwise"],
        notes="D3.8 — CKA between CPT'd 0.6B and base",
    ),

    # ====================================================================
    # ===== GPU 5 long chain (fires after current FT predict batch done) =
    # ====================================================================
    # D3.6 + D3.20 noise evals, then KV-BERT R2/R3 rerun chain.

    # D3.6 synthetic noise eval on 0.6B full FT
    QueueEntry(
        name="synthetic_noise_graceful_degradation",
        required_gpus=[5],
        tmux_window="d3_6_noise_eval",
        cmd=(
            f"clear; {ENV_PREFIX} && export CUDA_VISIBLE_DEVICES=5 && "
            f"python {REPO}/scripts/eval/eval_noise_matrix.py "
            f"--model kv_llm_06b_full {REPO}/model/ft/ft_kv_llm_06b_full_medstruct "
            f"--base {REPO}/model/kv_llm_qwen3_0.6b_full/final_model "
            f"--bench {REPO}/data_full/synthetic_noise_benchmark.jsonl "
            f"--output {REPO}/results/eval/d3_6_noise_06b_full.csv "
            f"--bf16 --limit-per-level 100 "
            f"2>&1 | tee {REPO}/logs/synthetic_noise_graceful_degradation.log"
        ),
        depends_on=["ft_kv_llm_17b_full_medstruct"],  # last FT on GPU 5
        notes="D3.6 — graceful degradation curve, 6 noise levels",
    ),
    # D3.20 component × noise cross-cut (4 models × 6 levels)
    QueueEntry(
        name="component_noise_cross_cut",
        required_gpus=[5],
        tmux_window="d3_20_component_noise",
        cmd=(
            f"clear; {ENV_PREFIX} && export CUDA_VISIBLE_DEVICES=5 && "
            f"python {REPO}/scripts/eval/eval_noise_matrix.py "
            f"--model full      {REPO}/model/ft/ft_kv_llm_06b_full_medstruct "
            f"--model no_kvnsp  {REPO}/model/ft/ft_kv_llm_06b_no_kvnsp_medstruct "
            f"--model no_noise  {REPO}/model/ft/ft_kv_llm_06b_no_noise_medstruct "
            f"--model no_span   {REPO}/model/ft/ft_kv_llm_06b_no_span_medstruct "
            f"--base {REPO}/model/kv_llm_qwen3_0.6b_full/final_model "
            f"--base {REPO}/model/kv_llm_qwen3_0.6b_no_kvnsp/span/final_model "
            f"--base {REPO}/model/kv_llm_qwen3_0.6b_no_noise/final_model "
            f"--base {REPO}/model/kv_llm_qwen3_0.6b_no_span/kv_nsp/final_model "
            f"--bench {REPO}/data_full/synthetic_noise_benchmark.jsonl "
            f"--output {REPO}/results/eval/d3_20_component_noise.csv "
            f"--bf16 --limit-per-level 100 "
            f"2>&1 | tee {REPO}/logs/component_noise_cross_cut.log"
        ),
        depends_on=["synthetic_noise_graceful_degradation"],
        notes="D3.20 — 4 ablation × 6 noise = 24 cells (strong RQ4)",
    ),

    # ====================================================================
    # ===== KV-BERT 重跑包 R1-R5 (split across GPU 4 + GPU 5) ===========
    # ====================================================================
    # GPU 4 chain (after CKA): R1×3 + R5 attention.
    QueueEntry(
        name="rerun_kv_bert_full_seed1",
        required_gpus=[4],
        tmux_window="rerun_kvbert_s1",
        cmd=f"bash {REPO}/experiments/rerun_kvbert/run_rerun.sh rerun_kv_bert_full_seed1 4",
        depends_on=["cka_similarity"],
        notes="R1 — KV-BERT full CPT seed=1",
    ),
    QueueEntry(
        name="rerun_kv_bert_full_seed2",
        required_gpus=[4],
        tmux_window="rerun_kvbert_s2",
        cmd=f"bash {REPO}/experiments/rerun_kvbert/run_rerun.sh rerun_kv_bert_full_seed2 4",
        depends_on=["rerun_kv_bert_full_seed1"],
        notes="R1 — KV-BERT full CPT seed=2",
    ),
    QueueEntry(
        name="rerun_kv_bert_full_seed3",
        required_gpus=[4],
        tmux_window="rerun_kvbert_s3",
        cmd=f"bash {REPO}/experiments/rerun_kvbert/run_rerun.sh rerun_kv_bert_full_seed3 4",
        depends_on=["rerun_kv_bert_full_seed2"],
        notes="R1 — KV-BERT full CPT seed=3",
    ),
    QueueEntry(
        name="rerun_kv_bert_attention",
        required_gpus=[4],
        tmux_window="rerun_kvbert_attn",
        cmd=f"bash {REPO}/experiments/rerun_kvbert/run_rerun.sh rerun_kv_bert_attention 4",
        depends_on=["rerun_kv_bert_full_seed3"],
        notes="R5(a) — Attention re-gen on R1 seed=1 checkpoint",
    ),

    # GPU 5 chain (after Component×Noise): R2×3 + R3×2 + R5 IG.
    QueueEntry(
        name="rerun_kv_bert_no_kvmlm",
        required_gpus=[5],
        tmux_window="rerun_kvbert_no_kvmlm",
        cmd=f"bash {REPO}/experiments/rerun_kvbert/run_rerun.sh rerun_kv_bert_no_kvmlm 5",
        depends_on=["component_noise_cross_cut"],
        notes="R2 — w/o KV-MLM",
    ),
    QueueEntry(
        name="rerun_kv_bert_no_kvnsp",
        required_gpus=[5],
        tmux_window="rerun_kvbert_no_kvnsp",
        cmd=f"bash {REPO}/experiments/rerun_kvbert/run_rerun.sh rerun_kv_bert_no_kvnsp 5",
        depends_on=["rerun_kv_bert_no_kvmlm"],
        notes="R2 — w/o KV-NSP",
    ),
    QueueEntry(
        name="rerun_kv_bert_no_noise",
        required_gpus=[5],
        tmux_window="rerun_kvbert_no_noise",
        cmd=f"bash {REPO}/experiments/rerun_kvbert/run_rerun.sh rerun_kv_bert_no_noise 5",
        depends_on=["rerun_kv_bert_no_kvnsp"],
        notes="R2 — w/o NoiseEmb",
    ),
    QueueEntry(
        name="rerun_kv_bert_noise_linear",
        required_gpus=[5],
        tmux_window="rerun_kvbert_lin",
        cmd=f"bash {REPO}/experiments/rerun_kvbert/run_rerun.sh rerun_kv_bert_noise_linear 5",
        depends_on=["rerun_kv_bert_no_noise"],
        notes="R3 — noise_mode=linear",
    ),
    QueueEntry(
        name="rerun_kv_bert_noise_mlp",
        required_gpus=[5],
        tmux_window="rerun_kvbert_mlp",
        cmd=f"bash {REPO}/experiments/rerun_kvbert/run_rerun.sh rerun_kv_bert_noise_mlp 5",
        depends_on=["rerun_kv_bert_noise_linear"],
        notes="R3 — noise_mode=mlp",
    ),
    QueueEntry(
        name="rerun_kv_bert_ig",
        required_gpus=[5],
        tmux_window="rerun_kvbert_ig",
        cmd=f"bash {REPO}/experiments/rerun_kvbert/run_rerun.sh rerun_kv_bert_ig 5",
        depends_on=["rerun_kv_bert_noise_mlp"],
        notes="R5(b) — IG re-gen on R1 seed=1 checkpoint",
    ),

    # P0 — D2.3 KV-LLM 0.6B variants × MedStruct-S FT.
    # Five variants, each LoRA r=8, GPUs 4-7 only per user. Launch on whichever
    # of 4/5 frees first; the watcher reads CPT artifact path to know if ready.
    QueueEntry(
        name="ft_kv_llm_06b_full_medstruct",
        required_gpus=[4],
        tmux_window="ft_06b_full",
        cmd=make_ft_cmd(name="ft_kv_llm_06b_full_medstruct",
                        cpt_dir=CPT_OUTPUTS["06b_full"][0], gpus=[4]),
        require_path=CPT_OUTPUTS["06b_full"][0],
        notes="D2.3 — main result FT",
    ),
    QueueEntry(
        name="ft_kv_llm_06b_no_noise_medstruct",
        required_gpus=[5],
        tmux_window="ft_06b_no_noise",
        cmd=make_ft_cmd(name="ft_kv_llm_06b_no_noise_medstruct",
                        cpt_dir=CPT_OUTPUTS["06b_no_noise"][0], gpus=[5]),
        require_path=CPT_OUTPUTS["06b_no_noise"][0],
        notes="D2.3 — noise embedding ablation FT",
    ),
    QueueEntry(
        name="ft_kv_llm_06b_no_kvnsp_medstruct",
        required_gpus=[4],
        tmux_window="ft_06b_no_kvnsp",
        cmd=make_ft_cmd(name="ft_kv_llm_06b_no_kvnsp_medstruct",
                        cpt_dir=CPT_OUTPUTS["06b_no_kvnsp"][0], gpus=[4]),
        depends_on=["ft_kv_llm_06b_full_medstruct"],
        require_path=CPT_OUTPUTS["06b_no_kvnsp"][0],
        notes="D2.3 — KV-NSP ablation FT (wait for GPU 4)",
    ),
    QueueEntry(
        name="ft_kv_llm_06b_no_span_medstruct",
        required_gpus=[5],
        tmux_window="ft_06b_no_span",
        cmd=make_ft_cmd(name="ft_kv_llm_06b_no_span_medstruct",
                        cpt_dir=CPT_OUTPUTS["06b_no_span"][0], gpus=[5]),
        depends_on=["ft_kv_llm_06b_no_noise_medstruct"],
        require_path=CPT_OUTPUTS["06b_no_span"][0],
        notes="D2.3 — span ablation FT (wait for GPU 5)",
    ),
    QueueEntry(
        name="ft_kv_llm_06b_plain_clm_medstruct",
        required_gpus=[4],
        tmux_window="ft_06b_plain_clm",
        cmd=make_ft_cmd(name="ft_kv_llm_06b_plain_clm_medstruct",
                        cpt_dir=CPT_OUTPUTS["06b_plain_clm"][0], gpus=[4]),
        depends_on=["ft_kv_llm_06b_no_kvnsp_medstruct"],
        require_path=CPT_OUTPUTS["06b_plain_clm"][0],
        notes="D2.4 — Plain CLM FT (wait for GPU 4)",
    ),
    QueueEntry(
        name="ft_kv_llm_17b_full_medstruct",
        required_gpus=[5],
        tmux_window="ft_17b_full",
        cmd=make_ft_cmd(name="ft_kv_llm_17b_full_medstruct",
                        cpt_dir=CPT_OUTPUTS["17b_full"][0], gpus=[5]),
        depends_on=["ft_kv_llm_06b_no_span_medstruct"],
        require_path=CPT_OUTPUTS["17b_full"][0],
        notes="D2.5 — 1.7B full FT (wait for GPU 5)",
    ),
]

# Tasks that will land here later (need code first, or further dependencies):
#   - SC0 MacBERT M0/M1/M2  (need KV-BERT staged training scripts adapted)
#   - SC1 MacBERT prefix-LM (need prefix-LM mask variant)
#   - SC2-A 0.6B random-mask CPT (need random-mask in span_corruption)
#   - SC3-B Qwen3-Instruct + KVNSP-SFT (need SFT prompt formatter)
#   - SC3-C Qwen3-Instruct base eval (need MedStruct-S eval harness)


# ============================ GPU watcher ====================================

def gpu_used_mb() -> dict[int, int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        text=True, timeout=10,
    )
    used = {}
    for line in out.strip().splitlines():
        idx, mem = [x.strip() for x in line.split(",")]
        used[int(idx)] = int(mem)
    return used


def all_free(gpus: list[int], used: dict[int, int], threshold_mb: int) -> bool:
    return all(used.get(g, 0) < threshold_mb for g in gpus)


def _live_processes_for_run(name: str) -> bool:
    """Return True iff a python/torchrun process whose --output_dir or
    --model_dir basename matches `name` is currently on the GPU. Mirrors
    dashboard.py _live_run_to_gpus() so the queue and the dashboard agree
    on 'is this run still alive'."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
    except Exception:
        return False
    pat = __import__("re").compile(r"--(?:output_dir|model_dir)(?:=|\s+)(\S+)")
    for line in out.strip().splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", errors="replace").replace("\0", " ")
        except Exception:
            continue
        m = pat.search(cmdline)
        if not m:
            continue
        if Path(m.group(1).rstrip("/")).name == name:
            return True
    return False


def is_run_finished(name: str, logs_dir: Path) -> bool:
    """A queue dependency is satisfied only once the named run has actually
    finished. Decision logic (in order):
      1. If nvidia-smi still has the run's process on a GPU → NOT finished.
         (Handles 1.7B FT predict that's silent for >5min between
          '[predict] N done' lines — would falsely look quiet otherwise.)
      2. log file must exist
      3. log must have been quiet for > 60s (process exited cleanly)
      4. log must contain either '[OK] KV-LLM full CPT saved' (CPT epilogue)
         or '[OK] KV-LLM SFT saved' (FT epilogue) or any 'train_runtime'
         (single-phase span/nsp/plain_clm)."""
    if _live_processes_for_run(name):
        return False
    log_path = logs_dir / f"{name}.log"
    if not log_path.exists():
        return False
    try:
        stat = log_path.stat()
        quiet_for = time.time() - stat.st_mtime
    except FileNotFoundError:
        return False
    if quiet_for < 60:
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return (
        "[OK] KV-LLM full CPT saved" in text
        or "[OK] KV-LLM SFT saved" in text
        or "'train_runtime'" in text
    )


def gpus_busy_with_other_runs(gpus: list[int], own_name: str, logs_dir: Path) -> bool:
    """Return True if any of the named GPUs are currently hosting another
    launched run that hasn't finished yet. This guards against the
    "depends_on satisfied but predecessor still occupies the GPU" race."""
    return False  # placeholder — the all_free check on raw memory.used already
    # gates this when threshold is tight enough; see free-threshold-mb default.


def ensure_tmux_window(session: str, window: str) -> None:
    """Create window if missing — no-op otherwise."""
    listing = subprocess.run(
        ["tmux", "list-windows", "-t", session, "-F", "#{window_name}"],
        capture_output=True, text=True, timeout=5,
    )
    existing = listing.stdout.split() if listing.returncode == 0 else []
    if window not in existing:
        subprocess.run(
            ["tmux", "new-window", "-t", session, "-n", window, "-c", REPO],
            check=True, timeout=5,
        )


def launch(entry: QueueEntry, session: str) -> None:
    ensure_tmux_window(session, entry.tmux_window)
    target = f"{session}:{entry.tmux_window}"
    subprocess.run(["tmux", "send-keys", "-t", target, entry.cmd, "Enter"],
                   check=True, timeout=5)


def log(msg: str, path: Path) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="cot", help="tmux session name to launch into")
    ap.add_argument("--poll-secs", type=int, default=60)
    ap.add_argument("--free-threshold-mb", type=int, default=20_000,
                    help="a GPU is considered 'free' if memory.used < this")
    ap.add_argument("--logfile", default=f"{REPO}/logs/auto_queue.log")
    ap.add_argument("--state", default=f"{REPO}/logs/auto_queue.state.json",
                    help="persist launched names so restart doesn't double-launch")
    ap.add_argument("--logs-dir", default=f"{REPO}/logs",
                    help="where to look for <name>.log files when evaluating depends_on completion")
    args = ap.parse_args()

    logpath = Path(args.logfile)
    logpath.parent.mkdir(parents=True, exist_ok=True)
    statepath = Path(args.state)
    launched_set: set[str] = set()
    if statepath.exists():
        try:
            launched_set = set(json.loads(statepath.read_text()))
        except Exception:
            pass

    log(f"queue size = {len(QUEUE)}, already launched = {sorted(launched_set)}", logpath)
    log(f"polling every {args.poll_secs}s; free threshold = {args.free_threshold_mb} MB", logpath)

    while True:
        try:
            used = gpu_used_mb()
        except Exception as e:
            log(f"nvidia-smi failed: {e!r}; sleeping and retrying", logpath)
            time.sleep(args.poll_secs)
            continue

        pending = [e for e in QUEUE if e.name not in launched_set]
        if not pending:
            log("queue empty — exiting watcher.", logpath)
            return

        logs_dir = Path(args.logs_dir)
        for entry in pending:
            # depends_on now means "the dependency has FINISHED", not "launched".
            # Without this, the watcher launches the dependent within the next
            # poll cycle while the predecessor is still hogging the GPU.
            deps_ok = all(is_run_finished(d, logs_dir) for d in entry.depends_on)
            if not deps_ok:
                continue
            if entry.require_path and not Path(entry.require_path).exists():
                continue
            if all_free(entry.required_gpus, used, args.free_threshold_mb):
                log(f"launching {entry.name} on GPUs {entry.required_gpus} (window {entry.tmux_window})", logpath)
                try:
                    launch(entry, args.session)
                    launched_set.add(entry.name)
                    statepath.write_text(json.dumps(sorted(launched_set), ensure_ascii=False, indent=2))
                    # don't launch multiple in same loop — wait for next poll
                    break
                except Exception as e:
                    log(f"launch {entry.name} failed: {e!r}", logpath)

        time.sleep(args.poll_secs)


if __name__ == "__main__":
    main()
