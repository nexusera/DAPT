#!/usr/bin/env python3
"""Auto-launch the next pending CPT / SC task whenever target GPUs free up.

Constraint (per user instruction 2026-05-13): NEW runs must land on GPUs
4-7 only. GPUs 0/1/3/2 currently host the first wave of 0.6B / 1.7B runs
that started on those cards and stay there until they finish — when they
finish those cards stay idle (don't repurpose them for new runs without
explicit user OK).

Queue is a hard-coded list below, in priority order. Each entry says
which GPUs it needs and the tmux window + command to launch it in. The
watcher loops every poll_secs and:

  1. checks `nvidia-smi` memory used for GPUs {4,5,6,7}
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


# ============================ Queue (priority order) =========================

QUEUE: list[QueueEntry] = [
    # P0 — 1.7B w/o NoiseEmb (Day 2 D2.1) — needs 2 GPUs DDP, target 6+7 after
    # the current 1.7B full DDP finishes
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

        for entry in pending:
            deps_ok = all(d in launched_set for d in entry.depends_on)
            if not deps_ok:
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
