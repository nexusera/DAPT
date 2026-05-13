#!/usr/bin/env python3
"""Harvest per-run final metrics from CPT/SC/FT logs and emit:

  1. /data/ocean/code/dapt/results/runs/<run>.json — machine-readable
  2. <paper-dir>/results/runs_summary.md           — markdown table to paste

For each log in --logs-dir we extract:
  - run name + plan_id (from CATALOGUE)
  - start timestamp (log mtime fallback)
  - state (completed / failed / running / stale / pending)
  - per-phase loss curve summary (initial, mid, final)
  - train_runtime (s), train_samples_per_second
  - last grad_norm and epoch
  - CPT config snapshot (per_device_batch, ga, lr, num_rounds...) from CLI line in log
  - artifact paths (final_model dir, checkpoint dirs)

Run locally (after rsyncing logs back) or on the server. The script imports
the run catalogue from dashboard.py so the two stay in sync.

Example:
  python scripts/monitoring/harvest_results.py \
    --logs-dir /data/ocean/code/dapt/logs \
    --models-dir /data/ocean/code/dapt/model \
    --json-out /data/ocean/code/dapt/results/runs \
    --markdown-out /Users/wy/Documents/data/KV_BERT/results/runs_summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# import RunSpec + CATALOGUE from dashboard.py
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dashboard import CATALOGUE, RunSpec, parse_run, RunStatus  # noqa: E402


LOSS_RE = re.compile(r"\{'loss':\s*([-\d.eE+]+),\s*'grad_norm':\s*([-\d.eE+]+).*?'learning_rate':\s*([-\d.eE+]+),\s*'epoch':\s*([\d.]+)\}")
RUNTIME_RE = re.compile(r"'train_runtime':\s*([\d.]+),\s*'train_samples_per_second':\s*([\d.]+),\s*'train_steps_per_second':\s*([\d.]+),\s*'train_loss':\s*([\d.eE+]+),\s*'epoch':\s*([\d.]+)\}")
CLI_FLAG_RE = re.compile(r"--(\w+)(?:\s+([^\s\\]+))?")
SPAN_START_RE = re.compile(r"\[KV-LLM\] round (\d+)/\d+: span corruption")
NSP_START_RE = re.compile(r"\[KV-LLM\] round (\d+)/\d+: KV-NSP")


def read_log(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")


def extract_loss_curve(text: str) -> dict:
    """Pull (step_idx, loss, grad_norm, lr, epoch) from each logged line."""
    points = []
    for m in LOSS_RE.finditer(text):
        try:
            loss = float(m.group(1))
            grad = float(m.group(2))
            lr = float(m.group(3))
            ep = float(m.group(4))
        except ValueError:
            continue
        points.append({"loss": loss, "grad_norm": grad, "lr": lr, "epoch": ep})
    if not points:
        return {"n_points": 0}
    return {
        "n_points": len(points),
        "first": points[0],
        "last": points[-1],
        "min_loss": min(p["loss"] for p in points),
        "max_loss": max(p["loss"] for p in points),
        "mean_grad_norm": sum(p["grad_norm"] for p in points) / len(points),
        "max_grad_norm": max(p["grad_norm"] for p in points),
    }


def extract_phase_runtimes(text: str) -> list[dict]:
    """Each train_runtime line marks end of one phase. Pair with most recent phase marker."""
    out = []
    phase_starts = []
    for m in SPAN_START_RE.finditer(text):
        phase_starts.append((m.start(), "span", int(m.group(1))))
    for m in NSP_START_RE.finditer(text):
        phase_starts.append((m.start(), "kv_nsp", int(m.group(1))))
    phase_starts.sort()
    for m in RUNTIME_RE.finditer(text):
        pos = m.start()
        # find latest phase_start before this train_runtime
        phase = ""
        round_idx = 0
        for ps_pos, p, r in reversed(phase_starts):
            if ps_pos < pos:
                phase, round_idx = p, r
                break
        out.append({
            "phase": phase,
            "round": round_idx,
            "runtime_s": float(m.group(1)),
            "samples_per_second": float(m.group(2)),
            "steps_per_second": float(m.group(3)),
            "train_loss": float(m.group(4)),
            "end_epoch": float(m.group(5)),
        })
    return out


def extract_cli_config(text: str) -> dict:
    """Find the python -m kv_llm.train_cpt invocation line and parse --flag args."""
    cfg = {}
    # find launch line
    for line in text.splitlines():
        if "kv_llm.train_cpt" in line and ("python" in line or "torchrun" in line):
            # parse flags
            tokens = re.split(r"\s+", line)
            i = 0
            while i < len(tokens):
                tok = tokens[i]
                if tok.startswith("--"):
                    key = tok[2:]
                    if i + 1 < len(tokens) and not tokens[i + 1].startswith("--") and not tokens[i + 1].endswith(".log"):
                        cfg[key] = tokens[i + 1]
                        i += 2
                    else:
                        cfg[key] = True
                        i += 1
                else:
                    i += 1
            if cfg:
                return cfg
    return cfg


def artifact_summary(models_dir: Path, run_name: str) -> dict:
    run_dir = models_dir / run_name
    out = {"output_dir": str(run_dir), "exists": run_dir.exists()}
    if not run_dir.exists():
        return out
    final_model = run_dir / "final_model"
    out["final_model_exists"] = final_model.exists()
    if final_model.exists():
        total = 0
        for p in final_model.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        out["final_model_size_mb"] = round(total / 1024 / 1024, 1)
    out["phase_subdirs"] = sorted(
        [p.name for p in run_dir.iterdir() if p.is_dir() and p.name != "final_model"]
    )
    return out


def harvest_one(spec: RunSpec, logs_dir: Path, models_dir: Path, stale_after_s: float) -> dict:
    log_path = logs_dir / f"{spec.name}.log"
    status = parse_run(spec.name, logs_dir, stale_after_s)
    rec = {
        "name": spec.name,
        "label": spec.label,
        "category": spec.category,
        "plan_id": spec.plan_id,
        "model": spec.model,
        "schedule": spec.schedule,
        "gpu": spec.gpu,
        "state": status.state,
        "log_path": str(log_path) if log_path.exists() else None,
    }
    if not log_path.exists():
        rec["artifacts"] = artifact_summary(models_dir, spec.name)
        return rec
    text = read_log(log_path)
    rec["loss_curve"] = extract_loss_curve(text)
    rec["phase_runtimes"] = extract_phase_runtimes(text)
    rec["config"] = extract_cli_config(text)
    rec["artifacts"] = artifact_summary(models_dir, spec.name)
    # status snapshot
    rec.update({
        "step": status.step,
        "total_steps": status.total,
        "pct": status.pct,
        "phase": status.phase,
        "round": status.round_idx,
        "last_loss": status.loss,
        "last_grad_norm": status.grad_norm,
        "last_epoch": status.epoch,
        "log_size_kb": status.log_size_kb,
        "last_modified_ago_s": status.last_modified_ago_s,
    })
    if status.error_snippet:
        rec["error_snippet"] = status.error_snippet
    return rec


# --- markdown rendering ------------------------------------------------------

STATE_EMOJI = {
    "completed": "✅", "running": "🏃", "stale": "⚠️",
    "failed": "❌", "pending": "·",
}


def render_markdown(records: list[dict]) -> str:
    out: list[str] = []
    out.append(f"# Run results summary — generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    out.append(
        "> 由 `scripts/monitoring/harvest_results.py` 自动生成。\n"
        "> 每次 run 状态变化（新跑完/失败/启动）后重新跑这个脚本，把下面的表格段落粘回 plan markdown。\n"
    )
    # summary counts
    counts: dict[str, int] = {}
    for r in records:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    out.append("\n**总览**：" + " · ".join(f"{STATE_EMOJI.get(k,'?')} {k}={v}" for k, v in sorted(counts.items())) + "\n")

    by_cat: dict[str, list[dict]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)

    for cat in ["CPT", "Sanity", "FT", "Eval", "Robustness", "Mechanism", "Efficiency", "Analysis", "Infra"]:
        if cat not in by_cat:
            continue
        out.append(f"\n## {cat}\n")
        out.append("| plan | task | state | model | schedule | runtime | final loss | last grad | end epoch | artifacts |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in by_cat[cat]:
            emoji = STATE_EMOJI.get(r["state"], "?")
            runtime = "—"
            if "phase_runtimes" in r and r["phase_runtimes"]:
                total = sum(p["runtime_s"] for p in r["phase_runtimes"])
                runtime = f"{total/60:.1f} min ({len(r['phase_runtimes'])} phase)"
            final_loss = "—"
            if "loss_curve" in r and r["loss_curve"].get("n_points"):
                final_loss = f"{r['loss_curve']['last']['loss']:.3f}"
                first_loss = r['loss_curve']['first']['loss']
                final_loss = f"{first_loss:.2f} → {final_loss}"
            last_grad = f"{r.get('last_grad_norm') or 0:.1f}" if r.get('last_grad_norm') is not None else "—"
            epoch = f"{r.get('last_epoch') or 0:.2f}" if r.get('last_epoch') is not None else "—"
            art = "—"
            if r.get("artifacts", {}).get("final_model_exists"):
                art = f"final_model ({r['artifacts'].get('final_model_size_mb', '?')} MB)"
            elif r.get("artifacts", {}).get("phase_subdirs"):
                art = ", ".join(r["artifacts"]["phase_subdirs"])
            out.append(
                f"| {r['plan_id'] or '—'} | {r['label']} | {emoji} {r['state']} | {r['model']} | `{r['schedule']}` | {runtime} | {final_loss} | {last_grad} | {epoch} | {art} |"
            )

    # detail section for completed/failed runs
    detailed = [r for r in records if r["state"] in {"completed", "failed"}]
    if detailed:
        out.append("\n\n## Detail per run\n")
        for r in detailed:
            out.append(f"\n### {STATE_EMOJI.get(r['state'])} `{r['name']}` ({r['label']})")
            out.append(f"- plan: {r['plan_id'] or '—'} · category: {r['category']}")
            out.append(f"- schedule: `{r['schedule']}` · model: {r['model']}")
            if r.get("config"):
                kvs = ", ".join(f"`{k}={v}`" for k, v in sorted(r['config'].items()) if k not in {"model_name_or_path", "nsp_data", "output_dir"})
                out.append(f"- config: {kvs}")
            if r.get("phase_runtimes"):
                for p in r["phase_runtimes"]:
                    out.append(f"- phase **{p['phase']}** round {p['round']}: {p['runtime_s']:.1f}s · samples/s {p['samples_per_second']:.1f} · train_loss {p['train_loss']:.3f} · end_epoch {p['end_epoch']}")
            if r.get("loss_curve", {}).get("n_points"):
                lc = r["loss_curve"]
                out.append(f"- loss curve: {lc['n_points']} logged points · first {lc['first']['loss']:.2f} → last {lc['last']['loss']:.2f} · min {lc['min_loss']:.2f} · max {lc['max_loss']:.2f} · mean_grad {lc['mean_grad_norm']:.1f}")
            if r.get("artifacts", {}).get("final_model_exists"):
                out.append(f"- artifact: `{r['artifacts']['output_dir']}/final_model/` ({r['artifacts'].get('final_model_size_mb', '?')} MB)")
            if r["state"] == "failed" and r.get("error_snippet"):
                out.append("- error tail:\n  ```\n  " + r["error_snippet"].replace("\n", "\n  ") + "\n  ```")
    return "\n".join(out) + "\n"


# --- main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="/data/ocean/code/dapt/logs")
    ap.add_argument("--models-dir", default="/data/ocean/code/dapt/model")
    ap.add_argument("--json-out", default="/data/ocean/code/dapt/results/runs")
    ap.add_argument("--markdown-out", default="/Users/wy/Documents/data/KV_BERT/results/runs_summary.md")
    ap.add_argument("--stale-after-s", type=float, default=300.0)
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    models_dir = Path(args.models_dir)
    json_dir = Path(args.json_out)
    md_path = Path(args.markdown_out)

    json_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for spec in CATALOGUE:
        rec = harvest_one(spec, logs_dir, models_dir, args.stale_after_s)
        records.append(rec)
        (json_dir / f"{spec.name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(records), encoding="utf-8")

    counts: dict[str, int] = {}
    for r in records:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    print(f"[harvest] {len(records)} records · " + " · ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"[harvest] JSON  → {json_dir}/")
    print(f"[harvest] MD    → {md_path}")


if __name__ == "__main__":
    main()
