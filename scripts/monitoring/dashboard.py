#!/usr/bin/env python3
"""KV-LLM CPT dashboard — single HTML page over HTTP for run-status at-a-glance.

Run on the server:
    python scripts/monitoring/dashboard.py --port 8889 --logs-dir /data/ocean/code/dapt/logs

Open http://<server-ip>:8889/ in a browser. The page auto-refreshes every 5s.

For each known run we look at the corresponding log file in logs-dir and
classify it:
  - completed  : log ends with "[OK] KV-LLM full CPT saved" (full schedule)
                 or contains a final "train_runtime" without any later phase line
  - failed     : log ends in a Traceback / RuntimeError / ChildFailedError
  - running    : recent tqdm progress line (within --stale-after seconds)
  - stale      : log exists but no progress for > stale-after seconds
  - pending    : no log file exists yet (run not started)

Progress is parsed from the last tqdm carriage-return-terminated line.
Loss is parsed from the last "{'loss': ..., 'grad_norm': ..., 'epoch': ...}"
trainer log dict.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional


# --- run catalogue -----------------------------------------------------------

@dataclass
class RunSpec:
    name: str
    label: str
    gpu: str
    model: str
    schedule: str
    notes: str = ""


CATALOGUE: list[RunSpec] = [
    # === KV-LLM CPT (Day 1 main + ablations) ===
    RunSpec("kv_llm_qwen3_0.6b_full",        "0.6B full (main)",          "4",   "Qwen3-0.6B", "full"),
    RunSpec("kv_llm_qwen3_0.6b_no_kvnsp",    "0.6B w/o KVNSP",            "5",   "Qwen3-0.6B", "span"),
    RunSpec("kv_llm_qwen3_0.6b_no_noise",    "0.6B w/o NoiseEmb",         "0",   "Qwen3-0.6B", "full+noise=none"),
    RunSpec("kv_llm_qwen3_0.6b_plain_clm",   "0.6B Plain CLM",            "1",   "Qwen3-0.6B", "plain_clm"),
    RunSpec("kv_llm_qwen3_0.6b_no_span",     "0.6B w/o SpanCorr",         "3",   "Qwen3-0.6B", "nsp"),
    RunSpec("kv_llm_qwen3_1.7b_full",        "1.7B full (DDP)",           "6,7", "Qwen3-1.7B", "full"),
    RunSpec("kv_llm_qwen3_1.7b_plain_clm",   "1.7B Plain CLM",            "2",   "Qwen3-1.7B", "plain_clm"),
    # === planned but not yet started — SC + remaining 1.7B variants ===
    RunSpec("kv_llm_qwen3_1.7b_no_noise",    "1.7B w/o NoiseEmb",         "?",   "Qwen3-1.7B", "full+noise=none",  "scheduled Day 2"),
    RunSpec("kv_llm_qwen3_0.6b_random_mask", "0.6B random-mask CPT (SC2-A)", "?", "Qwen3-0.6B", "random-mask",     "SC2-A baseline"),
    RunSpec("sc0_macbert_m0",                "SC0-M0  MC-BERT entity mask",  "?", "MacBERT 0.11B", "MLM",          "sanity check"),
    RunSpec("sc0_macbert_m1",                "SC0-M1  KV-MLM (k-v boundary)", "?", "MacBERT 0.11B", "MLM",         "sanity check"),
    RunSpec("sc0_macbert_m2",                "SC0-M2  KV-MLM + OCR sampling", "?", "MacBERT 0.11B", "MLM",         "sanity check"),
    RunSpec("sc1_macbert_prefix_lm",         "SC1  MacBERT prefix-LM",       "?", "MacBERT 0.11B", "prefix-LM",   "sanity check"),
    RunSpec("sc3b_qwen3_0.6b_instruct_kvnsp_sft", "SC3-B Qwen3-Instruct + KVNSP SFT", "?", "Qwen3-0.6B-Instruct", "SFT", "sanity check"),
]


# --- log parsers -------------------------------------------------------------

LOSS_RE = re.compile(r"\{'loss':\s*([-\d.eE+nainf]+),\s*'grad_norm':\s*([-\d.eE+nainf]+).*?'epoch':\s*([\d.]+)\}")
TQDM_RE = re.compile(
    r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[(\d{1,2}:\d{2}(?::\d{2})?)<(\d+:\d{2}(?::\d{2})?|\?+),\s*([\d.]+)([a-z/]+)\]"
)
PHASE_SPAN_RE = re.compile(r"\[KV-LLM\] round (\d+)/(\d+): span corruption")
PHASE_NSP_RE = re.compile(r"\[KV-LLM\] round (\d+)/(\d+): KV-NSP")
TRAIN_DONE_RE = re.compile(r"train_runtime':\s*([\d.]+),")
FULL_OK_RE = re.compile(r"\[OK\] KV-LLM full CPT saved")
ERROR_RE = re.compile(r"(Traceback|RuntimeError|TypeError|ChildFailedError|CUDA error|OutOfMemoryError|FAILED)")


@dataclass
class RunStatus:
    name: str
    state: str = "pending"  # pending / running / stale / completed / failed
    phase: str = ""         # "span" / "nsp" / ""
    round_idx: str = ""     # "1/3" etc
    step: int = 0
    total: int = 0
    pct: float = 0.0
    speed: str = ""         # "3.42s/it"
    eta: str = ""
    elapsed: str = ""
    loss: Optional[float] = None
    grad_norm: Optional[float] = None
    epoch: Optional[float] = None
    last_phase_completed: list[str] = field(default_factory=list)
    last_modified_ago_s: float = -1.0
    log_size_kb: float = 0.0
    error_snippet: str = ""


def _tail_normalized(path: Path, max_bytes: int = 200_000) -> str:
    """Read last `max_bytes` of file and replace CR with newline so tqdm progress
    lines become inspectable lines."""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(-max_bytes, os.SEEK_END)
        raw = f.read()
    text = raw.decode("utf-8", errors="replace").replace("\r", "\n")
    return text


def parse_run(name: str, logs_dir: Path, stale_after_s: float) -> RunStatus:
    log_path = logs_dir / f"{name}.log"
    st = RunStatus(name=name)
    if not log_path.exists():
        st.state = "pending"
        return st
    try:
        stat = log_path.stat()
        st.log_size_kb = stat.st_size / 1024
        st.last_modified_ago_s = time.time() - stat.st_mtime
    except FileNotFoundError:
        st.state = "pending"
        return st
    text = _tail_normalized(log_path)
    if not text.strip():
        st.state = "pending"
        return st

    # detect error / completion BEFORE classifying as running
    if FULL_OK_RE.search(text):
        st.state = "completed"
    elif ERROR_RE.search(text.split("[OK] KV-LLM full CPT saved")[-1]):
        # any error AFTER (or in absence of) the OK marker means failed
        m = ERROR_RE.search(text)
        snippet_lines = text.split("\n")[-30:]
        st.error_snippet = "\n".join(l for l in snippet_lines if l.strip())[-2000:]
        st.state = "failed"
    elif st.last_modified_ago_s > stale_after_s:
        st.state = "stale"
    else:
        st.state = "running"

    # phase tracking — look at last phase markers and any train_runtime markers
    spans = list(PHASE_SPAN_RE.finditer(text))
    nsps = list(PHASE_NSP_RE.finditer(text))
    if spans or nsps:
        last_span_pos = spans[-1].start() if spans else -1
        last_nsp_pos = nsps[-1].start() if nsps else -1
        if last_nsp_pos > last_span_pos:
            m = nsps[-1]
            st.phase = "kv_nsp"
            st.round_idx = f"{m.group(1)}/{m.group(2)}"
        else:
            m = spans[-1]
            st.phase = "span"
            st.round_idx = f"{m.group(1)}/{m.group(2)}"

    # also count completed phases via train_runtime markers
    st.last_phase_completed = TRAIN_DONE_RE.findall(text)

    # latest tqdm progress line (last match wins)
    tqdms = list(TQDM_RE.finditer(text))
    if tqdms:
        m = tqdms[-1]
        st.pct = float(m.group(1))
        st.step = int(m.group(2))
        st.total = int(m.group(3))
        st.elapsed = m.group(4)
        st.eta = m.group(5)
        st.speed = f"{m.group(6)}{m.group(7)}"

    losses = list(LOSS_RE.finditer(text))
    if losses:
        m = losses[-1]
        try:
            st.loss = float(m.group(1))
            st.grad_norm = float(m.group(2))
            st.epoch = float(m.group(3))
        except ValueError:
            pass

    return st


# --- GPU helpers -------------------------------------------------------------

def get_gpu_table() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            rows.append({
                "index": int(parts[0]),
                "name": parts[1],
                "mem_used_mb": int(parts[2]),
                "mem_total_mb": int(parts[3]),
                "util_pct": int(parts[4]),
            })
        except ValueError:
            continue
    return rows


# --- rendering ---------------------------------------------------------------

STATE_BADGE = {
    "pending":   ("#9aa0a6", "未开始"),
    "running":   ("#34a853", "运行中"),
    "stale":     ("#fbbc04", "停滞"),
    "completed": ("#1a73e8", "完成"),
    "failed":    ("#ea4335", "失败"),
}


def _fmt(v, prec=3, default="—"):
    if v is None:
        return default
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def render_html(specs: list[RunSpec], statuses: list[RunStatus], gpus: list[dict]) -> str:
    by_state: dict[str, list[tuple[RunSpec, RunStatus]]] = {k: [] for k in STATE_BADGE}
    for spec, st in zip(specs, statuses):
        by_state[st.state].append((spec, st))

    def table_for(state: str) -> str:
        rows = by_state.get(state, [])
        if not rows:
            return f"<p class=empty>无</p>"
        header = "<tr><th>任务</th><th>GPU</th><th>schedule</th><th>阶段</th><th>step / total</th><th>%</th><th>speed</th><th>elapsed / ETA</th><th>loss</th><th>grad</th><th>epoch</th><th>log size</th><th>last update</th></tr>"
        out = ["<table class=runs>", header]
        for spec, st in rows:
            phase_lbl = ""
            if st.phase:
                p = "Span" if st.phase == "span" else "KV-NSP"
                phase_lbl = f"{p} (round {st.round_idx})"
            done_phases = len(st.last_phase_completed)
            phase_extra = f" · finished {done_phases} phase(s)" if done_phases else ""
            phase_html = (phase_lbl or "—") + phase_extra
            step_total = f"{st.step}/{st.total}" if st.total else "—"
            pct = f"{st.pct:.1f}%" if st.total else "—"
            el_eta = f"{st.elapsed or '—'} / {st.eta or '—'}"
            update = "—"
            if st.last_modified_ago_s >= 0:
                if st.last_modified_ago_s < 60:
                    update = f"{int(st.last_modified_ago_s)}s ago"
                elif st.last_modified_ago_s < 3600:
                    update = f"{int(st.last_modified_ago_s / 60)}min ago"
                else:
                    update = f"{st.last_modified_ago_s / 3600:.1f}h ago"
            out.append(
                "<tr>"
                f"<td><div class=runname>{escape(spec.label)}</div><div class=hint>{escape(spec.name)}</div></td>"
                f"<td>{escape(spec.gpu)}</td>"
                f"<td><span class=sched>{escape(spec.schedule)}</span></td>"
                f"<td>{escape(phase_html)}</td>"
                f"<td>{escape(step_total)}</td>"
                f"<td>{escape(pct)}</td>"
                f"<td>{escape(st.speed)}</td>"
                f"<td>{escape(el_eta)}</td>"
                f"<td>{_fmt(st.loss, 3)}</td>"
                f"<td>{_fmt(st.grad_norm, 1)}</td>"
                f"<td>{_fmt(st.epoch, 3)}</td>"
                f"<td>{st.log_size_kb:.0f} KB</td>"
                f"<td>{escape(update)}</td>"
                "</tr>"
            )
            if state == "failed" and st.error_snippet:
                out.append(f"<tr class=err><td colspan=13><pre>{escape(st.error_snippet)}</pre></td></tr>")
        out.append("</table>")
        return "\n".join(out)

    gpu_rows = "".join(
        f"<tr><td>{g['index']}</td><td>{g['mem_used_mb']/1024:.1f} / {g['mem_total_mb']/1024:.0f} GB</td>"
        f"<td>{g['mem_used_mb']*100/g['mem_total_mb']:.0f}%</td><td>{g['util_pct']}%</td></tr>"
        for g in gpus
    )
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    counts = {k: len(by_state[k]) for k in STATE_BADGE}
    badges = " ".join(
        f"<span class=badge style='background:{color}'>{label}: {counts[k]}</span>"
        for k, (color, label) in STATE_BADGE.items()
    )

    return f"""<!doctype html>
<html lang="zh-cn"><meta charset="utf-8">
<title>KV-LLM CPT dashboard</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", Helvetica, sans-serif; margin: 16px; color: #202124; background: #fafafa; }}
  h1 {{ margin: 0 0 6px; font-size: 18px; }}
  h2 {{ margin: 24px 0 6px; font-size: 14px; color: #5f6368; text-transform: uppercase; letter-spacing: 0.5px; }}
  .summary {{ color: #5f6368; font-size: 12px; }}
  table.runs {{ border-collapse: collapse; width: 100%; font-size: 12px; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,.05); }}
  table.runs th, table.runs td {{ padding: 6px 10px; border-bottom: 1px solid #f1f3f4; text-align: left; }}
  table.runs th {{ background: #f8f9fa; font-weight: 600; color: #5f6368; font-size: 11px; }}
  table.runs tr.err pre {{ background: #fff4f4; color: #b3261e; padding: 8px; margin: 0; font-size: 11px; white-space: pre-wrap; }}
  .runname {{ font-weight: 600; }}
  .hint {{ color: #80868b; font-size: 10px; font-family: ui-monospace, monospace; }}
  .sched {{ font-family: ui-monospace, monospace; background: #eef; color: #1a4baf; padding: 1px 5px; border-radius: 4px; }}
  .badge {{ display:inline-block; color:#fff; padding: 2px 10px; border-radius: 999px; font-size: 11px; margin-right: 4px; }}
  table.gpus {{ font-size: 12px; }}
  table.gpus td, table.gpus th {{ padding: 3px 10px; }}
  p.empty {{ color: #80868b; font-size: 12px; }}
  button.refresh {{ background: #1a73e8; color: white; border: 0; border-radius: 4px; padding: 8px 16px; font-size: 13px; cursor: pointer; margin-left: 8px; }}
  button.refresh:hover {{ background: #1557b0; }}
  button.refresh:active {{ transform: translateY(1px); }}
  .top {{ display: flex; align-items: center; gap: 8px; }}
</style>
<body>
<div class=top>
  <h1>KV-LLM CPT runs · <span class=summary>last load {now}</span></h1>
  <button class=refresh onclick="location.reload()">🔄 刷新</button>
</div>
<div>{badges}</div>

<h2>运行中 (running)</h2>
{table_for("running")}

<h2>已完成 (completed)</h2>
{table_for("completed")}

<h2>失败 (failed)</h2>
{table_for("failed")}

<h2>停滞 (stale, &gt;5min 无更新)</h2>
{table_for("stale")}

<h2>未开始 (pending)</h2>
{table_for("pending")}

<h2>GPU 状态</h2>
<table class=gpus><tr><th>GPU</th><th>显存</th><th>占用</th><th>util</th></tr>{gpu_rows}</table>

<p class=summary>手动刷新模式 · 服务器时间 {now} · 按 R 键或点击上方按钮重新加载</p>
<script>
  document.addEventListener('keydown', function(e) {{
    if ((e.key === 'r' || e.key === 'R') && !e.ctrlKey && !e.metaKey && !e.altKey) {{
      location.reload();
    }}
  }});
</script>
</body></html>
"""


# --- HTTP server -------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    logs_dir: Path = Path("/data/ocean/code/dapt/logs")
    stale_after_s: float = 300.0

    def log_message(self, format: str, *args) -> None:
        return  # silence

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send(200, b"ok", "text/plain")
            return
        if self.path == "/json":
            statuses = [parse_run(s.name, self.logs_dir, self.stale_after_s) for s in CATALOGUE]
            payload = {
                "ts": int(time.time()),
                "runs": [{**s.__dict__, **st.__dict__} for s, st in zip(CATALOGUE, statuses)],
                "gpus": get_gpu_table(),
            }
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json")
            return
        statuses = [parse_run(s.name, self.logs_dir, self.stale_after_s) for s in CATALOGUE]
        html = render_html(CATALOGUE, statuses, get_gpu_table())
        self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8889)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--logs-dir", default="/data/ocean/code/dapt/logs")
    ap.add_argument("--stale-after-s", type=float, default=300.0)
    args = ap.parse_args()
    Handler.logs_dir = Path(args.logs_dir)
    Handler.stale_after_s = args.stale_after_s
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[dashboard] serving on http://{args.host}:{args.port}  (logs: {args.logs_dir})")
    print(f"[dashboard] open  http://<server-ip>:{args.port}/   or  /json for raw")
    srv.serve_forever()


if __name__ == "__main__":
    main()
