#!/usr/bin/env python3
"""Run a Surya OCR + layout smoke test on one image.

This helper assumes Surya is installed in a dedicated environment and shells
out to the official CLI entrypoints:

- surya_ocr
- surya_layout

It keeps OCR and layout outputs in separate directories so the raw Surya JSON
files remain untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_IMAGE = Path("/Users/wy/Documents/data/半结构化/病历/CT/record-image.jpeg")
DEFAULT_ENV_PREFIX = Path("/opt/miniconda3/envs/surya-ocr")
DEFAULT_OUTPUT_ROOT = Path("/Users/wy/Documents/code/DAPT/outputs/surya")


def _run(cmd: list[str], env: dict[str, str]) -> dict[str, object]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        captured.append(line)
    returncode = proc.wait()
    tail = "".join(captured[-200:])
    return {
        "returncode": returncode,
        "stdout_tail": tail[-4000:],
    }


def _results_json_path(out_dir: Path) -> Path:
    return out_dir / "results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Surya OCR + layout smoke test")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--env-prefix", type=Path, default=DEFAULT_ENV_PREFIX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--torch-device", default="mps", help="e.g. mps, cpu")
    parser.add_argument("--skip-layout", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = args.image.resolve()
    if not image.exists():
        print(f"[ERROR] image not found: {image}", file=sys.stderr)
        return 2

    env_prefix = args.env_prefix.resolve()
    surya_ocr = env_prefix / "bin" / "surya_ocr"
    surya_layout = env_prefix / "bin" / "surya_layout"
    if not surya_ocr.exists():
        print(f"[ERROR] surya_ocr not found: {surya_ocr}", file=sys.stderr)
        return 3
    if not args.skip_layout and not surya_layout.exists():
        print(f"[ERROR] surya_layout not found: {surya_layout}", file=sys.stderr)
        return 4

    output_root = args.output_root.resolve() / image.stem
    ocr_out = output_root / "ocr"
    layout_out = output_root / "layout"
    ocr_out.mkdir(parents=True, exist_ok=True)
    layout_out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("TORCH_DEVICE", args.torch_device)

    summary: dict[str, object] = {
        "image": str(image),
        "torch_device": env["TORCH_DEVICE"],
        "ocr": None,
        "layout": None,
    }

    if not args.skip_ocr:
        cmd = [str(surya_ocr), str(image), "--output_dir", str(ocr_out)]
        proc = _run(cmd, env)
        summary["ocr"] = {
            "cmd": cmd,
            "returncode": proc["returncode"],
            "stdout_tail": proc["stdout_tail"],
            "results_json": str(_results_json_path(ocr_out)),
        }

    if not args.skip_layout:
        cmd = [str(surya_layout), str(image), "--output_dir", str(layout_out)]
        proc = _run(cmd, env)
        summary["layout"] = {
            "cmd": cmd,
            "returncode": proc["returncode"],
            "stdout_tail": proc["stdout_tail"],
            "results_json": str(_results_json_path(layout_out)),
        }

    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
