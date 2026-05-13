#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def default_dapt_root() -> Path:
    remote_root = Path("/data/ocean/DAPT")
    if remote_root.is_dir():
        return remote_root
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the official MedStruct-S scorer from any working directory."
    )
    parser.add_argument(
        "--scorer_dir",
        type=Path,
        default=None,
        help="Directory containing the official MedStruct-S scorer.py",
    )
    args, scorer_args = parser.parse_known_args()

    dapt_root = default_dapt_root()
    scorer_dir = (
        args.scorer_dir.resolve()
        if args.scorer_dir
        else (dapt_root / "dapt_eval_package" / "MedStruct-S-master").resolve()
    )
    scorer_path = scorer_dir / "scorer.py"
    if not scorer_path.is_file():
        raise FileNotFoundError(f"Official MedStruct-S scorer not found: {scorer_path}")

    cmd = [sys.executable, str(scorer_path), *scorer_args]
    completed = subprocess.run(cmd, cwd=str(scorer_dir))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
