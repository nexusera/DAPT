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
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deprecated compatibility wrapper for the official MedStruct-S scorer."
    )
    parser.add_argument("--schema_file", default=None, help="Deprecated alias of --query_set")
    parser.add_argument("--query_set", default=None, help="Official MedStruct-S query set path")
    parser.add_argument("--scorer_dir", default=None, help="Directory containing official scorer.py")
    args, rest = parser.parse_known_args()

    dapt_root = default_dapt_root()
    scorer_dir = (
        Path(args.scorer_dir).resolve()
        if args.scorer_dir
        else (dapt_root / "dapt_eval_package" / "MedStruct-S-master").resolve()
    )
    scorer_path = scorer_dir / "scorer.py"
    if not scorer_path.is_file():
        raise FileNotFoundError(f"Official MedStruct-S scorer not found: {scorer_path}")

    query_set = args.query_set or args.schema_file
    forwarded = list(rest)
    if query_set:
        forwarded.extend(["--query_set", query_set])

    print(
        "[DEPRECATED] MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py now forwards to the official MedStruct-S scorer.",
        file=sys.stderr,
    )
    return subprocess.run([sys.executable, str(scorer_path), *forwarded], cwd=str(scorer_dir)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
