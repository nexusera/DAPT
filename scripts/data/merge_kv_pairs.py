#!/usr/bin/env python3
"""Merge + dedupe KV pairs from two sources for KV-NSP training.

Inputs:
  --old  /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json
         (JSON list of LS-style task dicts with annotations[].result;
          ~102,587 tasks, ~577,357 pairs after extraction)
  --new  /data/ocean/code/dapt/data_full/ls_kv_tasks.jsonl
         (JSONL of fresh LS Postgres export;
          ~16,530 tasks, ~110,469 pairs after extraction)

Output:
  --output /data/ocean/code/dapt/data_full/merged_kv_pairs.json
         (JSON list of `{"key": str, "value": str, "source": "old"|"new"|"both"}`
          entries; deduped by lowercase-stripped (key, value) tuple)

The output format is `kv_llm.kv_nsp.extract_direct_pairs` compatible:
each sample carries exactly one (key, value) pair, which is the simplest
schema the dataset accepts. This loses per-task grouping for negative
sampling, but the dataset's negative sampler already does cross-sample
selection so we are not losing anything material.

Run on the server inside the medical_bert conda env so we can reuse
kv_llm.kv_nsp.extract_label_studio_pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_repo_to_path(repo_root: str) -> None:
    sys.path.insert(0, repo_root)


def iter_old_pairs(path: Path):
    from kv_llm.kv_nsp import extract_label_studio_pairs

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for sample in data:
        for k, v in extract_label_studio_pairs(sample):
            yield k, v


def iter_new_pairs(path: Path):
    from kv_llm.kv_nsp import extract_label_studio_pairs

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            for k, v in extract_label_studio_pairs(sample):
                yield k, v


def normkey(k: str, v: str) -> tuple[str, str]:
    return k.strip().lower(), v.strip().lower()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="/data/ocean/code/dapt")
    p.add_argument("--old", default="/data/ocean/DAPT/data/pseudo_kv_labels_filtered.json")
    p.add_argument("--new", default="/data/ocean/code/dapt/data_full/ls_kv_tasks.jsonl")
    p.add_argument("--output", default="/data/ocean/code/dapt/data_full/merged_kv_pairs.json")
    args = p.parse_args()

    add_repo_to_path(args.repo)

    seen: dict[tuple[str, str], dict[str, object]] = {}

    old_pairs = 0
    for k, v in iter_old_pairs(Path(args.old)):
        if not (k and v):
            continue
        old_pairs += 1
        key = normkey(k, v)
        if key not in seen:
            seen[key] = {"key": k.strip(), "value": v.strip(), "source": "old"}
        else:
            # Keep the longest non-normalized strings (most informative)
            cur = seen[key]
            if len(k.strip()) > len(cur["key"]):  # type: ignore[arg-type]
                cur["key"] = k.strip()
            if len(v.strip()) > len(cur["value"]):  # type: ignore[arg-type]
                cur["value"] = v.strip()

    new_pairs = 0
    for k, v in iter_new_pairs(Path(args.new)):
        if not (k and v):
            continue
        new_pairs += 1
        key = normkey(k, v)
        if key not in seen:
            seen[key] = {"key": k.strip(), "value": v.strip(), "source": "new"}
        else:
            cur = seen[key]
            if cur["source"] == "old":
                cur["source"] = "both"
            if len(k.strip()) > len(cur["key"]):  # type: ignore[arg-type]
                cur["key"] = k.strip()
            if len(v.strip()) > len(cur["value"]):  # type: ignore[arg-type]
                cur["value"] = v.strip()

    merged = list(seen.values())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)

    by_source = {"old": 0, "new": 0, "both": 0}
    for it in merged:
        by_source[it["source"]] += 1  # type: ignore[index]
    print(f"[OK] merged: total_unique={len(merged)}")
    print(f"     old contributed: {old_pairs} raw, deduped to {by_source['old'] + by_source['both']}")
    print(f"     new contributed: {new_pairs} raw, deduped to {by_source['new'] + by_source['both']}")
    print(f"     source breakdown: only-old={by_source['old']}, only-new={by_source['new']}, both={by_source['both']}")
    print(f"     wrote {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
