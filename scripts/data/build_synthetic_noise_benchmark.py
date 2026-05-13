#!/usr/bin/env python3
"""Build the synthetic-noise graceful-degradation benchmark (plan D1.16 / §6.6).

Take N clean (or near-clean) medical OCR records and emit one record per
(source, noise_level) combination. Each record carries:

  - id              unique within the output file
  - source_id       record id in the source file (to dedupe / cross-ref)
  - category        document type from source
  - noise_level     fractional in [0, 1]
  - text_clean      original ocr_text from source
  - text_noisy      character-level perturbed text
  - pairs           K-V annotations (from transferred_annotations), unchanged
                    so downstream K-V extraction eval can score on noisy
                    inputs against the same gold structure
  - operations      counts of replace / delete / insert ops actually applied

Noise model (per character):
  - With prob = noise_level, the character is "affected"
  - Affected character is one of {replace, delete, insert} with the
    distribution --replace-prob / --delete-prob / --insert-prob
    (defaults 0.6 / 0.3 / 0.1, mirroring typical OCR error mix)
  - Replacement / insertion characters are drawn from a frequency-weighted
    pool built from the source corpus itself (no English / digit / punctuation
    drift)

Default source: /data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json
(the 358-page held-out test set, cleanly separated from CPT training data).

Default noise levels: 0%, 5%, 10%, 20%, 30%, 50% — six grid points for
the D3.6 graceful-degradation curve.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable


# ----- noise application -----------------------------------------------------

def apply_noise(
    text: str,
    *,
    noise_level: float,
    rng: random.Random,
    char_pool: list[str],
    char_weights: list[float],
    replace_prob: float = 0.6,
    delete_prob: float = 0.3,
    insert_prob: float = 0.1,
) -> tuple[str, dict[str, int]]:
    """Return (noisy_text, op_counts)."""
    if not text or noise_level <= 0:
        return text, {"replaced": 0, "deleted": 0, "inserted": 0, "original_len": len(text)}

    out_chars: list[str] = []
    counts = {"replaced": 0, "deleted": 0, "inserted": 0, "original_len": len(text)}
    pool_total = sum(char_weights)

    def sample_char() -> str:
        r = rng.random() * pool_total
        acc = 0.0
        for c, w in zip(char_pool, char_weights):
            acc += w
            if acc >= r:
                return c
        return char_pool[-1]

    for ch in text:
        if rng.random() < noise_level:
            r = rng.random()
            if r < replace_prob:
                out_chars.append(sample_char())
                counts["replaced"] += 1
            elif r < replace_prob + delete_prob:
                counts["deleted"] += 1  # skip char
            else:
                out_chars.append(sample_char())
                out_chars.append(ch)
                counts["inserted"] += 1
        else:
            out_chars.append(ch)
    return "".join(out_chars), counts


# ----- char pool from corpus -------------------------------------------------

def build_char_pool(records: Iterable[dict], top_k: int = 4000) -> tuple[list[str], list[float]]:
    """Frequency-weighted character pool from the source corpus. Restrict to CJK +
    common punctuation to avoid drifting into Latin / digits during noise."""
    counter: Counter[str] = Counter()
    for rec in records:
        text = rec.get("ocr_text") or rec.get("text") or ""
        for ch in text:
            # Keep CJK unified ideographs + CJK punctuation + ASCII basic
            if (
                "一" <= ch <= "鿿"
                or "　" <= ch <= "〿"
                or "＀" <= ch <= "￯"
                or ch in "0123456789()[],.;:/-+="
            ):
                counter[ch] += 1
    most = counter.most_common(top_k)
    if not most:
        # fallback to common Chinese chars if the corpus was empty
        most = [(c, 1.0) for c in "的一是了不在人有我"]
    chars, weights = zip(*most)
    return list(chars), [float(w) for w in weights]


# ----- annotation conversion -------------------------------------------------

def extract_pairs(rec: dict) -> list[dict]:
    """Pull the {key, value} pairs out of transferred_annotations alternating
    pattern. The source file is alternating 键名→值 per visual row, with
    occasional 医院名称 labels that we keep separate."""
    pairs: list[dict] = []
    anns = rec.get("transferred_annotations") or []
    pending_key: str | None = None
    hospital: str | None = None
    for ann in anns:
        labels = ann.get("labels") or []
        text = (ann.get("text") or "").strip()
        if not labels or not text:
            continue
        if labels[0] == "医院名称":
            hospital = text
        elif labels[0] == "键名":
            pending_key = text
        elif labels[0] == "值":
            if pending_key:
                pairs.append({"key": pending_key, "value": text})
                pending_key = None
    if hospital:
        pairs.insert(0, {"key": "_医院名称", "value": hospital})
    return pairs


# ----- main ------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json")
    p.add_argument("--output", default="/data/ocean/code/dapt/data_full/synthetic_noise_benchmark.jsonl")
    p.add_argument("--num-samples", type=int, default=1000, help="how many source records to keep (max if source is smaller)")
    p.add_argument("--noise-levels", default="0,0.05,0.10,0.20,0.30,0.50",
                   help="comma-separated noise levels in [0,1]")
    p.add_argument("--replace-prob", type=float, default=0.6)
    p.add_argument("--delete-prob", type=float, default=0.3)
    p.add_argument("--insert-prob", type=float, default=0.1)
    p.add_argument("--char-pool-size", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)

    src_path = Path(args.source)
    print(f"[noise-bench] loading {src_path}")
    with src_path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            records = json.load(f)
        else:
            records = [json.loads(line) for line in f if line.strip()]
    print(f"[noise-bench] loaded {len(records)} source records")

    rng.shuffle(records)
    records = records[: args.num_samples]
    print(f"[noise-bench] keeping {len(records)} records after shuffle")

    print("[noise-bench] building character pool from source corpus")
    char_pool, char_weights = build_char_pool(records, top_k=args.char_pool_size)
    print(f"[noise-bench] char pool size = {len(char_pool)}")

    noise_levels = [float(x) for x in args.noise_levels.split(",")]
    print(f"[noise-bench] noise levels = {noise_levels}")

    op_distrib_sum = args.replace_prob + args.delete_prob + args.insert_prob
    if abs(op_distrib_sum - 1.0) > 1e-3:
        print(f"[WARN] op distribution sums to {op_distrib_sum:.3f}, expected 1.0 — clipping at runtime")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for src_idx, rec in enumerate(records):
            clean = rec.get("ocr_text") or rec.get("text") or ""
            if not clean.strip():
                continue
            pairs = extract_pairs(rec)
            source_id = rec.get("record_id") or rec.get("id") or src_idx
            category = rec.get("category") or ""
            for lvl in noise_levels:
                noisy, counts = apply_noise(
                    clean, noise_level=lvl, rng=rng,
                    char_pool=char_pool, char_weights=char_weights,
                    replace_prob=args.replace_prob,
                    delete_prob=args.delete_prob,
                    insert_prob=args.insert_prob,
                )
                out_rec = {
                    "id": n_written,
                    "source_id": source_id,
                    "category": category,
                    "noise_level": lvl,
                    "text_clean": clean,
                    "text_noisy": noisy,
                    "pairs": pairs,
                    "operations": counts,
                }
                f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                n_written += 1
    print(f"[OK] wrote {n_written} records to {out_path}")
    print(f"     ({len(records)} sources × {len(noise_levels)} noise levels)")


if __name__ == "__main__":
    main()
