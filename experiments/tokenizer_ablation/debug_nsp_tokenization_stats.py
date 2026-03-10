#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from transformers import AutoTokenizer


def _read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_pairs_from_labelstudio(records: Sequence[Dict]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for sample in records:
        annotations = sample.get("annotations", [])
        valid_annos = [a for a in annotations if not a.get("was_cancelled")]
        if not valid_annos:
            continue
        latest = valid_annos[-1]
        results = latest.get("result", [])

        entities: Dict[str, Dict[str, str]] = {}
        relations: List[Tuple[str, str]] = []

        for res in results:
            res_type = res.get("type")
            if res_type == "labels":
                labels = res.get("value", {}).get("labels", [])
                if not labels:
                    continue
                label = labels[0]
                if label not in ("键名", "值"):
                    continue
                text = res.get("value", {}).get("text", "")
                if not text:
                    continue
                entities[res.get("id")] = {"label": label, "text": text}
            elif res_type == "relation":
                from_id = res.get("from_id")
                to_id = res.get("to_id")
                if from_id and to_id:
                    relations.append((from_id, to_id))

        for from_id, to_id in relations:
            key = entities.get(from_id)
            value = entities.get(to_id)
            if key and value and key["label"] == "键名" and value["label"] == "值":
                kt = key["text"].strip()
                vt = value["text"].strip()
                if kt and vt:
                    pairs.append((kt, vt))
    return pairs


def iter_nsp_files(nsp_path: str) -> List[Path]:
    p = Path(nsp_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([p / f for f in p.iterdir() if f.name.endswith(".json")])
    raise FileNotFoundError(nsp_path)


def main():
    ap = argparse.ArgumentParser(description="Debug NSP pair tokenization length/truncation under different tokenizers")
    ap.add_argument("--nsp_data", type=str, required=True, help="LabelStudio json file or dir")
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--num_pairs", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)

    files = iter_nsp_files(args.nsp_data)
    all_pairs: List[Tuple[str, str]] = []
    for fp in files:
        recs = _read_json(fp)
        all_pairs.extend(extract_pairs_from_labelstudio(recs))

    if not all_pairs:
        print("No pairs extracted. Check input format.")
        return

    rng = random.Random(args.seed)
    if args.num_pairs < len(all_pairs):
        pairs = rng.sample(all_pairs, args.num_pairs)
    else:
        pairs = list(all_pairs)

    lens_no_trunc: List[int] = []
    trunc_flags: List[int] = []
    unk_counts: List[int] = []

    unk_id = tok.unk_token_id

    for k, v in pairs:
        # length without truncation
        enc_full = tok(k, v, add_special_tokens=True, truncation=False)
        L = len(enc_full["input_ids"])
        lens_no_trunc.append(L)

        # with truncation
        enc = tok(k, v, add_special_tokens=True, truncation=True, max_length=args.max_length)
        trunc_flags.append(1 if L > args.max_length else 0)

        if unk_id is not None:
            unk_counts.append(sum(1 for tid in enc["input_ids"] if int(tid) == int(unk_id)))
        else:
            unk_counts.append(0)

    print("=" * 80)
    print(f"nsp_data={args.nsp_data}")
    print(f"tokenizer_path={args.tokenizer_path}")
    print(f"tokenizer_class={tok.__class__.__name__} is_fast={getattr(tok,'is_fast',False)} vocab_size={len(tok)}")
    print(f"pairs_used={len(pairs)} extracted_total={len(all_pairs)}")
    print("-" * 80)

    trunc_rate = 100.0 * sum(trunc_flags) / len(trunc_flags)
    print(f"len_no_trunc: mean={statistics.mean(lens_no_trunc):.1f} p50={statistics.median(lens_no_trunc):.0f} p90={statistics.quantiles(lens_no_trunc, n=10)[8]:.0f} max={max(lens_no_trunc)}")
    print(f"truncation(max_length={args.max_length}): rate={trunc_rate:.2f}%")

    if unk_counts:
        total_unk = sum(unk_counts)
        print(f"UNK(after trunc): mean={statistics.mean(unk_counts):.2f} max={max(unk_counts)} total={total_unk}")

    # simple histogram of lengths
    buckets = Counter(min((L // 64) * 64, 1024) for L in lens_no_trunc)
    common = ", ".join([f"{k}-{k+63}:{v}" for k, v in buckets.most_common(8)])
    print(f"len buckets(top): {common}")


if __name__ == "__main__":
    main()
