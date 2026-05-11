#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EBQA inference script for the trained Task3 model.

Features:
- Load HF QA checkpoint (noise-aware is fine) and tokenizer
- Run batched inference on precomputed EBQA JSONL (converted by convert_ebqa.py)
- Support optional noise_ids / token_type_ids if present
- Decode spans with EBQADecoder (dynamic cap, short-field boost)
- Optional EM/F1 scoring if ground truth answer_text is present

Example:
  CUDA_VISIBLE_DEVICES=2 \
  python pre_struct/ebqa/predict_ebqa.py \
    --model_dir runs/ebqa_dapt_noise/best \
    --tokenizer /data/ocean/DAPT/my-medical-tokenizer \
    --data_path data/kv_ner_prepared_comparison/ebqa_eval.jsonl \
    --output_preds runs/ebqa_eval_preds.jsonl

If you still need to prepare eval data, run convert_ebqa.py first:
  python pre_struct/ebqa/convert_ebqa.py \
    --input_file data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
    --struct_path data/kv_ner_prepared_comparison/keys_v2.json \
    --tokenizer_name /data/ocean/DAPT/my-medical-tokenizer \
    --noise_bins /data/ocean/DAPT/workspace/noise_bins.json \
    --output_file data/kv_ner_prepared_comparison/ebqa_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

# Ensure package roots are visible
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent
for _p in (_HERE, _PKG_ROOT, Path.cwd()):
    if str(_p) not in os.sys.path:
        os.sys.path.append(str(_p))

from pre_struct.ebqa.model_ebqa import EBQADecoder, NoiseAwareBertForQuestionAnswering  # type: ignore


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def pad_1d(seqs: List[List[int]], pad_val: int = 0) -> torch.Tensor:
    max_len = max(len(s) for s in seqs) if seqs else 0
    out = []
    for s in seqs:
        if len(s) < max_len:
            s = list(s) + [pad_val] * (max_len - len(s))
        out.append(s)
    return torch.tensor(out, dtype=torch.long)


def pad_noise(seqs: List[List[List[int]]]) -> torch.Tensor:
    max_len = max(len(s) for s in seqs) if seqs else 0
    out = []
    for s in seqs:
        s = s or []
        if len(s) < max_len:
            s = s + [[0] * 7] * (max_len - len(s))
        out.append(s)
    return torch.tensor(out, dtype=torch.long)


def pad_noise_values(seqs: List[List[List[float]]]) -> torch.Tensor:
    max_len = max(len(s) for s in seqs) if seqs else 0
    out = []
    for s in seqs:
        s = s or []
        if len(s) < max_len:
            s = s + [[0.0] * 7] * (max_len - len(s))
        out.append(s)
    return torch.tensor(out, dtype=torch.float32)


def batch_iter(data: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def score_em_f1(pred: str, gt: str) -> Dict[str, float]:
    # Simple exact match + char-level F1
    if gt is None:
        return {"em": 0.0, "f1": 0.0}
    if pred is None:
        pred = ""
    if pred == gt:
        return {"em": 1.0, "f1": 1.0}
    pred_chars = list(pred)
    gt_chars = list(gt)
    common = 0
    gt_count: Dict[str, int] = {}
    for c in gt_chars:
        gt_count[c] = gt_count.get(c, 0) + 1
    for c in pred_chars:
        if gt_count.get(c, 0) > 0:
            common += 1
            gt_count[c] -= 1
    if len(pred_chars) == 0 or len(gt_chars) == 0:
        return {"em": 0.0, "f1": 0.0}
    precision = common / len(pred_chars)
    recall = common / len(gt_chars)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"em": 0.0, "f1": f1}


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    print(f"[INFO] Loading model from {args.model_dir}")
    config = AutoConfig.from_pretrained(args.model_dir)
    if getattr(config, "use_noise", False):
        model = NoiseAwareBertForQuestionAnswering.from_pretrained(args.model_dir, config=config).to(device)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir, config=config).to(device)
    model.eval()

    print(f"[INFO] Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "Expected a fast tokenizer (is_fast=False). EBQA requires return_offsets_mapping."
        )
    _probe = "肿瘤标志物"
    _pieces = tokenizer.tokenize(_probe)
    if len(_pieces) == 1 and _pieces[0] == tokenizer.unk_token:
        raise RuntimeError(
            "Fast tokenizer appears misconfigured (probe tokenizes to a single [UNK]). "
            "Regenerate tokenizer.json: python DAPT/repair_fast_tokenizer.py --tokenizer_dir <TOKENIZER_DIR>"
        )

    decoder = EBQADecoder(
        tokenizer,
        max_answer_len=args.max_answer_len,
        short_field_boost=args.short_field_boost,
    )

    print(f"[INFO] Loading data: {args.data_path}")
    samples = list(load_jsonl(args.data_path))
    print(f"[INFO] Loaded {len(samples)} samples")

    preds = []
    metrics_accum = {"em_sum": 0.0, "f1_sum": 0.0, "n": 0}

    for batch in batch_iter(samples, args.batch_size):
        input_ids = [b["input_ids"] for b in batch]
        attn = [b["attention_mask"] for b in batch]
        token_type = [b.get("token_type_ids") for b in batch]
        noise = [b.get("noise_ids") for b in batch]
        noise_values = [b.get("noise_values") for b in batch]

        model_inputs: Dict[str, torch.Tensor] = {
            "input_ids": pad_1d(input_ids).to(device),
            "attention_mask": pad_1d(attn).to(device),
        }
        
        # Check if model accepts noise_ids
        forward_params = inspect.signature(model.forward).parameters
        
        if any(t for t in token_type) and "token_type_ids" in forward_params:
            tt = [t if t else [0] * len(input_ids[i]) for i, t in enumerate(token_type)]
            model_inputs["token_type_ids"] = pad_1d(tt).to(device)
            
        if any(n for n in noise) and "noise_ids" in forward_params:
            model_inputs["noise_ids"] = pad_noise(noise).to(device)
        if any(n for n in noise_values) and "noise_values" in forward_params:
            model_inputs["noise_values"] = pad_noise_values(noise_values).to(device)
            
        with torch.no_grad():
            out = model(**model_inputs)

        with torch.no_grad():
            out = model(**model_inputs)
        start_logits = out.start_logits.cpu().numpy()
        end_logits = out.end_logits.cpu().numpy()

        for sample, s_log, e_log in zip(batch, start_logits, end_logits):
            res = decoder.best_span_in_chunk(
                start_logits=s_log,
                end_logits=e_log,
                offset_mapping=sample["offset_mapping"],
                sequence_ids=sample["sequence_ids"],
                chunk_text=sample["chunk_text"],
                chunk_char_start=int(sample.get("chunk_char_start", 0)),
                is_short_field=bool(sample.get("is_short_field", False)),
                question_key=sample.get("question_key") or sample.get("key"),
            )
            pred_text = res.get("text", "") or ""
            rec = {
                "question_key": sample.get("question_key") or sample.get("key"),
                "report_index": sample.get("report_index"),
                "pred_text": pred_text,
                "score": res.get("score"),
                "start_char": res.get("start_char"),
                "end_char": res.get("end_char"),
            }
            # Optional scoring if ground truth present
            gt_text = sample.get("answer_text")
            if gt_text is not None:
                sc = score_em_f1(pred_text, gt_text)
                rec.update({"gt_text": gt_text, "em": sc["em"], "f1": sc["f1"]})
                metrics_accum["em_sum"] += sc["em"]
                metrics_accum["f1_sum"] += sc["f1"]
                metrics_accum["n"] += 1
            preds.append(rec)

    out_path = Path(args.output_preds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[OK] Saved {len(preds)} predictions to {out_path}")

    if metrics_accum["n"] > 0:
        em = metrics_accum["em_sum"] / metrics_accum["n"]
        f1 = metrics_accum["f1_sum"] / metrics_accum["n"]
        summary = {
            "num_scored": metrics_accum["n"],
            "em": em,
            "f1": f1,
        }
        summary_path = out_path.with_suffix(".summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved summary to {summary_path} | EM={em:.4f}, F1={f1:.4f}")
    else:
        print("[INFO] No ground truth found; skipped scoring")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EBQA inference for Task3 model")
    p.add_argument("--model_dir", type=str, default="runs/ebqa_dapt_noise/best", help="Path to trained QA checkpoint")
    p.add_argument("--tokenizer", type=str, default="/data/ocean/DAPT/my-medical-tokenizer", help="Tokenizer path")
    p.add_argument("--data_path", type=str, required=True, help="Prepared EBQA eval/test JSONL (from convert_ebqa.py)")
    p.add_argument("--output_preds", type=str, default="runs/ebqa_eval_preds.jsonl", help="Where to save predictions")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    p.add_argument("--max_answer_len", type=int, default=512, help="Decoder max answer length")
    p.add_argument("--short_field_boost", type=float, default=0.3, help="Decoder short field boost")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
