#!/usr/bin/env python3
"""SC3-C / D2.7 / D3.11 — Zero/few-shot K-V extraction eval for any Instruct
LLM (Qwen3-0.6B / Qwen3-8B / HuatuoGPT / DISC-MedLLM / etc.).

Loads a model, runs greedy K-V extraction on MedStruct-S test pairs (or
the synthetic-noise benchmark), scores with value-based NED. Supports
zero-shot or few-shot via --num-shots K (random demonstrations sampled
from train set).

Usage (SC3-C zero-shot eval of Qwen3-0.6B Instruct):
  python scripts/eval/zero_shot_kv_extract.py \
    --model /data/ocean/model/Qwen/Qwen3-0.6B \
    --test /data/ocean/code/dapt/data_full/medstruct_test_pairs.jsonl \
    --output /data/ocean/code/dapt/results/eval/sc3c_zero_shot.csv \
    --bf16

Few-shot baseline (D2.7 — Qwen3-8B-Instruct with 3-shot CoT):
  python scripts/eval/zero_shot_kv_extract.py \
    --model /data/ocean/model/Qwen/Qwen3-8B \
    --test /data/ocean/code/dapt/data_full/medstruct_test_pairs.jsonl \
    --train-for-shots /data/ocean/code/dapt/data_full/medstruct_train_pairs.jsonl \
    --num-shots 3 --use-cot \
    --output /data/ocean/code/dapt/results/eval/qwen3_8b_3shot_cot.csv \
    --bf16

LLM-TKIE-style strict JSON prompt (D3.11):
  add --prompt-style llm_tkie  → swaps the prompt template for the
  structured JSON instruction from LLM-TKIE (Scientific Reports 2025).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATES = {
    "default": (
        "Extract all medical key-value pairs from the OCR clinical report. "
        "Return compact JSON with a top-level 'pairs' list of objects with "
        "'key' and 'value' fields.\n\nReport:\n{text}\n\nJSON:\n"
    ),
    "llm_tkie": (
        "You are an information extraction model. Extract every (key, value) "
        "pair from the medical document below as a JSON list. Each entry "
        "must have exactly two fields: 'key' (the field name) and 'value' "
        "(the field value as it appears in the text). Output strict JSON, "
        "no commentary.\n\nDocument:\n{text}\n\nOutput:\n"
    ),
    "cot": (
        "Extract medical key-value pairs from the OCR clinical report. "
        "First reason briefly about what fields are present, then output "
        "the final JSON.\n\nReport:\n{text}\n\nReasoning + JSON:\n"
    ),
}


def load_records(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            rows.extend(json.load(f))
        else:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def build_few_shot_block(train_records: list[dict], k: int, rng: random.Random) -> str:
    if k <= 0 or not train_records:
        return ""
    shots = rng.sample(train_records, min(k, len(train_records)))
    out = []
    for i, r in enumerate(shots, 1):
        text = r.get("ocr_text") or r.get("text") or ""
        pairs = r.get("pairs") or []
        json_out = json.dumps({"pairs": [
            {"key": p["key"], "value": p["value"]} for p in pairs
            if p.get("key") and p.get("value")
        ]}, ensure_ascii=False)
        out.append(f"Example {i}:\nReport:\n{text}\n\nJSON:\n{json_out}\n")
    return "\n".join(out) + "\n\n"


def build_prompt(template: str, text: str, few_shot_block: str = "", use_chat_template=False, tokenizer=None) -> str:
    body = few_shot_block + template.format(text=text)
    if use_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            msgs = [{"role": "user", "content": body}]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return body


def parse_pairs(raw: str) -> list[dict]:
    raw = raw.strip()
    # strip markdown code fences
    raw = re.sub(r"```+\s*json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```+", "", raw).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("pairs"), list):
            return [
                {"key": str(p.get("key", "")).strip(), "value": str(p.get("value", "")).strip()}
                for p in obj["pairs"]
                if isinstance(p, dict) and p.get("key") and p.get("value")
            ]
        if isinstance(obj, list):
            return [
                {"key": str(p.get("key", "")).strip(), "value": str(p.get("value", "")).strip()}
                for p in obj
                if isinstance(p, dict) and p.get("key") and p.get("value")
            ]
    except Exception:
        pass
    out, seen = [], set()
    pat = re.compile(r'\{"key"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"value"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}')
    for m in pat.finditer(raw):
        k, v = m.group(1).strip(), m.group(2).strip()
        if (k, v) in seen or not k or not v:
            continue
        seen.add((k, v))
        out.append({"key": k, "value": v})
    return out


def score(pred, gold) -> dict:
    def norm(s): return "".join(str(s).split()).lower()
    def ned_ok(a, b, thr=0.8):
        if not a or not b: return False
        la, lb = len(a), len(b)
        prev = list(range(lb + 1))
        for i in range(1, la + 1):
            cur = [i] + [0] * lb
            for j in range(1, lb + 1):
                cost = 0 if a[i-1] == b[j-1] else 1
                cur[j] = min(cur[j-1]+1, prev[j]+1, prev[j-1]+cost)
            prev = cur
        sim = 1 - prev[lb] / max(la, lb)
        return sim >= thr + min(0.1, 5.0 / max(la, lb))
    p_kv = {(norm(p["key"]), norm(p["value"])) for p in pred}
    g_kv = {(norm(g["key"]), norm(g["value"])) for g in gold}
    p_k, g_k = {kv[0] for kv in p_kv}, {kv[0] for kv in g_kv}
    f_ke = 2 * len(p_k & g_k) / max(len(p_k) + len(g_k), 1)
    f_keve = 2 * len(p_kv & g_kv) / max(len(p_kv) + len(g_kv), 1)
    matched_va = 0; used = set()
    for p in pred:
        for i, g in enumerate(gold):
            if i in used: continue
            if norm(p["key"]) == norm(g["key"]) and ned_ok(norm(p["value"]), norm(g["value"])):
                matched_va += 1; used.add(i); break
    f_keva = 2 * matched_va / max(len(pred) + len(gold), 1)
    return {"Ke_F1": f_ke, "KeVe_F1": f_keve, "KeVa_F1": f_keva}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model dir (Instruct preferred)")
    p.add_argument("--test", required=True)
    p.add_argument("--train-for-shots", default=None)
    p.add_argument("--num-shots", type=int, default=0)
    p.add_argument("--prompt-style", choices=list(PROMPT_TEMPLATES.keys()), default="default")
    p.add_argument("--use-cot", action="store_true", help="alias for --prompt-style cot")
    p.add_argument("--use-chat-template", action="store_true", help="wrap prompt in tokenizer.apply_chat_template (Instruct models)")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True, help="CSV path (per-record + aggregate)")
    args = p.parse_args()

    template = PROMPT_TEMPLATES["cot"] if args.use_cot else PROMPT_TEMPLATES[args.prompt_style]
    rng = random.Random(args.seed)
    dtype = torch.bfloat16 if args.bf16 else None
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype)
    model.eval()
    if torch.cuda.is_available(): model = model.to("cuda")

    test_records = load_records(Path(args.test))
    if args.limit: test_records = test_records[: args.limit]
    train_records = load_records(Path(args.train_for_shots)) if args.train_for_shots else []
    few_shot_block = build_few_shot_block(train_records, args.num_shots, rng) if args.num_shots else ""

    per_record = []
    for i, r in enumerate(test_records):
        text = r.get("text_noisy") or r.get("ocr_text") or r.get("text") or ""
        gold = r.get("pairs") or []
        prompt = build_prompt(template, text, few_shot_block, args.use_chat_template, tok)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
        if torch.cuda.is_available(): enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                  do_sample=False, pad_token_id=tok.pad_token_id)
        raw = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_pairs(raw)
        sc = score(pred, gold)
        sc["record_id"] = r.get("record_id") or r.get("id") or i
        sc["n_pred"] = len(pred)
        sc["n_gold"] = len(gold)
        per_record.append(sc)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(test_records)}] Ke={sc['Ke_F1']:.3f}", file=sys.stderr)

    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["record_id", "Ke_F1", "KeVe_F1", "KeVa_F1", "n_pred", "n_gold"])
        w.writeheader()
        for r in per_record:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in w.fieldnames})

    n = len(per_record)
    agg = {k: sum(r[k] for r in per_record) / max(n, 1) for k in ("Ke_F1", "KeVe_F1", "KeVa_F1")}
    print(f"[OK] n={n}  Ke={agg['Ke_F1']:.3f}  KeVe={agg['KeVe_F1']:.3f}  KeVa={agg['KeVa_F1']:.3f}", file=sys.stderr)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
