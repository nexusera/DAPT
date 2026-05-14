#!/usr/bin/env python3
"""Generate KV-LLM predictions on MedStruct-S test set.

Loads a fine-tuned KV-LLM model (output of `kv_llm/fine_tune_sft.py` —
optionally LoRA-adapted), runs greedy generation on the OCR clinical
reports in --test-data, and writes one JSONL line per record:

    {"record_id": ..., "category": ..., "ocr_text": "...",
     "raw_response": "<LLM raw text>",
     "pred_pairs": [{"key": "...", "value": "..."}, ...],
     "gold_pairs": [{"key": "...", "value": "..."}, ...]}  # if available

The raw_response field stays so downstream `convert_llm_outputs.py` can
re-parse with its OCR-tolerant heuristics.

Example:
    python -m kv_llm.predict \
      --model_dir /data/ocean/code/dapt/model/ft/kv_llm_06b_full_medstruct \
      --base_model /data/ocean/code/dapt/model/kv_llm_qwen3_0.6b_full/final_model \
      --test_data /data/ocean/code/dapt/data_full/medstruct_test_pairs.jsonl \
      --output /data/ocean/code/dapt/results/preds/kv_llm_06b_full_medstruct.jsonl \
      --bf16 --max_new_tokens 1024
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_llm.data import read_json_or_jsonl


PROMPT_TEMPLATE = (
    "Extract all medical key-value pairs from the OCR clinical report. "
    "Return compact JSON with a top-level 'pairs' list.\n\n"
    "Report:\n{text}\n\nJSON:\n"
)


def build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(text=text)


_KV_RE = re.compile(r'\{"key"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"value"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}')


def parse_pairs_from_raw(raw: str) -> list[dict[str, str]]:
    """Best-effort: try strict JSON first, then regex fallback.

    The model is trained to emit {"pairs": [{"key": ..., "value": ...}, ...]}
    but real generations often have truncation, repeated chunks, or
    non-JSON suffixes. We extract every {"key": ..., "value": ...} object
    we can find and dedupe by (key, value)."""
    raw = raw.strip()
    # try strict parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "pairs" in obj and isinstance(obj["pairs"], list):
            return [
                {"key": str(p.get("key", "")).strip(), "value": str(p.get("value", "")).strip()}
                for p in obj["pairs"]
                if isinstance(p, dict) and p.get("key") and p.get("value")
            ]
    except Exception:
        pass
    # regex fallback
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for m in _KV_RE.finditer(raw):
        k, v = m.group(1).strip(), m.group(2).strip()
        if not k or not v:
            continue
        key = (k, v)
        if key in seen:
            continue
        seen.add(key)
        out.append({"key": k, "value": v})
    return out


def load_model(model_dir: str, base_model: str | None, bf16: bool):
    dtype = torch.bfloat16 if bf16 else None
    if base_model is not None:
        # LoRA adapter path: load base then attach adapter
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype=dtype)
        try:
            model = PeftModel.from_pretrained(base, model_dir)
        except Exception as e:
            print(f"[WARN] PeftModel.from_pretrained({model_dir}) failed: {e}", file=sys.stderr)
            print(f"[WARN] falling back to plain AutoModelForCausalLM at model_dir", file=sys.stderr)
            tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=dtype)
    else:
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=dtype)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tok, model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="FT/LoRA output directory")
    p.add_argument("--base_model", default=None,
                   help="If --model_dir is a LoRA adapter dir, base model path; "
                        "if --model_dir is a full FT model, leave unset.")
    p.add_argument("--test_data", required=True, help="JSONL with ocr_text + pairs per line")
    p.add_argument("--output", required=True, help="Output predictions JSONL")
    p.add_argument("--max_input_length", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    tok, model = load_model(args.model_dir, args.base_model, args.bf16)

    rows = read_json_or_jsonl(args.test_data)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in rows:
            ocr_text = rec.get("ocr_text") or rec.get("text") or rec.get("report") or ""
            if not ocr_text.strip():
                continue
            prompt = build_prompt(ocr_text)
            inp = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_length)
            if torch.cuda.is_available():
                inp = {k: v.to("cuda") for k, v in inp.items()}
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            new_ids = out[0, inp["input_ids"].shape[1]:]
            raw_response = tok.decode(new_ids, skip_special_tokens=True)
            pred_pairs = parse_pairs_from_raw(raw_response)
            f.write(json.dumps({
                "record_id": rec.get("record_id") or rec.get("id") or rec.get("report_index"),
                "category": rec.get("category") or rec.get("report_title"),
                "ocr_text": ocr_text,
                "raw_response": raw_response,
                "pred_pairs": pred_pairs,
                "gold_pairs": rec.get("pairs", []),
            }, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 25 == 0:
                print(f"[predict] {n_done} done", flush=True)
    print(f"[OK] wrote {n_done} predictions to {out_path}")


if __name__ == "__main__":
    main()
