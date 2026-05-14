#!/usr/bin/env python3
"""D3.21(a) + D3.21(b) — KV-LLM focused attention extraction.

Two sub-analyses on a CPT'd KV-LLM checkpoint:

  (a) Span Corruption mask attention pattern — at the position of a
      sentinel token, where does the model's attention go in the left
      context? Does it concentrate on medical entity boundaries?

  (b) KV-NSP last-token attention — at the final position of a
      [Key][SEP][Value] sequence (where the NSP head reads from), how
      is attention split across [Key] / [SEP] / [Value] segments?
      Validates the directional 'read K then judge V' hypothesis.

Output: JSONL, one record per analyzed sample. Each record has the
input text, the position(s) attended FROM, and a list of (token_index,
token_str, attention_weight, segment_label) for the top-20 attended
positions, averaged across heads of the final transformer layer.

Usage:
  python scripts/analysis/attention_extract.py --mode span_mask \
    --cpt-dir /data/ocean/code/dapt/model/kv_llm_qwen3_0.6b_full/final_model \
    --test-data /data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \
    --num-samples 20 \
    --output /data/ocean/code/dapt/results/eval/d3_21a_span_attn.jsonl \
    --bf16

  python scripts/analysis/attention_extract.py --mode kvnsp_lasttoken \
    --cpt-dir /data/ocean/code/dapt/model/kv_llm_qwen3_0.6b_full/final_model \
    --kv-pairs /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
    --num-samples 50 \
    --output /data/ocean/code/dapt/results/eval/d3_21b_kvnsp_attn.jsonl \
    --bf16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from kv_llm.constants import KV_NSP_SEP_TOKEN, build_sentinel_tokens  # noqa: E402


def load_model(cpt_dir: str, bf16: bool):
    dtype = torch.bfloat16 if bf16 else None
    tok = AutoTokenizer.from_pretrained(cpt_dir, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cpt_dir, trust_remote_code=True, torch_dtype=dtype, output_attentions=True,
    )
    model.eval()
    if torch.cuda.is_available(): model = model.to("cuda")
    return model, tok


def attn_from(model, tok, text: str, position: int, max_length: int = 512) -> Optional[list]:
    """Run a forward pass with output_attentions, return last-layer
    attention vector from `position` to all positions (averaged over heads)."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False)
    if torch.cuda.is_available():
        enc = {k: v.to("cuda") for k, v in enc.items()}
    if position < 0 or position >= enc["input_ids"].shape[1]:
        return None
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    last = out.attentions[-1].squeeze(0)  # (heads, S, S)
    vec = last.mean(dim=0)[position].float().cpu().tolist()  # length S
    tokens = tok.convert_ids_to_tokens(enc["input_ids"].squeeze(0).cpu().tolist())
    return list(zip(range(len(tokens)), tokens, vec))


def top_k(attn_pairs: list, k: int = 20) -> list:
    return sorted(attn_pairs, key=lambda x: x[2], reverse=True)[:k]


def analyze_span_mask(args):
    model, tok = load_model(args.cpt_dir, args.bf16)
    sentinels = build_sentinel_tokens(args.max_spans + 2)
    sentinel = sentinels[0]
    sent_id = tok.convert_tokens_to_ids(sentinel)
    if isinstance(sent_id, list): sent_id = sent_id[0]

    src_path = Path(args.test_data)
    with src_path.open("r", encoding="utf-8") as f:
        data = json.load(f) if src_path.suffix == ".json" else [json.loads(l) for l in f if l.strip()]

    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for i, r in enumerate(data[: args.num_samples]):
            text = r.get("ocr_text") or r.get("text", "")
            if len(text) < 20: continue
            # insert sentinel at midpoint to simulate a mask
            mid = len(text) // 2
            masked = text[: mid] + sentinel + text[mid + 5 :]
            enc = tok(masked, return_tensors="pt", truncation=True, max_length=args.max_length, add_special_tokens=False)
            ids = enc["input_ids"].squeeze(0).cpu().tolist()
            try:
                pos = ids.index(sent_id)
            except ValueError:
                continue
            attn_list = attn_from(model, tok, masked, position=pos, max_length=args.max_length)
            if attn_list is None: continue
            fout.write(json.dumps({
                "sample_index": i,
                "record_id": r.get("record_id"),
                "mode": "span_mask",
                "sentinel_position": pos,
                "input_text": masked,
                "top_attended": [(idx, tok_str, float(w)) for idx, tok_str, w in top_k(attn_list, args.top_k)],
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 5 == 0:
                print(f"  [D3.21a] {i+1}/{args.num_samples}", file=sys.stderr)
    print(f"[OK] wrote {out_path}")


def analyze_kvnsp_lasttoken(args):
    model, tok = load_model(args.cpt_dir, args.bf16)
    sep_id = tok.convert_tokens_to_ids(KV_NSP_SEP_TOKEN)
    if isinstance(sep_id, list): sep_id = sep_id[0]
    if sep_id is None or sep_id == tok.unk_token_id:
        sep_id = tok.sep_token_id or tok.eos_token_id

    src_path = Path(args.kv_pairs)
    with src_path.open("r", encoding="utf-8") as f:
        pairs_data = json.load(f)
    # extract first N labeled (k,v) pairs
    samples = []
    for rec in pairs_data:
        for ann in rec.get("annotations") or []:
            if ann.get("was_cancelled"): continue
            entities = {}; rels = []
            for res in ann.get("result", []):
                if res.get("type") == "labels":
                    lbls = res.get("value", {}).get("labels") or []
                    if lbls and lbls[0] in ("键名", "值"):
                        entities[res.get("id")] = (lbls[0], res.get("value", {}).get("text", "").strip())
                elif res.get("type") == "relation":
                    rels.append((res.get("from_id"), res.get("to_id")))
            for fid, tid in rels:
                if fid in entities and tid in entities and entities[fid][0] == "键名" and entities[tid][0] == "值":
                    samples.append((entities[fid][1], entities[tid][1]))
                    if len(samples) >= args.num_samples: break
            if len(samples) >= args.num_samples: break
        if len(samples) >= args.num_samples: break

    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for i, (k, v) in enumerate(samples):
            text = f"{k}{KV_NSP_SEP_TOKEN}{v}"
            enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length, add_special_tokens=False)
            ids = enc["input_ids"].squeeze(0).cpu().tolist()
            last_pos = len(ids) - 1
            attn_list = attn_from(model, tok, text, position=last_pos, max_length=args.max_length)
            if attn_list is None: continue
            # segment label per token
            sep_indices = [idx for idx, tid in enumerate(ids) if tid == sep_id]
            split_at = sep_indices[0] if sep_indices else len(ids) // 2
            segments = []
            for j, _, _ in attn_list:
                if j < split_at: segments.append("K")
                elif j == split_at: segments.append("SEP")
                else: segments.append("V")
            tagged = [(idx, tok_str, float(w), seg) for (idx, tok_str, w), seg in zip(attn_list, segments)]
            seg_sum = {"K": 0.0, "SEP": 0.0, "V": 0.0}
            for _, _, w, seg in tagged: seg_sum[seg] += w
            fout.write(json.dumps({
                "sample_index": i,
                "mode": "kvnsp_lasttoken",
                "key": k, "value": v,
                "last_position": last_pos,
                "segment_attention_sums": seg_sum,
                "top_attended": [(idx, tok_str, w, seg) for idx, tok_str, w, seg in sorted(tagged, key=lambda x: x[2], reverse=True)[:args.top_k]],
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  [D3.21b] {i+1}/{len(samples)}", file=sys.stderr)
    print(f"[OK] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["span_mask", "kvnsp_lasttoken"])
    ap.add_argument("--cpt-dir", required=True)
    ap.add_argument("--test-data", default=None, help="for span_mask")
    ap.add_argument("--kv-pairs", default=None, help="for kvnsp_lasttoken")
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-spans", type=int, default=24)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.mode == "span_mask":
        if not args.test_data:
            raise SystemExit("--test-data required for span_mask mode")
        analyze_span_mask(args)
    else:
        if not args.kv_pairs:
            raise SystemExit("--kv-pairs required for kvnsp_lasttoken mode")
        analyze_kvnsp_lasttoken(args)


if __name__ == "__main__":
    main()
