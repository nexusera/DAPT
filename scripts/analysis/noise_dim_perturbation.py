#!/usr/bin/env python3
"""D3.21(c) — Noise embedding 7-dim component perturbation (causal ablation).

For a CPT KV-LLM checkpoint that contains the 7-dim BucketNoiseEmbedder,
zero out one dim at a time at inference and measure downstream task
performance drop on the held-out test set. The causal ablation is more
decisive than IG correlation: if zeroing dim k hurts the metric materially,
dim k is *causally* useful.

Approach:
  - Load the CPT model + reload kv_llm_noise_module.pt / kv_llm_nsp_head.pt
  - Snapshot the noise embedding weights, then for each condition
    (baseline, zero_<feat_0>, ..., zero_<feat_6>, zero_all) overwrite the
    target embedding with zeros, generate, score, restore.
  - Output CSV: condition × Ke_F1 / KeVe_F1 / delta-vs-baseline

Usage:
  python scripts/analysis/noise_dim_perturbation.py \
    --cpt-dir /data/ocean/code/dapt/model/kv_llm_qwen3_0.6b_full/final_model \
    --test    /data/ocean/code/dapt/data_full/medstruct_test_pairs.jsonl \
    --output  /data/ocean/code/dapt/results/eval/d3_21c_noise_dim_06b_full.csv \
    --bf16
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from kv_llm.modeling import KvLlmForCausalPreTraining, BucketNoiseEmbedder  # noqa: E402
from noise_feature_processor import FEATURES  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


PROMPT = (
    "Extract all medical key-value pairs from the OCR clinical report. "
    "Return compact JSON with a top-level 'pairs' list.\n\nReport:\n{text}\n\nJSON:\n"
)


def patch_zero_dim(model: KvLlmForCausalPreTraining, dim: Optional[str]) -> None:
    if dim is None:
        return
    if not isinstance(model.noise_module, BucketNoiseEmbedder):
        raise ValueError(f"need BucketNoiseEmbedder, got {type(model.noise_module).__name__}")
    feats = list(FEATURES) if dim == "all" else [dim]
    with torch.no_grad():
        for feat in feats:
            model.noise_module.noise_embeddings[feat].weight.zero_()


def snapshot(model: KvLlmForCausalPreTraining) -> dict[str, torch.Tensor]:
    return {f: model.noise_module.noise_embeddings[f].weight.detach().clone() for f in FEATURES}


def restore(model: KvLlmForCausalPreTraining, snap: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for f in FEATURES:
            model.noise_module.noise_embeddings[f].weight.copy_(snap[f])


def load_kv_llm(cpt_dir: str, bf16: bool):
    dtype = torch.bfloat16 if bf16 else None
    model = KvLlmForCausalPreTraining.from_pretrained(
        cpt_dir, trust_remote_code=True, torch_dtype=dtype, noise_mode="bucket",
    )
    noise_path = Path(cpt_dir) / "kv_llm_noise_module.pt"
    if noise_path.exists():
        model.noise_module.load_state_dict(torch.load(noise_path, map_location="cpu"))
    nsp_path = Path(cpt_dir) / "kv_llm_nsp_head.pt"
    if nsp_path.exists():
        model.nsp_head.load_state_dict(torch.load(nsp_path, map_location="cpu"))
    tok = AutoTokenizer.from_pretrained(cpt_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tok


def load_records(path: Path, noise_filter: Optional[float] = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            data = json.load(f)
            iter_rows = data
        else:
            iter_rows = (json.loads(l) for l in f if l.strip())
        for r in iter_rows:
            if noise_filter is not None and abs(r.get("noise_level", 0) - noise_filter) > 1e-6:
                continue
            rows.append(r)
    return rows


def generate(model, tok, text: str, max_new_tokens: int = 512) -> str:
    enc = tok(PROMPT.format(text=text), return_tensors="pt", truncation=True, max_length=2048)
    if torch.cuda.is_available():
        enc = {k: v.to("cuda") for k, v in enc.items()}
    with torch.no_grad():
        out = model.base_model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_pairs(raw: str) -> list[dict]:
    import re
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("pairs"), list):
            return [
                {"key": str(p.get("key", "")).strip(), "value": str(p.get("value", "")).strip()}
                for p in obj["pairs"]
                if isinstance(p, dict) and p.get("key") and p.get("value")
            ]
    except Exception:
        pass
    out, seen = [], set()
    for m in re.finditer(r'\{"key"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"value"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}', raw):
        k, v = m.group(1).strip(), m.group(2).strip()
        if (k, v) in seen or not k or not v:
            continue
        seen.add((k, v)); out.append({"key": k, "value": v})
    return out


def score(pred, gold) -> dict[str, float]:
    def norm(s: str) -> str:
        return "".join(s.split()).lower()
    p_kv = {(norm(p["key"]), norm(p["value"])) for p in pred}
    g_kv = {(norm(g["key"]), norm(g["value"])) for g in gold}
    p_k = {kv[0] for kv in p_kv}
    g_k = {kv[0] for kv in g_kv}
    f_ke = 2 * len(p_k & g_k) / max(len(p_k) + len(g_k), 1)
    f_keve = 2 * len(p_kv & g_kv) / max(len(p_kv) + len(g_kv), 1)
    return {"Ke_F1": f_ke, "KeVe_F1": f_keve}


def run_condition(model, tok, cond: Optional[str], records, snap, max_new_tokens: int) -> dict:
    restore(model, snap)
    patch_zero_dim(model, cond)
    per_rec = []
    for i, r in enumerate(records):
        text = r.get("text_noisy") or r.get("ocr_text") or r.get("text", "")
        gold = r.get("pairs") or []
        raw = generate(model, tok, text, max_new_tokens=max_new_tokens)
        per_rec.append(score(parse_pairs(raw), gold))
        if (i + 1) % 50 == 0:
            print(f"    cond={cond} {i+1}/{len(records)}", file=sys.stderr)
    if not per_rec:
        return {"Ke_F1": 0, "KeVe_F1": 0, "n": 0}
    return {
        "Ke_F1": sum(r["Ke_F1"] for r in per_rec) / len(per_rec),
        "KeVe_F1": sum(r["KeVe_F1"] for r in per_rec) / len(per_rec),
        "n": len(per_rec),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpt-dir", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--noise-level", type=float, default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model, tok = load_kv_llm(args.cpt_dir, args.bf16)
    snap = snapshot(model)
    records = load_records(Path(args.test), noise_filter=args.noise_level)
    if args.limit:
        records = records[: args.limit]
    print(f"[D3.21c] {len(records)} records, conditions = baseline + 7 single-dim + zero_all", file=sys.stderr)

    conditions: list[tuple[str, Optional[str]]] = [("baseline", None)]
    for feat in FEATURES:
        conditions.append((f"zero_{feat}", feat))
    conditions.append(("zero_all", "all"))

    baseline = None
    rows = []
    for cname, cond in conditions:
        print(f"[D3.21c] === {cname} ===", file=sys.stderr)
        s = run_condition(model, tok, cond, records, snap, args.max_new_tokens)
        if cname == "baseline":
            baseline = s
        rows.append({
            "condition": cname,
            "n": s["n"],
            "Ke_F1": s["Ke_F1"],
            "KeVe_F1": s["KeVe_F1"],
            "delta_Ke": s["Ke_F1"] - (baseline["Ke_F1"] if baseline else 0.0),
            "delta_KeVe": s["KeVe_F1"] - (baseline["KeVe_F1"] if baseline else 0.0),
        })
        print(f"    {cname} Ke={s['Ke_F1']:.3f} KeVe={s['KeVe_F1']:.3f}", file=sys.stderr)

    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "n", "Ke_F1", "KeVe_F1", "delta_Ke", "delta_KeVe"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
