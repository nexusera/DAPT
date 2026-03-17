#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# Path bootstrap
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PKG_ROOT = REPO_ROOT / "dapt_eval_package"
for p in (REPO_ROOT, PKG_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _to_tensor_1d(x: List[int], dtype=None):
    import torch
    if dtype is None:
        dtype = torch.long
    return torch.tensor([x], dtype=dtype)


def _pad_noise_ids(x: List[List[int]], seq_len: int) -> List[List[int]]:
    x = x[:seq_len]
    if len(x) < seq_len:
        x = x + [[0] * 7] * (seq_len - len(x))
    return x


def _pad_noise_values(x: List[List[float]], seq_len: int) -> List[List[float]]:
    x = x[:seq_len]
    if len(x) < seq_len:
        x = x + [[0.0] * 7] * (seq_len - len(x))
    return x


def _topk_token_scores(scores: torch.Tensor, k: int) -> List[Dict[str, Any]]:
    import torch
    vals, idxs = torch.topk(scores, k=min(k, scores.numel()))
    return [{"token_index": int(i), "score": float(v)} for v, i in zip(vals.tolist(), idxs.tolist())]


def run(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.data_path)

    if args.dry_run:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for i, r in enumerate(rows[: max(1, min(args.max_samples, 3))]):
                rec = {
                    "analysis_id": f"ebqa_dry_{i}",
                    "report_index": r.get("report_index"),
                    "question_key": r.get("question_key") or r.get("key"),
                    "mode": "dry_run",
                    "target": {"start": 0, "end": 0},
                    "top_tokens": [{"token_index": 0, "score": 0.0}],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[OK] dry_run output saved to {out_path}")
        return

    import torch
    from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
    from pre_struct.ebqa.model_ebqa import NoiseAwareBertForQuestionAnswering

    try:
        from captum.attr import LayerIntegratedGradients, IntegratedGradients
    except Exception as e:
        raise RuntimeError("captum 未安装，请先执行: pip install captum") from e

    device = torch.device(args.device)

    config = AutoConfig.from_pretrained(args.model_dir)
    if getattr(config, "use_noise", False):
        model = NoiseAwareBertForQuestionAnswering.from_pretrained(args.model_dir, config=config).to(device)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir, config=config).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    emb_layer = model.bert.get_input_embeddings()

    def forward_joint(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        noise_ids: Optional[torch.Tensor],
        noise_values: Optional[torch.Tensor],
        start_idx: int,
        end_idx: int,
    ) -> torch.Tensor:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            noise_ids=noise_ids,
            noise_values=noise_values,
            return_dict=True,
        )
        return out.start_logits[:, start_idx] + out.end_logits[:, end_idx]

    lig = LayerIntegratedGradients(forward_joint, emb_layer)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = rows if args.max_samples <= 0 else rows[: args.max_samples]

    with out_path.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(total):
            input_ids = row.get("input_ids")
            attention_mask = row.get("attention_mask")
            if not isinstance(input_ids, list) or not isinstance(attention_mask, list):
                continue

            seq_len = min(len(input_ids), len(attention_mask))
            input_ids = input_ids[:seq_len]
            attention_mask = attention_mask[:seq_len]

            token_type_ids = row.get("token_type_ids") or [0] * seq_len
            token_type_ids = token_type_ids[:seq_len]
            if len(token_type_ids) < seq_len:
                token_type_ids = token_type_ids + [0] * (seq_len - len(token_type_ids))

            ids_t = _to_tensor_1d(input_ids, torch.long).to(device)
            mask_t = _to_tensor_1d(attention_mask, torch.long).to(device)
            ttype_t = _to_tensor_1d(token_type_ids, torch.long).to(device)

            noise_ids_t = None
            nids = row.get("noise_ids")
            if isinstance(nids, list) and nids:
                nids = _pad_noise_ids(nids, seq_len)
                noise_ids_t = torch.tensor([nids], dtype=torch.long, device=device)

            noise_values_t = None
            nvals = row.get("noise_values")
            if isinstance(nvals, list) and nvals:
                nvals = _pad_noise_values(nvals, seq_len)
                noise_values_t = torch.tensor([nvals], dtype=torch.float32, device=device)

            with torch.no_grad():
                out = model(
                    input_ids=ids_t,
                    attention_mask=mask_t,
                    token_type_ids=ttype_t,
                    noise_ids=noise_ids_t,
                    noise_values=noise_values_t,
                    return_dict=True,
                )
            start_idx = int(torch.argmax(out.start_logits, dim=-1)[0].item())
            end_idx = int(torch.argmax(out.end_logits, dim=-1)[0].item())

            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else pad_id
            baseline_id = pad_id if args.baseline == "pad" else mask_id
            base_t = torch.full_like(ids_t, fill_value=int(baseline_id), device=device)

            attrs = lig.attribute(
                inputs=ids_t,
                baselines=base_t,
                additional_forward_args=(mask_t, ttype_t, noise_ids_t, noise_values_t, start_idx, end_idx),
                n_steps=args.ig_steps,
                internal_batch_size=args.internal_batch_size,
            )
            token_scores = attrs.sum(dim=-1).squeeze(0).abs().detach().cpu()
            top_tokens = _topk_token_scores(token_scores, args.top_k)
            decoded_tokens = tokenizer.convert_ids_to_tokens(ids_t.squeeze(0).detach().cpu().tolist())
            for t in top_tokens:
                idx = int(t["token_index"])
                t["token"] = decoded_tokens[idx] if 0 <= idx < len(decoded_tokens) else ""

            rec: Dict[str, Any] = {
                "analysis_id": f"ebqa_{i}",
                "report_index": row.get("report_index"),
                "question_key": row.get("question_key") or row.get("key"),
                "target": {"start": start_idx, "end": end_idx},
                "top_tokens": top_tokens,
                "noise_attr": None,
            }

            if noise_values_t is not None and getattr(config, "use_noise", False) and str(getattr(config, "noise_mode", "bucket")).lower() in {"linear", "mlp"}:
                ig_noise = IntegratedGradients(
                    lambda nv, ii, am, tt, ni, sidx, eidx: forward_joint(ii, am, tt, ni, nv, sidx, eidx)
                )
                noise_baseline = torch.zeros_like(noise_values_t) if args.noise_baseline == "zero" else torch.tensor(
                    [[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * noise_values_t.size(1)],
                    dtype=torch.float32,
                    device=device,
                )
                n_attr = ig_noise.attribute(
                    inputs=noise_values_t,
                    baselines=noise_baseline,
                    additional_forward_args=(ids_t, mask_t, ttype_t, noise_ids_t, start_idx, end_idx),
                    n_steps=args.ig_steps,
                    internal_batch_size=args.internal_batch_size,
                )
                dim_score = n_attr.abs().mean(dim=1).squeeze(0).detach().cpu().tolist()
                rec["noise_attr"] = {
                    "mode": "continuous",
                    "feature_scores": [float(x) for x in dim_score],
                }
            elif getattr(config, "use_noise", False) and str(getattr(config, "noise_mode", "bucket")).lower() == "bucket":
                rec["noise_attr"] = {
                    "mode": "bucket",
                    "note": "minimal script: bucket noise attribution is not implemented in this version",
                }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] EBQA IG saved to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Integrated Gradients for EBQA model")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--ig_steps", type=int, default=64)
    p.add_argument("--internal_batch_size", type=int, default=8)
    p.add_argument("--baseline", type=str, choices=["pad", "mask"], default="pad")
    p.add_argument("--noise_baseline", type=str, choices=["zero", "perfect"], default="perfect")
    p.add_argument("--max_samples", type=int, default=20)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
