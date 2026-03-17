#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# Path bootstrap
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PKG_ROOT = REPO_ROOT / "dapt_eval_package"
for p in (REPO_ROOT, PKG_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pre_struct.kv_ner.config_io import load_config, tokenizer_name_from, max_seq_length
try:
    from noise_fusion import uses_continuous_noise
except Exception:
    def uses_continuous_noise(mode: str) -> bool:
        return str(mode).lower() in {"linear", "mlp"}


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _tokenize(text: str, tokenizer: Any, max_len: int) -> Dict[str, torch.Tensor]:
    import torch
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    out = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "offset_mapping": enc["offset_mapping"],
    }
    if "token_type_ids" in enc:
        out["token_type_ids"] = enc["token_type_ids"]
    else:
        out["token_type_ids"] = torch.zeros_like(enc["input_ids"])
    return out


def _compute_emissions(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    noise_ids: Optional[torch.Tensor] = None,
    noise_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    import torch
    seq_length = input_ids.shape[1]
    device = input_ids.device
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(input_ids.shape[0], -1)

    outputs = model.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
    )
    sequence_output = model.dropout(outputs[0])

    if model.use_noise:
        if model.noise_mode == "bucket" and noise_ids is not None and model.noise_embeddings is not None:
            noise_vecs = []
            for i, emb in enumerate(model.noise_embeddings):
                ids_i = noise_ids[:, :, i].clamp(min=0, max=emb.num_embeddings - 1)
                noise_vecs.append(emb(ids_i))
            noise_cat = torch.cat(noise_vecs, dim=-1)
            noise_h = model.noise_proj(noise_cat)
            noise_h = model.noise_dropout(noise_h)
            sequence_output = sequence_output + noise_h
        elif uses_continuous_noise(model.noise_mode) and noise_values is not None and model.noise_projector is not None:
            noise_h = model.noise_projector(noise_values.to(sequence_output.device, dtype=torch.float32))
            sequence_output = sequence_output + noise_h

    if model.use_bilstm and model.lstm is not None:
        original_seq_length = sequence_output.size(1)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                sequence_output, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_output, _ = model.lstm(packed)
            lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                lstm_output, batch_first=True, total_length=original_seq_length
            )
        else:
            lstm_output, _ = model.lstm(sequence_output)
        sequence_output = model.dropout(lstm_output)

    return model.classifier(sequence_output)


def _choose_target(emissions: torch.Tensor, id2label: Dict[int, str], mask: torch.Tensor) -> Tuple[int, int]:
    import torch
    key_value_label_ids = [idx for idx, name in id2label.items() if ("KEY" in name or "VALUE" in name)]
    if not key_value_label_ids:
        key_value_label_ids = list(id2label.keys())

    valid_positions = torch.where(mask[0] > 0)[0]
    if len(valid_positions) == 0:
        return 0, key_value_label_ids[0]

    best_score = None
    best_pos = int(valid_positions[0].item())
    best_label = int(key_value_label_ids[0])

    for pos in valid_positions.tolist():
        for lid in key_value_label_ids:
            score = float(emissions[0, pos, lid].item())
            if (best_score is None) or (score > best_score):
                best_score = score
                best_pos = int(pos)
                best_label = int(lid)
    return best_pos, best_label


def _topk_token_scores(scores: torch.Tensor, k: int) -> List[Dict[str, Any]]:
    import torch
    vals, idxs = torch.topk(scores, k=min(k, scores.numel()))
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        out.append({"token_index": int(i), "score": float(v)})
    return out


def run(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.analysis_set)

    if args.dry_run:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for i, r in enumerate(rows[: max(1, min(args.max_samples, 3))]):
                rec = {
                    "analysis_id": r.get("analysis_id", f"dry_{i}"),
                    "mode": "dry_run",
                    "target": {"token_index": 0, "label_id": 0, "label_name": "O"},
                    "top_tokens": [{"token_index": 0, "score": 0.0}],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[OK] dry_run output saved to {out_path}")
        return

    import torch
    from transformers import AutoTokenizer
    from pre_struct.kv_ner.modeling import BertCrfTokenClassifier

    try:
        from captum.attr import LayerIntegratedGradients, IntegratedGradients
    except Exception as e:
        raise RuntimeError("captum 未安装，请先执行: pip install captum") from e

    device = torch.device(args.device)
    cfg = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_from(cfg), use_fast=True)
    model = BertCrfTokenClassifier.from_pretrained(args.model_dir).to(device)
    model.eval()

    label_name_by_id = {int(k): v for k, v in model.id2label.items()} if isinstance(model.id2label, dict) else model.id2label

    def forward_token(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        noise_ids: Optional[torch.Tensor],
        noise_values: Optional[torch.Tensor],
        target_pos: int,
        target_label: int,
    ) -> torch.Tensor:
        em = _compute_emissions(model, input_ids, attention_mask, token_type_ids, noise_ids, noise_values)
        return em[:, target_pos, target_label]

    lig = LayerIntegratedGradients(forward_token, model.bert.get_input_embeddings())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = rows if args.max_samples <= 0 else rows[: args.max_samples]

    with out_path.open("w", encoding="utf-8") as fout:
        for row in total:
            text = str(row.get("text") or "")
            if not text:
                continue
            enc = _tokenize(text, tokenizer, max_seq_length(cfg))
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            token_type_ids = enc["token_type_ids"].to(device)

            noise_ids = None
            noise_values = None
            if model.use_noise and uses_continuous_noise(model.noise_mode):
                nvals = row.get("noise_values")
                if isinstance(nvals, list) and nvals:
                    seq_len = int(input_ids.size(1))
                    nv = nvals[:seq_len]
                    if len(nv) < seq_len:
                        nv = nv + [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * (seq_len - len(nv))
                    noise_values = torch.tensor([nv], dtype=torch.float32, device=device)

            with torch.no_grad():
                emissions = _compute_emissions(model, input_ids, attention_mask, token_type_ids, noise_ids, noise_values)
            target_pos, target_label = _choose_target(emissions, label_name_by_id, attention_mask)

            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else pad_id
            baseline_id = pad_id if args.baseline == "pad" else mask_id
            baselines = torch.full_like(input_ids, fill_value=int(baseline_id), device=device)

            attributions = lig.attribute(
                inputs=input_ids,
                baselines=baselines,
                additional_forward_args=(attention_mask, token_type_ids, noise_ids, noise_values, target_pos, target_label),
                n_steps=args.ig_steps,
                internal_batch_size=args.internal_batch_size,
            )

            token_scores = attributions.sum(dim=-1).squeeze(0).abs().detach().cpu()
            top_tokens = _topk_token_scores(token_scores, args.top_k)
            decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu().tolist())
            for t in top_tokens:
                idx = int(t["token_index"])
                t["token"] = decoded_tokens[idx] if 0 <= idx < len(decoded_tokens) else ""

            rec: Dict[str, Any] = {
                "analysis_id": row.get("analysis_id"),
                "report_index": row.get("report_index"),
                "target": {
                    "token_index": int(target_pos),
                    "label_id": int(target_label),
                    "label_name": str(label_name_by_id.get(int(target_label), "UNK")),
                },
                "top_tokens": top_tokens,
                "noise_attr": None,
            }

            if model.use_noise and uses_continuous_noise(model.noise_mode) and noise_values is not None:
                ig_noise = IntegratedGradients(
                    lambda nv, ids, attn, ttype, pos, lid: forward_token(ids, attn, ttype, None, nv, pos, lid)
                )
                noise_baseline = torch.zeros_like(noise_values) if args.noise_baseline == "zero" else torch.tensor(
                    [[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * noise_values.size(1)],
                    dtype=torch.float32,
                    device=device,
                )
                n_attr = ig_noise.attribute(
                    inputs=noise_values,
                    baselines=noise_baseline,
                    additional_forward_args=(input_ids, attention_mask, token_type_ids, target_pos, target_label),
                    n_steps=args.ig_steps,
                    internal_batch_size=args.internal_batch_size,
                )
                dim_score = n_attr.abs().mean(dim=1).squeeze(0).detach().cpu().tolist()
                rec["noise_attr"] = {
                    "mode": "continuous",
                    "feature_scores": [float(x) for x in dim_score],
                }
            elif model.use_noise and model.noise_mode == "bucket":
                rec["noise_attr"] = {
                    "mode": "bucket",
                    "note": "minimal script: bucket noise attribution is not implemented in this version",
                }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] KV-NER IG saved to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Integrated Gradients for KV-NER model")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--analysis_set", type=str, required=True)
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
