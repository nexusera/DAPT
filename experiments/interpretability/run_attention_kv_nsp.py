#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Path bootstrap
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from transformers import AutoConfig, AutoTokenizer

from noise_feature_processor import FEATURES, NoiseFeatureProcessor
from train_dapt_macbert_staged import BertForDaptMTL


def _load_state_dict_from_dir(model_dir: Path, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    safetensors_path = model_dir / "model.safetensors"
    pytorch_path = model_dir / "pytorch_model.bin"

    if safetensors_path.is_file():
        try:
            from safetensors.torch import load_file

            return load_file(str(safetensors_path), device=map_location)
        except Exception as e:
            raise RuntimeError(f"Failed to load safetensors at {safetensors_path}: {e}") from e

    if pytorch_path.is_file():
        state = torch.load(str(pytorch_path), map_location=map_location)
        if not isinstance(state, dict):
            raise RuntimeError(f"Invalid state dict format in {pytorch_path}")
        return state

    raise FileNotFoundError(
        f"No weight file found in {model_dir}. Expected one of: {safetensors_path.name}, {pytorch_path.name}"
    )


def _load_dapt_model(
    model_dir: Path,
    device: str,
    base_model_name: str,
    noise_mode_hint: str,
    noise_bin_edges: Optional[Dict[str, List[float]]],
) -> BertForDaptMTL:
    config_path = model_dir / "config.json"
    if config_path.is_file():
        cfg = AutoConfig.from_pretrained(str(model_dir))
        model = BertForDaptMTL.from_pretrained(str(model_dir), config=cfg)
        return model.to(device)

    # Fallback path: recover config from base model, then load local weights directly.
    cfg = AutoConfig.from_pretrained(base_model_name)
    cfg.noise_mode = str(noise_mode_hint or "bucket").lower()
    if noise_bin_edges is not None:
        cfg.noise_bin_edges = noise_bin_edges

    model = BertForDaptMTL(cfg)
    state = _load_state_dict_from_dir(model_dir, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[warn] missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys when loading checkpoint: {len(unexpected)}")

    return model.to(device)


def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for k in ("data", "items", "samples", "records"):
            v = data.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        return [data]
    return []


@dataclass
class PairSample:
    sample_id: str
    key_text: str
    value_text: str
    label: int
    group: str  # positive | reverse | random | negative_other
    meta: Dict[str, Any]


def _normalize_group(label: int, raw_group: Optional[str]) -> str:
    if label == 1:
        return "positive"
    g = (raw_group or "").lower()
    if "reverse" in g or "swap" in g or "倒序" in g:
        return "reverse"
    if "random" in g or "rand" in g or "随机" in g:
        return "random"
    return "negative_other"


def _extract_label(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        x = v.strip().lower()
        if x in {"1", "true", "match", "positive", "pos", "yes"}:
            return 1
        if x in {"0", "false", "mismatch", "negative", "neg", "no"}:
            return 0
    return None


def _extract_pairs_from_labelstudio_obj(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    # Minimal parser to avoid hard dependency on KVDataset internals.
    annos = row.get("annotations", [])
    if not isinstance(annos, list):
        return []
    valid_annos = [a for a in annos if isinstance(a, dict) and not a.get("was_cancelled")]
    if not valid_annos:
        return []
    latest = valid_annos[-1]
    results = latest.get("result", [])
    if not isinstance(results, list):
        return []

    entities: Dict[str, Dict[str, str]] = {}
    relations: List[Tuple[str, str]] = []

    for res in results:
        if not isinstance(res, dict):
            continue
        rtype = res.get("type")
        if rtype == "labels":
            labels = (res.get("value") or {}).get("labels", [])
            if not labels:
                continue
            label = str(labels[0])
            if label not in ("键名", "值"):
                continue
            text = str((res.get("value") or {}).get("text", "") or "").strip()
            rid = res.get("id")
            if rid and text:
                entities[str(rid)] = {"label": label, "text": text}
        elif rtype == "relation":
            fid = res.get("from_id")
            tid = res.get("to_id")
            if fid and tid:
                relations.append((str(fid), str(tid)))

    out: List[Tuple[str, str]] = []
    for fid, tid in relations:
        a = entities.get(fid)
        b = entities.get(tid)
        if a and b and a.get("label") == "键名" and b.get("label") == "值":
            out.append((a["text"], b["text"]))
    return out


def _load_pair_samples(path: Path) -> List[PairSample]:
    rows = _read_json_or_jsonl(path)
    samples: List[PairSample] = []

    for idx, row in enumerate(rows):
        key = row.get("key") or row.get("key_text") or row.get("text_a") or row.get("question_key")
        val = row.get("value") or row.get("value_text") or row.get("text_b") or row.get("value_text_pred")

        if key is None or val is None:
            # try Label Studio raw format
            ls_pairs = _extract_pairs_from_labelstudio_obj(row)
            if ls_pairs:
                for j, (k, v) in enumerate(ls_pairs):
                    sid = row.get("id") or row.get("task_id") or f"ls_{idx}_{j}"
                    samples.append(
                        PairSample(
                            sample_id=str(sid),
                            key_text=k,
                            value_text=v,
                            label=1,
                            group="positive",
                            meta={"source_index": idx, "source": "labelstudio"},
                        )
                    )
            continue

        label = (
            _extract_label(row.get("label"))
            or _extract_label(row.get("is_match"))
            or _extract_label(row.get("matched"))
            or _extract_label(row.get("nsp_label"))
        )
        if label is None:
            label = 1

        raw_group = row.get("negative_type") or row.get("pair_type") or row.get("sample_type")
        group = _normalize_group(label=label, raw_group=raw_group)

        sid = row.get("id") or row.get("sample_id") or row.get("uid") or f"row_{idx}"
        samples.append(
            PairSample(
                sample_id=str(sid),
                key_text=str(key),
                value_text=str(val),
                label=int(label),
                group=group,
                meta={"source_index": idx, "raw_group": raw_group},
            )
        )

    return samples


def _has_negatives(samples: Sequence[PairSample]) -> bool:
    return any(s.label == 0 for s in samples)


def _auto_generate_negatives(samples: Sequence[PairSample], seed: int) -> List[PairSample]:
    rng = random.Random(seed)
    positives = [s for s in samples if s.label == 1]
    if not positives:
        return list(samples)

    value_pool = [s.value_text for s in positives]
    out = list(samples)

    for i, p in enumerate(positives):
        out.append(
            PairSample(
                sample_id=f"{p.sample_id}_reverse",
                key_text=p.value_text,
                value_text=p.key_text,
                label=0,
                group="reverse",
                meta={"derived_from": p.sample_id},
            )
        )

        # random negative: same key, random value from other sample
        candidate = p.value_text
        retries = 0
        while candidate == p.value_text and retries < 30:
            candidate = rng.choice(value_pool)
            retries += 1
        if candidate != p.value_text:
            out.append(
                PairSample(
                    sample_id=f"{p.sample_id}_random",
                    key_text=p.key_text,
                    value_text=candidate,
                    label=0,
                    group="random",
                    meta={"derived_from": p.sample_id},
                )
            )

        if i % 1000 == 0 and i > 0:
            pass

    return out


def _subsample_by_group(samples: Sequence[PairSample], max_per_group: int, seed: int) -> List[PairSample]:
    if max_per_group <= 0:
        return list(samples)
    rng = random.Random(seed)
    buckets: Dict[str, List[PairSample]] = {}
    for s in samples:
        buckets.setdefault(s.group, []).append(s)

    out: List[PairSample] = []
    for g, rows in buckets.items():
        if len(rows) <= max_per_group:
            out.extend(rows)
        else:
            out.extend(rng.sample(rows, max_per_group))
    return out


def _infer_noise_level(meta: Dict[str, Any], default: str = "unknown") -> str:
    # Priority 1: explicit field
    nl = meta.get("noise_level")
    if isinstance(nl, str) and nl.strip():
        x = nl.strip().lower()
        if x in {"high", "medium", "low"}:
            return x

    # Priority 2: conf_avg style scalar in metadata
    for k in ("conf_avg", "ocr_conf_avg", "confidence", "avg_conf"):
        v = meta.get(k)
        if isinstance(v, (int, float)):
            v = float(v)
            if v >= 0.95:
                return "high"
            if v >= 0.85:
                return "medium"
            return "low"

    # Priority 3: from noise_values 7-d vectors
    nvals = meta.get("noise_values")
    if isinstance(nvals, list) and nvals:
        vals: List[float] = []
        for r in nvals:
            if isinstance(r, list) and r and isinstance(r[0], (int, float)):
                vals.append(float(r[0]))
        if vals:
            avg = float(sum(vals) / max(1, len(vals)))
            if avg >= 0.95:
                return "high"
            if avg >= 0.85:
                return "medium"
            return "low"

    return default


def _prepare_encoding(
    tokenizer,
    key_text: str,
    value_text: str,
    max_length: int,
) -> Tuple[Dict[str, Any], List[int], List[int], List[str]]:
    # Need fast tokenizer for sequence_ids.
    enc = tokenizer(
        key_text,
        value_text,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_offsets_mapping=True,
        return_tensors=None,
    )

    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Fast tokenizer is required to recover pair spans. Please provide a fast tokenizer.")

    ids = enc["input_ids"]
    special = enc.get("special_tokens_mask", [0] * len(ids))
    token_texts = tokenizer.convert_ids_to_tokens(ids)

    # sequence_ids() only available on fast tokenizer encoding object.
    seq_ids = enc.encodings[0].sequence_ids
    key_idx: List[int] = []
    value_idx: List[int] = []
    for i, sid in enumerate(seq_ids):
        if special[i] == 1:
            continue
        if sid == 0:
            key_idx.append(i)
        elif sid == 1:
            value_idx.append(i)

    return enc, key_idx, value_idx, token_texts


def _to_tensor_2d(ids: List[int], dtype=torch.long, device: str = "cpu") -> torch.Tensor:
    return torch.tensor([ids], dtype=dtype, device=device)


def _build_perfect_noise_tensors(
    seq_len: int,
    device: str,
    noise_mode: str,
    noise_processor: Optional[NoiseFeatureProcessor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    perfect_values_row = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    noise_values = torch.tensor(
        [[perfect_values_row for _ in range(seq_len)]], dtype=torch.float32, device=device
    )

    if noise_mode == "bucket":
        if noise_processor is None:
            row_ids = [0] * len(FEATURES)
        else:
            row_ids = noise_processor.map_batch([perfect_values_row])[0]
        noise_ids = torch.tensor([[row_ids for _ in range(seq_len)]], dtype=torch.long, device=device)
        return noise_ids, noise_values

    return None, noise_values


def _aggregate_attention(attentions: Sequence[torch.Tensor], last_n_layers: int) -> np.ndarray:
    # attentions: tuple[L], each shape [B,H,S,S]. Use first item in batch.
    L = len(attentions)
    n = max(1, min(last_n_layers, L))
    sel = attentions[L - n : L]

    mats = []
    for a in sel:
        x = a[0].detach().float().cpu().numpy()  # [H,S,S]
        mats.append(x.mean(axis=0))  # [S,S]
    A = np.mean(np.stack(mats, axis=0), axis=0)  # [S,S]
    return A


def _attention_rollout(attentions: Sequence[torch.Tensor], last_n_layers: int) -> np.ndarray:
    L = len(attentions)
    n = max(1, min(last_n_layers, L))
    sel = attentions[L - n : L]

    seq_len = attentions[0].shape[-1]
    R = np.eye(seq_len, dtype=np.float64)
    I = np.eye(seq_len, dtype=np.float64)

    for a in sel:
        m = a[0].detach().float().cpu().numpy().mean(axis=0).astype(np.float64)  # [S,S]
        m = (m + I) / 2.0
        row_sum = m.sum(axis=-1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        m = m / row_sum
        R = R @ m
    return R


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def _compute_csam(A: np.ndarray, key_idx: List[int], value_idx: List[int]) -> float:
    if not key_idx or not value_idx:
        return 0.0
    numer = float(A[np.ix_(key_idx, value_idx)].sum())
    denom = float(A[np.ix_(key_idx, list(range(A.shape[1])))].sum())
    return _safe_div(numer, denom)


def _compute_topk_align_rate(
    A: np.ndarray,
    key_idx: List[int],
    value_idx: List[int],
    k: int,
    valid_token_idx: Optional[List[int]] = None,
) -> float:
    if not key_idx or not value_idx:
        return 0.0
    if k <= 0:
        return 0.0

    valid = valid_token_idx if valid_token_idx is not None else list(range(A.shape[1]))
    if not valid:
        return 0.0

    value_set = set(value_idx)
    rates: List[float] = []

    for qi in key_idx:
        row = A[qi, valid]
        if row.size == 0:
            continue
        kk = min(k, row.size)
        top_local = np.argpartition(-row, kk - 1)[:kk]
        top_global = [valid[int(i)] for i in top_local.tolist()]
        hit = sum(1 for x in top_global if x in value_set)
        rates.append(float(hit) / float(kk))

    if not rates:
        return 0.0
    return float(sum(rates) / len(rates))


def _plot_heatmap(matrix: np.ndarray, out_file: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 4.6), dpi=160)
    plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Value token index")
    plt.ylabel("Key token index")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"n": 1, "mean": float(values[0]), "std": 0.0}
    return {
        "n": int(len(values)),
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
    }


def _cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    mx = statistics.mean(x)
    my = statistics.mean(y)
    vx = statistics.variance(x)
    vy = statistics.variance(y)
    nx = len(x)
    ny = len(y)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(1, (nx + ny - 2))
    if pooled <= 0:
        return 0.0
    return float((mx - my) / math.sqrt(pooled))


def _mann_whitney_u(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    # Prefer scipy if available; fallback to a permutation test approximation.
    try:
        from scipy.stats import mannwhitneyu

        u, p = mannwhitneyu(x, y, alternative="two-sided")
        return {"u": float(u), "p_value": float(p), "method": "scipy_mannwhitneyu"}
    except Exception:
        return _permutation_test(x, y)


def _permutation_test(x: Sequence[float], y: Sequence[float], n_perm: int = 5000, seed: int = 2026) -> Dict[str, float]:
    if not x or not y:
        return {"u": 0.0, "p_value": 1.0, "method": "perm_approx_empty"}

    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    obs = abs(x.mean() - y.mean())
    z = np.concatenate([x, y])
    n = len(x)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(z)
        a = z[:n]
        b = z[n:]
        if abs(a.mean() - b.mean()) >= obs:
            count += 1
    p = (count + 1) / float(n_perm + 1)
    return {"u": 0.0, "p_value": float(p), "method": f"perm_mean_diff_{n_perm}"}


def _select_case_indices(samples: Sequence[Dict[str, Any]], topn: int, key_name: str) -> List[int]:
    buckets: Dict[str, List[Tuple[int, float]]] = {}
    for i, s in enumerate(samples):
        g = s["group"]
        score = float(s.get(key_name, 0.0))
        buckets.setdefault(g, []).append((i, score))

    out: List[int] = []
    for g, rows in buckets.items():
        rows_sorted = sorted(rows, key=lambda t: t[1], reverse=True)
        out.extend([i for i, _ in rows_sorted[:topn]])
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention visualization and metrics for KV-NSP")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to staged model (e.g., final_staged_model)")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path; default uses model_dir")
    parser.add_argument("--input_file", type=str, required=True, help="JSON/JSONL with key-value pairs")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="hfl/chinese-macbert-base",
        help="Fallback backbone config when model_dir has no config.json",
    )

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_samples_per_group", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--last_n_layers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--case_topn_per_group", type=int, default=6)
    parser.add_argument("--run_rollout", action="store_true")

    parser.add_argument("--noise_bins_json", type=str, default=None)
    parser.add_argument(
        "--noise_mode_hint",
        type=str,
        default="bucket",
        choices=["bucket", "linear", "mlp"],
        help="Used only when model_dir lacks config.json",
    )
    parser.add_argument("--inject_perfect_noise", action="store_true", help="Inject perfect noise vectors when forwarding")

    parser.add_argument("--auto_generate_negatives", action="store_true", help="If input has no negatives, auto-build reverse/random negatives")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_file)
    samples = _load_pair_samples(input_path)
    if not samples:
        raise RuntimeError(f"No pair samples found in {input_path}")

    if (not _has_negatives(samples)) and args.auto_generate_negatives:
        samples = _auto_generate_negatives(samples, seed=args.seed)

    samples = _subsample_by_group(samples, max_per_group=args.max_samples_per_group, seed=args.seed)

    tok_path = args.tokenizer_path or args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for span extraction. Please provide tokenizer.json enabled tokenizer.")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"model_dir not found: {model_dir}. Please check remote path and run 'ls -lah {model_dir.parent}' to verify."
        )
    if not model_dir.is_dir():
        raise NotADirectoryError(f"model_dir is not a directory: {model_dir}")

    noise_processor = None
    noise_bin_edges = None
    if args.noise_bins_json:
        p = Path(args.noise_bins_json)
        if p.is_file():
            noise_processor = NoiseFeatureProcessor.load(str(p))
            noise_bin_edges = noise_processor.bin_edges

    model = _load_dapt_model(
        model_dir=model_dir,
        device=args.device,
        base_model_name=args.base_model_name,
        noise_mode_hint=args.noise_mode_hint,
        noise_bin_edges=noise_bin_edges,
    )
    model.eval()

    noise_mode = str(getattr(model.config, "noise_mode", args.noise_mode_hint) or args.noise_mode_hint).lower()

    per_sample: List[Dict[str, Any]] = []

    with torch.no_grad():
        for s in samples:
            enc, key_idx, value_idx, token_texts = _prepare_encoding(
                tokenizer=tokenizer,
                key_text=s.key_text,
                value_text=s.value_text,
                max_length=args.max_length,
            )

            if not key_idx or not value_idx:
                continue

            ids = enc["input_ids"]
            mask = enc["attention_mask"]
            ttypes = enc.get("token_type_ids")
            if ttypes is None:
                ttypes = [0] * len(ids)

            input_ids_t = _to_tensor_2d(ids, dtype=torch.long, device=args.device)
            mask_t = _to_tensor_2d(mask, dtype=torch.long, device=args.device)
            ttype_t = _to_tensor_2d(ttypes, dtype=torch.long, device=args.device)

            noise_ids_t = None
            noise_values_t = None
            if args.inject_perfect_noise:
                noise_ids_t, noise_values_t = _build_perfect_noise_tensors(
                    seq_len=len(ids),
                    device=args.device,
                    noise_mode=noise_mode,
                    noise_processor=noise_processor,
                )

            out = model(
                input_ids=input_ids_t,
                attention_mask=mask_t,
                token_type_ids=ttype_t,
                noise_ids=noise_ids_t,
                noise_values=noise_values_t,
                output_attentions=True,
                return_dict=True,
            )
            attentions = out.attentions
            if attentions is None:
                continue

            A = _aggregate_attention(attentions, last_n_layers=args.last_n_layers)
            valid_idx = [i for i, m in enumerate(mask) if int(m) == 1]

            csam = _compute_csam(A, key_idx=key_idx, value_idx=value_idx)
            topk_rate = _compute_topk_align_rate(
                A,
                key_idx=key_idx,
                value_idx=value_idx,
                k=args.topk,
                valid_token_idx=valid_idx,
            )

            rec: Dict[str, Any] = {
                "sample_id": s.sample_id,
                "label": int(s.label),
                "group": s.group,
                "noise_level": _infer_noise_level(s.meta),
                "key_text": s.key_text,
                "value_text": s.value_text,
                "key_len": len(key_idx),
                "value_len": len(value_idx),
                "seq_len": int(len(ids)),
                "csam": float(csam),
                "topk_align_rate": float(topk_rate),
                "tokens": token_texts,
                "key_span_idx": key_idx,
                "value_span_idx": value_idx,
            }

            if args.run_rollout:
                R = _attention_rollout(attentions, last_n_layers=args.last_n_layers)
                rec["rollout_csam"] = float(_compute_csam(R, key_idx=key_idx, value_idx=value_idx))
                rec["rollout_topk_align_rate"] = float(
                    _compute_topk_align_rate(
                        R,
                        key_idx=key_idx,
                        value_idx=value_idx,
                        k=args.topk,
                        valid_token_idx=valid_idx,
                    )
                )

            # Keep sub-matrix for selected case rendering.
            sub = A[np.ix_(key_idx, value_idx)]
            rec["key_to_value_submatrix"] = sub.tolist()

            if args.run_rollout:
                rsub = R[np.ix_(key_idx, value_idx)]
                rec["rollout_key_to_value_submatrix"] = rsub.tolist()

            per_sample.append(rec)

    if not per_sample:
        raise RuntimeError("No valid samples were processed. Check tokenizer spans and input format.")

    # Save per-sample metrics jsonl
    per_sample_path = out_dir / "per_sample_metrics.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as f:
        for r in per_sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Group summaries
    groups = sorted({r["group"] for r in per_sample})
    summary: Dict[str, Any] = {
        "num_samples": len(per_sample),
        "groups": {},
        "noise_groups": {},
        "tests": {},
        "config": {
            "model_dir": args.model_dir,
            "tokenizer_path": tok_path,
            "input_file": str(input_path),
            "last_n_layers": args.last_n_layers,
            "topk": args.topk,
            "run_rollout": bool(args.run_rollout),
            "inject_perfect_noise": bool(args.inject_perfect_noise),
            "noise_mode": noise_mode,
        },
    }

    for g in groups:
        rows = [r for r in per_sample if r["group"] == g]
        summary["groups"][g] = {
            "csam": _mean_std([float(r["csam"]) for r in rows]),
            "topk_align_rate": _mean_std([float(r["topk_align_rate"]) for r in rows]),
        }
        if args.run_rollout:
            summary["groups"][g]["rollout_csam"] = _mean_std([float(r.get("rollout_csam", 0.0)) for r in rows])
            summary["groups"][g]["rollout_topk_align_rate"] = _mean_std(
                [float(r.get("rollout_topk_align_rate", 0.0)) for r in rows]
            )

    noise_levels = sorted({r["noise_level"] for r in per_sample})
    for nl in noise_levels:
        rows = [r for r in per_sample if r["noise_level"] == nl]
        summary["noise_groups"][nl] = {
            "csam": _mean_std([float(r["csam"]) for r in rows]),
            "topk_align_rate": _mean_std([float(r["topk_align_rate"]) for r in rows]),
        }

    # Significance tests for key contrasts.
    pos = [float(r["csam"]) for r in per_sample if r["group"] == "positive"]
    rev = [float(r["csam"]) for r in per_sample if r["group"] == "reverse"]
    rnd = [float(r["csam"]) for r in per_sample if r["group"] == "random"]

    if pos and rev:
        t = _mann_whitney_u(pos, rev)
        summary["tests"]["csam_positive_vs_reverse"] = {
            **t,
            "cohens_d": _cohens_d(pos, rev),
            "n_pos": len(pos),
            "n_reverse": len(rev),
        }
    if pos and rnd:
        t = _mann_whitney_u(pos, rnd)
        summary["tests"]["csam_positive_vs_random"] = {
            **t,
            "cohens_d": _cohens_d(pos, rnd),
            "n_pos": len(pos),
            "n_random": len(rnd),
        }

    # Also test topk align rates.
    pos_t = [float(r["topk_align_rate"]) for r in per_sample if r["group"] == "positive"]
    rev_t = [float(r["topk_align_rate"]) for r in per_sample if r["group"] == "reverse"]
    rnd_t = [float(r["topk_align_rate"]) for r in per_sample if r["group"] == "random"]

    if pos_t and rev_t:
        t = _mann_whitney_u(pos_t, rev_t)
        summary["tests"]["topk_positive_vs_reverse"] = {
            **t,
            "cohens_d": _cohens_d(pos_t, rev_t),
            "n_pos": len(pos_t),
            "n_reverse": len(rev_t),
        }
    if pos_t and rnd_t:
        t = _mann_whitney_u(pos_t, rnd_t)
        summary["tests"]["topk_positive_vs_random"] = {
            **t,
            "cohens_d": _cohens_d(pos_t, rnd_t),
            "n_pos": len(pos_t),
            "n_random": len(rnd_t),
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Case heatmaps: top csam per group.
    case_idx = _select_case_indices(per_sample, topn=args.case_topn_per_group, key_name="csam")
    for idx in case_idx:
        rec = per_sample[idx]
        sub = np.array(rec["key_to_value_submatrix"], dtype=np.float64)
        if sub.size > 0:
            fn = cases_dir / f"{rec['group']}_{rec['sample_id']}_k2v.png"
            _plot_heatmap(sub, fn, title=f"{rec['group']} | CSAM={rec['csam']:.4f}")

        if args.run_rollout and "rollout_key_to_value_submatrix" in rec:
            rsub = np.array(rec["rollout_key_to_value_submatrix"], dtype=np.float64)
            if rsub.size > 0:
                fn = cases_dir / f"{rec['group']}_{rec['sample_id']}_k2v_rollout.png"
                _plot_heatmap(rsub, fn, title=f"{rec['group']} | Rollout CSAM={rec.get('rollout_csam', 0.0):.4f}")

    # Lightweight markdown report
    report_lines = []
    report_lines.append("# KV-NSP Attention Visualization Report")
    report_lines.append("")
    report_lines.append(f"- Processed samples: {len(per_sample)}")
    report_lines.append(f"- Model: {args.model_dir}")
    report_lines.append(f"- Noise mode: {noise_mode}")
    report_lines.append("")
    report_lines.append("## Group Summary")
    for g in groups:
        info = summary["groups"][g]
        report_lines.append(
            f"- {g}: CSAM mean={info['csam']['mean']:.4f} std={info['csam']['std']:.4f}; "
            f"Top-k mean={info['topk_align_rate']['mean']:.4f} std={info['topk_align_rate']['std']:.4f}"
        )
    report_lines.append("")
    report_lines.append("## Significance Tests")
    if not summary["tests"]:
        report_lines.append("- No valid group pairs for testing.")
    else:
        for name, t in summary["tests"].items():
            report_lines.append(
                f"- {name}: p={t.get('p_value', 1.0):.6g}, d={t.get('cohens_d', 0.0):.4f}, method={t.get('method', 'n/a')}"
            )

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[OK] per-sample metrics: {per_sample_path}")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] markdown report: {report_path}")
    print(f"[OK] case heatmaps dir: {cases_dir}")


if __name__ == "__main__":
    main()
