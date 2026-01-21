#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

if __package__ in (None, ""):
    _PACKAGE_ROOT = Path(__file__).resolve().parents[2]
    if str(_PACKAGE_ROOT) not in os.sys.path:
        os.sys.path.insert(0, str(_PACKAGE_ROOT))
    from pre_struct.kv_ner import config_io
    from pre_struct.kv_ner.data_utils import build_bio_label_list, load_labelstudio_export
    from pre_struct.kv_ner.dataset import TokenClassificationDataset, collate_batch
    from pre_struct.kv_ner.metrics import char_spans, compute_ner_metrics
    from pre_struct.kv_ner.modeling import BertCrfTokenClassifier
else:
    from . import config_io
    from .data_utils import build_bio_label_list, load_labelstudio_export
    from .dataset import TokenClassificationDataset, collate_batch
    from .metrics import char_spans, compute_ner_metrics
    from .modeling import BertCrfTokenClassifier

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def _entity_records(
    label_ids: List[int],
    mask: List[bool],
    offsets: List[List[int]],
    id2label: Dict[int, str],
    text: str,
) -> List[Dict[str, Any]]:
    spans = char_spans(label_ids, mask, offsets, id2label)
    entities: List[Dict[str, Any]] = []
    for ent_type, start, end in spans:
        snippet = text[start:end] if 0 <= start < end <= len(text) else ""
        entities.append(
            {
                "type": ent_type,
                "start": start,
                "end": end,
                "text": snippet.strip(),
            }
        )
    entities.sort(key=lambda x: (x["start"], x["end"]))
    return entities


def _postprocess_value_for_key(
    key_text: str,
    value_text: str,
    start: int,
    end: int,
    full_text: str,
    cfg: Dict[str, Any],
) -> Tuple[str, int, int]:
    try:
        cap = int(cfg.get("value_len_cap", 64))
        stop_chars = set(str(cfg.get("value_stop_punct", "。；;，,\n")))
        phone_keys = list(cfg.get("value_phone_keys", []))
        phone_regex = str(cfg.get("value_phone_regex", r"(?:\\+?86[- ]?)?1[3-9]\\d{9}"))
    except Exception:
        cap = 64
        stop_chars = set("。；;，,\n")
        phone_keys = ["电话", "联系方式", "联系电话"]
        phone_regex = r"(?:\+?86[- ]?)?1[3-9]\d{9}"

    kt = key_text or ""
    if any(s in kt for s in phone_keys):
        import re
        m = re.search(phone_regex, value_text)
        if m:
            off = m.start()
            new_start = start + off
            new_end = new_start + (m.end() - m.start())
            return m.group(0), new_start, new_end
    if len(value_text) > cap:
        for i, ch in enumerate(value_text):
            if i >= 6 and ch in stop_chars:
                return value_text[:i].strip(), start, start + i
    return value_text, start, end


def _assemble_pairs(
    entities: List[Dict[str, Any]],
    *,
    value_attach_window: int = 50,
    value_same_line_only: bool = True,
    value_crossline_fallback_len: int = 0,
    full_text: Optional[str] = None,
) -> Dict[str, Any]:
    hospital = [e for e in entities if e["type"] == "HOSPITAL"]
    seq = [e for e in entities if e["type"] in {"KEY", "VALUE"}]
    seq.sort(key=lambda x: (x["start"], x["end"]))

    pairs: List[Dict[str, Any]] = []
    key_without_value: List[Dict[str, Any]] = []
    value_without_key: List[Dict[str, Any]] = []

    pending: Optional[Dict[str, Any]] = None
    for idx, ent in enumerate(seq):
        if ent["type"] == "KEY":
            if pending:
                if pending["values"]:
                    v0 = pending["values"][0]
                    v1 = pending["values"][-1]
                    s = int(v0.get("start", 0)); e = int(v1.get("end", 0))
                    if full_text is not None:
                        s = max(0, min(len(full_text), s))
                        e = max(s, min(len(full_text), e))
                        if bool(int(os.environ.get("KVNER_EXPAND_SENTENCE", "0"))):
                            stopset = set("。；;.!？！?\n")
                            limit = int(os.environ.get("KVNER_EXPAND_MAX", "120"))
                            i = e
                            steps = 0
                            while i < len(full_text) and steps < limit:
                                ch = full_text[i]
                                i += 1
                                steps += 1
                                if ch in stopset:
                                    e = i
                                    break
                        slice_text = full_text[s:e]
                    else:
                        slice_text = "；".join(v.get("text", "") for v in pending["values"])
                    pairs.append(
                        {
                            "key": pending["key"],
                            "values": [{"type": "VALUE", "start": s, "end": e, "text": slice_text}],
                            "value_text": slice_text,
                        }
                    )
                else:
                    key_without_value.append(pending["key"])
            pending = {"key": ent, "values": []}
        elif ent["type"] == "VALUE":
            if pending:
                pending["values"].append(ent)
            else:
                attached = False
                if full_text is not None and value_attach_window > 0:
                    for k in range(idx - 1, -1, -1):
                        prev = seq[k]
                        if prev["type"] != "KEY":
                            continue
                        if ent["start"] - prev["end"] <= value_attach_window:
                            if value_same_line_only:
                                middle = full_text[prev["end"]:ent["start"]]
                                if "\n" in middle and len(ent.get("text", "")) > int(value_crossline_fallback_len):
                                    continue
                            if full_text is not None:
                                s = int(ent.get("start", 0)); e = int(ent.get("end", 0))
                                s = max(0, min(len(full_text), s))
                                e = max(s, min(len(full_text), e))
                                vtxt = full_text[s:e]
                            else:
                                vtxt = ent.get("text", "")
                            pairs.append({"key": prev, "values": [ent], "value_text": vtxt})
                            attached = True
                            break
                if not attached:
                    value_without_key.append(ent)
    if pending:
        if pending["values"]:
            v0 = pending["values"][0]
            v1 = pending["values"][-1]
            s = int(v0.get("start", 0)); e = int(v1.get("end", 0))
            if full_text is not None:
                s = max(0, min(len(full_text), s))
                e = max(s, min(len(full_text), e))
                if bool(int(os.environ.get("KVNER_EXPAND_SENTENCE", "0"))):
                    stopset = set("。；;.!？！?\n")
                    limit = int(os.environ.get("KVNER_EXPAND_MAX", "120"))
                    i = e
                    steps = 0
                    while i < len(full_text) and steps < limit:
                        ch = full_text[i]
                        i += 1
                        steps += 1
                        if ch in stopset:
                            e = i
                            break
                slice_text = full_text[s:e]
            else:
                slice_text = "；".join(v.get("text", "") for v in pending["values"])
            pairs.append(
                {
                    "key": pending["key"],
                    "values": [{"type": "VALUE", "start": s, "end": e, "text": slice_text}],
                    "value_text": slice_text,
                }
            )
        else:
            key_without_value.append(pending["key"])

    all_kvs = []
    
    for entry in pairs:
        key_text = entry["key"]["text"]
        value_text = entry["value_text"]
        if not key_text:
            continue
        key_start = entry["key"].get("start", 0)
        all_kvs.append((key_text, value_text, key_start, False))
    
    for key_ent in key_without_value:
        key_text = key_ent.get("text", "").strip()
        if key_text:
            key_start = key_ent.get("start", 0)
            all_kvs.append((key_text, "", key_start, False))
    
    for hospital_ent in hospital:
        hospital_text = hospital_ent.get("text", "").strip()
        if hospital_text:
            hospital_start = hospital_ent.get("start", 0)
            all_kvs.append(("医院名称", hospital_text, hospital_start, True))
    
    all_kvs.sort(key=lambda x: (0 if x[3] else 1, x[2]))
    
    structured: Dict[str, Any] = {}
    for key_text, value_text, _, _ in all_kvs:
        existing = structured.get(key_text)
        if existing is None:
            structured[key_text] = value_text
        else:
            if isinstance(existing, list):
                existing.append(value_text)
            else:
                structured[key_text] = [existing, value_text]

    return {
        "pairs": pairs,
        "key_without_value": key_without_value,
        "value_without_key": value_without_key,
        "structured": structured,
        "hospital": hospital,
    }


def predict(args: argparse.Namespace) -> None:
    cfg = config_io.load_config(args.config)
    predict_block = config_io.ensure_block(cfg, "predict")
    label_map = config_io.label_map_from(cfg)
    label_list = build_bio_label_list(label_map)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    o_id = next((idx for idx, name in id2label.items() if name == "O"), 0)

    input_path = args.input or predict_block.get("input_path")
    if not input_path:
        raise ValueError("Input path must be provided via --input or predict.input_path")
    output_path = args.output or predict_block.get("output_path") or "runs/kv_ner/predictions.json"
    model_dir = args.model_dir or predict_block.get("model_dir") or cfg.get("model_dir")
    if not model_dir:
        raise ValueError("Model directory must be provided (predict.model_dir or top-level model_dir).")

    tokenizer_path = Path(model_dir) / "tokenizer"
    if tokenizer_path.is_dir():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config_io.tokenizer_name_from(cfg), use_fast=True)

    samples = load_labelstudio_export(
        input_path,
        label_map,
        include_unlabeled=True,
    )
    dataset = TokenClassificationDataset(
        samples,
        tokenizer,
        label2id,
        max_seq_length=config_io.max_seq_length(cfg),
        label_all_tokens=config_io.label_all_tokens(cfg),
        include_labels=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(predict_block.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(predict_block.get("num_workers", 0)),
        pin_memory=bool(predict_block.get("pin_memory", False)),
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertCrfTokenClassifier.from_pretrained(model_dir).to(device)
    model.eval()

    results: List[Dict[str, Any]] = []
    metric_inputs = {"pred": [], "ref": [], "mask": [], "offset": []}

    value_attach_window = int(cfg.get("value_attach_window", 50))
    value_same_line_only = bool(cfg.get("value_same_line_only", True))
    adjust_boundaries = bool(cfg.get("adjust_boundaries", False))
    adjust_max_shift = int(cfg.get("adjust_max_shift", 1))
    adjust_chars = set(str(cfg.get("adjust_chars", "")))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            token_type_ids = batch.token_type_ids.to(device)
            labels = batch.labels

            decoded = model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            seq_len = batch.labels.size(1)
            attn_cpu = attention_mask.cpu().tolist()
            offsets_cpu = batch.offset_mapping.cpu().tolist()
            labels_cpu = labels.tolist()

            for i, seq in enumerate(decoded):
                seq_list = list(seq)
                if len(seq_list) < seq_len:
                    seq_list = seq_list + [o_id] * (seq_len - len(seq_list))
                elif len(seq_list) > seq_len:
                    seq_list = seq_list[:seq_len]

                mask_list = [bool(v) for v in attn_cpu[i]]
                offset_list = offsets_cpu[i]
                full_text = batch.texts[i]
                entities = _entity_records(seq_list, mask_list, offset_list, id2label, full_text)
                if adjust_boundaries and entities:
                    L = len(full_text)
                    adj = []
                    for ent in entities:
                        if str(ent.get('type')) == 'KEY':
                            adj.append(ent)
                            continue
                        s = int(ent.get('start', -1))
                        epos = int(ent.get('end', -1))
                        if 0 <= s < epos <= L:
                            for _ in range(adjust_max_shift):
                                if s > 0 and full_text[s-1] in adjust_chars:
                                    s -= 1
                                else:
                                    break
                            for _ in range(adjust_max_shift):
                                if epos < L and full_text[epos] in adjust_chars:
                                    epos += 1
                                else:
                                    break
                            ent = dict(ent)
                            ent['start'] = s
                            ent['end'] = epos
                            ent['text'] = full_text[s:epos].strip()
                        adj.append(ent)
                    entities = adj
                structure = _assemble_pairs(
                    entities,
                    value_attach_window=value_attach_window,
                    value_same_line_only=value_same_line_only,
                    value_crossline_fallback_len=int(cfg.get("value_crossline_fallback_len", 0)),
                    full_text=full_text,
                )

                for p in structure["pairs"]:
                    kt = p["key"]["text"]
                    if p["values"]:
                        v0 = p["values"][0]
                        vt, ns, ne = _postprocess_value_for_key(
                            kt, v0.get("text", ""), int(v0.get("start", -1)), int(v0.get("end", -1)), full_text, cfg
                        )
                        p["values"] = [{"type": "VALUE", "start": ns, "end": ne, "text": vt}]
                        p["value_text"] = vt
                        if kt in structure["structured"]:
                            structure["structured"][kt] = vt

                record = {
                    "task_id": batch.task_ids[i],
                    "title": batch.titles[i],
                    "text": batch.texts[i],
                    "entities": entities,
                    "structured": structure["structured"],
                    "pairs": structure["pairs"],
                    "key_without_value": structure["key_without_value"],
                    "value_without_key": structure["value_without_key"],
                    "hospital": structure["hospital"],
                }
                results.append(record)

                if batch.entities[i]:
                    metric_inputs["pred"].append(seq_list)
                    metric_inputs["ref"].append(labels_cpu[i])
                    metric_inputs["mask"].append(mask_list)
                    metric_inputs["offset"].append(offset_list)

    output = {"results": results}

    if metric_inputs["pred"]:
        metrics = compute_ner_metrics(
            metric_inputs["pred"],
            metric_inputs["ref"],
            metric_inputs["mask"],
            id2label,
            offsets=metric_inputs["offset"],
        )
        output["metrics"] = metrics
        logger.info(
            "Evaluation on annotated subset: F1=%.4f (precision=%.4f, recall=%.4f)",
            metrics["overall"]["f1"],
            metrics["overall"]["precision"],
            metrics["overall"]["recall"],
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Predictions written to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the KV-NER model.")
    parser.add_argument(
        "--config",
        type=str,
        default=config_io.default_config_path(),
        help="Path to kv_ner_config.json",
    )
    parser.add_argument("--input", type=str, help="Override input path for inference")
    parser.add_argument("--output", type=str, help="Override output path")
    parser.add_argument("--model-dir", type=str, help="Override model directory")
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
