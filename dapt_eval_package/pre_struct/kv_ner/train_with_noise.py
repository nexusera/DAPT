#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KV-NER 训练脚本（支持 DAPT 噪声特征）

支持功能：
1. 从 JSONL 加载数据（含 noise_values 字段）
2. 使用 NoiseCollator 自动处理 noise_ids 对齐
3. 将 noise_ids 传入模型
4. 保存最佳模型到新的输出目录（不破坏原模型）

使用方式：
    python train_with_noise.py \\
        --config kv_ner_config.json \\
        --noise_bins /path/to/noise_bins.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
try:
    from noise_fusion import uses_bucket_noise, uses_continuous_noise
except Exception:  # pragma: no cover
    import sys
    from pathlib import Path as _Path
    _ROOT = _Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from noise_fusion import uses_bucket_noise, uses_continuous_noise

if __package__ in (None, ""):
    _PACKAGE_ROOT = Path(__file__).resolve().parents[2]
    if str(_PACKAGE_ROOT) not in os.sys.path:
        os.sys.path.insert(0, str(_PACKAGE_ROOT))
    from pre_struct.kv_ner import config_io
    from pre_struct.kv_ner.data_utils import (
        build_bio_label_list,
        load_labelstudio_export,
        split_samples,
        Entity,
        Relation,
        Sample,
        _select_latest_annotation,
        # _normalize_label,  # Use local version
    )
    from pre_struct.kv_ner.dataset import TokenClassificationDataset, collate_batch
    from pre_struct.kv_ner.metrics import compute_ner_metrics
    from pre_struct.kv_ner.modeling import BertCrfTokenClassifier
    from pre_struct.kv_ner.noise_utils import (
        NoiseFeatureProcessor,
        NoiseCollator,
        prepare_noise_ids_for_model,
        PERFECT_VALUES,
    )
else:
    from . import config_io
    from .data_utils import (
        build_bio_label_list,
        load_labelstudio_export,
        split_samples,
        Entity,
        Relation,
        Sample,
        _select_latest_annotation,
        # _normalize_label, # Use local version
    )
    from .dataset import TokenClassificationDataset, collate_batch
    from .metrics import compute_ner_metrics
    from .modeling import BertCrfTokenClassifier
    from .noise_utils import (
        NoiseFeatureProcessor,
        NoiseCollator,
        prepare_noise_ids_for_model,
        PERFECT_VALUES,
    )

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def _normalize_label(raw: str, label_map: Dict[str, str]) -> Optional[str]:
    # Debug: Print unrecognized labels to help troubleshooting 
    if not raw: return None
    
    # 1. Exact match in config map
    if raw in label_map:
        return label_map[raw]
        
    # 2. Heuristic normalization (Case insensitive)
    # 许多数据里的 label 是 "KEY"/"Value" 或中文 "键"/"值"
    # 我们不仅查表，还尝试模糊匹配
    upper_raw = raw.upper()
    
    # Check if upper case version is in map
    if upper_raw in label_map:
        return label_map[upper_raw]
        
    # Check common hardcoded targets
    if upper_raw in ["KEY", "VALUE", "HOSPITAL"]: 
        return upper_raw
        
    # Keywords check
    if "key" in raw.lower() or "键" in raw: return "KEY"
    if "value" in raw.lower() or "值" in raw: return "VALUE"
    
    return None

def _expand_word_noise_to_chars(ocr_raw, noise_values_per_word):
    """Expand per-word 7-d noise to per-character list using ocr_raw.words_result."""
    if not (isinstance(ocr_raw, dict) and isinstance(noise_values_per_word, list)):
        return None
    words_result = ocr_raw.get("words_result")
    if not isinstance(words_result, list):
        return None
    char_noise = []
    for wr, nv in zip(words_result, noise_values_per_word):
        if not (isinstance(wr, dict) and isinstance(nv, (list, tuple)) and len(nv) == 7):
            continue
        w = wr.get("words", "")
        if not isinstance(w, str):
            continue
        repeat = max(1, len(w))
        char_noise.extend([list(nv)] * repeat)
    return char_noise if char_noise else None


def _broadcast_global_noise(noise_values, text_len: int):
    """If noise is a single 7-d vector, broadcast to text length."""
    if (
        isinstance(noise_values, list)
        and len(noise_values) == 7
        and all(not isinstance(v, (list, tuple)) for v in noise_values)
        and text_len > 0
    ):
        return [list(noise_values) for _ in range(text_len)]
    return noise_values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl_with_noise(path: str | Path, label_map: Dict[str, str], include_unlabeled: bool = False) -> List[Sample]:
    """
    从 JSONL 文件加载样本，支持两类格式：

    1) 逐行普通 JSONL，包含 text/title/key_value_pairs/noise_values 字段。
    2) Label Studio 按行导出的任务 JSON（包含 annotations/data 等字段）。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    
    samples: List[Sample] = []

    def _parse_labelstudio_task(task: dict) -> Optional[Sample]:
        data = task.get("data", {}) if isinstance(task, dict) else {}
        # 标注行：若有 per-word 噪声，先与 ocr_raw.words_result 对齐后再展开到字符
        ocr_raw = task.get("ocr_raw") or data.get("ocr_raw")
        per_word_noise = data.get("noise_values_per_word") or task.get("noise_values_per_word")
        text = str(data.get("ocr_text") or data.get("text") or "").strip()
        if not text:
            return None

        results = _select_latest_annotation(task)
        entities: List[Entity] = []
        relations: List[Relation] = []

        for res in results:
            r_type = res.get("type")
            if r_type == "labels":
                value = res.get("value") or {}
                raw_labels = value.get("labels") or []
                if not raw_labels:
                    continue
                normalized = _normalize_label(raw_labels[0], label_map)
                if not normalized:
                    continue
                start = int(value.get("start") or 0)
                end = int(value.get("end") or 0)
                if end <= start or start < 0:
                    continue
                entities.append(
                    Entity(
                        start=start,
                        end=end,
                        label=normalized,
                        result_id=res.get("id"),
                        text=value.get("text"),
                    )
                )
            elif r_type == "relation":
                from_id = res.get("from_id")
                to_id = res.get("to_id")
                if isinstance(from_id, str) and isinstance(to_id, str):
                    relations.append(
                        Relation(
                            from_id=from_id,
                            to_id=to_id,
                            direction=str(res.get("direction") or "right"),
                        )
                    )

        if not entities and not include_unlabeled:
            return None

        expanded_noise = _expand_word_noise_to_chars(ocr_raw, per_word_noise)
        noise_values = expanded_noise or data.get("noise_values") or task.get("noise_values")
        noise_values = _broadcast_global_noise(noise_values, len(text))
        return Sample(
            task_id=str(task.get("id")),
            text=text,
            title=str(data.get("category") or data.get("title") or ""),
            entities=sorted(entities, key=lambda e: (e.start, e.end)),
            relations=relations,
            noise_values=noise_values,
        )
    
    with p.open("r", encoding="utf-8-sig") as f:
        # Robust check: Read first non-whitespace char
        pos = 0
        while True:
            char = f.read(1)
            pos += 1
            if not char: break # EOF
            if not char.isspace():
                break
        
        f.seek(0)
        objects = []
        
        # Debug log
        if char:
            logger.info(f"First non-whitespace char: '{char}' (ord: {ord(char)})")
        else:
            logger.warning("File seems empty or only whitespace")

        if char == "[":
            try:
                # Seek to 0 ONLY if we didn't advance too much? No, seek 0 is safe usually.
                # If we skipped whitespace, we might want to seek back to 'pos' not 0?
                # Actually, standard JSON ignores whitespace, so seeking 0 is fine.
                f.seek(0)
                objects = json.load(f)
                logger.info(f"Loaded {len(objects)} items from JSON array.")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON array from {p}: {e}")
                raise
        else:
             # Regular JSONL or Single Object
             f.seek(0)
             for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # If line is a list (e.g. single-line array file missed by '[' check due to BOM/encoding issues)
                    if isinstance(obj, list):
                        objects.extend(obj)
                    else:
                        objects.append(obj)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_no}: JSON parse error - {e}")
                    continue

        for obj in objects:
            # Common handling for both formats
            # Label Studio JSONL（单行一个任务）
            if isinstance(obj, dict) and "annotations" in obj and "data" in obj:
                sample = _parse_labelstudio_task(obj)
                if sample:
                    samples.append(sample)
                continue

            if not isinstance(obj, dict):
                continue

            data_block = obj.get("data") if isinstance(obj.get("data"), dict) else {}

            task_id = str(
                obj.get("id")
                or obj.get("record_id")
                or data_block.get("id")
                or data_block.get("record_id")
                or "unknown"
            )

            text = str(
                obj.get("ocr_text")
                or data_block.get("ocr_text")
                or obj.get("text")
                or data_block.get("text")
                or ""
            ).strip()

            title = str(
                obj.get("category")
                or data_block.get("category")
                or obj.get("title")
                or data_block.get("title")
                or ""
            )

            # 普通 JSONL 行：同样优先处理 per-word 噪声，回退到 noise_values
            ocr_raw = obj.get("ocr_raw") or data_block.get("ocr_raw")
            per_word_noise = obj.get("noise_values_per_word") or data_block.get("noise_values_per_word")
            
            if not text:
                continue

            expanded_noise = _expand_word_noise_to_chars(ocr_raw, per_word_noise)
            noise_values = expanded_noise or obj.get("noise_values") or data_block.get("noise_values")
            noise_values = _broadcast_global_noise(noise_values, len(text))
            
            # 解析 transferred_annotations 为 Entity (兼容旧的 key_value_pairs)
            entities: List[Entity] = []
            
            # Case 1: Standard key_value_pairs
            kv_list = obj.get("key_value_pairs") or data_block.get("key_value_pairs") or []
            if isinstance(kv_list, list) and kv_list:
                for kv in kv_list:
                    if not isinstance(kv, dict): continue
                    # ... processing key_value_pairs (omitted for brevity, assume similar logic)
                    key_info = kv.get("key", {})
                    key_start = int(key_info.get("start", -1)) if isinstance(key_info, dict) else -1
                    key_end = int(key_info.get("end", -1)) if isinstance(key_info, dict) else -1
                    if 0 <= key_start < key_end <= len(text):
                        entities.append(Entity(start=key_start, end=key_end, label="KEY", text=key_info.get("text")))
                    
                    val_info = kv.get("value", {})
                    val_start = int(val_info.get("start", -1)) if isinstance(val_info, dict) else -1
                    val_end = int(val_info.get("end", -1)) if isinstance(val_info, dict) else -1
                    if 0 <= val_start < val_end <= len(text):
                        entities.append(Entity(start=val_start, end=val_end, label="VALUE", text=val_info.get("text")))

            # Case 2: transferred_annotations (New Format)
            annos = obj.get("transferred_annotations") or data_block.get("transferred_annotations") or []
            if not isinstance(annos, list):
                annos = []
            if isinstance(annos, list):
                 for ann in annos:
                     if not isinstance(ann, dict):
                         continue

                     # Format A: Label Studio result item
                     # e.g. {"type":"labels","value":{"start":0,"end":5,"labels":["KEY"],"text":"姓名"}}
                     if "type" in ann and "value" in ann and isinstance(ann.get("value"), dict):
                         r_type = ann.get("type")
                         if r_type == "labels":
                             value = ann.get("value") or {}
                             raw_labels = value.get("labels") or []
                             if isinstance(raw_labels, str):
                                 raw_labels = [raw_labels]
                             if not raw_labels:
                                 continue
                             normalized = _normalize_label(str(raw_labels[0]), label_map)
                             if not normalized:
                                 continue
                             start = int(value.get("start") or 0)
                             end = int(value.get("end") or 0)
                             if 0 <= start < end <= len(text):
                                 entities.append(
                                     Entity(
                                         start=start,
                                         end=end,
                                         label=normalized,
                                         text=value.get("text"),
                                     )
                                 )
                         continue

                     # e.g. {"start": 0, "end": 5, "label": "KEY", "text": "姓名:"}
                     # 兼容两种 offset 写法，并处理可能的空值
                     s = ann.get("start")
                     e = ann.get("end")
                     if s is None: s = ann.get("start_offset", -1)
                     if e is None: e = ann.get("end_offset", -1)
                     
                     start = int(s) if s is not None else -1
                     end = int(e) if e is not None else -1
                     
                     label = ann.get("label")
                     if label is None and isinstance(ann.get("labels"), list) and ann.get("labels"):
                         label = ann.get("labels")[0]
                     
                     normalized = _normalize_label(label, label_map)
                     
                     # 动态索引修补：如果缺少 start/end 但有 text，尝试在全文中搜索
                     entity_text = ann.get("text") or ann.get("value")
                     if (start < 0 or end < 0) and entity_text and normalized:
                         # 简单的首次匹配策略
                         idx = text.find(str(entity_text))
                         if idx != -1:
                             start = idx
                             end = idx + len(str(entity_text))
                     
                     # 放宽校验：如果 label 存在且 normalize 后非空，即使 text=None 也尝试添加
                     if normalized:
                         if 0 <= start < end <= len(text):
                             entities.append(Entity(
                                 start=start, 
                                 end=end, 
                                 label=normalized, 
                                 text=ann.get("text")
                             ))

            # Case 3: relations
            relations: List[Relation] = []
            rels = obj.get("relations") or data_block.get("relations") or []
            if isinstance(rels, list):
                for r in rels:
                   # e.g. {"from": 3, "to": 4, "type": "key_val_pair"} or {"from_id": "...", "to_id": "..."}
                   # For "real_train_with_ocr.json", relations seem to be list of objects
                   f_id = r.get("from_id")
                   t_id = r.get("to_id")
                   if f_id and t_id:
                       relations.append(Relation(from_id=str(f_id), to_id=str(t_id), direction="right"))

            entities.sort(key=lambda e: (e.start, e.end))
            
            sample = Sample(
                task_id=task_id,
                text=text,
                title=title,
                entities=entities,
                relations=relations,
                noise_values=noise_values,
            )
            samples.append(sample)
    
    logger.info(f"Loaded {len(samples)} samples from {p}")
    return samples


def _prepare_dataloaders_with_noise(
    cfg: Dict[str, Any],
    tokenizer,
    label2id: Dict[str, int],
    train_samples: List[Sample],
    val_samples: List[Sample],
    test_samples: List[Sample],
    noise_processor: Optional[NoiseFeatureProcessor] = None,
    noise_mode: str = "bucket",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    准备数据加载器，支持 noise_ids
    """
    logger.info(f"[_prepare_dataloaders] Starting with {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples")
    
    max_len = config_io.max_seq_length(cfg)
    label_all_tokens = config_io.label_all_tokens(cfg)
    chunk_size = int(cfg.get("chunk_size", max_len))
    if chunk_size > max_len:
        chunk_size = max_len
    chunk_overlap = int(cfg.get("chunk_overlap", 0))

    logger.info(f"[_prepare_dataloaders] Creating train dataset (noise_processor={noise_processor is not None})")
    train_dataset = TokenClassificationDataset(
        train_samples,
        tokenizer,
        label2id,
        max_seq_length=max_len,
        label_all_tokens=label_all_tokens,
        include_labels=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_chunking=True,
        noise_processor=noise_processor,
        noise_mode=noise_mode,
    )
    logger.info(f"[_prepare_dataloaders] Train dataset created with {len(train_dataset)} features")
    
    logger.info(f"[_prepare_dataloaders] Creating val dataset")
    val_dataset = TokenClassificationDataset(
        val_samples,
        tokenizer,
        label2id,
        max_seq_length=max_len,
        label_all_tokens=label_all_tokens,
        include_labels=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_chunking=True,
        noise_processor=noise_processor,
        noise_mode=noise_mode,
    )
    logger.info(f"[_prepare_dataloaders] Val dataset created with {len(val_dataset)} features")
    
    logger.info(f"[_prepare_dataloaders] Creating test dataset")
    test_dataset = TokenClassificationDataset(
        test_samples,
        tokenizer,
        label2id,
        max_seq_length=max_len,
        label_all_tokens=label_all_tokens,
        include_labels=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_chunking=True,
        noise_processor=noise_processor,
        noise_mode=noise_mode,
    )
    logger.info(f"[_prepare_dataloaders] Test dataset created with {len(test_dataset)} features")
    
    train_block = config_io.ensure_block(cfg, "train")
    batch_size = int(train_block.get("train_batch_size", 16))
    eval_batch_size = int(train_block.get("eval_batch_size", batch_size))
    num_workers = int(train_block.get("num_workers", 0))
    pin_memory = bool(train_block.get("pin_memory", False))

    # 使用标准collate（数据集中已预先生成noise_ids）
    collate_fn = collate_batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    
    return train_loader, eval_loader, test_loader


def _evaluate_model(
    model: BertCrfTokenClassifier,
    dataloader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
    use_noise: bool = False,
    noise_mode: str = "bucket",
) -> Dict[str, Dict[str, float]]:
    """
    评估模型，可选支持 noise_ids
    """
    model.eval()
    predictions: List[List[int]] = []
    references: List[List[int]] = []
    masks: List[List[bool]] = []
    offsets: List[List[Tuple[int, int]]] = []

    o_id = next((idx for idx, name in id2label.items() if name == "O"), 0)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.input_ids.to(device)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.long) if isinstance(batch, dict) else batch.attention_mask.to(device, dtype=torch.long)
            token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids)).to(device, dtype=torch.long) if isinstance(batch, dict) else batch.token_type_ids.to(device, dtype=torch.long)
            labels = batch["labels"].to(device) if isinstance(batch, dict) else batch.labels.to(device)

            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            
            # 如果有noise_ids且模型支持，传入（允许样本缺噪声时跳过）
            if use_noise:
                if uses_bucket_noise(noise_mode) and "noise_ids" in (batch if isinstance(batch, dict) else batch.__dict__):
                    raw_noise = batch.get("noise_ids") if isinstance(batch, dict) else batch.noise_ids
                    noise_ids = raw_noise.to(device) if raw_noise is not None else None
                    if noise_ids is not None and model.use_noise and model.noise_embeddings:
                        with torch.no_grad():
                            for fi, emb in enumerate(model.noise_embeddings):
                                max_id = int(torch.max(noise_ids[:, :, fi]).item())
                                if max_id >= emb.num_embeddings:
                                    raise ValueError(
                                        f"noise_ids feature {fi} has max id {max_id} >= num_embeddings {emb.num_embeddings}"
                                    )
                        kwargs["noise_ids"] = noise_ids
                elif uses_continuous_noise(noise_mode) and "noise_values" in (batch if isinstance(batch, dict) else batch.__dict__):
                    raw_noise = batch.get("noise_values") if isinstance(batch, dict) else batch.noise_values
                    noise_values = raw_noise.to(device, dtype=torch.float32) if raw_noise is not None else None
                    if noise_values is not None:
                        kwargs["noise_values"] = noise_values

            decoded = model.predict(**kwargs)

            seq_len = labels.size(1)
            for i, seq in enumerate(decoded):
                seq_list = list(seq)
                if len(seq_list) < seq_len:
                    seq_list = seq_list + [o_id] * (seq_len - len(seq_list))
                elif len(seq_list) > seq_len:
                    seq_list = seq_list[:seq_len]
                predictions.append(seq_list)
            references.extend(labels.cpu().tolist())
            
            # 处理attention_mask（可能是dict或对象）
            if isinstance(batch, dict):
                masks.extend(batch["attention_mask"].cpu().tolist())
                offsets.extend(batch.get("offset_mapping", torch.zeros_like(input_ids)).cpu().tolist())
            else:
                masks.extend(batch.attention_mask.cpu().tolist())
                offsets.extend(batch.offset_mapping.cpu().tolist())

    return compute_ner_metrics(predictions, references, masks, id2label, offsets=offsets)


def train(args: argparse.Namespace) -> None:
    cfg = config_io.load_config(args.config)
    train_block = config_io.ensure_block(cfg, "train")

    # Optional CLI overrides (keep default behavior when args are None)
    if getattr(args, "learning_rate", None) is not None:
        train_block["learning_rate"] = float(args.learning_rate)
    if getattr(args, "num_train_epochs", None) is not None:
        train_block["num_train_epochs"] = int(args.num_train_epochs)
    if getattr(args, "no_bilstm", False):
        train_block["use_bilstm"] = False
    if getattr(args, "token_ce_loss_weight", None) is not None:
        train_block["token_ce_loss_weight"] = float(args.token_ce_loss_weight)
    if getattr(args, "token_ce_value_class_weight", None) is not None:
        train_block["token_ce_value_class_weight"] = float(args.token_ce_value_class_weight)
    if getattr(args, "token_ce_key_class_weight", None) is not None:
        train_block["token_ce_key_class_weight"] = float(args.token_ce_key_class_weight)
    if getattr(args, "noise_mode", None):
        train_block["noise_mode"] = str(args.noise_mode)
    if getattr(args, "noise_mlp_hidden_dim", None) is not None:
        train_block["noise_mlp_hidden_dim"] = int(args.noise_mlp_hidden_dim)

    label_map = config_io.label_map_from(cfg)
    label_list = build_bio_label_list(label_map)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    set_seed(int(train_block.get("seed", 42)))

    # 优先从 pretrained_model 加载 tokenizer（如果是 DAPT 模型，词表已扩展）
    if args.pretrained_model:
        tokenizer_name = args.pretrained_model
        logger.info(f"Loading tokenizer from pretrained model: {tokenizer_name}")
    else:
        tokenizer_name = config_io.tokenizer_name_from(cfg)
        logger.info(f"Loading tokenizer from config: {tokenizer_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    # Fast tokenizer sanity-check: downstream NER relies on offset_mapping.
    # If vocab.txt was edited but tokenizer.json is stale/mismatched, fast tokenization may degenerate
    # (e.g., a whole Chinese phrase becomes a single [UNK]), silently breaking training/eval.
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "Expected a fast tokenizer (is_fast=False). "
            "KV-NER requires return_offsets_mapping. Please provide a tokenizer with fast backend."
        )
    try:
        _probe = "肿瘤标志物"
        _pieces = tokenizer.tokenize(_probe)
        if len(_pieces) == 1 and _pieces[0] == tokenizer.unk_token:
            raise RuntimeError(
                "Fast tokenizer appears misconfigured: a Chinese probe string tokenizes to a single [UNK]. "
                "This often happens when vocab.txt was modified but tokenizer.json wasn't regenerated. "
                "Run: python DAPT/repair_fast_tokenizer.py --tokenizer_dir <TOKENIZER_DIR>"
            )
        _enc = tokenizer(_probe, add_special_tokens=False, return_offsets_mapping=True)
        if not _enc.get("offset_mapping"):
            raise RuntimeError(
                "Fast tokenizer did not return offset_mapping. "
                "Please ensure you're using a fast tokenizer and that tokenizer.json is valid."
            )
    except Exception as e:
        raise RuntimeError(
            f"Fast tokenizer sanity-check failed: {e}. "
            "If you edited vocab.txt, regenerate tokenizer.json with DAPT/repair_fast_tokenizer.py"
        )

    # 加载数据
    data_path = Path(train_block.get("data_path"))
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    # 根据文件类型选择加载方式
    # 优先尝试使用 load_jsonl_with_noise，因为它支持 JSON Array 和 JSONL 两种格式，且容错性更好
    if str(data_path).endswith(".jsonl") or str(data_path).endswith(".json"):
        try:
            train_samples = load_jsonl_with_noise(data_path, label_map)
        except Exception as e:
            logger.warning(f"load_jsonl_with_noise faied for {data_path}, falling back to load_labelstudio_export: {e}")
            train_samples = load_labelstudio_export(data_path, label_map, include_unlabeled=False)
    else:
        train_samples = load_labelstudio_export(data_path, label_map, include_unlabeled=False)
    
    logger.info(f"Training set: {data_path} ({len(train_samples)} samples)")
    
    def _load_samples(path_value: str | Path) -> List[Sample]:
        p = Path(str(path_value))
        if not p.exists():
            raise FileNotFoundError(f"Data path not found: {p}")
        if str(p).endswith((".jsonl", ".json")):
            try:
                return load_jsonl_with_noise(p, label_map)
            except Exception as e:
                logger.warning(
                    f"load_jsonl_with_noise failed for {p}, falling back to load_labelstudio_export: {e}"
                )
                return load_labelstudio_export(p, label_map, include_unlabeled=False)
        return load_labelstudio_export(p, label_map, include_unlabeled=False)

    # 加载验证集 / 测试集
    val_path = train_block.get("val_data_path")
    test_path = train_block.get("test_data_path")
    seed = int(train_block.get("seed", 42))

    if test_path:
        test_pool = _load_samples(test_path)
        logger.info(f"Test set: {test_path} ({len(test_pool)} samples)")
        test_pool_labeled = [s for s in test_pool if s.has_labels]
        if not test_pool_labeled:
            raise ValueError(
                f"Loaded 0 labeled samples from test_data_path={test_path}. "
                "Check that each item has text (ocr_text/text or data.ocr_text/data.text) and labels "
                "(Label Studio annotations/transferred_annotations/key_value_pairs)."
            )
        test_samples = test_pool_labeled

        if val_path:
            val_pool = _load_samples(val_path)
            logger.info(f"Validation set: {val_path} ({len(val_pool)} samples)")
            val_pool_labeled = [s for s in val_pool if s.has_labels]
            if not val_pool_labeled:
                raise ValueError(
                    f"Loaded 0 labeled samples from val_data_path={val_path}. "
                    "Check that each item has text (ocr_text/text or data.ocr_text/data.text) and labels "
                    "(Label Studio annotations/transferred_annotations/key_value_pairs)."
                )
            val_samples = val_pool_labeled
        else:
            # 没有单独 val_data_path：从训练集中切出一部分做验证（保持 test 独立）
            val_ratio = float(train_block.get("val_ratio", 0.1))
            if val_ratio <= 0 or len(train_samples) < 2:
                val_samples = []
                logger.warning(
                    "val_data_path not configured and val_ratio<=0 or train too small; disabling validation"
                )
            else:
                train_samples, val_samples = train_test_split(
                    train_samples,
                    test_size=val_ratio,
                    random_state=seed,
                    shuffle=True,
                )
                logger.info(
                    "Split training set for validation: train=%d, val=%d (val_ratio=%.3f)",
                    len(train_samples),
                    len(val_samples),
                    val_ratio,
                )
    elif val_path:
        val_pool = _load_samples(val_path)
        logger.info(f"Validation set pool: {val_path} ({len(val_pool)} samples)")

        # 从验证集池划分验证集和测试集（旧行为，向后兼容）
        test_split_ratio = float(train_block.get("test_split_ratio", 0.5))
        val_pool_labeled = [s for s in val_pool if s.has_labels]

        if not val_pool_labeled:
            raise ValueError(
                f"Loaded 0 labeled samples from val_data_path={val_path}. "
                "If this file stores text under `data`, ensure it has data.ocr_text or data.text."
            )

        if not (0.0 < test_split_ratio < 1.0):
            raise ValueError(f"test_split_ratio must be in (0,1), got {test_split_ratio}")

        if len(val_pool_labeled) < 2:
            val_samples = val_pool_labeled
            test_samples = []
            logger.warning(
                "Validation pool too small to split (n=%d); using all as val and empty test",
                len(val_pool_labeled),
            )
        else:
            val_samples, test_samples = train_test_split(
                val_pool_labeled,
                test_size=test_split_ratio,
                random_state=seed,
                shuffle=True,
            )
            logger.info(
                "Split validation pool: val=%d (%.1f%%), test=%d (%.1f%%)",
                len(val_samples),
                (1 - test_split_ratio) * 100,
                len(test_samples),
                test_split_ratio * 100,
            )
    else:
        logger.warning("val_data_path not configured; will split from training data (not recommended)")
        train_ratio = float(train_block.get("train_ratio", 0.8))
        val_ratio = float(train_block.get("val_ratio", 0.1))
        train_samples, val_samples, test_samples = split_samples(
            train_samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        logger.info(
            "Dataset split: train=%d, val=%d, test=%d",
            len(train_samples),
            len(val_samples),
            len(test_samples),
        )

    # 初始化 noise 处理器（如果提供了 noise_bins）
    noise_mode = str(train_block.get("noise_mode", "bucket") or "bucket").lower()
    noise_processor: Optional[NoiseFeatureProcessor] = None
    use_noise = False
    if args.noise_bins:
        try:
            noise_processor = NoiseFeatureProcessor.load(args.noise_bins)
            use_noise = True
            logger.info(f"Loaded noise feature processor from {args.noise_bins}")
        except Exception as e:
            logger.warning(f"Failed to load noise bins: {e}; training without noise support")
            use_noise = False
    if uses_continuous_noise(noise_mode):
        use_noise = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Creating dataloaders...")
    train_loader, eval_loader, test_loader = _prepare_dataloaders_with_noise(
        cfg,
        tokenizer,
        label2id,
        train_samples,
        val_samples,
        test_samples,
        noise_processor=noise_processor,
        noise_mode=noise_mode,
    )
    logger.info(f"DataLoaders created: train_loader has {len(train_loader)} batches")

    # 初始化模型（从原模型或新模型）
    model_name = config_io.model_name_from(cfg)
    if args.pretrained_model:
        model_name = args.pretrained_model
        logger.info(f"Loading pretrained model from {model_name}")
    
    model = BertCrfTokenClassifier(
        model_name_or_path=model_name,
        label2id=label2id,
        id2label=id2label,
        dropout=float(train_block.get("dropout", 0.1)),
        freeze_encoder=bool(train_block.get("freeze_encoder", False)),
        unfreeze_last_n_layers=train_block.get("unfreeze_last_n_layers"),
        use_noise=use_noise,
        noise_embed_dim=int(train_block.get("noise_embed_dim", 16)),
        noise_mode=noise_mode,
        noise_mlp_hidden_dim=train_block.get("noise_mlp_hidden_dim"),
        noise_bin_edges=getattr(noise_processor, "bins", {}) if noise_processor is not None else None,
        use_bilstm=bool(train_block.get("use_bilstm", False)),
        lstm_hidden_size=train_block.get("lstm_hidden_size"),
        lstm_num_layers=int(train_block.get("lstm_num_layers", 1)),
        lstm_dropout=float(train_block.get("lstm_dropout", 0.0)),
        boundary_loss_weight=float(train_block.get("boundary_loss_weight", 0.0)),
        boundary_positive_weight=float(train_block.get("boundary_positive_weight", 1.0)),
        include_hospital_boundary=bool(train_block.get("include_hospital_boundary", True)),
        token_ce_loss_weight=float(train_block.get("token_ce_loss_weight", 0.0)),
        token_ce_label_smoothing=float(train_block.get("token_ce_label_smoothing", 0.0)),
        boundary_ce_label_smoothing=float(train_block.get("boundary_ce_label_smoothing", 0.0)),
        token_ce_value_class_weight=float(train_block.get("token_ce_value_class_weight", 3.0)),
        token_ce_key_class_weight=float(train_block.get("token_ce_key_class_weight", 1.0)),
        end_boundary_loss_weight=float(train_block.get("end_boundary_loss_weight", 0.0)),
        end_boundary_positive_weight=float(train_block.get("end_boundary_positive_weight", 1.0)),
    ).to(device)

    # 优化器和调度器
    lr = float(train_block.get("learning_rate", 3e-5))
    weight_decay = float(train_block.get("weight_decay", 0.01))
    enc_lr = float(train_block.get("encoder_learning_rate", lr))
    head_lr = float(train_block.get("head_learning_rate", enc_lr * 5.0))

    if ("encoder_learning_rate" in train_block) or ("head_learning_rate" in train_block):
        bert_params = []
        head_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("bert."):
                bert_params.append(p)
            else:
                head_params.append(p)
        
        if bert_params and head_params:
            optimizer = AdamW(
                [
                    {"params": bert_params, "lr": enc_lr},
                    {"params": head_params, "lr": head_lr},
                ],
                weight_decay=weight_decay,
            )
        elif bert_params:
            optimizer = AdamW(bert_params, lr=enc_lr, weight_decay=weight_decay)
        else:
            optimizer = AdamW(head_params, lr=head_lr, weight_decay=weight_decay)
        
        logger.info("Optimizer with discriminative LRs: encoder_lr=%.2e, head_lr=%.2e", enc_lr, head_lr)
    else:
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    num_epochs = int(train_block.get("num_train_epochs", 3))
    grad_accum = int(train_block.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(train_block.get("max_grad_norm", 1.0))
    total_steps = int(math.ceil((num_epochs * len(train_loader)) / max(1, grad_accum)))
    warmup_ratio = float(train_block.get("warmup_ratio", 0.1))
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # 输出目录（不破坏原模型）
    output_dir = Path(train_block.get("output_dir", "runs/kv_ner_finetuned"))
    best_dir = output_dir / "best"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Original model preserved at: {model_name}")
    logger.info(f"New best model will be saved to: {best_dir}")

    best_f1 = -1.0
    global_step = 0
    history: List[Dict[str, float]] = []

    logger.info("Starting training loop...")
    epoch_pbar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        logger.info(f"Starting epoch {epoch}/{num_epochs}")
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        steps_in_epoch = 0
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        
        for step, batch in enumerate(batch_pbar, start=1):
            steps_in_epoch = step
            # Debug on first batch
            if step == 1 and epoch == 1:
                logger.info("First batch received from DataLoader")
                logger.info(f"Batch type: {type(batch)}")
                if isinstance(batch, dict):
                    logger.info(f"Batch keys: {list(batch.keys())}")
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                else:
                    logger.info(f"Batch attributes: {[attr for attr in dir(batch) if not attr.startswith('_')]}")
                    for attr in ['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'noise_ids']:
                        if hasattr(batch, attr):
                            v = getattr(batch, attr)
                            if isinstance(v, torch.Tensor):
                                logger.info(f"  {attr}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
            
            input_ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.input_ids.to(device)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.long) if isinstance(batch, dict) else batch.attention_mask.to(device, dtype=torch.long)
            token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids)).to(device, dtype=torch.long) if isinstance(batch, dict) else batch.token_type_ids.to(device, dtype=torch.long)
            labels = batch["labels"].to(device) if isinstance(batch, dict) else batch.labels.to(device)

            # Debug: log batch shapes and dtypes on first batch
            if step == 1 and epoch == 1:
                logger.info(
                    "Batch 1 debug: input_ids shape=%s dtype=%s, attention_mask shape=%s dtype=%s min=%s max=%s, "
                    "token_type_ids shape=%s dtype=%s min=%s max=%s, labels shape=%s dtype=%s min=%s max=%s",
                    input_ids.shape, input_ids.dtype,
                    attention_mask.shape, attention_mask.dtype, attention_mask.min().item(), attention_mask.max().item(),
                    token_type_ids.shape, token_type_ids.dtype, token_type_ids.min().item(), token_type_ids.max().item(),
                    labels.shape, labels.dtype, labels.min().item(), labels.max().item(),
                )
                # Check input_ids validity
                with torch.no_grad():
                    vocab_size = model.config.vocab_size
                    input_id_min = int(input_ids.min().item())
                    input_id_max = int(input_ids.max().item())
                    logger.info(f"Input IDs range: min={input_id_min}, max={input_id_max}, vocab_size={vocab_size}")
                    if input_id_max >= vocab_size:
                        raise ValueError(f"input_ids max {input_id_max} >= vocab_size {vocab_size}")
                    if input_id_min < 0:
                        raise ValueError(f"input_ids min {input_id_min} < 0")
                if hasattr(batch, "noise_ids") and batch.noise_ids is not None:
                    logger.info("noise_ids shape=%s dtype=%s", batch.noise_ids.shape, batch.noise_ids.dtype)
                if hasattr(batch, "noise_values") and batch.noise_values is not None:
                    logger.info("noise_values shape=%s dtype=%s", batch.noise_values.shape, batch.noise_values.dtype)

            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": labels,
            }
            # 如果有noise_ids，传入模型
            if uses_bucket_noise(noise_mode) and hasattr(batch, "noise_ids") and batch.noise_ids is not None:
                noise_ids = batch.noise_ids.to(device)
                if model.use_noise and model.noise_embeddings:
                    with torch.no_grad():
                        for fi, emb in enumerate(model.noise_embeddings):
                            max_id = int(torch.max(noise_ids[:, :, fi]).item())
                            if max_id >= emb.num_embeddings:
                                raise ValueError(
                                    f"noise_ids feature {fi} has max id {max_id} >= num_embeddings {emb.num_embeddings}"
                                )
                kwargs["noise_ids"] = noise_ids
            elif uses_continuous_noise(noise_mode) and hasattr(batch, "noise_values") and batch.noise_values is not None:
                kwargs["noise_values"] = batch.noise_values.to(device, dtype=torch.float32)

            # labels range check for CRF
            with torch.no_grad():
                lbl_min = int(labels.min().item())
                lbl_max = int(labels.max().item())
                if lbl_min < 0 or lbl_max >= model.num_labels:
                    raise ValueError(
                        f"Label id out of range: min={lbl_min}, max={lbl_max}, num_labels={model.num_labels}"
                    )

            loss = model(**kwargs)
            loss = loss / grad_accum
            loss.backward()
            running_loss += loss.item()

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            
            current_lr = scheduler.get_last_lr()[0]
            avg_batch_loss = running_loss * grad_accum / max(1, step)
            batch_pbar.set_postfix({"loss": f"{avg_batch_loss:.4f}", "lr": f"{current_lr:.2e}"})

        # Handle remainder micro-batches when steps_in_epoch is not divisible by grad_accum
        # (otherwise the last few batches' gradients would never be applied)
        if steps_in_epoch > 0 and (steps_in_epoch % grad_accum) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        avg_loss = running_loss * grad_accum / max(1, len(train_loader))
        logger.info("Epoch %d/%d - train loss: %.4f", epoch, num_epochs, avg_loss)

        metrics = _evaluate_model(model, eval_loader, device, id2label, use_noise=use_noise, noise_mode=noise_mode)
        overall_f1 = metrics["overall"]["f1"]
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_f1": overall_f1})
        logger.info(
            "Validation F1: %.4f (KEY=%.4f, VALUE=%.4f, HOSPITAL=%.4f)",
            overall_f1,
            metrics.get("KEY", {}).get("f1", 0.0),
            metrics.get("VALUE", {}).get("f1", 0.0),
            metrics.get("HOSPITAL", {}).get("f1", 0.0),
        )

        if overall_f1 > best_f1:
            best_f1 = overall_f1
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(best_dir / "tokenizer")
            logger.info("Saved new best model to %s (F1=%.4f)", best_dir, best_f1)
        
        epoch_pbar.set_postfix({"best_f1": f"{best_f1:.4f}", "val_f1": f"{overall_f1:.4f}"})

    # 最终评估
    logger.info("Training finished. Evaluating on test set...")
    metrics = _evaluate_model(model, test_loader, device, id2label, use_noise=use_noise, noise_mode=noise_mode)
    logger.info("Test F1: %.4f", metrics["overall"]["f1"])

    # 保存总结
    summary = {
        "best_val_f1": best_f1,
        "test_f1": metrics["overall"]["f1"],
        "history": history,
        "test_metrics": metrics,
        "model_dir": str(best_dir),
        "original_model": model_name,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "use_noise": use_noise,
        "noise_mode": noise_mode,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Results saved to %s/training_summary.json", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KV-NER with DAPT noise support")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to kv_ner_config.json",
    )
    parser.add_argument(
        "--noise_bins",
        type=str,
        default=None,
        help="Path to noise_bins.json for noise feature processing",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Path to pretrained model (e.g., /data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu_noise_v2/final_model)",
    )

    # Optional overrides to avoid editing JSON configs (useful for ablations / debugging)
    parser.add_argument("--learning_rate", type=float, default=None, help="Override train.learning_rate")
    parser.add_argument("--num_train_epochs", type=int, default=None, help="Override train.num_train_epochs")
    parser.add_argument("--no_bilstm", action="store_true", help="Force disable BiLSTM head (train.use_bilstm=false)")
    parser.add_argument("--token_ce_loss_weight", type=float, default=None, help="Override train.token_ce_loss_weight")
    parser.add_argument("--token_ce_value_class_weight", type=float, default=None, help="Override train.token_ce_value_class_weight")
    parser.add_argument("--token_ce_key_class_weight", type=float, default=None, help="Override train.token_ce_key_class_weight")
    parser.add_argument("--noise_mode", type=str, default=None, choices=["bucket", "linear", "mlp"], help="Override train.noise_mode")
    parser.add_argument("--noise_mlp_hidden_dim", type=int, default=None, help="Override train.noise_mlp_hidden_dim")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.config:
        args.config = config_io.default_config_path()
    train(args)
