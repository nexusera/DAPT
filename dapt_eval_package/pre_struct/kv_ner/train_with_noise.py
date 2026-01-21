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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

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
        _normalize_label,
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
        _normalize_label,
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

        noise_values = data.get("noise_values") or task.get("noise_values")
        return Sample(
            task_id=str(task.get("id")),
            text=text,
            title=str(data.get("category") or data.get("title") or ""),
            entities=sorted(entities, key=lambda e: (e.start, e.end)),
            relations=relations,
            noise_values=noise_values,
        )
    
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_no}: JSON parse error - {e}")
                continue

            # Label Studio JSONL（单行一个任务）
            if isinstance(obj, dict) and "annotations" in obj and "data" in obj:
                sample = _parse_labelstudio_task(obj)
                if sample:
                    samples.append(sample)
                continue
            
            task_id = str(obj.get("id", line_no))
            text = str(obj.get("text", "")).strip()
            title = str(obj.get("title", ""))
            
            if not text:
                continue

            noise_values = obj.get("noise_values") or obj.get("data", {}).get("noise_values")
            
            # 解析 key_value_pairs 为 Entity
            entities: List[Entity] = []
            kv_list = obj.get("key_value_pairs", [])
            if isinstance(kv_list, list):
                for kv in kv_list:
                    if not isinstance(kv, dict):
                        continue
                    
                    key_info = kv.get("key", {})
                    key_start = int(key_info.get("start", -1)) if isinstance(key_info, dict) else -1
                    key_end = int(key_info.get("end", -1)) if isinstance(key_info, dict) else -1
                    if 0 <= key_start < key_end <= len(text):
                        entities.append(Entity(
                            start=key_start,
                            end=key_end,
                            label="KEY",
                            text=key_info.get("text") if isinstance(key_info, dict) else None,
                        ))
                    
                    val_info = kv.get("value", {})
                    val_start = int(val_info.get("start", -1)) if isinstance(val_info, dict) else -1
                    val_end = int(val_info.get("end", -1)) if isinstance(val_info, dict) else -1
                    if 0 <= val_start < val_end <= len(text):
                        entities.append(Entity(
                            start=val_start,
                            end=val_end,
                            label="VALUE",
                            text=val_info.get("text") if isinstance(val_info, dict) else None,
                        ))
            
            entities.sort(key=lambda e: (e.start, e.end))
            
            sample = Sample(
                task_id=task_id,
                text=text,
                title=title,
                entities=entities,
                relations=[],
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
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    准备数据加载器，支持 noise_ids
    """
    max_len = config_io.max_seq_length(cfg)
    label_all_tokens = config_io.label_all_tokens(cfg)
    chunk_size = int(cfg.get("chunk_size", max_len))
    if chunk_size > max_len:
        chunk_size = max_len
    chunk_overlap = int(cfg.get("chunk_overlap", 0))

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
    )
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
    )
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
    )
    
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
            attention_mask = batch["attention_mask"].to(device) if isinstance(batch, dict) else batch.attention_mask.to(device)
            token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids)).to(device) if isinstance(batch, dict) else batch.token_type_ids.to(device)
            labels = batch["labels"].to(device) if isinstance(batch, dict) else batch.labels.to(device)

            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            
            # 如果有noise_ids且模型支持，传入
            if use_noise and "noise_ids" in (batch if isinstance(batch, dict) else batch.__dict__):
                noise_ids = batch["noise_ids"].to(device) if isinstance(batch, dict) else batch.noise_ids.to(device)
                kwargs["noise_ids"] = noise_ids

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

    label_map = config_io.label_map_from(cfg)
    label_list = build_bio_label_list(label_map)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    set_seed(int(train_block.get("seed", 42)))

    tokenizer_name = config_io.tokenizer_name_from(cfg)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # 加载数据
    data_path = Path(train_block.get("data_path"))
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    # 根据文件类型选择加载方式
    if str(data_path).endswith(".jsonl"):
        train_samples = load_jsonl_with_noise(data_path, label_map)
    else:
        train_samples = load_labelstudio_export(data_path, label_map, include_unlabeled=False)
    
    logger.info(f"Training set: {data_path} ({len(train_samples)} samples)")
    
    # 加载验证集
    val_path = train_block.get("val_data_path")
    if val_path:
        if str(val_path).endswith(".jsonl"):
            val_pool = load_jsonl_with_noise(val_path, label_map)
        else:
            val_pool = load_labelstudio_export(Path(val_path), label_map, include_unlabeled=False)
        
        logger.info(f"Validation set pool: {val_path} ({len(val_pool)} samples)")
        
        # 从验证集池划分验证集和测试集
        test_split_ratio = float(train_block.get("test_split_ratio", 0.5))
        val_pool_labeled = [s for s in val_pool if s.has_labels]
        
        val_samples, test_samples = train_test_split(
            val_pool_labeled,
            test_size=test_split_ratio,
            random_state=int(train_block.get("seed", 42)),
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
            seed=int(train_block.get("seed", 42)),
        )
        logger.info(
            "Dataset split: train=%d, val=%d, test=%d",
            len(train_samples),
            len(val_samples),
            len(test_samples),
        )

    # 初始化 noise 处理器（如果提供了 noise_bins）
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_loader, eval_loader, test_loader = _prepare_dataloaders_with_noise(
        cfg,
        tokenizer,
        label2id,
        train_samples,
        val_samples,
        test_samples,
        noise_processor=noise_processor,
    )

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
        use_noise=bool(noise_processor is not None),
        noise_embed_dim=int(train_block.get("noise_embed_dim", 16)),
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
    total_steps = num_epochs * len(train_loader) // max(1, grad_accum)
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

    epoch_pbar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        
        for step, batch in enumerate(batch_pbar, start=1):
            input_ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.input_ids.to(device)
            attention_mask = batch["attention_mask"].to(device) if isinstance(batch, dict) else batch.attention_mask.to(device)
            token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids)).to(device) if isinstance(batch, dict) else batch.token_type_ids.to(device)
            labels = batch["labels"].to(device) if isinstance(batch, dict) else batch.labels.to(device)

            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": labels,
            }
            # 如果有noise_ids，传入模型
            if hasattr(batch, "noise_ids") and batch.noise_ids is not None:
                kwargs["noise_ids"] = batch.noise_ids.to(device)

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

        avg_loss = running_loss * grad_accum / max(1, len(train_loader))
        logger.info("Epoch %d/%d - train loss: %.4f", epoch, num_epochs, avg_loss)

        metrics = _evaluate_model(model, eval_loader, device, id2label, use_noise=use_noise)
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
    metrics = _evaluate_model(model, test_loader, device, id2label, use_noise=use_noise)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.config:
        args.config = config_io.default_config_path()
    train(args)
