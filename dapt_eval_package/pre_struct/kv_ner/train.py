#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
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
    )
    from pre_struct.kv_ner.dataset import TokenClassificationDataset, collate_batch
    from pre_struct.kv_ner.metrics import compute_ner_metrics
    from pre_struct.kv_ner.modeling import BertCrfTokenClassifier
else:
    from . import config_io
    from .data_utils import (
        build_bio_label_list,
        load_labelstudio_export,
        split_samples,
    )
    from .dataset import TokenClassificationDataset, collate_batch
    from .metrics import compute_ner_metrics
    from .modeling import BertCrfTokenClassifier

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_dataloaders(
    cfg: Dict[str, Any],
    tokenizer,
    label2id: Dict[str, int],
    train_samples,
    val_samples,
    test_samples,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
    )
    train_block = config_io.ensure_block(cfg, "train")
    batch_size = int(train_block.get("train_batch_size", 16))
    eval_batch_size = int(train_block.get("eval_batch_size", batch_size))
    num_workers = int(train_block.get("num_workers", 0))
    pin_memory = bool(train_block.get("pin_memory", False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )
    eval_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )
    return train_loader, eval_loader, test_loader


def _evaluate_model(
    model: BertCrfTokenClassifier,
    dataloader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
) -> Dict[str, Dict[str, float]]:
    model.eval()
    predictions: List[List[int]] = []
    references: List[List[int]] = []
    masks: List[List[bool]] = []
    offsets: List[List[Tuple[int, int]]] = []

    o_id = next((idx for idx, name in id2label.items() if name == "O"), 0)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            token_type_ids = batch.token_type_ids.to(device)
            labels = batch.labels.to(device)

            decoded = model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            seq_len = labels.size(1)
            for i, seq in enumerate(decoded):
                seq_list = list(seq)
                if len(seq_list) < seq_len:
                    seq_list = seq_list + [o_id] * (seq_len - len(seq_list))
                elif len(seq_list) > seq_len:
                    seq_list = seq_list[:seq_len]
                predictions.append(seq_list)
            references.extend(labels.cpu().tolist())
            masks.extend(attention_mask.cpu().tolist())
            offsets.extend(batch.offset_mapping.cpu().tolist())

    return compute_ner_metrics(predictions, references, masks, id2label, offsets=offsets)


def train(args: argparse.Namespace) -> None:
    cfg = config_io.load_config(args.config)
    train_block = config_io.ensure_block(cfg, "train")

    label_map = config_io.label_map_from(cfg)
    label_list = build_bio_label_list(label_map)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    data_path = Path(train_block["data_path"])
    set_seed(int(train_block.get("seed", 42)))

    tokenizer_name = config_io.tokenizer_name_from(cfg)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Fast tokenizer sanity-check: training pipeline relies on offset_mapping.
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "Expected a fast tokenizer (is_fast=False). "
            "KV-NER requires return_offsets_mapping; please provide a fast tokenizer backend."
        )
    _probe = "肿瘤标志物"
    _pieces = tokenizer.tokenize(_probe)
    if len(_pieces) == 1 and _pieces[0] == tokenizer.unk_token:
        raise RuntimeError(
            "Fast tokenizer appears misconfigured (probe tokenizes to a single [UNK]). "
            "If vocab.txt was edited, regenerate tokenizer.json: "
            "python DAPT/repair_fast_tokenizer.py --tokenizer_dir <TOKENIZER_DIR>"
        )

    train_samples = load_labelstudio_export(data_path, label_map, include_unlabeled=False)
    logger.info(f"训练集: {data_path} ({len(train_samples)} 条)")
    
    val_path = train_block.get("val_data_path")
    
    if val_path:
        val_pool = load_labelstudio_export(Path(val_path), label_map, include_unlabeled=False)
        logger.info(f"验证集池: {val_path} ({len(val_pool)} 条)")
        
        test_split_ratio = float(train_block.get("test_split_ratio", 0.5))
        
        val_pool_labeled = [s for s in val_pool if s.has_labels]
        
        val_samples, test_samples = train_test_split(
            val_pool_labeled,
            test_size=test_split_ratio,
            random_state=int(train_block.get("seed", 42)),
            shuffle=True,
        )
        
        logger.info(
            "从验证集池动态划分: val=%d (%.1f%%), test=%d (%.1f%%)",
            len(val_samples),
            (1 - test_split_ratio) * 100,
            len(test_samples),
            test_split_ratio * 100,
        )
    else:
        logger.warning("未配置 val_data_path，将从训练数据自动划分（不推荐）")
        logger.warning("建议先运行 prepare_data.py 生成已划分的数据")
        
        samples = load_labelstudio_export(data_path, label_map, include_unlabeled=False)
        train_ratio_compat = float(train_block.get("train_ratio", 0.8))
        val_ratio_compat = float(train_block.get("val_ratio", 0.1))
        train_samples, val_samples, test_samples = split_samples(
            samples,
            train_ratio=train_ratio_compat,
            val_ratio=val_ratio_compat,
            seed=int(train_block.get("seed", 42)),
        )
        logger.info(
            "Dataset split: train=%d, val=%d, test=%d",
            len(train_samples),
            len(val_samples),
            len(test_samples),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, eval_loader, test_loader = _prepare_dataloaders(
        cfg,
        tokenizer,
        label2id,
        train_samples,
        val_samples,
        test_samples,
    )

    model = BertCrfTokenClassifier(
        model_name_or_path=config_io.model_name_from(cfg),
        label2id=label2id,
        id2label=id2label,
        dropout=float(train_block.get("dropout", 0.1)),
        freeze_encoder=bool(train_block.get("freeze_encoder", False)),
        unfreeze_last_n_layers=train_block.get("unfreeze_last_n_layers"),
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
        if not bert_params and head_params:
            optimizer = AdamW(head_params, lr=head_lr, weight_decay=weight_decay)
        elif bert_params and not head_params:
            optimizer = AdamW(bert_params, lr=enc_lr, weight_decay=weight_decay)
        else:
            optimizer = AdamW(
                [
                    {"params": bert_params, "lr": enc_lr},
                    {"params": head_params, "lr": head_lr},
                ],
                lr=enc_lr,
                weight_decay=weight_decay,
            )
        logger.info(
            "Optimizer param groups: encoder_lr=%.2e, head_lr=%.2e (weight_decay=%.3g)",
            enc_lr, head_lr, weight_decay,
        )
    else:
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
    num_epochs = int(train_block.get("num_train_epochs", 5))
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

    output_dir = Path(train_block.get("output_dir", "runs/kv_ner"))
    best_dir = output_dir / "best"
    output_dir.mkdir(parents=True, exist_ok=True)

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
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            token_type_ids = batch.token_type_ids.to(device)
            labels = batch.labels.to(device)

            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = loss / grad_accum
            loss.backward()
            running_loss += loss.item()

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else lr
            avg_batch_loss = running_loss * grad_accum / max(1, step)
            batch_pbar.set_postfix({
                'loss': f'{avg_batch_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })

        avg_loss = running_loss * grad_accum / max(1, len(train_loader))
        logger.info("Epoch %d/%d - train loss: %.4f", epoch, num_epochs, avg_loss)

        metrics = _evaluate_model(model, eval_loader, device, id2label)
        overall_f1 = metrics["overall"]["f1"]
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_f1": overall_f1})
        logger.info("Validation F1: %.4f (KEY=%.4f, VALUE=%.4f, HOSPITAL=%.4f)",
                    overall_f1,
                    metrics.get('KEY', {}).get('f1', 0.0),
                    metrics.get('VALUE', {}).get('f1', 0.0),
                    metrics.get('HOSPITAL', {}).get('f1', 0.0))

        if overall_f1 > best_f1:
            best_f1 = overall_f1
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(best_dir / "tokenizer")
            logger.info("Saved new best model to %s (F1=%.4f)", best_dir, best_f1)
        else:
            logger.info("F1 did not improve; best remains %.4f", best_f1)

    logger.info("Training finished. Best F1=%.4f", best_f1)
    metrics = _evaluate_model(model, test_loader, device, id2label)
    logger.info("Test F1: %.4f", metrics["overall"]["f1"])

    summary = {
        "best_f1": best_f1,
        "history": history,
        "test": metrics,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KV-NER model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to KV-NER config JSON. Defaults to pre_struct/kv_ner/kv_ner_config.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
