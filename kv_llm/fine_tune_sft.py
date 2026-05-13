#!/usr/bin/env python3
"""Minimal decoder-only SFT/LoRA entrypoint for MedStruct-S-style JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from kv_llm.constants import DEFAULT_MODEL_NAME
from kv_llm.data import read_json_or_jsonl


def pairs_to_json_text(pairs: list[dict[str, Any]]) -> str:
    clean = []
    for p in pairs:
        key = str(p.get("key", "")).strip()
        value = str(p.get("value", "")).strip()
        if key and value:
            clean.append({"key": key, "value": value})
    return json.dumps({"pairs": clean}, ensure_ascii=False)


class MedStructSftDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, max_length: int = 2048, max_samples: int | None = None) -> None:
        rows = read_json_or_jsonl(path)
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        self.examples = []
        for item in rows:
            text = item.get("ocr_text") or item.get("text") or item.get("report_text")
            pairs = item.get("pairs") or []
            if not isinstance(text, str) or not isinstance(pairs, list):
                continue
            prompt = (
                "Extract all medical key-value pairs from the OCR clinical report. "
                "Return compact JSON with a top-level 'pairs' list.\n\n"
                f"Report:\n{text}\n\nJSON:\n"
            )
            answer = pairs_to_json_text(pairs)
            full = prompt + answer
            enc = tokenizer(full, truncation=True, max_length=max_length, padding=False)
            prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            labels = list(enc["input_ids"])
            labels[: min(prompt_len, len(labels))] = [-100] * min(prompt_len, len(labels))
            enc["labels"] = labels
            self.examples.append(enc)
        if not self.examples:
            raise ValueError(f"No SFT examples found in {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


class SftCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        input_ids = []
        attention_mask = []
        labels = []
        pad = self.tokenizer.pad_token_id
        for item in features:
            n = len(item["input_ids"])
            pad_n = max_len - n
            input_ids.append(item["input_ids"] + [pad] * pad_n)
            attention_mask.append(item["attention_mask"] + [0] * pad_n)
            labels.append(item["labels"] + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA SFT for decoder-only MedStruct-S extraction")
    p.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    p.add_argument("--train_data", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def maybe_apply_lora(model, args):
    if not args.use_lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:
        raise RuntimeError("Install peft to use --use_lora") from exc
    cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, cfg)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.bf16 else None
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=dtype)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = maybe_apply_lora(model, args)
    ds = MedStructSftDataset(args.train_data, tokenizer, max_length=args.max_length, max_samples=args.max_samples)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            bf16=bool(args.bf16),
            remove_unused_columns=False,
            report_to="none",
            save_total_limit=2,
        ),
        train_dataset=ds,
        data_collator=SftCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
