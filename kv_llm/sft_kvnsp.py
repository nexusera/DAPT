#!/usr/bin/env python3
"""SC3-B — Instruct-style KV-NSP SFT injection (plan §10 SC3 setting B).

Same KV-NSP signal as the CPT route (kv_llm/kv_nsp.py), but injected via
prompted SFT on the Instruct checkpoint instead of as a CPT classification
head. The comparison vs SC3-A (= our 0.6B full CPT) answers RQ5:
'Should KV-NSP prior live in CPT or in SFT?'

Each KV-NSP pair (key, value, label) becomes one SFT example:

    [system] You are a clinical KV verifier.
    [user]   Given key '{KEY}' and value '{VALUE}',
             are they a matched clinical key-value pair? Answer Yes or No.
    [assistant] {Yes|No}

Loss is computed only on the answer token(s). Trains a LoRA adapter on
Qwen3-0.6B (the "Instruct" variant per Qwen3 naming: no -Base suffix).

Usage:
  python -m kv_llm.sft_kvnsp \
    --model_name_or_path /data/ocean/model/Qwen/Qwen3-0.6B \
    --nsp_data /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
    --output_dir /data/ocean/code/dapt/model/sc3b_kvnsp_sft \
    --use_lora --lora_rank 8 --bf16
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from kv_nsp.negative_sampling import build_negative_sampling_config, sample_kv_nsp_text_pair
from kv_llm.kv_nsp import extract_label_studio_pairs, extract_direct_pairs
from kv_llm.data import find_json_files, read_json_or_jsonl


SYSTEM_PROMPT = (
    "You are a clinical key-value verifier. Decide whether a given key and "
    "value form a valid medical key-value pair extracted from an OCR'd "
    "clinical report."
)

USER_TEMPLATE = (
    "Given key '{key}' and value '{value}', are they a matched clinical "
    "key-value pair? Answer 'Yes' or 'No' only."
)


class KvNspSftDataset(Dataset):
    """Materialize a list of (key, value, label) examples via the same
    negative sampler as the CPT KV-NSP loop, then format each with the
    Instruct chat template; loss only on the answer."""

    def __init__(self, data_path: str | Path, tokenizer, *, max_length: int = 256,
                 negative_prob: float = 0.5,
                 reverse_negative_ratio: float = 1.0,
                 random_negative_ratio: float = 1.0,
                 max_easy_retries: int = 10,
                 seed: int = 42,
                 max_samples: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rng = random.Random(seed)
        self.sampling_config = build_negative_sampling_config(
            negative_prob=negative_prob,
            reverse_negative_ratio=reverse_negative_ratio,
            random_negative_ratio=random_negative_ratio,
            max_easy_retries=max_easy_retries,
        )
        pairs: list[tuple[str, str]] = []
        for p in find_json_files(data_path):
            for rec in read_json_or_jsonl(p):
                direct = extract_direct_pairs(rec)
                if direct:
                    pairs.extend(direct)
                else:
                    pairs.extend(extract_label_studio_pairs(rec))
        pairs = [(k, v) for k, v in pairs if k and v]
        if max_samples is not None:
            pairs = pairs[: int(max_samples)]
        if not pairs:
            raise ValueError(f"no kv pairs found in {data_path}")
        self.pairs = pairs
        self.value_pool = [v for _, v in pairs]
        self.valid = set(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        k, v = self.pairs[idx]
        key, value, label, _ = sample_kv_nsp_text_pair(
            key_text=k, value_text=v,
            value_pool=self.value_pool, valid_pairs_set=self.valid,
            config=self.sampling_config, pair_pool=self.pairs, rng=self.rng,
        )
        answer = "Yes" if int(label) == 1 else "No"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(key=key, value=value)},
            {"role": "assistant", "content": answer},
        ]
        # If tokenizer has chat_template (Qwen3 small does), use it; else fall back
        try:
            full = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_only = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        except Exception:
            full = (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"<|user|>\n{USER_TEMPLATE.format(key=key, value=value)}\n"
                f"<|assistant|>\n{answer}"
            )
            prompt_only = full[: -len(answer)]
        enc = self.tokenizer(full, truncation=True, max_length=self.max_length, padding=False)
        prompt_len = len(self.tokenizer(prompt_only, add_special_tokens=False)["input_ids"])
        labels = list(enc["input_ids"])
        labels[: min(prompt_len, len(labels))] = [-100] * min(prompt_len, len(labels))
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


def collate(batch, pad_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    def pad(x, p): return x + [p] * (max_len - len(x))
    return {
        "input_ids": torch.tensor([pad(b["input_ids"], pad_id) for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch], dtype=torch.long),
        "labels": torch.tensor([pad(b["labels"], -100) for b in batch], dtype=torch.long),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="/data/ocean/model/Qwen/Qwen3-0.6B",
                   help="Qwen3-0.6B (Instruct in Qwen3 naming, no -Base suffix)")
    p.add_argument("--nsp_data", default="/data/ocean/DAPT/data/pseudo_kv_labels_filtered.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--save_steps", type=int, default=2000)
    args = p.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if args.bf16 else None
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=dtype)
    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank * 2,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                         lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    ds = KvNspSftDataset(args.nsp_data, tok, max_length=args.max_length,
                         seed=args.seed, max_samples=args.max_samples)
    print(f"[SC3-B] kv-NSP SFT dataset size = {len(ds)}")

    ta = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=bool(args.bf16),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        save_safetensors=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=ta, train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()
    if trainer.is_world_process_zero():
        final = Path(args.output_dir) / "final"
        model.save_pretrained(final, safe_serialization=False)
        tok.save_pretrained(final)
        print(f"[OK] SC3-B saved to {final}")


if __name__ == "__main__":
    main()
