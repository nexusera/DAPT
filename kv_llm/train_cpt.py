#!/usr/bin/env python3
"""KV-LLM continued pretraining entrypoint for Qwen3-0.6B.

Examples:
    python -m kv_llm.train_cpt \
      --model_name_or_path Qwen/Qwen3-0.6B \
      --span_data /data/ocean/DAPT/workspace/train_chunked.txt \
      --entity_dict /data/ocean/DAPT/vocab_for_jieba.txt \
      --nsp_data /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
      --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
      --output_dir /data/ocean/DAPT/workspace/kv_llm_full \
      --schedule full
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

from kv_nsp.negative_sampling import format_negative_sampling_summary
from kv_llm.constants import DEFAULT_MODEL_NAME, KV_NSP_SEP_TOKEN, build_sentinel_tokens
from kv_llm.data import TextFileDataset
from kv_llm.kv_nsp import LlmKvnspCollator, LlmKvnspDataset
from kv_llm.modeling import KvLlmForCausalPreTraining
from kv_llm.span_corruption import SpanCorruptionCollator, load_entity_dictionary
from noise_feature_processor import NoiseFeatureProcessor
import paths_config as PC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KV-LLM continued pretraining for Qwen3-0.6B")
    p.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--span_data", default=PC.TRAIN_CHUNKED_PATH, help="TXT/JSON/JSONL text data for span corruption or plain CLM")
    p.add_argument("--entity_dict", default=None, help="Optional single entity dictionary path; defaults to BERT route dictionaries")
    p.add_argument("--keys_file", default=PC.KEYS_FILE, help="Business key dictionary, aligned with BERT route")
    p.add_argument("--vocab_for_jieba", default=PC.JIEBA_VOCAB_PATH, help="WordPiece mined vocabulary, aligned with BERT route")
    p.add_argument("--nsp_data", default=PC.NSP_DATA_PATH, help="Label Studio JSON/JSONL or directory for KV-NSP")
    p.add_argument("--noise_bins_json", default=PC.NOISE_BINS_JSON)
    p.add_argument("--schedule", choices=["full", "span", "nsp", "plain_clm"], default="full")
    p.add_argument("--noise_mode", choices=["bucket", "linear", "mlp", "concat_linear", "none"], default="bucket")
    p.add_argument("--noise_alpha", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--nsp_max_length", type=int, default=256)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--max_spans", type=int, default=24)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_rounds", type=int, default=3, help="Aligned with staged BERT route: span -> nsp alternation rounds")
    p.add_argument("--span_epochs_per_round", type=float, default=1.0)
    p.add_argument("--nsp_epochs_per_round", type=float, default=3.0)
    p.add_argument("--nsp_negative_prob", type=float, default=0.5)
    p.add_argument("--nsp_reverse_negative_ratio", type=float, default=1.0)
    p.add_argument("--nsp_random_negative_ratio", type=float, default=1.0)
    p.add_argument("--nsp_max_easy_retries", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def prepare_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens: dict[str, object] = {}
    additional_special_tokens: list[str] = []
    if args.schedule in {"full", "span"}:
        additional_special_tokens.extend(build_sentinel_tokens(args.max_spans + 2))
    if getattr(tokenizer, "sep_token", None) is None:
        special_tokens["sep_token"] = KV_NSP_SEP_TOKEN
    if additional_special_tokens:
        special_tokens["additional_special_tokens"] = additional_special_tokens
    if special_tokens:
        added = tokenizer.add_special_tokens(special_tokens)
        if added:
            print(f"[INFO] Added {added} special tokens.")
    return tokenizer


def build_model(args: argparse.Namespace, tokenizer) -> KvLlmForCausalPreTraining:
    dtype = torch.bfloat16 if args.bf16 else None
    model = KvLlmForCausalPreTraining.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        noise_mode=args.noise_mode,
        noise_alpha=args.noise_alpha,
    )
    model.base_model.resize_token_embeddings(len(tokenizer))
    model.base_model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.base_model.gradient_checkpointing_enable()
    return model


def training_args(args: argparse.Namespace, output_dir: Path, epochs: float) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=bool(args.bf16),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
    )


def load_noise_processor(path: str | None) -> NoiseFeatureProcessor | None:
    if path and os.path.exists(path):
        return NoiseFeatureProcessor.load(path)
    return None


def resolve_entity_dicts(args: argparse.Namespace) -> list[str]:
    if args.entity_dict:
        return [str(args.entity_dict)]
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        str(args.keys_file),
        str(args.vocab_for_jieba),
        str(repo_root / "biaozhu_keys_only_min5.txt"),
        str(repo_root / "vocab_for_jieba.txt"),
    ]
    seen: list[str] = []
    for p in candidates:
        if p and p not in seen and os.path.exists(p):
            seen.append(p)
    return seen


def run_span_phase(args, model, tokenizer, noise_processor, *, plain_clm: bool = False, round_idx: int | None = None) -> None:
    if not args.span_data:
        raise ValueError("--span_data is required for span/plain_clm training")
    ds = TextFileDataset(args.span_data, max_samples=args.max_samples)
    entity_dicts = resolve_entity_dicts(args)
    if not plain_clm:
        print(f"[KV-LLM] span dictionaries: {entity_dicts}")
    collator = SpanCorruptionCollator(
        tokenizer=tokenizer,
        entity_terms=load_entity_dictionary(entity_dicts),
        max_length=args.max_length,
        mask_prob=args.mask_prob,
        max_spans=args.max_spans,
        seed=args.seed,
        plain_clm=plain_clm,
        noise_mode=args.noise_mode,
        noise_processor=noise_processor,
    )
    phase_name = "plain_clm" if plain_clm else "span"
    out = Path(args.output_dir) / (f"round_{round_idx}_{phase_name}" if round_idx is not None else phase_name)
    trainer = Trainer(
        model=model,
        args=training_args(args, out, args.span_epochs_per_round),
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(out / "final_model")
    tokenizer.save_pretrained(out / "final_model")


def run_nsp_phase(args, model, tokenizer, noise_processor, *, round_idx: int | None = None) -> None:
    if not args.nsp_data:
        raise ValueError("--nsp_data is required for KV-NSP training")
    ds = LlmKvnspDataset(
        args.nsp_data,
        seed=args.seed,
        max_samples=args.max_samples,
        negative_prob=args.nsp_negative_prob,
        reverse_negative_ratio=args.nsp_reverse_negative_ratio,
        random_negative_ratio=args.nsp_random_negative_ratio,
        max_easy_retries=args.nsp_max_easy_retries,
    )
    print(f"[KV-LLM] NSP negative sampling: {format_negative_sampling_summary(ds.sampling_config)}")
    collator = LlmKvnspCollator(
        tokenizer=tokenizer,
        max_length=args.nsp_max_length,
        noise_mode=args.noise_mode,
        noise_processor=noise_processor,
    )
    out = Path(args.output_dir) / (f"round_{round_idx}_kv_nsp" if round_idx is not None else "kv_nsp")
    trainer = Trainer(
        model=model,
        args=training_args(args, out, args.nsp_epochs_per_round),
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(out / "final_model")
    tokenizer.save_pretrained(out / "final_model")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = prepare_tokenizer(args)
    model = build_model(args, tokenizer)
    noise_processor = load_noise_processor(args.noise_bins_json)

    if args.schedule == "plain_clm":
        run_span_phase(args, model, tokenizer, noise_processor, plain_clm=True)
    elif args.schedule == "span":
        run_span_phase(args, model, tokenizer, noise_processor)
    elif args.schedule == "nsp":
        run_nsp_phase(args, model, tokenizer, noise_processor)
    else:
        for round_idx in range(1, int(args.num_rounds) + 1):
            print(f"[KV-LLM] round {round_idx}/{args.num_rounds}: span corruption")
            run_span_phase(args, model, tokenizer, noise_processor, round_idx=round_idx)
            print(f"[KV-LLM] round {round_idx}/{args.num_rounds}: KV-NSP")
            run_nsp_phase(args, model, tokenizer, noise_processor, round_idx=round_idx)
        final_dir = Path(args.output_dir) / "final_model"
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"[OK] KV-LLM full CPT saved to {final_dir}")


if __name__ == "__main__":
    main()
