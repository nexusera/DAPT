# KV-LLM Decoder-Only Route

This folder implements the Qwen3 experimental path for migrating KV-BERT
ideas to decoder-only LLMs.

**Use the `-Base` suffix.** Qwen3 small repos without the `-Base` suffix
(`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`) are the instruction-tuned variants
— not valid CPT starting points per plan §10 SC3.

Non-MLM logic is intentionally aligned with the BERT route:

- same default corpus root from `paths_config.py`
- same `pseudo_kv_labels_filtered.json` for KV-NSP
- same `noise_bins.json`
- same dual dictionaries: `biaozhu_keys_only_min5.txt` + `vocab_for_jieba.txt`
- same staged alternation idea: span-corruption round then KV-NSP round
- same KV-NSP negative sampling defaults: `0.5 / 1.0 / 1.0 / 10`
- same KV-NSP negative sampling implementation via `kv_nsp/negative_sampling.py`

## What Is Implemented

- **Span Corruption**: whole medical-entity masking with sentinel tokens
  (`<extra_id_0>`, `<extra_id_1>`, ...), trained as causal text infilling.
- **KV-NSP**: `[Key][SEP][Value]` last-token classification, aligned with the
  BERT route except for the decoder-only backbone.
- **Noise-Embedding**: bucket / linear / mlp / concat-linear noise features
  added to Qwen token embeddings before the Transformer.
- **CPT Entry Point**: staged training (`span` then `kv_nsp`) or individual
  phases through `train_cpt.py`.
- **SFT Entry Point**: minimal MedStruct-S style JSON generation fine-tuning
  through `fine_tune_sft.py`, with optional LoRA rank 8.

## Full CPT Pilot

```bash
python -m kv_llm.train_cpt \
  --model_name_or_path Qwen/Qwen3-0.6B-Base \
  --nsp_data /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
  --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
  --output_dir /data/ocean/DAPT/workspace/kv_llm_full \
  --schedule full \
  --noise_mode bucket \
  --bf16 \
  --gradient_checkpointing \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_rounds 3 \
  --max_samples 1000
```

Remove `--max_samples` for full data. Use `--schedule plain_clm` for the
Plain CLM CPT baseline, `--schedule span` for w/o KV-NSP, and `--schedule nsp`
for isolated KV-NSP.

If you do not pass `--span_data`, `--keys_file`, or `--vocab_for_jieba`, the
script falls back to the same default paths used by the BERT route.

## Downstream SFT Pilot

```bash
python -m kv_llm.fine_tune_sft \
  --model_name_or_path /data/ocean/DAPT/workspace/kv_llm_full/final_model \
  --train_data /data/ocean/DAPT/runs/gt.jsonl \
  --output_dir /data/ocean/DAPT/workspace/kv_llm_sft_task13 \
  --use_lora \
  --lora_rank 8 \
  --bf16
```

Expected input JSON/JSONL records contain `ocr_text` and `pairs`. The script
trains the model to generate compact JSON:

```json
{"pairs":[{"key":"...", "value":"..."}]}
```

## Files

- `span_corruption.py`: entity span masking and collator.
- `kv_nsp.py`: Label Studio pair extraction, negative sampling, NSP collator.
- `modeling.py`: Qwen causal LM wrapper with noise embeddings and NSP head.
- `train_cpt.py`: continued pretraining entrypoint.
- `fine_tune_sft.py`: downstream decoder-only SFT/LoRA entrypoint.
- `configs/qwen3_0_6b_full.json`: default experiment configuration.

## Notes

- The package is independent from the existing KV-BERT/MacBERT path.
- For bucket/concat noise modes, pass `--noise_bins_json`; otherwise all noise
  ids default to anchor bins.
- The SFT script is intentionally minimal. For official numbers, convert
  generated JSONL to MedStruct-S format and score with the existing scorer.
