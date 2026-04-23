# KV-BERT / DAPT — Code Review Report (English)

**Repository**: `/Users/user/Documents/DAPT`
**Branch**: `feature/noise-embedding` (296 files / ~2M lines ahead of `main`; real source ≈ 15–20k LoC)
**HEAD**: `fcf1d61` "docs: add OCR_TEXT_AND_NOISE_ALIGNMENT guide"
**Date**: 2026-04-23
**Scope**: Pretraining pipeline + downstream NER/EBQA + FastAPI serving + ablation experiments.

---

## 1. Executive Summary

**KV-BERT** is a noise-robust, domain-adaptive pretraining framework for semi-structured Key-Value extraction in OCR-derived Chinese clinical reports, by Hao Li et al. (AI Starfish). It contributes three techniques: **KV-MLM** (entity-boundary WWM), **KV-NSP** (binary classification of Key↔Value with hard/easy negatives), and **Noise-Embedding** (7-D OCR-quality features discretised via binning and summed into the embedding layer). The project now targets MacBERT as base model (swapped from RoBERTa) and has a full pipeline from raw OCR dumps to a Dockerised FastAPI service.

**Overall quality**: *senior research-engineer level with publication-track ambition, but engineering hygiene has not kept up with feature velocity*.

| Dimension | Score (1–5) | Notes |
|-----------|:-----------:|-------|
| Architecture & ML design | **4.5** | Thoughtful noise-binning design, clean 3-contribution story, strong ablation coverage. |
| Correctness | **3** | Critical `main` bugs fixed, but one dimension-mismatch bug introduced; shape checks still thin. |
| Code hygiene / DRY | **2** | ≥60 % copy-paste across 11 training scripts; no shared utilities module. |
| Documentation | **4** | Bilingual docstrings, paper draft, pipeline docs, integration guides. |
| Portability | **2** | Hard-coded `/data/ocean/DAPT/...` paths everywhere; sys.path hacks in 6+ scripts. |
| Test coverage | **1.5** | Only 2 serving integration tests; no unit tests for noise, modeling, or evaluation. |
| Production readiness (serving) | **2** | Works, but no auth, no rate limit, permissive CORS, unpinned Docker base. |

---

## 2. What Changed Since `main`

The `main` branch review identified five issues; four of them are now resolved:

| Issue on `main` | Status on `feature/noise-embedding` |
|-----------------|--------------------------------------|
| Per-split index / OCR alignment bug in `add_noise_features.py` | ✅ **Fixed** — `global_idx = split_offsets[split] + idx` (lines 165–182, 268–271). |
| SymSpell built with `max_dictionary_edit_distance=32` | ✅ **Removed** — whole SymSpell / Levenshtein path dropped in favour of binning. |
| `char_break_norm=24` saturating the feature | ✅ **Removed** — `char_break_ratio` now hard-clipped to 0.25 (`noise_feature_processor.py:42`). |
| Missing seq-len assertion in `RobertaNoiseEmbeddings.forward` | ✅ **Fixed** — length is inferred from `input_shape[1]` (`noise_bert_model.py:123`). |
| Deprecated `evaluation_strategy=` in `kv_nsp/run_train.py` | ⚠️ **Partial** — main pretraining scripts migrated, but `kv_nsp/run_train.py:192` and `kv_nsp/run_train_with_noise.py:472` still use the old kwarg. |

---

## 3. Project Architecture

```
 raw JSON / OCR dumps (multi-source)
        │
        ▼                                     extract_and_dedup_json_v2.py (MD5 dedupe)
 train.txt + per-source splits
        │
        ▼                                     resample_mix.py  + chunk_long_lines.py
 train_resampled.txt → train_chunked.txt
        │
        ▼                                     train_ocr_clean.py  +  filter_vocab_with_llm.py
 medical_vocab_ocr_only/vocab.txt  → kept_vocab.txt
        │
        ▼                                     final_merge_v9_regex_split_slim.py
 my-medical-tokenizer/  +  vocab_for_jieba.txt (generate_jieba_vocab.py)
        │
        ▼                                     build_dataset_final_slim.py (Jieba-aligned WWM)
 processed_dataset/  (input_ids + word_ids)
        │
        ▼                                     NoiseFeatureProcessor.fit_bins  → noise_bins.json
        ▼                                     add_noise_features.py
 processed_dataset_with_noise/  (+ noise_values = 7-D per token)
        │
        ▼                                     train_dapt_macbert_staged.py  (8×H200, staged MLM↔NSP)
 output_macbert_kvmlm_staged/final_staged_model/
        │
        ├─────►  Downstream fine-tuning:  dapt_eval_package/pre_struct/kv_ner/train_with_noise.py
        │                                 dapt_eval_package/pre_struct/ebqa/train_ebqa.py
        │
        ├─────►  Ablations: experiments/{mlm,noise,nsp_ratio,tokenizer}_ablation/*
        │
        ├─────►  Interpretability: experiments/interpretability/{attention,IG}/*
        │
        └─────►  Serving: serving/ (FastAPI + BertCRF + batching + noise extraction)
```

**Scale** (evaluated on paper claims): 982 k raw lines → 225 k resampled → 537 k chunks; 3 582 OCR pages annotated (3 224 train / 358 test).

---

## 4. Strengths

1. **Principled noise modeling** — the binning design (`NoiseFeatureProcessor`) with an anchor bin for "perfect text" is a clean way to map OCR metadata into a BERT embedding space while staying compatible with clean-text samples.
2. **Three well-separated contributions** (KV-MLM / KV-NSP / Noise-Embedding) each with dedicated ablation directories — `no_mlm`, `no_nsp`, `no_noise`, `noise_{bucket,linear,mlp,concat_linear}`, `nsp_ratio_{1:1,3:1,1:3}`.
3. **Fixed the critical alignment bug** from `main`: per-split offsets now keep OCR alignment correct across train/test splits.
4. **Multiple training regimes** — staged, multi-task, hybrid-masking — covering common DAPT strategies.
5. **Complete lifecycle** — data cleaning, tokenizer training, pretraining, downstream fine-tuning (KV-NER, EBQA), interpretability (attention, IG), and productionised serving.
6. **Paper-ready documentation** — `paper.md` (LLNCS LaTeX), `pipeline_new.md`, `PRETRAINED_MODELS_SUMMARY.md`, `interview_prep.md`, `KV_BERT_预训练与下游推理指南.md`.
7. **Pragmatic training tricks** — `bf16`, `tf32`, `group_by_length`, gradient checkpointing, staged curriculum, `ddp_find_unused_parameters=True` where heads are conditionally active.

---

## 5. Issues by Severity

### 🔴 Critical

| # | File : Line | Issue |
|---|-------------|-------|
| **C1** | `add_noise_features.py:96` | `build_zero_feats` still returns `[[0.0] * 5 …]` and `[[False] * 5 …]`, but `FEATURES` now has **7** entries. On the fallback path (missing / malformed OCR object at line 195) the sample ends up with 5-D `noise_values` while all normal samples have 7-D. HuggingFace `save_to_disk` either raises on schema drift or silently coerces ragged rows — both outcomes are bad. Fix: `[[0.0] * len(FEATURES) for _ in range(seq_len)]`. |
| **C2** | All `train_dapt_*.py` | **≈60–70 % duplicated code** across 11 training scripts. `PrecomputedWWMCollator`, `PerplexityCallback`, `RobertaModelWithNoise`, `MLMStageCollator`, `DynamicNSPDataset` are each re-defined 3–6 times. Every bug fix must be applied N times; at least one has already drifted (see M1). Extract into a `pretraining_common.py`. |

### 🟠 High

| # | File : Line | Issue |
|---|-------------|-------|
| **H1** | `noise_bert_model.py:147–154` | Bucket-mode forward does not assert `noise_ids.shape[-1] == len(FEATURES)`. A 5-D fallback vector (see C1) or a truncated cache will silently index the wrong embedding matrix. |
| **H2** | `noise_bert_model.py:156–162` | Continuous/linear/mlp mode does not validate `noise_values` dtype or shape either. |
| **H3** | `noise_fusion.py:~126` | `nan_to_num` is applied **after** `clamp`, so NaNs propagate through clamp first; should be reordered. |
| **H4** | `dapt_eval_package/pre_struct/kv_ner/evaluate.py` vs `evaluate_with_dapt_noise.py` | ~1 000 lines of near-identical prediction / assembly / metrics code across two files. High risk of metric drift between noise-on and noise-off experiments. |
| **H5** | `data_utils.py:226–228`, `train_with_noise.py:325–333`, `compare_models.py:81–98` | `_expand_word_noise_to_chars()` / `_broadcast_global_noise()` is reimplemented **three times** in three files. Any fix to noise-expansion must be repeated. |
| **H6** | `train_with_noise.py:602–603` | Type-unsafe batch extraction: `batch.get("noise_ids") if isinstance(batch, dict) else batch.noise_ids`. Crashes if batch is a tuple or custom collator output. |
| **H7** | `kv_nsp/run_train.py:192`, `kv_nsp/run_train_with_noise.py:472` | Deprecated `evaluation_strategy=` — removed in `transformers ≥ 4.46`. |
| **H8** | `da_core/dataset.py:901, 919` | Debug `print()` statements on every sample with `ridx < 5`; left from debugging session. |
| **H9** | `serving/app.py:114` | `allow_origins=["*"]` + `allow_headers=["*"]` — any origin can call the GPU inference endpoint. |
| **H10** | `serving/` (all routers) | No authentication, no rate limiting, no request-ID propagation. The endpoint fronting an 8-GPU model is effectively open. |
| **H11** | `serving/routers/extract.py:178` | `detail=str(exc)` returns raw exception text — leaks model paths, CUDA errors, internal structure. |
| **H12** | `serving/core/postprocessor.py:149` | `full_text[v_start:v_end]` is reached after filtering without re-checking `vals` is non-empty → potential `IndexError`. |
| **H13** | `serving/core/batch_engine.py:132–140` | Batch deadline is anchored to the **first** request. Slow trickles can miss the window because the timer never resets on new arrivals. |
| **H14** | Whole repo | **No unit tests** outside `serving/test_api.py` + `serving/tests/test_extract_from_file.py`. Noise binning, WWM collator, NSP sampling, noise processor, evaluator — all untested. |

### 🟡 Medium

| # | File : Line | Issue |
|---|-------------|-------|
| M1 | Various training scripts | `ddp_find_unused_parameters` is `False` in `train_dapt_distributed.py:395`, `True` in `train_dapt_mtl.py:566` and `train_dapt_staged.py:477`. Because heads may be conditionally active in staged/MTL runs, `False` can raise a DDP error at step > 0 on those. |
| M2 | `train_dapt_mtl.py:568` | `save_safetensors=False` hard-coded — disables safer serialisation without explanation. |
| M3 | `train_ebqa.py` (1 829 loc), `da_core/dataset.py` (1 371), `train_with_noise.py` (1 169), `evaluate.py` (1 154), `model_ebqa.py` (994), `compare_models.py` (969), `train_dapt_macbert_staged.py` (749) | Very long monolithic files; need decomposition (dataset / model / metrics / training-loop). |
| M4 | Repo-wide | Hard-coded absolute paths `/data/ocean/DAPT/...`, `/home/ocean/...`, `/data/hxzh/...`. Move to env vars or a single `config.yaml`. |
| M5 | 6+ scripts | `sys.path.append(current_dir)` scattered across scripts — fragile to CWD; breaks if imported as package. |
| M6 | `serving/Dockerfile` | Base image `nvcr.io/nvidia/pytorch:24.03-py3` is **unpinned** (no digest). |
| M7 | `serving/requirements.txt` | Dependencies **unpinned** — no `==X.Y.Z`. |
| M8 | `serving/Dockerfile` | `COPY . /app` pulls entire repo (notebooks, fixtures, tools) into the production image — use `.dockerignore`. |
| M9 | `serving/schemas/request.py` | `noise_values` input has no numeric range validation — NaN/inf could pollute downstream calculations. |
| M10 | `serving/app.py:121–135` | Generic `Exception` handler may mask FastAPI's own 422 validation responses. |
| M11 | `serving/app.py:67–69` | Lifespan-hook exceptions are swallowed silently. |
| M12 | `serving/routers/extract.py:70` | `request_id` is generated locally; no upstream tracing correlation. |
| M13 | `noise_feature_processor.py:157` vs `add_noise_features.py` | `compute_noise_from_ocr.py` reimplements the same feature extraction third time — a third copy of features logic. |
| M14 | `train_dapt_distributed.py:384` vs `train_dapt_macbert_staged.py:591` | Inconsistent guards when loading noise processor — one file checks existence, another does not. |
| M15 | Commit history | Four consecutive "fix" commits on sync / alignment (`78476fa`, `f01f729`, `fcf1d61`, `54ac0a3`) indicate the OCR↔dataset alignment invariant has been adjusted multiple times — consider adding an automated alignment check in CI. |

### 🟢 Minor / Nits

| # | File : Line | Issue |
|---|-------------|-------|
| N1 | `dapt_eval_package/.../modeling.py:97–98` | Stale comment: "noise_embed_dim preserved but no longer used" — parameter still in signature, creates reader confusion. |
| N2 | Error-handling style | Some functions `raise ValueError`, others silently `return None` (e.g. `dataset.py:223–224`). |
| N3 | `noise_feature_processor.py:157` | `to_id` returns `int(digitize) + 1` for non-zero values; callers must size embedding tables as `NUM_BINS + 1` (0 = anchor). Document this invariant. |
| N4 | `requirements.txt` at repo root | Missing — only `serving/requirements.txt` exists; training has no lock file. |
| N5 | `paper.md` | Still anonymised for submission but author block commented-in just above; clean up before non-anonymous export. |
| N6 | `.DS_Store` committed under `dapt_eval_package/pre_struct/` | Add to `.gitignore` (already in root `.gitignore`). |

---

## 6. File-By-File Highlights

| Group | Representative files | Health |
|-------|----------------------|--------|
| Data ingestion | `extract_and_dedup_json_v2.py`, `resample_mix.py`, `chunk_long_lines.py` | OK |
| Vocab / Tokenizer | `train_ocr_clean.py`, `filter_vocab_with_llm.py`, `final_merge_v9_regex_split_slim.py`, `generate_jieba_vocab.py` | OK (LLM filter lacks checkpoint) |
| Dataset construction | `build_dataset_final_slim.py`, `add_noise_features.py` | **C1 bug** |
| Noise core | `noise_feature_processor.py`, `noise_embeddings.py`, `noise_bert_model.py`, `noise_fusion.py` | **H1–H3** |
| Pretraining scripts (11) | `train_dapt_distributed.py`, `train_dapt_macbert_staged.py`, `…_no_mlm/no_nsp/no_noise/mtl/hybrid/staged` | **C2 duplication**, M1, M2 |
| KV-NSP | `kv_nsp/{dataset_with_noise.py,run_train_with_noise.py,negative_sampling.py}` | H7 |
| Downstream fine-tuning | `dapt_eval_package/pre_struct/kv_ner/{modeling,train,train_with_noise,evaluate,compare_models}.py`, `ebqa/{train_ebqa,model_ebqa,predict_ebqa}.py` | **H4, H5, H6, H8**, M3 |
| Ablation shells | `experiments/{mlm,noise,nsp_ratio,tokenizer}_ablation/*.sh` | Numerous; shell-heavy |
| Interpretability | `experiments/interpretability/{attention,IG}_*.py` | Research grade |
| Serving | `serving/app.py`, `serving/core/*`, `serving/routers/*` | **H9–H13, M6–M12** |
| Verification | `verify_noise_alignment.py`, `scripts/validate_ner_spans_after_ocr_sync.py`, `scripts/check_pretrain_test_leakage.py` | Good that these exist |

---

## 7. Prioritised Recommendations

**Do before the next pretraining / ablation re-run**

1. **C1** — Fix `add_noise_features.py:96` to use `len(FEATURES)` (= 7).
2. **H1** — Add `assert noise_ids.shape[-1] == len(FEATURES)` in `noise_bert_model.py` forward.
3. **H7** — Migrate `kv_nsp/run_train*.py` from `evaluation_strategy=` to `eval_strategy=`.
4. **H8** — Remove debug `print()` from `da_core/dataset.py`.
5. Confirm **M1** — audit `ddp_find_unused_parameters` flag against each script's actual parameter graph.

**Before handing off / collaborating outside the current team**

6. **C2** — Create `pretraining_common.py` (collator, callback, model wrappers) and delete the duplicates.
7. **H4 / H5** — Single shared `noise_utils.py` and a single `evaluate_core.py`.
8. **H14** — Add `tests/` with at minimum: noise-processor quantile round-trip, WWM collator shape/count, NSP negative-sampling distribution, end-to-end `add_noise_features` on a 5-sample fixture.
9. **M4 / M5** — Introduce `config.yaml` / `.env` for paths; remove `sys.path.append` hacks.
10. **M6 / M7 / M8** — Pin Docker base and dependencies; add `.dockerignore`.

**Before exposing `serving/` outside the internal network**

11. **H9** — Tighten CORS to specific origins.
12. **H10** — Add API-key auth and a token-bucket rate limit.
13. **H11** — Sanitise error responses; log internally, return opaque errors externally.
14. **H12 / H13** — Fix postprocessor `IndexError` path and reset batch-deadline on each new arrival.
15. **M9** — Validate `noise_values` numeric range; reject NaN/inf.

**Quality of life**

16. Split 1 000+ line files (`train_ebqa.py`, `da_core/dataset.py`, `train_with_noise.py`, `evaluate.py`) into cohesive modules.
17. Add a `CI.yml` that at minimum runs `python -m pyflakes`, the new unit tests, and a lint on all `.py` files.
18. Squash the four consecutive sync/alignment "fix" commits into one logical change before the next PR.

---

## 8. Verdict

**For a research group**: this is a strong, publication-ready codebase. The ML story is coherent, the ablation grid is thorough, and the paper draft is substantive. The critical `main`-branch bugs are fixed.

**For a production handover**: it is not yet ready. The single new dimension-mismatch bug (C1) can silently corrupt a training run, the ~70 % duplication across training scripts means any fix must be applied in many places, the FastAPI service has no auth, and the whole pipeline has no unit tests. Addressing items C1, C2, H1, H4/H5, H9–H11, and H14 would bring the project from "works on my GPU cluster" to "safe for another team to operate".

**Short recommendation**: fix C1 and H1 today, then budget 2–3 days to de-duplicate the training scripts and add a minimal test suite before the next full-scale pretraining run.
