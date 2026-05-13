# -*- coding: utf-8 -*-
"""
M4: 集中管理所有绝对路径，通过环境变量覆盖默认值，消除硬编码。

用法：
    import paths_config as PC
    tokenizer = AutoTokenizer.from_pretrained(PC.TOKENIZER_PATH)

远端 H200 使用默认值无需额外设置。本地或其他环境可通过 .env 或命令行覆盖：
    export DAPT_WORKSPACE=/my/workspace
    export DAPT_TOKENIZER=/my/tokenizer
"""
import os

# ── 顶级工作区 ────────────────────────────────────────────────────────────────
DAPT_ROOT = os.environ.get("DAPT_ROOT", "/data/ocean/DAPT")
WORKSPACE_DIR = os.environ.get("DAPT_WORKSPACE", os.path.join(DAPT_ROOT, "workspace"))

# ── 模型 / Tokenizer ─────────────────────────────────────────────────────────
TOKENIZER_PATH = os.environ.get(
    "DAPT_TOKENIZER",
    os.path.join(DAPT_ROOT, "my-medical-tokenizer"),
)
MODEL_CHECKPOINT = os.environ.get("DAPT_MODEL_CHECKPOINT", "hfl/chinese-roberta-wwm-ext")
MACBERT_CHECKPOINT = os.environ.get(
    "DAPT_MACBERT_CHECKPOINT",
    "/data/hxzh/models/chinese-macbert-base",
)
KV_NER_BEST_MODEL = os.environ.get(
    "DAPT_KV_NER_MODEL",
    os.path.join(DAPT_ROOT, "runs/kv_ner_finetuned_noise_bucket/best"),
)

# ── 数据集 / 预计算产物 ───────────────────────────────────────────────────────
DATASET_PATH = os.environ.get(
    "DAPT_DATASET",
    os.path.join(WORKSPACE_DIR, "processed_dataset"),
)
TRAIN_CHUNKED_PATH = os.environ.get(
    "DAPT_TRAIN_CHUNKED",
    os.path.join(WORKSPACE_DIR, "train_chunked.txt"),
)
NOISE_BINS_JSON = os.environ.get(
    "DAPT_NOISE_BINS",
    os.path.join(WORKSPACE_DIR, "noise_bins.json"),
)
NSP_DATA_PATH = os.environ.get(
    "DAPT_NSP_DATA",
    os.path.join(DAPT_ROOT, "data/pseudo_kv_labels_filtered.json"),
)
KEYS_FILE = os.environ.get(
    "DAPT_KEYS_FILE",
    os.path.join(DAPT_ROOT, "biaozhu_keys_only_min5.txt"),
)
JIEBA_VOCAB_PATH = os.environ.get(
    "DAPT_JIEBA_VOCAB",
    os.path.join(DAPT_ROOT, "vocab_for_jieba.txt"),
)

# ── 输出目录（各脚本可通过 --output_dir 覆盖） ────────────────────────────────
DEFAULT_OUTPUT_DIR = os.environ.get(
    "DAPT_OUTPUT_DIR",
    os.path.join(WORKSPACE_DIR, "output_medical_bert_v2_8gpu"),
)
