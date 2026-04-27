import os
import math
import torch
import sys

# ===========================
# 0. 设置环境变量
# ===========================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("TORCHINDUCTOR_BENCHMARK_KERNEL", "0")

from dataclasses import dataclass
from typing import Any, Dict, List
from datasets import load_from_disk
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
# C2: 公共组件，消除训练脚本间重复
from pretraining_common import PerplexityCallback

# ===========================
# 1. 配置路径与参数
# ===========================
WORKSPACE_DIR = "/data/ocean/bpe_workspace"
# Tokenizer 统一指向 /data/ocean/DAPT/my-medical-tokenizer
TOKENIZER_PATH = "/data/ocean/DAPT/my-medical-tokenizer"
DATASET_PATH = os.path.join(WORKSPACE_DIR, "processed_dataset")
MODEL_CHECKPOINT = "hfl/chinese-roberta-wwm-ext"
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "output_medical_bert_add_vocab_mlm")  # 新输出目录

# ===========================
# 8卡 H200 极速配置
# ===========================
# Global Batch Size = 16 * 8(GPUs) * 4(Accum) = 512
PER_DEVICE_BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 8e-5
MAX_SEQ_LEN = 896

def is_main_process():
    return int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]

# ===========================
# 2. 位置Embedding扩展（如有需要）
# ===========================
def resize_position_embeddings(model, new_max_len=1024):
    config = model.config
    old_max_len = config.max_position_embeddings
    if new_max_len <= old_max_len:
        return model
    if is_main_process():
        print(f"Executing Weight Cloning: Context Length {old_max_len} -> {new_max_len} ...")
    old_pos_embeddings = model.bert.embeddings.position_embeddings.weight.data
    embedding_dim = old_pos_embeddings.shape[1]
    new_pos_embeddings = torch.nn.Embedding(new_max_len, embedding_dim).weight.data
    new_pos_embeddings[:old_max_len, :] = old_pos_embeddings
    remaining_len = new_max_len - old_max_len
    copy_len = min(old_max_len, remaining_len)
    new_pos_embeddings[old_max_len : old_max_len + copy_len, :] = old_pos_embeddings[:copy_len, :]
    model.bert.embeddings.position_embeddings.weight.data = new_pos_embeddings
    model.bert.embeddings.position_embeddings.num_embeddings = new_max_len
    model.config.max_position_embeddings = new_max_len
    model.bert.embeddings.register_buffer("position_ids", torch.arange(new_max_len).expand((1, -1)))
    return model

# ===========================
# 3. Perplexity Callback
# ===========================
# C2: PerplexityCallback 已提取到 pretraining_common.py，通过顶部 import 引入。

def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"数据集未找到: {DATASET_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    dataset = load_from_disk(DATASET_PATH)

    if is_main_process():
        print(f"Loading base model {MODEL_CHECKPOINT}...")

    model = AutoModelForMaskedLM.from_pretrained(MODEL_CHECKPOINT)
    model.resize_token_embeddings(len(tokenizer))
    model = resize_position_embeddings(model, new_max_len=MAX_SEQ_LEN)
    model.gradient_checkpointing_enable()

    # 官方标准MLM collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        ddp_find_unused_parameters=False,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.05,
        bf16=True,
        tf32=True,
        dataloader_num_workers=8,
        torch_compile=False,
        gradient_checkpointing=True,
        remove_unused_columns=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        logging_steps=10,
        report_to="tensorboard",
        group_by_length=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[PerplexityCallback]
    )

    if is_main_process():
        print("Starting 8-GPU Distributed Training (Official MLM)...")

    trainer.train()

    if is_main_process():
        print("Saving final model...")
        trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

if __name__ == "__main__":
    main()