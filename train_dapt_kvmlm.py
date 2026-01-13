import os
import math
import torch
import sys
import random

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
    TrainerCallback
)

# ---------------------------
# 注释说明（快速参考）
# ---------------------------
# - `AutoTokenizer.from_pretrained(path_or_id)`：加载 tokenizer（词表 + 分词逻辑）。path_or_id 可以是本地目录或 HuggingFace hub id。
# - `AutoModelForMaskedLM.from_pretrained(path_or_id)`：加载带 MLM 头的预训练模型权重。
# - `datasets.load_from_disk(path)`：加载已保存到磁盘的 HuggingFace Dataset（用于高效重复使用）。
# - `Trainer` / `TrainingArguments`：transformers 封装的训练循环与配置；用法见下文代码。
# 
# 核心要点（确保“不扩展词表”）：
# 1) 不要在脚本里调用 `model.resize_token_embeddings(new_size)`，这会修改模型 embedding 的行数，从而等同于扩表或缩表。
# 2) 加载后可以通过 `len(tokenizer)` 与 `model.get_input_embeddings().weight.size(0)` 比较，二者应一致；若不一致说明 tokenizer/model 词表不匹配（可能已经扩表或用错 tokenizer）。
# 3) 如果你确实需要扩表（不同实验），应该显式调用 `model.resize_token_embeddings(...)` 并保存新的 checkpoint 以供下游使用。
# 
# 本脚本为 kv-MLM 实验：使用 `word_ids` 做按词掩码（mask），但**不**修改模型词表。
# ---------------------------
# ===========================
# 1. 配置路径与参数
# ===========================
WORKSPACE_DIR = "/data/ocean/DAPT/workspace"
# Tokenizer 存放在 /data/ocean/DAPT/my-medical-tokenizer
TOKENIZER_PATH = "/data/ocean/DAPT/my-medical-tokenizer"
DATASET_PATH = os.path.join(WORKSPACE_DIR, "processed_dataset")
MODEL_CHECKPOINT = "hfl/chinese-roberta-wwm-ext"
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "output_medical_bert_kvmlm")

# ===========================
# 8卡 配置（可按需修改）
# ===========================
PER_DEVICE_BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 8e-5
MAX_SEQ_LEN = 896

def is_main_process():
    return int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]

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

@dataclass
class PrecomputedWWMCollator:
    tokenizer: Any
    mlm_probability: float = 0.15

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_input_ids = [f["input_ids"] for f in features]
        batch_word_ids = [f.get("word_ids") for f in features]
        batch = self.tokenizer.pad(
            {"input_ids": batch_input_ids},
            padding="longest",
            max_length=MAX_SEQ_LEN,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        for i in range(len(features)):
            word_ids = batch_word_ids[i]
            if word_ids is None:
                # fallback to random token-level masking if word_ids absent
                continue
            current_ids = input_ids[i]
            mapping = {}
            for idx, wid in enumerate(word_ids):
                if wid is None: continue
                if idx >= len(current_ids): break
                mapping.setdefault(wid, []).append(idx)

            unique_words = list(mapping.keys())
            if len(unique_words) == 0:
                continue
            num_to_mask = max(1, int(len(unique_words) * self.mlm_probability))
            masked_words = set(random.sample(unique_words, num_to_mask))
            mask_indices = torch.zeros(len(current_ids), dtype=torch.bool)
            for wid in masked_words:
                for idx in mapping[wid]:
                    mask_indices[idx] = True

            special_tokens_mask = torch.tensor(
                self.tokenizer.get_special_tokens_mask(current_ids.tolist(), already_has_special_tokens=True),
                dtype=torch.bool
            )
            mask_indices.masked_fill_(special_tokens_mask, value=False)
            if self.tokenizer.pad_token_id is not None:
                mask_indices.masked_fill_(current_ids == self.tokenizer.pad_token_id, value=False)

            probability_matrix[i, :] = 0.0
            probability_matrix[i, mask_indices] = 1.0

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.is_local_process_zero:
            loss = metrics.get("eval_loss")
            if loss:
                ppl = math.exp(loss)
                print(f"\n[Evaluation] Perplexity (PPL): {ppl:.4f}\n")
                metrics["perplexity"] = ppl

def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"数据集未找到: {DATASET_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    dataset = load_from_disk(DATASET_PATH)

    if is_main_process():
        print(f"Loading base model {MODEL_CHECKPOINT}...")
    # 从预训练检查点加载模型（基座模型）。
    # 说明：MODEL_CHECKPOINT 可以是 HuggingFace 的 model id（如 'hfl/chinese-roberta-wwm-ext'）或本地 checkpoint 路径。
    model = AutoModelForMaskedLM.from_pretrained(MODEL_CHECKPOINT)

    # ---- 核心检查：确认 tokenizer 与 model embedding 大小匹配（以证明没有扩展词表） ----
    try:
        tokenizer_vocab_size = len(tokenizer)
        model_emb_size = model.get_input_embeddings().weight.size(0)
        if is_main_process():
            print(f"Tokenizer vocab size: {tokenizer_vocab_size}, Model embedding rows: {model_emb_size}")
        if tokenizer_vocab_size != model_emb_size:
            if is_main_process():
                print("WARNING: tokenizer vocab size != model embedding size. "
                      "这通常意味着 tokenizer 或模型已被修改（词表不匹配）。\n"
                      "如果你想严格不扩展词表，请确保使用与模型对应的 tokenizer，或者在扩表实验中显式调用 resize_token_embeddings 并保存新 checkpoint。")
    except Exception:
        if is_main_process():
            print("Note: 无法比较 tokenizer 与 model embedding 大小（可能模型类型非标准）。跳过该检查。")

    # 注意：此处**不**调用 model.resize_token_embeddings(len(tokenizer))，以保持词表不变。
    # 若需要扩表（另一个实验），可在这里显式调用并随后保存新模型。

    # 可选：扩展 position embeddings（如果需要更长 context）
    model = resize_position_embeddings(model, new_max_len=MAX_SEQ_LEN)
    model.gradient_checkpointing_enable()
    data_collator = PrecomputedWWMCollator(tokenizer=tokenizer)

    # 额外安全检查：确认 dataset 中实际的 token id 不会超出模型 embedding 范围
    # 这是导致 "CUDA error: device-side assert triggered" 的常见原因（embedding 索引越界）
    try:
        if is_main_process():
            print("Checking dataset token id ranges (sample up to 1000 examples)...")
        max_id_seen = -1
        sample_limit = min(1000, len(dataset["train"]))
        for i in range(sample_limit):
            ids = dataset["train"][i].get("input_ids")
            if ids is None:
                continue
            if isinstance(ids, list):
                cur_max = max(ids) if len(ids)>0 else -1
            else:
                try:
                    cur_max = int(max(ids))
                except Exception:
                    cur_max = -1
            if cur_max > max_id_seen:
                max_id_seen = cur_max
        if is_main_process():
            print(f"Max token id in sampled train examples: {max_id_seen}")
        if max_id_seen >= model.get_input_embeddings().weight.size(0):
            if is_main_process():
                print("ERROR: Found token ids >= model embedding size. 这会在 embedding lookup 时导致 CUDA 断言失败。\n"
                      "可能原因：dataset 是用扩展后的 tokenizer 生成，但当前加载的 model 未扩表。\n"
                      "可选修复：1) 使用与 dataset 对应的 tokenizer+模型（已扩表的 checkpoint）；\n"
                      "         2) 重新用当前 tokenizer 重新生成 processed_dataset（re-tokenize）；\n"
                      "         3) 若你确实想扩表，允许在脚本中调用 model.resize_token_embeddings(len(tokenizer)) 并重启训练。\n"
                      "在继续训练前请修正上述问题。\n")
            # 退出以避免 GPU crash
            sys.exit(1)
    except Exception as e:
        if is_main_process():
            print("Warning: dataset token id range check failed:", e)

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
        remove_unused_columns=False,
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
        print("Starting kv-MLM Distributed Training (no vocab resize)...")

    trainer.train()

    if is_main_process():
        print("Saving final model...")
        trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

if __name__ == "__main__":
    main()
