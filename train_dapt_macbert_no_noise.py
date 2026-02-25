import os
import torch
import sys
import random
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
from torch.utils.data import Dataset

# ===========================
# 0. 环境与依赖设置（与带噪声版本保持一致）
# ===========================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 防止 DataLoader 死锁
os.environ.setdefault("TORCHINDUCTOR_BENCHMARK_KERNEL", "0")

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BertForPreTraining,
)

# 引入本地模块（仅使用 KV-NSP 数据集，不再使用噪声相关模块）
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 引入 KV-NSP 模块
kv_nsp_dir = os.path.join(current_dir, "kv_nsp")
if os.path.isdir(kv_nsp_dir):
    sys.path.append(kv_nsp_dir)
from dataset import KVDataset

# 常量定义
MAX_SEQ_LEN = 512


# ===========================
# 1. 数据处理（不使用噪声特征）
# ===========================

# --- A. MLM 专用 Collator（去掉 noise_ids 相关逻辑） ---
@dataclass
class MLMStageCollator:
    """仅用于标准 MLM 的 Collator，不依赖噪声特征。

    - 输入来自预处理后的 HF Dataset（processed_dataset），字段至少包含 input_ids / word_ids（可选）
    - 实现 Whole Word Masking（WWM）策略，与带噪声版本保持一致
    """

    tokenizer: Any
    mlm_probability: float = 0.15
    max_length: int = 512

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. 提取基础数据
        batch_input_ids = [f["input_ids"] for f in features]
        batch_word_ids = [f.get("word_ids") for f in features]

        # 2. Padding input_ids
        batch = self.tokenizer.pad(
            {"input_ids": batch_input_ids},
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # 3. WWM Masking Strategy
        for i in range(len(features)):
            word_ids = batch_word_ids[i]
            current_ids = input_ids[i]
            if word_ids:
                mapping = {}
                for idx, wid in enumerate(word_ids):
                    if wid is None or idx >= len(current_ids):
                        continue
                    mapping.setdefault(wid, []).append(idx)

                unique_words = list(mapping.keys())
                num_to_mask = max(1, int(len(unique_words) * self.mlm_probability))
                masked_words = set(random.sample(unique_words, num_to_mask))
                mask_indices = torch.zeros(len(current_ids), dtype=torch.bool)
                for wid in masked_words:
                    for idx in mapping[wid]:
                        mask_indices[idx] = True
            else:
                # 若没有 word_ids 信息，则退化为 token-level 随机掩码
                mask_indices = torch.bernoulli(probability_matrix[i]).bool()

            # Special tokens & Pad tokens mask
            special_tokens_mask = torch.tensor(
                self.tokenizer.get_special_tokens_mask(
                    current_ids.tolist(), already_has_special_tokens=True
                ),
                dtype=torch.bool,
            )
            mask_indices.masked_fill_(special_tokens_mask, value=False)
            if self.tokenizer.pad_token_id is not None:
                mask_indices.masked_fill_(
                    current_ids == self.tokenizer.pad_token_id, value=False
                )
            probability_matrix[i, :] = 0.0
            probability_matrix[i, mask_indices] = 1.0

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        batch["input_ids"] = input_ids
        batch["labels"] = labels

        # 注意：MLM 阶段完全不需要 NSP 标签，直接不传该字段，
        # 这样 BertForPreTraining 会根据 labels 自动计算 MLM loss。

        return batch


# --- B. NSP 专用 Dataset 和 Collator（不使用噪声特征） ---

class DynamicNSPDataset(Dataset):
    """包装原始 KVDataset，实现实时动态负采样 (Dynamic Negative Sampling)。"""

    def __init__(self, raw_kv_dataset: KVDataset):
        self.ds = raw_kv_dataset
        self.valid_pairs_set = set(self.ds.pairs)

    def __len__(self):
        return len(self.ds.pairs)

    def __getitem__(self, idx):
        key_text, value_text = self.ds.pairs[idx]
        label = 1  # Positive

        if random.random() < self.ds.negative_prob:
            label = 0
            if random.random() < self.ds.hard_negative_prob:
                # Hard negative: swap
                key_text, value_text = value_text, key_text
            else:
                # Easy negative: random value，避免采到真实正样本
                max_retries = 10
                for _ in range(max_retries):
                    candidate_value = random.choice(self.ds.value_pool)
                    if (key_text, candidate_value) not in self.valid_pairs_set:
                        value_text = candidate_value
                        break
                else:
                    value_text = candidate_value

        return {"text_a": key_text, "text_b": value_text, "label": label}


@dataclass
class NSPStageCollator:
    """NSP 阶段 Collator：仅构造句对输入与 NSP 标签，不依赖噪声特征。"""

    tokenizer: Any
    max_length: int = 512

    def __call__(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        text_a_list = [x["text_a"] for x in batch_items]
        text_b_list = [x["text_b"] for x in batch_items]
        labels = [x["label"] for x in batch_items]

        enc = self.tokenizer(
            text_a_list,
            text_b_list,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc["token_type_ids"],
            "next_sentence_label": torch.tensor(labels, dtype=torch.long),
            # NSP 阶段不做 MLM：完全不传 labels 字段，
            # 这样 BertForPreTraining 只会计算 NSP loss。
        }


# ===========================
# 2. 主流程（标准 DAPT：MLM + NSP，无噪声嵌入）
# ===========================


def main():
    parser = argparse.ArgumentParser()
    # 基础配置
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/ocean/DAPT/workspace/processed_dataset",
        help="HF Dataset 路径（仅包含文本，不要求噪声字段）",
    )
    parser.add_argument(
        "--nsp_data_dir",
        type=str,
        default="/data/ocean/DAPT/data/pseudo_kv_labels_filtered.json",
        help="KV-NSP 训练用的伪标签 JSON 或目录",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/data/ocean/DAPT/my-medical-tokenizer",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="可选：从已有 checkpoint 继续训练",
    )

    # 训练超参（与带噪声 staged 版本保持一致）
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=3,
        help="交替训练的总轮数 (MLM -> NSP -> MLM -> NSP ...)",
    )
    parser.add_argument(
        "--mlm_epochs_per_round",
        type=int,
        default=1,
        help="每轮 MLM 训练的 epoch 数",
    )
    parser.add_argument(
        "--nsp_epochs_per_round",
        type=int,
        default=3,
        help="每轮 NSP 训练的 epoch 数",
    )

    args = parser.parse_args()

    # 1. 资源准备
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # 2. 准备数据集
    print(f"Loading MLM Dataset from {args.dataset_path}...")
    mlm_dataset_disk = load_from_disk(args.dataset_path)
    mlm_dataset = (
        mlm_dataset_disk["train"] if "train" in mlm_dataset_disk else mlm_dataset_disk
    )

    print(f"Loading NSP Dataset from {args.nsp_data_dir}...")
    p = Path(args.nsp_data_dir)
    nsp_files = [p] if p.is_file() else [p / f for f in os.listdir(p) if f.endswith(".json")]
    raw_kv_dataset = KVDataset(nsp_files, tokenizer)
    nsp_dataset = DynamicNSPDataset(raw_kv_dataset)

    print(f"MLM Samples: {len(mlm_dataset)}, NSP Samples: {len(nsp_dataset)}")

    # 3. 初始化模型（标准 BertForPreTraining，不带噪声嵌入）
    model_path = "hfl/chinese-macbert-base"
    if args.resume_from_checkpoint:
        model_path = args.resume_from_checkpoint
        print(f"Resuming from checkpoint: {model_path}")

    model = BertForPreTraining.from_pretrained(model_path)

    # Resize Token Embeddings：适配自定义 tokenizer
    if len(tokenizer) > model.config.vocab_size:
        print(f"Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # 4. 准备 Collator
    mlm_collator = MLMStageCollator(tokenizer)
    nsp_collator = NSPStageCollator(tokenizer)

    # 5. 交替训练循环（结构与带噪声 staged 版本保持一致）
    for round_idx in range(1, args.num_rounds + 1):
        print(f"\n{'=' * 40}\n Round {round_idx}/{args.num_rounds} Start \n{'=' * 40}")

        # --- Phase A: MLM Training ---
        print(f"--- [Round {round_idx}] Phase A: MLM Training ---")
        mlm_output_dir = os.path.join(args.output_dir, f"round_{round_idx}_mlm_no_noise")

        training_args_mlm = TrainingArguments(
            output_dir=mlm_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.mlm_epochs_per_round,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            learning_rate=args.learning_rate,
            logging_steps=50,
            save_strategy="epoch",
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            ddp_find_unused_parameters=True,
            dataloader_num_workers=0,
            save_safetensors=False,
            remove_unused_columns=False,
            report_to="tensorboard",
            run_name=f"dapt_no_noise_round_{round_idx}_mlm",
        )

        trainer_mlm = Trainer(
            model=model,
            args=training_args_mlm,
            train_dataset=mlm_dataset,
            data_collator=mlm_collator,
        )
        trainer_mlm.train()

        trainer_mlm.save_model(mlm_output_dir)
        tokenizer.save_pretrained(mlm_output_dir)

        # --- Phase B: NSP Training ---
        print(f"--- [Round {round_idx}] Phase B: NSP Training ---")
        nsp_output_dir = os.path.join(args.output_dir, f"round_{round_idx}_nsp_no_noise")

        training_args_nsp = TrainingArguments(
            output_dir=nsp_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.nsp_epochs_per_round,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            learning_rate=args.learning_rate,
            logging_steps=20,
            save_strategy="epoch",
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            ddp_find_unused_parameters=True,
            dataloader_num_workers=0,
            save_safetensors=False,
            remove_unused_columns=False,
            report_to="tensorboard",
            run_name=f"dapt_no_noise_round_{round_idx}_nsp",
        )

        trainer_nsp = Trainer(
            model=model,
            args=training_args_nsp,
            train_dataset=nsp_dataset,
            data_collator=nsp_collator,
        )
        trainer_nsp.train()

        trainer_nsp.save_model(nsp_output_dir)
        tokenizer.save_pretrained(nsp_output_dir)

        print(f"Round {round_idx} completed. Checkpoint saved at {nsp_output_dir}")

    # Final Save
    final_output_dir = os.path.join(args.output_dir, "final_no_noise_model")
    print(f"All rounds finished. Saving final model to {final_output_dir}...")
    trainer_nsp.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)


if __name__ == "__main__":
    main()
