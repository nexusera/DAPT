"""
KV-NSP 训练脚本
----------------
目标：在 DAPT 阶段训练一个二分类模型，判断“键-值”文本对是否匹配。

特点：
- 使用 Hugging Face Trainer + BertForSequenceClassification（num_labels=2）。
- 数据来自标注 JSON（Label Studio 格式），通过自定义 KVDataset 动态生成负样本。
- 评估指标：Accuracy / Precision / Recall / F1。

使用示例：
python run_train.py \\
  --model_name_or_path bert-base-chinese \\
  --data_dir /Users/shanqi/Documents/BERT_DAPT/my-bert-finetune/data \\
  --output_dir ./kv_nsp_ckpt \\
  --num_train_epochs 3
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from dataset import KVDataset


# ---------------------------------------------------------------------- #
# 工具函数
# ---------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KV-NSP (Key-Value Next Sentence Prediction) 训练脚本")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-chinese",
        help="预训练模型名称或本地路径（必须可被 AutoModel / AutoTokenizer 直接加载）。",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/shanqi/Documents/BERT_DAPT/my-bert-finetune/data",
        help="标注 JSON 所在目录，默认指向当前仓库的本地数据副本。",
    )
    parser.add_argument(
        "--data_files",
        type=str,
        nargs="+",
        default=[
            "ruyuanjilu1119.json",
            "menzhenbingli1119.json",
            "shuhoubingli1119.json",
            "huojianbingli1119.json",
            "huizhenbingli1119.json",
        ],
        help="要使用的标注文件名列表（位于 data_dir 下）。",
    )
    parser.add_argument("--output_dir", type=str, default="./kv_nsp_outputs", help="模型与日志的输出目录。")
    parser.add_argument("--max_length", type=int, default=256, help="输入最大长度，句对将被截断/填充到该长度。")
    parser.add_argument("--negative_prob", type=float, default=0.5, help="生成负样本的概率。")
    parser.add_argument("--hard_negative_prob", type=float, default=0.5, help="负样本中使用倒序策略的比例。")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集占比，剩余作为验证集。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，便于复现。")

    # 训练相关超参
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数。")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="训练 batch 大小。")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="验证 batch 大小。")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="学习率。")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减。")
    parser.add_argument("--logging_steps", type=int, default=50, help="日志打印步数。")
    parser.add_argument("--save_total_limit", type=int, default=2, help="最多保留的 checkpoint 数量。")
    parser.add_argument("--eval_strategy", type=str, default="epoch", help="评估策略：'epoch' 或 'steps'。")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup 比例。")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_file_list(data_dir: Path, names: Sequence[str]) -> List[Path]:
    files = []
    for name in names:
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"未找到数据文件：{path}")
        files.append(path)
    return files


def compute_metrics(eval_pred):
    """
    Trainer 回调：计算 Accuracy / Precision / Recall / F1
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------- #
# 主流程
# ---------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    data_files = build_file_list(data_dir, args.data_files)

    # 1) 加载分词器与模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    
    # 从 MLM 模型（可能包含噪声嵌入）加载权重到 SequenceClassification 模型
    # 使用 ignore_mismatched_sizes=True 自动处理 MLM head 和 Classification head 的不匹配
    print(f"正在从预训练模型加载: {args.model_name_or_path}")
    print("注意：如果原模型是 MLM 类型，将自动提取 backbone 权重，忽略噪声嵌入层和 MLM head")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        ignore_mismatched_sizes=True,  # 自动处理 MLM head vs Classification head 的大小不匹配
    )
    print("✅ 模型加载完成（已自动处理不匹配的层）")

    # 2) 构建 Dataset（动态负采样在 __getitem__ 中完成）
    full_dataset = KVDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        max_length=args.max_length,
        negative_prob=args.negative_prob,
        hard_negative_prob=args.hard_negative_prob,
        seed=args.seed,
    )

    # 3) 划分训练 / 验证
    indices = list(range(len(full_dataset)))
    train_idx, eval_idx = train_test_split(
        indices,
        test_size=1 - args.train_ratio,
        random_state=args.seed,
        shuffle=True,
    )
    train_dataset = Subset(full_dataset, train_idx)
    eval_dataset = Subset(full_dataset, eval_idx)

    # 4) DataCollator：自动 padding（虽然 Dataset 已 max_length padding，但该 collator 不会破坏 token_type_ids）
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)

    # 5) 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluation_strategy=args.eval_strategy,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",  # 默认关闭 wandb / mlflow，避免环境依赖
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7) 开始训练与评估
    trainer.train()
    eval_metrics = trainer.evaluate()
    print("验证指标:", eval_metrics)

    # 8) 保存最终模型
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))


if __name__ == "__main__":
    main()


