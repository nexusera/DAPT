"""
KV-NSP 训练脚本（带噪声嵌入版本）
----------------
目标：在 DAPT 阶段训练一个二分类模型，判断"键-值"文本对是否匹配。
使用带噪声嵌入的模型，保持与 DAPT 训练阶段的一致性。

特点：
- 使用带噪声嵌入的 RobertaForSequenceClassificationWithNoise 模型
- 数据来自标注 JSON（Label Studio 格式），通过自定义 KVDatasetWithNoise 动态生成负样本
- 所有样本使用完美噪声值（KV-NSP 数据不是 OCR，没有真实噪声特征）
- 评估指标：Accuracy / Precision / Recall / F1

使用示例：
python run_train_with_noise.py \\
  --model_name_or_path /data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu_noise_v2/final_model \\
  --data_dir /data/ocean/FT_workspace/ner-finetune/data \\
  --output_dir ./kv_nsp_ckpt_with_noise \\
  --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \\
  --num_train_epochs 3
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Sequence, Dict, Any, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    RobertaForSequenceClassification,
    RobertaModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

# 导入噪声相关模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from noise_embeddings import RobertaNoiseEmbeddings
from noise_feature_processor import NoiseFeatureProcessor, FEATURES

from dataset_with_noise import KVDatasetWithNoise
from negative_sampling import format_negative_sampling_summary

# 完美物理值（非 OCR 样本用），由 processor 映射到桶 ID
PERFECT_VALUES = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------- #
# 带噪声嵌入的模型类
# ---------------------------------------------------------------------- #
class RobertaModelWithNoise(RobertaModel):
    """
    轻量包装：用 RobertaNoiseEmbeddings，并将 noise_ids 传入 embeddings。
    """

    def __init__(self, config, add_pooling_layer: bool = True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.embeddings = RobertaNoiseEmbeddings(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        noise_ids: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_extended_attention_mask = None
        if encoder_hidden_states is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            noise_ids=noise_ids,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class RobertaForSequenceClassificationWithNoise(RobertaForSequenceClassification):
    """
    带噪声嵌入的序列分类模型
    """

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModelWithNoise(config, add_pooling_layer=True)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        noise_ids: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            noise_ids=noise_ids,
        )

        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ---------------------------------------------------------------------- #
# 带噪声处理的 Collator
# ---------------------------------------------------------------------- #
class NoiseAwareDataCollator:
    """
    处理噪声特征的 DataCollator
    - 自动 padding noise_ids 到 batch 的最大长度
    - 确保所有样本都有 noise_ids（如果没有，使用完美噪声值）
    """

    def __init__(self, tokenizer, noise_processor: Optional[NoiseFeatureProcessor] = None):
        self.tokenizer = tokenizer
        self.noise_processor = noise_processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 提取所有字段
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        token_type_ids = [f["token_type_ids"] for f in features]
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        # Padding
        max_len = max(len(ids) for ids in input_ids)
        batch_size = len(features)

        # Padding input_ids, attention_mask, token_type_ids
        padded_input_ids = []
        padded_attention_mask = []
        padded_token_type_ids = []
        for i in range(batch_size):
            pad_len = max_len - len(input_ids[i])
            padded_input_ids.append(torch.cat([input_ids[i], torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)]))
            padded_attention_mask.append(torch.cat([attention_mask[i], torch.zeros(pad_len, dtype=torch.long)]))
            padded_token_type_ids.append(torch.cat([token_type_ids[i], torch.zeros(pad_len, dtype=torch.long)]))

        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "token_type_ids": torch.stack(padded_token_type_ids),
            "labels": labels,
        }

        # 处理 noise_ids
        noise_ids_list = []
        for f in features:
            if "noise_ids" in f:
                noise_ids = f["noise_ids"]  # [seq_len, 7]
            else:
                # 如果没有 noise_ids，使用完美噪声值
                seq_len = len(f["input_ids"])
                if self.noise_processor:
                    # 使用 processor 映射完美值到桶 ID
                    perfect_vals = [PERFECT_VALUES for _ in range(seq_len)]
                    noise_ids = torch.tensor(self.noise_processor.map_batch(perfect_vals), dtype=torch.long)
                else:
                    # 如果没有 processor，使用全 0（anchor bin）
                    noise_ids = torch.zeros((seq_len, len(FEATURES)), dtype=torch.long)

            # Padding noise_ids
            pad_len = max_len - noise_ids.shape[0]
            if pad_len > 0:
                pad_noise = torch.zeros((pad_len, len(FEATURES)), dtype=torch.long)
                noise_ids = torch.cat([noise_ids, pad_noise], dim=0)
            noise_ids_list.append(noise_ids)

        batch["noise_ids"] = torch.stack(noise_ids_list)  # [batch, max_len, 7]

        return batch


# ---------------------------------------------------------------------- #
# 工具函数
# ---------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KV-NSP (Key-Value Next Sentence Prediction) 训练脚本（带噪声嵌入）")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="预训练模型路径（DAPT 训练的 MLM 模型，包含噪声嵌入层）",
    )
    parser.add_argument(
        "--noise_bins_json",
        type=str,
        required=True,
        help="噪声分桶边界 JSON 文件路径（与 DAPT 训练时使用的相同）",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/ocean/FT_workspace/ner-finetune/data",
        help="标注 JSON 所在目录",
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
        help="要使用的标注文件名列表（位于 data_dir 下）",
    )
    parser.add_argument("--output_dir", type=str, default="./kv_nsp_outputs_with_noise", help="模型与日志的输出目录")
    parser.add_argument("--max_length", type=int, default=256, help="输入最大长度")
    parser.add_argument("--negative_prob", type=float, default=0.5, help="生成负样本的概率")
    parser.add_argument("--hard_negative_prob", type=float, default=0.5, help="负样本中使用倒序策略的比例（兼容旧参数；若设置 ratio 参数则会被覆盖）")
    parser.add_argument("--reverse_negative_ratio", type=float, default=None, help="reverse 倒序负样本权重，例如 3 表示 3:1 里的 3")
    parser.add_argument("--random_negative_ratio", type=float, default=None, help="random 随机 value 负样本权重，例如 1 表示 3:1 里的 1")
    parser.add_argument("--max_easy_retries", type=int, default=10, help="构造 random 负样本时避免真实正例的最大重试次数")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集占比")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 训练相关超参
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="训练 batch 大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="验证 batch 大小")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--logging_steps", type=int, default=50, help="日志打印步数")
    parser.add_argument("--save_total_limit", type=int, default=2, help="最多保留的 checkpoint 数量")
    parser.add_argument("--eval_strategy", type=str, default="epoch", help="评估策略：'epoch' 或 'steps'")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup 比例")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random
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

    # 1) 加载分词器
    print(f"正在加载分词器: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    # 2) 加载噪声处理器
    print(f"正在加载噪声分桶配置: {args.noise_bins_json}")
    noise_processor = NoiseFeatureProcessor.load(args.noise_bins_json)

    # 3) 计算完美噪声值对应的桶 ID
    # PERFECT_VALUES = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # conf_avg=1.0 和 conf_min=1.0 会映射到最大桶 ID（通常是 64 或 65）
    # 其他 5 个 0.0 会映射到 0（anchor bin）
    perfect_noise_vals = [PERFECT_VALUES]  # 单个样本的完美值
    perfect_noise_ids = noise_processor.map_batch(perfect_noise_vals)[0]  # [7] 维桶 ID
    print(f"完美噪声值 {PERFECT_VALUES} 映射到桶 ID: {perfect_noise_ids}")
    print(f"  说明: conf_avg 和 conf_min 的 1.0 映射到最大桶 ID，其他 0.0 映射到 anchor bin (0)")

    # 4) 加载模型（从 MLM 模型加载，提取 backbone 权重）
    print(f"正在从预训练模型加载: {args.model_name_or_path}")
    print("注意：将从 MLM 模型提取 backbone 权重，创建带噪声嵌入的分类模型")

    # 先加载配置
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = 2

    # 创建带噪声嵌入的分类模型
    model = RobertaForSequenceClassificationWithNoise(config)

    # 从 MLM 模型加载权重（包括噪声嵌入层）
    try:
        from transformers import AutoModel
        mlm_model = AutoModel.from_pretrained(args.model_name_or_path)
        # 提取 roberta 部分的权重（包括噪声嵌入层）
        roberta_state = {}
        for k, v in mlm_model.state_dict().items():
            if k.startswith("roberta."):
                # 移除 "roberta." 前缀，保留所有权重（包括噪声嵌入层）
                new_key = k.replace("roberta.", "")
                roberta_state[new_key] = v
        # 加载到分类模型的 roberta 部分（包括噪声嵌入层）
        missing_keys, unexpected_keys = model.roberta.load_state_dict(roberta_state, strict=False)
        if missing_keys:
            print(f"⚠️  部分权重未加载: {len(missing_keys)} 个键")
            if len(missing_keys) <= 10:
                print(f"   缺失的键: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️  意外的键（将被忽略）: {len(unexpected_keys)} 个")
        print("✅ 已从 MLM 模型加载权重（包括噪声嵌入层）")
    except Exception as e:
        print(f"⚠️  从 MLM 模型加载权重失败: {e}")
        print("将使用随机初始化的权重")

    # 5) 构建 Dataset（带噪声）
    full_dataset = KVDatasetWithNoise(
        data_files=data_files,
        tokenizer=tokenizer,
        max_length=args.max_length,
        negative_prob=args.negative_prob,
        hard_negative_prob=args.hard_negative_prob,
        reverse_negative_ratio=args.reverse_negative_ratio,
        random_negative_ratio=args.random_negative_ratio,
        max_easy_retries=args.max_easy_retries,
        seed=args.seed,
        perfect_noise_ids=perfect_noise_ids.tolist() if isinstance(perfect_noise_ids, np.ndarray) else perfect_noise_ids,
    )
    print(f"KV-NSP negative sampling: {format_negative_sampling_summary(full_dataset.sampling_config)}")

    # 6) 划分训练 / 验证
    indices = list(range(len(full_dataset)))
    train_idx, eval_idx = train_test_split(
        indices,
        test_size=1 - args.train_ratio,
        random_state=args.seed,
        shuffle=True,
    )
    train_dataset = Subset(full_dataset, train_idx)
    eval_dataset = Subset(full_dataset, eval_idx)

    # 7) DataCollator：处理噪声特征
    data_collator = NoiseAwareDataCollator(tokenizer=tokenizer, noise_processor=noise_processor)

    # 8) 训练参数
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
        report_to="none",
        remove_unused_columns=False,  # 重要：保留 noise_ids，否则 Trainer 会移除它
    )

    # 9) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 10) 开始训练与评估
    trainer.train()
    eval_metrics = trainer.evaluate()
    print("验证指标:", eval_metrics)

    # 11) 保存最终模型
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    print(f"✅ 模型已保存到: {os.path.join(args.output_dir, 'final_model')}")


if __name__ == "__main__":
    main()

