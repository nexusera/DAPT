import os
import math
import torch
import sys
import random
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from torch.utils.data import Dataset

# ===========================
# 0. 环境与依赖设置
# ===========================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false" # 关键修复：防止 DataLoader 死锁
os.environ.setdefault("TORCHINDUCTOR_BENCHMARK_KERNEL", "0")

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    RobertaPreTrainedModel,
    RobertaModel
)
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead,
    RobertaClassificationHead,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from torch.nn import CrossEntropyLoss

# 引入本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import paths_config as PC
from noise_embeddings import RobertaNoiseEmbeddings
from noise_feature_processor import NoiseFeatureProcessor, FEATURES

# 引入 KV-NSP 模块
kv_nsp_dir = os.path.join(current_dir, "kv_nsp")
if os.path.isdir(kv_nsp_dir) and kv_nsp_dir not in sys.path:
    sys.path.insert(0, kv_nsp_dir)
from dataset import KVDataset
from negative_sampling import format_negative_sampling_summary

# 常量定义
PERFECT_VALUES = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
MAX_SEQ_LEN = 512

def is_main_process():
    return int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]

# ===========================
# 1. 模型定义 (复用 MTL 结构以保持双头状态)
# ===========================

class RobertaModelWithNoise(RobertaModel):
    def __init__(self, config, add_pooling_layer: bool = True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.embeddings = RobertaNoiseEmbeddings(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, 
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, 
                past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, 
                return_dict=None, noise_ids=None):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        input_shape = input_ids.size() if input_ids is not None else inputs_embeds.size()[:-1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=0,
            noise_ids=noise_ids,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=self.get_extended_attention_mask(attention_mask, input_shape, device),
            head_mask=self.get_head_mask(head_mask, self.config.num_hidden_layers),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
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
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

@dataclass
class MultiTaskOutput(MaskedLMOutput):
    nsp_loss: Optional[torch.Tensor] = None
    mlm_loss: Optional[torch.Tensor] = None

class RobertaForDaptMTL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModelWithNoise(config, add_pooling_layer=True)
        self.lm_head = RobertaLMHead(config)
        self.nsp_head = RobertaClassificationHead(config)
        self.nsp_head.out_proj = torch.nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
        inputs_embeds=None, labels=None, next_sentence_label=None, noise_ids=None,
        output_attentions=None, output_hidden_states=None, return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, 
            return_dict=return_dict, noise_ids=noise_ids,
        )

        sequence_output = outputs[0]
        
        total_loss = None
        mlm_loss = None
        nsp_loss = None

        # 1. MLM Loss
        prediction_scores = self.lm_head(sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = mlm_loss

        # 2. NSP Loss
        if next_sentence_label is not None:
            nsp_scores = self.nsp_head(sequence_output)
            valid_mask = (next_sentence_label != -100)
            if valid_mask.any():
                loss_fct = CrossEntropyLoss()
                nsp_loss = loss_fct(nsp_scores[valid_mask].view(-1, 2), next_sentence_label[valid_mask].view(-1))
                if total_loss is None:
                    total_loss = nsp_loss
                elif mlm_loss is not None:
                    # 如果同时计算（虽然本脚本倾向于交替），则相加
                    total_loss += nsp_loss
                else:
                    total_loss = nsp_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MultiTaskOutput(
            loss=total_loss, mlm_loss=mlm_loss, nsp_loss=nsp_loss, logits=prediction_scores,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )

# ===========================
# 2. 数据处理 (分阶段独立)
# ===========================

# --- A. MLM 专用的 Collator (从 NoiseAwareCollator 简化而来) ---
@dataclass
class MLMStageCollator:
    tokenizer: Any
    noise_processor: NoiseFeatureProcessor
    mlm_probability: float = 0.15
    max_length: int = 512

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. 提取基础数据
        batch_input_ids = [f["input_ids"] for f in features]
        batch_word_ids = [f.get("word_ids") for f in features]
        
        # 2. Noise Processing
        batch_noise_ids = []
        perfect_noise = [PERFECT_VALUES for _ in range(self.max_length)] # Pre-alloc
        for feat in features:
            nv = feat.get("noise_values") or []
            if not nv: nv = [PERFECT_VALUES for _ in range(len(feat["input_ids"]))]
            # 截断或填充
            nv = (nv + perfect_noise)[:len(feat["input_ids"])]
            ids = self.noise_processor.map_batch(nv) if self.noise_processor else [[0]*len(FEATURES)]*len(feat["input_ids"])
            batch_noise_ids.append(ids)

        # 3. Padding Input Ids
        batch = self.tokenizer.pad(
            {"input_ids": batch_input_ids},
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # 4. WWM Masking Strategy
        for i in range(len(features)):
            word_ids = batch_word_ids[i]
            current_ids = input_ids[i]
            if word_ids:
                mapping = {}
                for idx, wid in enumerate(word_ids):
                    if wid is None or idx >= len(current_ids): continue
                    mapping.setdefault(wid, []).append(idx)
                
                unique_words = list(mapping.keys())
                num_to_mask = max(1, int(len(unique_words) * self.mlm_probability))
                masked_words = set(random.sample(unique_words, num_to_mask))
                mask_indices = torch.zeros(len(current_ids), dtype=torch.bool)
                for wid in masked_words:
                    for idx in mapping[wid]: mask_indices[idx] = True
            else:
                mask_indices = torch.bernoulli(probability_matrix[i]).bool()

            # Special tokens & Pad tokens mask
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
        
        # 5. Handle Noise Padding & Tensor conversion
        seq_len = input_ids.shape[1]
        final_noise_ids = torch.zeros((len(features), seq_len, len(FEATURES)), dtype=torch.long)
        for i, row in enumerate(batch_noise_ids):
             l = min(len(row), seq_len)
             final_noise_ids[i, :l, :] = torch.tensor(row[:l], dtype=torch.long)
        batch["noise_ids"] = final_noise_ids
        
        # MLM 阶段不需要 NSP Label
        batch["next_sentence_label"] = None 

        return batch


# --- B. NSP 专用的 Dataset 和 Collator ---

class DynamicNSPDataset(Dataset):
    """
    包装原始 KVDataset，实现实时动态负采样 (Dynamic Negative Sampling)
    """
    def __init__(self, raw_kv_dataset: KVDataset):
        self.ds = raw_kv_dataset
        # Optimize: 预先构建 ground truth lookup set，防止 False Negative
        self.valid_pairs_set = getattr(self.ds, "valid_pairs_set", set(self.ds.pairs))
    
    def __len__(self):
        return len(self.ds.pairs)
    
    def __getitem__(self, idx):
        key_text, value_text, label, _ = self.ds.sample_text_pair(idx, valid_pairs_set=self.valid_pairs_set)
        
        return {
            "text_a": key_text,
            "text_b": value_text,
            "label": label
        }

@dataclass
class NSPStageCollator:
    tokenizer: Any
    noise_processor: NoiseFeatureProcessor
    max_length: int = 512

    def __call__(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        text_a_list = [x["text_a"] for x in batch_items]
        text_b_list = [x["text_b"] for x in batch_items]
        labels = [x["label"] for x in batch_items]

        # Tokenize pair
        # 核心修改：利用 Tokenizer 自动生成 token_type_ids (000...111...)
        # BERT 类 Tokenizer 这里会自动处理：
        # [CLS] A [SEP] B [SEP]
        #  0    0   0   1   1
        enc = self.tokenizer(
            text_a_list,
            text_b_list,
            max_length=self.max_length,
            truncation=True, # 默认只截断第二个序列，或按 max_length 截断
            padding="longest", # 显式指定 padding 策略，避免模糊
            return_tensors="pt"
        )
        
        # 强制检查：如果 Tokenizer 是 RoBERTa 类型，它可能默认不返回 token_type_ids 或全返回 0
        # 我们需要手动修正 token_type_ids 以提供结构信息
        input_ids = enc["input_ids"]
        token_type_ids = enc.get("token_type_ids")

        if token_type_ids is None or token_type_ids.sum() == 0:
            # 手动构造 Segment Embeddings
            # RoBERTa 的 sep_token_id 通常是 2
            sep_id = self.tokenizer.sep_token_id
            token_type_ids = torch.zeros_like(input_ids)
            
            for i in range(input_ids.shape[0]):
                # 找到第一个 [SEP] 的位置
                sep_indices = (input_ids[i] == sep_id).nonzero(as_tuple=True)[0]
                if len(sep_indices) >= 2:
                    # 第一个 SEP 之后的内容全部设为 1
                    first_sep_idx = sep_indices[0]
                    # 从 first_sep_idx + 1 开始到结尾（或即使是 padding 其实也不影响，Attention mask 会屏蔽）
                    # 但严谨起见，我们只把第二个分段设为 1
                    token_type_ids[i, first_sep_idx + 1 :] = 1
                    
                    # 将 padding 部分还原为 0 (可选，Standard BERT 里 Padding 的 type id 通常也是 0)
                    if self.tokenizer.pad_token_id is not None:
                         token_type_ids[i, input_ids[i] == self.tokenizer.pad_token_id] = 0
            
            # 将手动构造的 type ids 塞回
            enc["token_type_ids"] = token_type_ids
        
        # Prepare Inputs
        bsz, seq_len = enc["input_ids"].shape
        
        # Noise Ids: 对于 NSP 这种纯文本任务，我们统一使用 "Perfect" 噪声特征
        # 确保其输入分布与 MLM 阶段的“Text Only”部分一致
        perfect_ids_row = [0] * len(FEATURES) # Default bin 0 for perfect
        if self.noise_processor:
             # 如果 processor 存在，映射一次 Perfect Values 确保 ID 正确
             perfect_ids_row = self.noise_processor.map_batch([PERFECT_VALUES])[0]
             
        # 修复：pin_memory 报错 "more than one element... refers to a single memory location"
        # .expand() 创建的是共享内存的视图，这在 PyTorch pin_memory 中是不允许的。
        # 必须使用 .repeat() 来物理复制数据，或者确保每个样本都有独立的内存空间。
        noise_ids = torch.tensor([perfect_ids_row], dtype=torch.long).repeat(bsz, seq_len, 1)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc["token_type_ids"],
            "next_sentence_label": torch.tensor(labels, dtype=torch.long),
            "labels": None, # MLM Label 为空
            "noise_ids": noise_ids
        }


# ===========================
# 3. 主流程
# ===========================

def main():
    parser = argparse.ArgumentParser()
    # 基础配置
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=PC.DATASET_PATH)
    parser.add_argument("--nsp_data_dir", type=str, default=PC.NSP_DATA_PATH)
    parser.add_argument("--tokenizer_path", type=str, default=PC.TOKENIZER_PATH)
    parser.add_argument("--noise_bins_json", type=str, default=PC.NOISE_BINS_JSON)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    # 训练超参
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_rounds", type=int, default=3, help="交替训练的总轮数 (MLM -> NSP -> MLM -> NSP ...)")
    parser.add_argument("--mlm_epochs_per_round", type=int, default=1, help="每轮 MLM 训练的 epoch 数")
    parser.add_argument("--nsp_epochs_per_round", type=int, default=3, help="每轮 NSP 训练的 epoch 数")
    parser.add_argument("--nsp_negative_prob", type=float, default=0.5, help="KV-NSP 中把正样本改造成负样本的总概率。")
    parser.add_argument("--nsp_reverse_negative_ratio", type=float, default=1.0, help="KV-NSP 负样本里 reverse 倒序策略的权重。")
    parser.add_argument("--nsp_random_negative_ratio", type=float, default=1.0, help="KV-NSP 负样本里 random 随机 value 策略的权重。")
    parser.add_argument("--nsp_max_easy_retries", type=int, default=10, help="构造 random 负样本时避免真实正例的最大重试次数。")
    
    args = parser.parse_args()

    # 1. 资源准备
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    noise_processor = NoiseFeatureProcessor()
    if os.path.exists(args.noise_bins_json):
        noise_processor.load(args.noise_bins_json)
    
    # 2. 准备数据集
    print(f"Loading MLM Dataset from {args.dataset_path}...")
    mlm_dataset_disk = load_from_disk(args.dataset_path)
    mlm_dataset = mlm_dataset_disk["train"] if "train" in mlm_dataset_disk else mlm_dataset_disk
    
    print(f"Loading NSP Dataset from {args.nsp_data_dir}...")
    # 兼容单个文件或目录
    p = Path(args.nsp_data_dir)
    nsp_files = [p] if p.is_file() else [p / f for f in os.listdir(p) if f.endswith(".json")]
    raw_kv_dataset = KVDataset(
        nsp_files,
        tokenizer,
        negative_prob=args.nsp_negative_prob,
        reverse_negative_ratio=args.nsp_reverse_negative_ratio,
        random_negative_ratio=args.nsp_random_negative_ratio,
        max_easy_retries=args.nsp_max_easy_retries,
    )
    nsp_dataset = DynamicNSPDataset(raw_kv_dataset) # 包装成 torch Dataset
    print(f"NSP negative sampling: {format_negative_sampling_summary(raw_kv_dataset.sampling_config)}")
    
    print(f"MLM Samples: {len(mlm_dataset)}, NSP Samples: {len(nsp_dataset)}")

    # 3. 初始化模型
    # 尝试加载最新 checkpint 或者 基座
    model_path = "hfl/chinese-roberta-wwm-ext"
    if args.resume_from_checkpoint:
        model_path = args.resume_from_checkpoint
        print(f"Resuming form checkpoint: {model_path}")

    model = RobertaForDaptMTL.from_pretrained(model_path)
    
    # Resize Token Embeddings (Safe Logic)
    if len(tokenizer) > model.config.vocab_size:
        print(f"Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    # 4. 准备 Collators
    mlm_collator = MLMStageCollator(tokenizer, noise_processor)
    nsp_collator = NSPStageCollator(tokenizer, noise_processor)

    # 5. 交替训练循环
    for round_idx in range(1, args.num_rounds + 1):
        print(f"\n{'='*40}\n Round {round_idx}/{args.num_rounds} Start \n{'='*40}")

        # --- Phase A: MLM Training ---
        print(f"--- [Round {round_idx}] Phase A: MLM Training ---")
        mlm_output_dir = os.path.join(args.output_dir, f"round_{round_idx}_mlm")
        
        training_args_mlm = TrainingArguments(
            output_dir=mlm_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.mlm_epochs_per_round,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            learning_rate=args.learning_rate,
            logging_steps=50,
            save_strategy="epoch", # 每轮只在结束时保存，避免中间文件过多
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            # DDP Settings
            ddp_find_unused_parameters=True, 
            dataloader_num_workers=0, # 彻底禁用多进程 Loader，解决死锁/IPC崩溃问题
            save_safetensors=False,    # M2: 自定义模型含共享权重，safetensors 会因 shared-tensor 检查报错；待上游修复后可移除
            remove_unused_columns=False, 
            report_to="tensorboard",
            run_name=f"dapt_round_{round_idx}_mlm"
        )

        trainer_mlm = Trainer(
            model=model,
            args=training_args_mlm,
            train_dataset=mlm_dataset,
            data_collator=mlm_collator,
        )
        trainer_mlm.train()
        
        # 保存 Phase A 结果作为中间态
        trainer_mlm.save_model(mlm_output_dir)
        tokenizer.save_pretrained(mlm_output_dir)
        
        # --- Phase B: NSP Training ---
        print(f"--- [Round {round_idx}] Phase B: NSP Training ---")
        nsp_output_dir = os.path.join(args.output_dir, f"round_{round_idx}_nsp")
        
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
            dataloader_num_workers=0, # 同步禁用
            save_safetensors=False,    # M2: 自定义模型含共享权重，safetensors 会因 shared-tensor 检查报错；待上游修复后可移除
            remove_unused_columns=False, # 关键修复
            report_to="tensorboard",
            run_name=f"dapt_round_{round_idx}_nsp"
        )
        
        trainer_nsp = Trainer(
            model=model, # 复用同一个 model 对象，权重持续更新
            args=training_args_nsp,
            train_dataset=nsp_dataset,
            data_collator=nsp_collator,
        )
        trainer_nsp.train()
        
        # 保存 Phase B 结果
        trainer_nsp.save_model(nsp_output_dir)
        tokenizer.save_pretrained(nsp_output_dir)
        
        print(f"Round {round_idx} completed. Checkpoint saved at {nsp_output_dir}")

    # Final Save
    final_output_dir = os.path.join(args.output_dir, "final_staged_model")
    print(f"All rounds finished. Saving final model to {final_output_dir}...")
    trainer_nsp.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

if __name__ == "__main__":
    main()
