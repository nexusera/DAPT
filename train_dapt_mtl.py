import os
import math
import torch
import sys
import random
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# ===========================
# 0. 环境与依赖设置
# ===========================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
sys.path.append(current_dir)
from noise_embeddings import RobertaNoiseEmbeddings
from noise_feature_processor import NoiseFeatureProcessor, FEATURES

# 引入 KV-NSP 模块
kv_nsp_dir = os.path.join(current_dir, "kv_nsp")
if os.path.isdir(kv_nsp_dir):
    sys.path.append(kv_nsp_dir)
from dataset import KVDataset

# 常量定义
PERFECT_VALUES = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
MAX_SEQ_LEN = 512

def is_main_process():
    return int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]

# ===========================
# 1. 模型定义 (MTL 支持)
# ===========================

class RobertaModelWithNoise(RobertaModel):
    """
    带 Noise Embeddings 的 Roberta Model
    """
    def __init__(self, config, add_pooling_layer: bool = True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        # 替换 embeddings 为支持噪声输入的版本
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

@dataclass
class MultiTaskOutput(MaskedLMOutput):
    nsp_loss: Optional[torch.Tensor] = None
    mlm_loss: Optional[torch.Tensor] = None

class RobertaForDaptMTL(RobertaPreTrainedModel):
    """
    多任务学习模型：同时支持 Masked LM 和 KV-NSP。
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        
        # 1. 基础模型 (带 OCR 噪声感知)
        self.roberta = RobertaModelWithNoise(config, add_pooling_layer=True)
        
        # 2. MLM Head
        self.lm_head = RobertaLMHead(config)
        
        # 3. NSP Head (KV 匹配二分类)
        # 使用简单的分类头: Dropout -> Dense -> Tanh -> Dropout -> Out_Proj
        # (通常复用 RobertaClassificationHead 结构)
        self.nsp_head = RobertaClassificationHead(config)
        
        # 修改 nsp_head 的输出为 2 类 (Match / Not Match)
        self.nsp_head.out_proj = torch.nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,              # MLM Labels
        next_sentence_label: Optional[torch.Tensor] = None, # NSP Labels
        noise_ids: Optional[torch.Tensor] = None,           # OCR Noise
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
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

        sequence_output = outputs[0] # [batch, seq_len, hidden]
        pooled_output = outputs[1]   # [batch, hidden] (CLS token after pooler)

        total_loss = None
        mlm_loss = None
        nsp_loss = None

        # --- Task 1: MLM ---
        prediction_scores = self.lm_head(sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = mlm_loss

        # --- Task 2: KV-NSP ---
        # 并不是所有样本都有 NSP 标签，仅当 next_sentence_label 不为 None (或 -100?) 时计算
        nsp_scores = None
        if next_sentence_label is not None:
            nsp_scores = self.nsp_head(sequence_output) # RobertaClassificationHead 内部取 CLS (idx 0)
            
            # 过滤掉不需要计算 NSP 的样本 (label = -100)
            valid_mask = (next_sentence_label != -100)
            if valid_mask.any():
                loss_fct = CrossEntropyLoss()
                active_logits = nsp_scores[valid_mask]
                active_labels = next_sentence_label[valid_mask]
                nsp_loss = loss_fct(active_logits.view(-1, 2), active_labels.view(-1))
                
                if total_loss is None:
                    total_loss = nsp_loss
                else:
                    total_loss += nsp_loss # 简单的 Loss 相加，可引入权重

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MultiTaskOutput(
            loss=total_loss,
            mlm_loss=mlm_loss,
            nsp_loss=nsp_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# ===========================
# 2. 数据处理 (Collator)
# ===========================

@dataclass
class MultiTaskCollator:
    """
    负责同时处理 MLM 数据和 KV-NSP 数据。
    - MLM数据来源：传入的 dataset features (dict list)
    - NSP数据来源：self.nsp_dataset (KVDataset)
    
    策略：
    - 遍历 batch 中的每个样本
    - 以 nsp_probability 的概率，将该样本替换为从 nsp_dataset 随机抽取的一个样本
    - 否则保持 MLM 样本逻辑 (WWM + Noise)
    """
    tokenizer: Any
    nsp_dataset: KVDataset
    noise_processor: NoiseFeatureProcessor
    mlm_probability: float = 0.15
    nsp_probability: float = 0.1 # 10% 的样本被替换为 NSP 任务
    max_length: int = 512

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        batch_input_ids = []
        batch_labels = []            # For MLM
        batch_nsp_labels = []        # For NSP
        batch_token_type_ids = []
        batch_noise_ids = []
        
        # 预先生成 Perfect Noise IDs (Bucket 0)
        perfect_noise_ids = [0] * len(FEATURES)

        for feat in features:
            # 决策：是做 NSP 还是 MLM？
            is_nsp_task = (random.random() < self.nsp_probability) and (len(self.nsp_dataset.pairs) > 0)
            
            if is_nsp_task:
                # --- 构建 KV-NSP 样本 ---
                # 从 NSP Dataset 随机采一个
                # distinct from __getitem__ to allow random access during collation
                pair_idx = random.randint(0, len(self.nsp_dataset.pairs) - 1)
                
                # KVDataset 逻辑重现 (需要动态负采样)
                # 为了简便，直接调用 dataset 内部逻辑或手动从 pairs 构建
                # 这里手动构建以复用 tokenizer
                key_text, value_text = self.nsp_dataset.pairs[pair_idx]
                label = 1 # 正样本
                
                # 负采样逻辑
                if random.random() < self.nsp_dataset.negative_prob:
                    label = 0
                    if random.random() < self.nsp_dataset.hard_negative_prob:
                        # Hard negative: swap
                        key_text, value_text = value_text, key_text
                    else:
                        # Easy negative: random value
                        value_text = random.choice(self.nsp_dataset.value_pool)

                enc = self.tokenizer(
                    key_text,
                    value_text,
                    max_length=self.max_length,
                    truncation=True,
                    padding=False, # 稍后统一 pad
                    return_token_type_ids=True
                )
                
                input_ids = enc["input_ids"]
                token_type_ids = enc["token_type_ids"]
                
                # MLM Label -> -100 (Ignore)
                mlm_label = [-100] * len(input_ids)
                
                # NSP Label -> 0 or 1
                nsp_label = label
                
                # Noise -> Perfect
                # 关键修正：必须使用 PERFECT_VALUES 并经过 noise_processor 映射
                # 这样才能保证 NSP 的纯文本特征与 MLM 中的纯文本特征（非OCR路）不仅在物理意义上一致，
                # 在 ID 编码上也完全一致（例如 conf=1.0 映射到高分桶，而不是 ID=0）。
                if self.noise_processor:
                    nv_batch = [PERFECT_VALUES for _ in range(len(input_ids))]
                    noise_row = self.noise_processor.map_batch(nv_batch)
                else:
                    noise_row = [perfect_noise_ids] * len(input_ids)

            else:
                # --- 构建 MLM 样本 ---
                # 原始逻辑：WWM Masking + Noise Feature Mapping
                input_ids = feat["input_ids"]
                word_ids = feat.get("word_ids")
                
                # 1. Noise Processing
                nv = feat.get("noise_values") or []
                if not nv:
                    nv = [PERFECT_VALUES for _ in range(len(input_ids))]
                # 截断或填充
                nv = (nv + [PERFECT_VALUES] * self.max_length)[:len(input_ids)]
                noise_row = self.noise_processor.map_batch(nv) if self.noise_processor else [perfect_noise_ids] * len(input_ids)

                # 2. WWM Masking (KV-MLM 核心逻辑)
                # 复现 PrecomputedWWMCollator 的全词掩码策略：
                # 基于 word_ids 聚合 token，确保属于同一个词/实体的 tokens 同时被 mask，
                # 从而保持模型对 OCR 实体和 Key-Value 结构的敏感性。
                
                input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
                labels_tensor = input_ids_tensor.clone()
                prob_matrix = torch.full(labels_tensor.shape, self.mlm_probability)
                
                # 应用 word_ids mask
                if word_ids:
                    mapping = {}
                    for idx, wid in enumerate(word_ids):
                        if wid is None or idx >= len(input_ids): continue
                        mapping.setdefault(wid, []).append(idx)
                    
                    # 选词 mask (Whole Word Masking)
                    unique_words = list(mapping.keys())
                    num_to_mask = max(1, int(len(unique_words) * self.mlm_probability))
                    masked_words = set(random.sample(unique_words, num_to_mask))
                    mask_indices = torch.zeros(len(input_ids), dtype=torch.bool)
                    for wid in masked_words:
                        for idx in mapping[wid]:
                            mask_indices[idx] = True
                else:
                    # Fallback to random token mask (当 word_ids 缺失时)
                     mask_indices = torch.bernoulli(prob_matrix).bool()

                # Special tokens mask (CLS, SEP, etc.)
                special_tokens_mask = torch.tensor(
                    self.tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True),
                    dtype=torch.bool
                )
                mask_indices.masked_fill_(special_tokens_mask, value=False)

                # Pad tokens mask (双重保险，虽然此处通常未 Pad)
                if self.tokenizer.pad_token_id is not None:
                     mask_indices.masked_fill_(input_ids_tensor == self.tokenizer.pad_token_id, value=False)

                
                # 生成 labels (-100 for unmasked)
                labels_tensor[~mask_indices] = -100
                
                # 80% mask, 10% random, 10% original
                indices_replaced = torch.bernoulli(torch.full(labels_tensor.shape, 0.8)).bool() & mask_indices
                input_ids_tensor[indices_replaced] = self.tokenizer.mask_token_id
                indices_random = torch.bernoulli(torch.full(labels_tensor.shape, 0.5)).bool() & mask_indices & ~indices_replaced
                random_words = torch.randint(len(self.tokenizer), labels_tensor.shape, dtype=torch.long)
                input_ids_tensor[indices_random] = random_words[indices_random]
                
                input_ids = input_ids_tensor.tolist()
                mlm_label = labels_tensor.tolist()
                
                # NSP Label -> -100 (Ignore)
                nsp_label = -100
                # MLM 通常是单句或长文档，token_type_ids 全 0
                token_type_ids = [0] * len(input_ids)

            # 收集结果
            batch_input_ids.append(input_ids)
            batch_labels.append(mlm_label)
            batch_nsp_labels.append(nsp_label)
            batch_token_type_ids.append(token_type_ids)
            batch_noise_ids.append(noise_row)

        # Pad 所有序列到最大长度 (batch 内的最长 或者 self.max_length)
        # 使用 tokenizer.pad 方便
        padded = self.tokenizer.pad(
            {"input_ids": batch_input_ids, "token_type_ids": batch_token_type_ids},
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 手动 Pad labels, noise_ids
        final_input_ids = padded["input_ids"]
        bsz, seq_len = final_input_ids.shape
        
        # Pad MLM Labels (-100)
        final_labels = torch.full((bsz, seq_len), -100, dtype=torch.long)
        for i, lab in enumerate(batch_labels):
            l = min(len(lab), seq_len)
            final_labels[i, :l] = torch.tensor(lab[:l], dtype=torch.long)
            
        # Pad Noise IDs (0)
        final_noise_ids = torch.zeros((bsz, seq_len, len(FEATURES)), dtype=torch.long)
        for i, row in enumerate(batch_noise_ids):
             l = min(len(row), seq_len)
             # row is list of list, convert to tensor
             row_tensor = torch.tensor(row[:l], dtype=torch.long)
             final_noise_ids[i, :l, :] = row_tensor

        # NSP Labels (Vector)
        final_nsp_labels = torch.tensor(batch_nsp_labels, dtype=torch.long)

        return {
            "input_ids": final_input_ids,
            "attention_mask": padded["attention_mask"],
            "token_type_ids": padded["token_type_ids"],
            "labels": final_labels,
            "next_sentence_label": final_nsp_labels,
            "noise_ids": final_noise_ids
        }

# ===========================
# 3. 训练入口
# ===========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="/data/ocean/DAPT/workspace/processed_dataset")
    parser.add_argument("--nsp_data_dir", type=str, default="/data/ocean/FT_workspace/ner-finetune/data")
    parser.add_argument("--tokenizer_path", type=str, default="/data/ocean/DAPT/my-medical-tokenizer")
    parser.add_argument("--noise_bins_json", type=str, default="/data/ocean/DAPT/workspace/noise_bins.json")
    parser.add_argument("--nsp_prob", type=float, default=0.1, help="NSP 任务样本占比")
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    args = parser.parse_args()

    # 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # 准备 Noise Processor
    noise_processor = NoiseFeatureProcessor()
    if os.path.exists(args.noise_bins_json):
        noise_processor.load(args.noise_bins_json)
    elif is_main_process():
        print(f"Warning: Noise bins file {args.noise_bins_json} not found. Using default bins.")

    # 1. 加载 MLM 数据集
    dataset = load_from_disk(args.dataset_path)
    if "train" in dataset:
        train_dataset = dataset["train"]
    else:
        train_dataset = dataset

    # 2. 准备 KV-NSP 数据集
    nsp_files = []
    nsp_path_arg = Path(args.nsp_data_dir)
    
    if nsp_path_arg.is_dir():
        print(f"Scanning directory for NSP JSONs: {nsp_path_arg}")
        nsp_files = [nsp_path_arg / f for f in os.listdir(nsp_path_arg) if f.endswith(".json")]
    elif nsp_path_arg.is_file():
        print(f"Using single file for NSP: {nsp_path_arg}")
        nsp_files = [nsp_path_arg]
    elif not nsp_path_arg.exists():
        print(f"Error: NSP path does not exist: {nsp_path_arg}")
    else:
        print(f"Error: NSP path is neither file nor dir (special file?): {nsp_path_arg}")
    
    if not nsp_files:
        print(f"No NSP JSON files found in {args.nsp_data_dir}, dummy NSP dataset will be used (empty).")
        # 创建一个空 KVDataset，不再报错
        try:
            nsp_dataset = KVDataset([], tokenizer) 
        except Exception as e:
            # Fallback for old code if KVDataset still raises error
            print(f"Failed to create empty KVDataset: {e}. Using raw object.")
            class DummyDS:
                pairs = []
                value_pool = []
                negative_prob = 0.5
                hard_negative_prob = 0.5
            nsp_dataset = DummyDS()
    else:
        print(f"Loading NSP data from {len(nsp_files)} files...")
        nsp_dataset = KVDataset(nsp_files, tokenizer)
        print(f"Loaded {len(nsp_dataset.pairs)} KV pairs.")

    # 3. 初始化模型
    print("Initializing Multi-Task Model...")
    model = RobertaForDaptMTL.from_pretrained("hfl/chinese-roberta-wwm-ext")
    
    # 修改词表大小，以适应扩充了的 tokenizer
    # 关键修复：确保 embedding 层和 lm_head 的输出层都被扩展
    # 通过在 RobertaForDaptMTL 中实现了 get_output_embeddings/set_output_embeddings，
    # model.resize_token_embeddings 会自动处理 lm_head 的维度调整。
    if len(tokenizer) > model.config.vocab_size:
        print(f"Resizing token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))


    # 4. Collator
    collator = MultiTaskCollator(
        tokenizer=tokenizer,
        nsp_dataset=nsp_dataset,
        noise_processor=noise_processor,
        nsp_probability=args.nsp_prob
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_steps=2000,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        ddp_find_unused_parameters=True, # 必须为 True，因为多任务学习中某些 Head 可能在当前 Batch 不参与 Loss 计算
        remove_unused_columns=False, # 必须保留，否则 Collator 返回的自定义列会被过滤
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    print("Starting Multi-Task Training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
