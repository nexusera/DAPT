import os
import math
import torch
import sys
import random
import argparse
import numpy as np
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, RandomSampler

# ===========================
# 0. 基础环境设置
# ===========================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 开启一些 PyTorch 优化标志
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    RobertaForMaskedLM,
    RobertaModel,
    PreTrainedTokenizerBase
)
# 引入本地模块 (假设脚本在 DAPT 目录下)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from noise_embeddings import RobertaNoiseEmbeddings
from noise_feature_processor import NoiseFeatureProcessor, FEATURES

# 复用我们自定义的带噪声嵌入的模型类
class RobertaForMaskedLMWithNoise(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        # 替换 embeddings 层为支持噪声输入的版本
        self.roberta.embeddings = RobertaNoiseEmbeddings(config)

PERFECT_VALUES = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# ===========================
# 1. 定义两类数据集
# ===========================

class GeneralMLMDataset(Dataset):
    """
    包装之前的通用预训练数据集 (Apache Arrow格式)
    """
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        # 返回原始 item, 打上标记
        item = self.hf_dataset[idx]
        return {
            "type": "general_mlm",
            "data": item
        }

class KVStructuralDataset(Dataset):
    """
    读取 JSON 标注数据，专门用于 KV 对抗训练
    """
    def __init__(self, data_files: List[str], max_limit=None):
        self.pairs: List[Tuple[str, str]] = [] # (Key, Value)
        self.load_data(data_files)
        if max_limit:
            self.pairs = self.pairs[:max_limit]
        print(f"Loaded {len(self.pairs)} KV pairs from json files.")

    def load_data(self, files):
        for fpath in files:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    self.extract_kv(item)
    
    def extract_kv(self, item):
        # 简化的 Label Studio 解析逻辑
        annotations = item.get("annotations", [])
        if not annotations: return
        try:
            res = annotations[0].get("result", [])
            # 建立 id -> {text, label} 映射
            entities = {}
            relations = []
            for r in res:
                if r['type'] == 'labels':
                    lbl = r['value']['labels'][0]
                    txt = r['value']['text']
                    entities[r['id']] = {'label': lbl, 'text': txt}
                elif r['type'] == 'relation':
                    relations.append((r['from_id'], r['to_id']))
            
            # 找到 Key->Value 关系
            for fid, tid in relations:
                src = entities.get(fid)
                tgt = entities.get(tid)
                if src and tgt and src['label'] == '键名' and tgt['label'] == '值':
                    self.pairs.append((src['text'], tgt['text']))
        except:
            pass

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        key_text, value_text = self.pairs[idx]
        return {
            "type": "kv_span",
            "data": {
                "key": key_text,
                "value": value_text
            }
        }

# ===========================
# 2. 混合策略 DataCollator (核心逻辑)
# ===========================

@dataclass
class HybridMaskingCollator:
    tokenizer: PreTrainedTokenizerBase
    noise_processor: NoiseFeatureProcessor
    mlm_probability: float = 0.15
    max_length: int = 512
    
    def __call__(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        处理混合的 batch。
        - 对于 type='general_mlm'：执行标准的随机 Token Masking。
        - 对于 type='kv_span'：执行结构化对抗 Masking (全Key或全Value掩码)。
        """
        
        # 1. 分离数据
        general_items = [b['data'] for b in batch_items if b['type'] == 'general_mlm']
        kv_items = [b['data'] for b in batch_items if b['type'] == 'kv_span']
        
        # 结果容器
        final_input_ids = []
        final_labels = []
        final_noise_ids = []
        
        # --- 处理 A: 通用 MLM 数据 (随机 Mask) ---
        if general_items:
            # 复用之前的逻辑 (简化版)
            batch_input_ids = [f["input_ids"] for f in general_items]
            
            # 处理噪声特征
            batch_noise_vals = []
            perfect_noise = [PERFECT_VALUES for _ in range(self.max_length)]
            for feat in general_items:
                nv = feat.get("noise_values") or []
                nv = (nv + perfect_noise)[:len(feat["input_ids"])]
                batch_noise_vals.append(self.noise_processor.map_batch(nv))

            # Padding
            padded = self.tokenizer.pad(
                {"input_ids": batch_input_ids},
                padding="longest",
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = padded["input_ids"]
            labels = input_ids.clone()
            
            # 随机 Mask 逻辑 (WWMM - Whole Word Masking 这里简化为随机Token以节省篇幅，核心是混合)
            # 生产环境建议带上 Word Masking 逻辑
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
            if self.tokenizer.pad_token_id is not None:
                probability_matrix.masked_fill_(input_ids == self.tokenizer.pad_token_id, value=0.0)
            
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100 # -100 表示不计算 Loss
            
            # 80% [MASK], 10% Random, 10% Original
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            input_ids[indices_replaced] = self.tokenizer.mask_token_id
            
            # 处理 Noise Ids padding
            seq_len = input_ids.shape[1]
            noise_ids = torch.zeros((len(general_items), seq_len, len(FEATURES)), dtype=torch.long)
            for i, row in enumerate(batch_noise_vals):
                l = min(len(row), seq_len)
                noise_ids[i, :l, :] = torch.tensor(row[:l], dtype=torch.long)

            final_input_ids.append(input_ids)
            final_labels.append(labels)
            final_noise_ids.append(noise_ids)

        # --- 处理 B: KV 对抗数据 (Span Mask) ---
        if kv_items:
            # 构造句子: "Key: Value"
            texts = [f"{item['key']}:{item['value']}" for item in kv_items]
            
            # Tokenize
            enc = self.tokenizer(
                texts, 
                max_length=self.max_length, 
                padding="longest", 
                truncation=True, 
                return_tensors="pt",
                return_offsets_mapping=True # 关键：我们需要知道 token 对应的原始字符位置
            )
            kv_input_ids = enc["input_ids"]
            kv_labels = kv_input_ids.clone() # 先复制一份作为 label
            kv_labels[:] = -100 # 默认全部不计算 Loss
            
            bsz, seq_len = kv_input_ids.shape
            
            # 遍历每个样本，决定是 Mask Key 还是 Mask Value
            for i in range(bsz):
                k_text = kv_items[i]['key']
                v_text = kv_items[i]['value']
                
                # 策略：50% 概率 Mask Key，50% 概率 Mask Value
                mask_target = "value" if random.random() > 0.5 else "key"
                
                # 寻找 Key 或 Value 在 token 序列中的范围
                # 简单做法：我们知道格式是 "Key:Value"，我们利用 offsets_mapping
                offsets = enc["offset_mapping"][i]
                
                # 确定字符级边界
                # 文本 = Key + ":" + Value
                # Key 范围: [0, len(Key))
                # Value 范围: [len(Key)+1, len(Key)+1+len(Value))
                
                k_start_char, k_end_char = 0, len(k_text)
                v_start_char, v_end_char = len(k_text) + 1, len(k_text) + 1 + len(v_text)
                
                target_start = v_start_char if mask_target == "value" else k_start_char
                target_end = v_end_char if mask_target == "value" else k_end_char
                
                # 在 Token 序列中寻找对应的 Token
                # 遍历所有 token 的 char_span，如果有交集，则 Mask
                mask_indices = []
                for token_idx, (start, end) in enumerate(offsets):
                    if start == end == 0: continue # Skip special tokens or padding
                    
                    # 判断 token 是否落在目标区间内
                    # 只要 token 的大部分内容在 target 区间里，就 mask
                    token_center = (start + end) / 2
                    if target_start <= token_center < target_end:
                         mask_indices.append(token_idx)
                
                if mask_indices:
                    # 执行 Mask 操作
                    for idx in mask_indices:
                        kv_input_ids[i, idx] = self.tokenizer.mask_token_id # 替换为 [MASK]
                        kv_labels[i, idx] = enc["input_ids"][i, idx] # 把原始 ID 放入 Label
                
                # 注意：我们这里不需要处理 random replacement (10%) 等策略
                # 因为这是一个强对抗任务，我们希望模型确切地填出被 Mask 的部分
            
            # 处理这部分的 Noise Ids (使用 Perfect Noise)
            # 因为这里是纯净的 KV 对，我们假设它是高质量的
            full_noise_row = self.noise_processor.map_batch([PERFECT_VALUES])[0]
            kv_noise_ids = torch.tensor([full_noise_row], dtype=torch.long).repeat(bsz, seq_len, 1)

            final_input_ids.append(kv_input_ids)
            final_labels.append(kv_labels)
            final_noise_ids.append(kv_noise_ids)
        
        # 3. 合并 Batch
        # 注意：General 和 KV 的 seq_len 可能不同 (因为 padding="longest")
        # 需要再次 pad 到统一长度
        max_batch_len = 0
        for t in final_input_ids: max_batch_len = max(max_batch_len, t.shape[1])
        
        def pad_tensor(t, target_len, pad_val):
            # t: [bsz, len, ...]
            if t.shape[1] >= target_len: return t
            pad_shape = list(t.shape)
            pad_shape[1] = target_len - t.shape[1]
            padding = torch.full(pad_shape, pad_val, dtype=t.dtype, device=t.device)
            return torch.cat([t, padding], dim=1)

        if len(final_input_ids) > 1:
            input_ids = torch.cat([pad_tensor(t, max_batch_len, self.tokenizer.pad_token_id) for t in final_input_ids])
            labels = torch.cat([pad_tensor(t, max_batch_len, -100) for t in final_labels])
            noise_ids = torch.cat([pad_tensor(t, max_batch_len, 0) for t in final_noise_ids]) # noise bin 0 usually fine or specific pad bin
        else:
            input_ids = final_input_ids[0]
            labels = final_labels[0]
            noise_ids = final_noise_ids[0]

        return {
            "input_ids": input_ids,
            "labels": labels,  # HuggingFace 会自动计算 labels 中非 -100 部分的 CrossEntropy
            "noise_ids": noise_ids
        }


# ===========================
# 3. 主流程
# ===========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="/data/ocean/DAPT/workspace/processed_dataset")
    parser.add_argument("--kv_json_path", type=str, default="/data/ocean/DAPT/data/pseudo_kv_labels_filtered.json")
    parser.add_argument("--tokenizer_path", type=str, default="/data/ocean/DAPT/my-medical-tokenizer")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="上一轮 MLM 的 Checkpoint")
    parser.add_argument("--noise_bins_json", type=str, default="/data/ocean/DAPT/workspace/noise_bins.json")
    
    # 混合比例: 多少个 KV 样本对应多少个 General 样本
    # 简单起见，我们通过 Upsampling (重复采样) KV 数据集来控制比例
    parser.add_argument("--mix_ratio", type=float, default=0.2, help="KV 数据占总 Batch 的比例 (0.0 - 1.0)")
    parser.add_argument("--num_epochs", type=int, default=3)
    
    args = parser.parse_args()

    # 1. 加载资源
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    noise_processor = NoiseFeatureProcessor()
    if os.path.exists(args.noise_bins_json):
        noise_processor.load(args.noise_bins_json)

    # 2. 加载数据集
    print("Loading General MLM Dataset...")
    disk_ds = load_from_disk(args.dataset_path)
    train_ds = disk_ds["train"] if "train" in disk_ds else disk_ds
    general_ds = GeneralMLMDataset(train_ds)
    
    print("Loading KV Structural Dataset...")
    # 支持单个文件或目录
    p = Path(args.kv_json_path)
    files = [p] if p.is_file() else [p / f for f in os.listdir(p) if f.endswith(".json")]
    kv_ds = KVStructuralDataset(files)
    
    # 3. 混合数据集 (通过 ConcatDataset 实现)
    # 计算需要的 KV 样本数量以达到 mix_ratio
    # N_kv / (N_gen + N_kv) = ratio  => N_kv = N_gen * ratio / (1 - ratio)
    current_kv_len = len(kv_ds)
    match_kv_len = int(len(general_ds) * args.mix_ratio / (1 - args.mix_ratio))
    
    if current_kv_len < match_kv_len:
        repeat_times = match_kv_len // current_kv_len + 1
        print(f"Upsampling KV data {repeat_times} times to match mix ratio {args.mix_ratio}...")
        kv_ds = ConcatDataset([kv_ds] * repeat_times)
        # 裁剪到精确数量
        # kv_ds = torch.utils.data.Subset(kv_ds, range(match_kv_len)) 
    
    print(f"Final Dataset Size: General={len(general_ds)}, KV={len(kv_ds)}")
    full_dataset = ConcatDataset([general_ds, kv_ds])
    
    # 4. 初始化模型
    model = RobertaForMaskedLMWithNoise.from_pretrained(args.model_name_or_path or args.tokenizer_path)

    # 5. Trainer 配置 (对齐 train_dapt_distributed.py 的训练参数以进行消融实验)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=16,          # Aligned: 16
        gradient_accumulation_steps=4,           # Aligned: 4
        learning_rate=8e-5,                      # Aligned: 8e-5
        weight_decay=0.01,                       # Aligned: 0.01
        warmup_ratio=0.05,                       # Aligned: 0.05
        bf16=True,                               # Aligned: 使用 bf16 (H200)
        tf32=True,                               # Aligned
        gradient_checkpointing=True,             # Aligned
        save_strategy="steps",
        save_steps=100,                          # Aligned: 100
        save_total_limit=3,                      # Aligned
        logging_steps=10,                        # Aligned: 10
        remove_unused_columns=False,             # 必须为 False
        dataloader_num_workers=4,                # 适当开启多进程加速数据处理，如遇死锁请改回0
        ddp_find_unused_parameters=False,        # Aligned
        report_to="none"
    )

    collator = HybridMaskingCollator(
        tokenizer=tokenizer,
        noise_processor=noise_processor,
        max_length=512
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        data_collator=collator
    )

    print("Starting Hybrid Training...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}/final_hybrid_model")
    trainer.save_model(os.path.join(args.output_dir, "final_hybrid_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_hybrid_model"))

if __name__ == "__main__":
    main()
