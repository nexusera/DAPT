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
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("TORCHINDUCTOR_BENCHMARK_KERNEL", "0")

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BertPreTrainedModel,
    BertModel,
    BertConfig
)
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead,
    MaskedLMOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertPreTrainingHeads
)
from torch.nn import CrossEntropyLoss

# 引入本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from noise_embeddings import RobertaNoiseEmbeddings
from noise_feature_processor import NoiseFeatureProcessor, FEATURES

# 常量定义
PERFECT_VALUES = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
MAX_SEQ_LEN = 512

# ===========================
# 1. 模型定义 (No NSP, 仅保留 MLM Head)
# ===========================

# 定义 NUM_BINS (与 noise_embeddings.py 保持一致)
NUM_BINS = {
    "conf_avg": 64,
    "conf_min": 64,
    "conf_var_log": 32,
    "conf_gap": 32,
    "punct_err_ratio": 16,
    "char_break_ratio": 32,
    "align_score": 64,
}

class BertNoiseEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.noise_dim = len(FEATURES)
        self.noise_embeddings = torch.nn.ModuleDict()
        for feat in FEATURES:
            n_bins = NUM_BINS.get(feat, 64)
            self.noise_embeddings[feat] = torch.nn.Embedding(n_bins + 1, config.hidden_size)
        self.alpha = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self._reset_noise_parameters()

    def _reset_noise_parameters(self):
        for emb in self.noise_embeddings.values():
            torch.nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0, noise_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            if input_ids is not None:
                 device = input_ids.device
            else:
                 device = inputs_embeds.device
            
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
             device = input_ids.device if input_ids is not None else inputs_embeds.device
             token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        if noise_ids is not None:
             if noise_ids.dim() == 2:
                noise_ids = noise_ids.unsqueeze(0)
             noise_ids = noise_ids.to(embeddings.device)
             noise_embed = 0.0
             for i, feat in enumerate(FEATURES):
                emb_layer = self.noise_embeddings[feat]
                ids_clamped = noise_ids[:, :, i].clamp(min=0, max=emb_layer.num_embeddings - 1)
                noise_embed = noise_embed + emb_layer(ids_clamped)
             embeddings = embeddings + self.alpha * noise_embed

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertModelWithNoise(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.embeddings = BertNoiseEmbeddings(config)
    
    # Forward 方法与原版一致，透传 noise_ids
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, 
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, 
                past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, 
                return_dict=None, noise_ids=None):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        input_shape = input_ids.size() if input_ids is not None else inputs_embeds.size()[:-1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :input_shape[1]]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], input_shape[1])
                token_type_ids = buffered_token_type_ids_expanded
            else:
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

# [消融实验] 仅保留 MLM 头的模型
class BertForDaptNoNSP(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelWithNoise(config)
        self.cls = BertPreTrainingHeads(config) # 依然使用 PreTrainingHeads 但忽略 NSP 输出
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
        inputs_embeds=None, labels=None, noise_ids=None, # 移除 next_sentence_label
        output_attentions=None, output_hidden_states=None, return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, 
            return_dict=return_dict, noise_ids=noise_ids,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        
        prediction_scores, _ = self.cls(sequence_output, pooled_output) # 忽略 NSP logits
        
        total_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            total_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MaskedLMOutput(
            loss=total_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# ===========================
# 2. 数据处理 (仅需 MLM Stage)
# ===========================

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
        perfect_noise = [PERFECT_VALUES for _ in range(self.max_length)]
        for feat in features:
            nv = feat.get("noise_values") or []
            if not nv: nv = [PERFECT_VALUES for _ in range(len(feat["input_ids"]))]
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
        
        # 5. Handle Noise Padding
        seq_len = input_ids.shape[1]
        final_noise_ids = torch.zeros((len(features), seq_len, len(FEATURES)), dtype=torch.long)
        for i, row in enumerate(batch_noise_ids):
             l = min(len(row), seq_len)
             final_noise_ids[i, :l, :] = torch.tensor(row[:l], dtype=torch.long)
        batch["noise_ids"] = final_noise_ids
        
        return batch

# ===========================
# 3. 主流程
# ===========================

def main():
    parser = argparse.ArgumentParser()
    # 基础配置
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="/data/ocean/DAPT/workspace/processed_dataset")
    parser.add_argument("--tokenizer_path", type=str, default="/data/ocean/DAPT/my-medical-tokenizer")
    parser.add_argument("--noise_bins_json", type=str, default="/data/ocean/DAPT/workspace/noise_bins.json")
    # NSP相关参数在此脚本中被忽略，但保留ArgumentParser以兼容命令行调用
    parser.add_argument("--nsp_data_dir", type=str, default=None) 
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    # 训练超参
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3, help="MLM 训练的总轮数")
    
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
    print(f"MLM Samples: {len(mlm_dataset)}")

    # 3. 初始化模型
    model_path = "hfl/chinese-macbert-base"
    if args.resume_from_checkpoint:
        model_path = args.resume_from_checkpoint
        print(f"Resuming form checkpoint: {model_path}")

    model = BertForDaptNoNSP.from_pretrained(model_path)
    
    if len(tokenizer) > model.config.vocab_size:
        print(f"Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    # 4. 准备 Collators
    mlm_collator = MLMStageCollator(tokenizer, noise_processor)

    # 5. 训练循环 (只进行 MLM)
    print(f"\n{'='*40}\n [Ablation: No NSP] MLM Only Training Start \n{'='*40}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        ddp_find_unused_parameters=True, 
        dataloader_num_workers=0,
        save_safetensors=False,
        remove_unused_columns=False, 
        report_to="tensorboard",
        run_name=f"dapt_no_nsp"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mlm_dataset,
        data_collator=mlm_collator,
    )
    trainer.train()
    
    # Final Save
    final_output_dir = os.path.join(args.output_dir, "final_no_nsp_model")
    print(f"Training finished. Saving final model to {final_output_dir}...")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

if __name__ == "__main__":
    main()