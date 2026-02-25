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
# 0. 环境依赖 (与主脚本保持一致)
# ===========================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("TORCHINDUCTOR_BENCHMARK_KERNEL", "0")

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BertPreTrainedModel,
    BertModel,
    BertConfig
)
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertPreTrainingHeads
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, ModelOutput
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

PERFECT_VALUES = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# ===========================
# 1. 模型定义 (复用主脚本的模型结构，但只使用 NSP Loss)
# ===========================
# 必须复用相同的 Nosie Embeddings 定义，以免权重不兼容
NUM_BINS = {
    "conf_avg": 64, "conf_min": 64, "conf_var_log": 32, "conf_gap": 32,
    "punct_err_ratio": 16, "char_break_ratio": 32, "align_score": 64,
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
        if input_ids is not None: input_shape = input_ids.size()
        else: input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        
        if position_ids is None:
             device = input_ids.device if input_ids is not None else inputs_embeds.device
             position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device).unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
             device = input_ids.device if input_ids is not None else inputs_embeds.device
             token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if inputs_embeds is None: inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds + self.token_type_embeddings(token_type_ids) + self.position_embeddings(position_ids)

        if noise_ids is not None:
             noise_ids = noise_ids.to(embeddings.device)
             if noise_ids.dim() == 2: noise_ids = noise_ids.unsqueeze(0)
             noise_embed = 0.0
             for i, feat in enumerate(FEATURES):
                noise_embed += self.noise_embeddings[feat](noise_ids[:, :, i].clamp(min=0, max=self.noise_embeddings[feat].num_embeddings - 1))
             embeddings += self.alpha * noise_embed

        return self.dropout(self.LayerNorm(embeddings))

class BertModelWithNoise(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.embeddings = BertNoiseEmbeddings(config)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, noise_ids=None):
        # BertModel.forward does not pass noise_ids to self.embeddings, so we must manually handle it or copy the forward logic.
        # Here we copy the logic from BertModel.forward but pass noise_ids to embeddings.
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            if input_ids is not None:
                attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
            else:
                attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            noise_ids=noise_ids, # Custom arg passed here
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
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

@dataclass
class OnlyNSPOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

class BertForDaptNoMLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelWithNoise(config)
        self.cls = BertPreTrainingHeads(config)
        self.init_weights()
    
    def forward(
        self,
        input_ids=None, attention_mask=None, token_type_ids=None, next_sentence_label=None, noise_ids=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
            noise_ids=noise_ids, return_dict=True
        )
        # 依然经过 heads，但只取 seq_relationship_score (NSP)
        _, seq_relationship_score = self.cls(outputs.last_hidden_state, outputs.pooler_output)
        
        total_loss = None
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            total_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))

        return OnlyNSPOutput(loss=total_loss, logits=seq_relationship_score)

# ===========================
# 2. 数据处理 (仅需 NSP Stage)
# ===========================

class DynamicNSPDataset(Dataset):
    def __init__(self, raw_kv_dataset: KVDataset):
        self.ds = raw_kv_dataset
        self.valid_pairs_set = set(self.ds.pairs)
    def __len__(self): return len(self.ds.pairs)
    def __getitem__(self, idx):
        key_text, value_text = self.ds.pairs[idx]
        label = 1 
        if random.random() < self.ds.negative_prob:
            label = 0
            if random.random() < self.ds.hard_negative_prob:
                key_text, value_text = value_text, key_text
            else:
                for _ in range(10):
                    candidate = random.choice(self.ds.value_pool)
                    if (key_text, candidate) not in self.valid_pairs_set:
                        value_text = candidate; break
                else: value_text = candidate
        return {"text_a": key_text, "text_b": value_text, "label": label}

@dataclass
class NSPStageCollator:
    tokenizer: Any
    noise_processor: NoiseFeatureProcessor
    max_length: int = 512

    def __call__(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        text_a_list = [x["text_a"] for x in batch_items]
        text_b_list = [x["text_b"] for x in batch_items]
        labels = [x["label"] for x in batch_items]
        enc = self.tokenizer(text_a_list, text_b_list, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")
        
        bsz, seq_len = enc["input_ids"].shape
        perfect_ids_row = [0] * len(FEATURES)
        if self.noise_processor:
             perfect_ids_row = self.noise_processor.map_batch([PERFECT_VALUES])[0]
        noise_ids = torch.tensor([perfect_ids_row], dtype=torch.long).repeat(bsz, seq_len, 1)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc["token_type_ids"],
            "next_sentence_label": torch.tensor(labels, dtype=torch.long),
            "noise_ids": noise_ids
        }

# ===========================
# 3. 主流程
# ===========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--nsp_data_dir", type=str, default="/data/ocean/DAPT/data/pseudo_kv_labels_filtered.json")
    parser.add_argument("--tokenizer_path", type=str, default="/data/ocean/DAPT/my-medical-tokenizer")
    parser.add_argument("--noise_bins_json", type=str, default="/data/ocean/DAPT/workspace/noise_bins.json")
    parser.add_argument("--dataset_path", type=str, default=None) # 忽略 MLM 数据路径
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=5, help="NSP 训练的总轮数")
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    noise_processor = NoiseFeatureProcessor()
    if os.path.exists(args.noise_bins_json):
        noise_processor.load(args.noise_bins_json)
    
    print(f"Loading NSP Dataset from {args.nsp_data_dir}...")
    p = Path(args.nsp_data_dir)
    nsp_files = [p] if p.is_file() else [p / f for f in os.listdir(p) if f.endswith(".json")]
    raw_kv_dataset = KVDataset(nsp_files, tokenizer)
    nsp_dataset = DynamicNSPDataset(raw_kv_dataset)
    print(f"NSP Samples: {len(nsp_dataset)}")

    model_path = "hfl/chinese-macbert-base"
    if args.resume_from_checkpoint:
        model_path = args.resume_from_checkpoint
    
    model = BertForDaptNoMLM.from_pretrained(model_path)
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    
    nsp_collator = NSPStageCollator(tokenizer, noise_processor)

    print(f"\n{'='*40}\n [Ablation: No MLM] NSP Only Training Start \n{'='*40}")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        ddp_find_unused_parameters=True,
        dataloader_num_workers=0,
        save_safetensors=False,
        remove_unused_columns=False,
        report_to="tensorboard",
        run_name=f"dapt_no_mlm"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=nsp_dataset,
        data_collator=nsp_collator,
    )
    trainer.train()
    
    final_output_dir = os.path.join(args.output_dir, "final_no_mlm_model")
    print(f"Training finished. Saving final model to {final_output_dir}...")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

if __name__ == "__main__":
    main()