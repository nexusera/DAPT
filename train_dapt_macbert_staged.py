import os
import math
import torch
import sys
import random
import argparse
import logging
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from torch.utils.data import Dataset

# ===========================
# 0. 环境与依赖设置
# ===========================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# PyTorch allocator config (new name: PYTORCH_ALLOC_CONF; old name kept for compatibility)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_ALLOC_CONF", "expandable_segments:True"))
os.environ["TOKENIZERS_PARALLELISM"] = "false" # 关键修复：防止 DataLoader 死锁
os.environ.setdefault("TORCHINDUCTOR_BENCHMARK_KERNEL", "0")

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    RobertaPreTrainedModel,
    RobertaModel,
    BertPreTrainedModel,
    BertModel,
    BertConfig
)
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead,
    RobertaClassificationHead,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertPreTrainingHeads
)
from torch.nn import CrossEntropyLoss

# 抑制 tokenization 的高频截断提示，避免日志 I/O 拖慢训练
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# 引入本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from noise_embeddings import RobertaNoiseEmbeddings
from noise_feature_processor import NoiseFeatureProcessor, FEATURES
from noise_fusion import ContinuousNoiseProjector, uses_bucket_noise, uses_continuous_noise

# 引入 KV-NSP 模块
kv_nsp_dir = os.path.join(current_dir, "kv_nsp")
if os.path.isdir(kv_nsp_dir):
    sys.path.append(kv_nsp_dir)
from dataset import KVDataset
from negative_sampling import format_negative_sampling_summary

# 常量定义
PERFECT_VALUES = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
MAX_SEQ_LEN = 512


@contextmanager
def _temporarily_unset_env(name: str):
    old = os.environ.get(name)
    if name in os.environ:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        if old is not None:
            os.environ[name] = old


def _export_fast_tokenizer_if_possible(tokenizer_dir: str, probe_text: str = "肿瘤标志物") -> None:
    """Generate tokenizer.json for downstream offset_mapping.

    We pretrain with slow tokenizers (use_fast=False) for stability.
    Downstream KV-NER/EBQA uses fast tokenizers for return_offsets_mapping.
    This function tries to ensure the saved model directory contains a healthy fast backend.

    Never raises: on failure, prints a warning and continues.
    """
    try:
        with _temporarily_unset_env("TRANSFORMERS_NO_FAST_TOKENIZER"):
            tok_fast = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)

        if not getattr(tok_fast, "is_fast", False):
            print(f"(warn) fast tokenizer export skipped: is_fast=False for {tokenizer_dir}")
            return

        pieces = tok_fast.tokenize(probe_text)
        if len(pieces) == 1 and pieces[0] == tok_fast.unk_token:
            print(
                f"(warn) fast tokenizer looks broken for {tokenizer_dir}: "
                f"probe={probe_text!r} -> {pieces}. tokenizer.json not written."
            )
            return

        # Ensure offset_mapping is available
        try:
            enc = tok_fast(probe_text, add_special_tokens=False, return_offsets_mapping=True)
            if not enc.get("offset_mapping"):
                print(
                    f"(warn) fast tokenizer missing offset_mapping for {tokenizer_dir}; "
                    "tokenizer.json not written."
                )
                return
        except Exception as e:
            print(f"(warn) fast tokenizer return_offsets_mapping failed for {tokenizer_dir}: {e}")
            return

        tok_fast.save_pretrained(tokenizer_dir)
        print(f"Exported fast tokenizer.json to: {tokenizer_dir}")
    except Exception as e:
        print(f"(warn) failed to export fast tokenizer for {tokenizer_dir}: {e}")


def _looks_like_tokenizer(obj: Any) -> bool:
    return hasattr(obj, "tokenize") and hasattr(obj, "convert_tokens_to_ids")


def _load_tokenizer_with_fallback(tokenizer_path: str):
    """Load tokenizer robustly.

    Preferred for pretraining is slow tokenizer (use_fast=False) for stability.
    However, some environments may behave unexpectedly (e.g. returning a non-tokenizer object).
    In that case, fall back to a fast tokenizer.
    """
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        if not isinstance(tok, bool) and _looks_like_tokenizer(tok):
            return tok
        print(
            f"(warn) slow tokenizer load returned {type(tok).__name__}; falling back to fast tokenizer: {tokenizer_path}"
        )
    except Exception as e:
        print(f"(warn) failed to load slow tokenizer ({e}); falling back to fast tokenizer: {tokenizer_path}")

    tok_fast = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if not _looks_like_tokenizer(tok_fast):
        raise TypeError(f"Failed to load a usable tokenizer from: {tokenizer_path}")
    return tok_fast


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def is_main_process():
    return int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]

# ===========================
# 1. 模型定义 (复用 MTL 结构以保持双头状态)
# ===========================

# ===========================
# 1. 模型定义 (适配 MacBERT/BERT 架构)
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
        self.noise_mode = str(getattr(config, "noise_mode", "bucket") or "bucket").lower()
        self.noise_mlp_hidden_dim = getattr(config, "noise_mlp_hidden_dim", None)
        self.noise_bin_edges = getattr(config, "noise_bin_edges", None)
        self.noise_embeddings = None
        self.noise_projector = None
        if uses_bucket_noise(self.noise_mode):
            self.noise_embeddings = torch.nn.ModuleDict()
            for feat in FEATURES:
                n_bins = NUM_BINS.get(feat, 64)
                self.noise_embeddings[feat] = torch.nn.Embedding(n_bins + 1, config.hidden_size)
        elif uses_continuous_noise(self.noise_mode):
            self.noise_projector = ContinuousNoiseProjector(
                config.hidden_size,
                mode=self.noise_mode,
                dropout=getattr(config, "hidden_dropout_prob", 0.1),
                mlp_hidden_dim=self.noise_mlp_hidden_dim,
                feature_ranges=self.noise_bin_edges,
            )
        else:
            raise ValueError(f"Unsupported noise_mode: {self.noise_mode}")
        self.alpha = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self._reset_noise_parameters()

    def _reset_noise_parameters(self):
        if self.noise_embeddings is not None:
            for emb in self.noise_embeddings.values():
                torch.nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0, noise_ids=None, noise_values=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            if input_ids is not None:
                 # Create position ids from input ids usually, simpler approach for standard BERT:
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

        if uses_bucket_noise(self.noise_mode) and noise_ids is not None:
            if noise_ids.dim() == 2:
                noise_ids = noise_ids.unsqueeze(0)
            noise_ids = noise_ids.to(embeddings.device)
            noise_embed = 0.0
            for i, feat in enumerate(FEATURES):
                emb_layer = self.noise_embeddings[feat]
                ids_clamped = noise_ids[:, :, i].clamp(min=0, max=emb_layer.num_embeddings - 1)
                noise_embed = noise_embed + emb_layer(ids_clamped)
            embeddings = embeddings + self.alpha * noise_embed
        elif uses_continuous_noise(self.noise_mode) and noise_values is not None:
            if noise_values.dim() == 2:
                noise_values = noise_values.unsqueeze(0)
            noise_values = noise_values.to(embeddings.device, dtype=torch.float32)
            noise_embed = self.noise_projector(noise_values)
            embeddings = embeddings + self.alpha * noise_embed

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertModelWithNoise(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.embeddings = BertNoiseEmbeddings(config)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, 
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, 
                past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, 
                return_dict=None, noise_ids=None, noise_values=None):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        
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
            noise_values=noise_values,
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

class BertForDaptMTL(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelWithNoise(config)
        self.cls = BertPreTrainingHeads(config)
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
        inputs_embeds=None, labels=None, next_sentence_label=None, noise_ids=None, noise_values=None,
        output_attentions=None, output_hidden_states=None, return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, 
            return_dict=return_dict, noise_ids=noise_ids, noise_values=noise_values,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        
        total_loss = None
        mlm_loss = None
        nsp_loss = None

        # 1. MLM Loss
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = mlm_loss

        # 2. NSP Loss
        if next_sentence_label is not None:
            valid_mask = (next_sentence_label != -100)
            if valid_mask.any():
                loss_fct = CrossEntropyLoss()
                nsp_loss = loss_fct(seq_relationship_score[valid_mask].view(-1, 2), next_sentence_label[valid_mask].view(-1))
                if total_loss is None:
                    total_loss = nsp_loss
                elif mlm_loss is not None:
                    total_loss += nsp_loss
                else:
                    total_loss = nsp_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        if total_loss is not None and (not torch.isfinite(total_loss)):
            raise ValueError(f"Non-finite loss detected: {total_loss}")

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
    noise_mode: str = "bucket"
    mlm_probability: float = 0.15
    max_length: int = 512
    # kv_wwm: use word_ids-based whole-word masking (KV-aware via jieba dict)
    # token: standard token-level MLM masking (ignore word_ids)
    mlm_masking: str = "kv_wwm"
    _cached_perfect_noise_values: Optional[torch.Tensor] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. 提取基础数据
        batch_input_ids = [f["input_ids"] for f in features]
        batch_word_ids = [f.get("word_ids") for f in features]
        
        # 2. Noise Processing
        batch_noise_ids = []
        batch_noise_values = []
        noise_lengths = []
        flat_noise_values = []
        for feat in features:
            nv = feat.get("noise_values") or []
            target_len = len(feat["input_ids"])
            if not nv:
                nv = [list(PERFECT_VALUES) for _ in range(target_len)]
            else:
                cleaned = []
                for row in nv[:target_len]:
                    if isinstance(row, (list, tuple)) and len(row) == len(FEATURES):
                        cleaned.append([float(v) for v in row])
                    else:
                        cleaned.append(list(PERFECT_VALUES))
                if len(cleaned) < target_len:
                    cleaned.extend([list(PERFECT_VALUES) for _ in range(target_len - len(cleaned))])
                nv = cleaned
            batch_noise_values.append(nv)
            if uses_bucket_noise(self.noise_mode):
                noise_lengths.append(target_len)
                flat_noise_values.extend(nv)

        if uses_bucket_noise(self.noise_mode):
            if self.noise_processor and flat_noise_values:
                mapped_all = self.noise_processor.map_batch(flat_noise_values)
            else:
                mapped_all = [[0] * len(FEATURES) for _ in range(len(flat_noise_values))]

            offset = 0
            for ln in noise_lengths:
                batch_noise_ids.append(mapped_all[offset: offset + ln])
                offset += ln

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
        
        # 4. Masking Strategy
        for i in range(len(features)):
            word_ids = batch_word_ids[i]
            current_ids = input_ids[i]
            use_wwm = (self.mlm_masking == "kv_wwm")
            if use_wwm and word_ids:
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

            # Ensure at least one token is masked per sample; otherwise labels become all -100 and loss becomes 0.
            if not bool(mask_indices.any()):
                candidate = ~special_tokens_mask
                if self.tokenizer.pad_token_id is not None:
                    candidate = candidate & (current_ids != self.tokenizer.pad_token_id)
                cand_idx = torch.nonzero(candidate, as_tuple=False).view(-1)
                if cand_idx.numel() > 0:
                    j = int(cand_idx[torch.randint(0, cand_idx.numel(), (1,)).item()].item())
                    mask_indices[j] = True

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
        final_noise_values = torch.zeros((len(features), seq_len, len(FEATURES)), dtype=torch.float32)
        for i, row in enumerate(batch_noise_values):
            l = min(len(row), seq_len)
            final_noise_values[i, :l, :] = torch.tensor(row[:l], dtype=torch.float32)
        batch["noise_values"] = final_noise_values
        if uses_bucket_noise(self.noise_mode):
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
        # Optimize: Pre-build a set for fast lookup of valid pairs to avoid False Negatives
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
    noise_mode: str = "bucket"
    max_length: int = 512
    _cached_perfect_ids_row: Optional[List[int]] = None

    def __call__(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        text_a_list = [x["text_a"] for x in batch_items]
        text_b_list = [x["text_b"] for x in batch_items]
        labels = [x["label"] for x in batch_items]

        # Tokenize pair
        enc = self.tokenizer(
            text_a_list,
            text_b_list,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Prepare Inputs
        bsz, seq_len = enc["input_ids"].shape
        
        # Noise Ids: 对于 NSP 这种纯文本任务，我们统一使用 "Perfect" 噪声特征
        # 确保其输入分布与 MLM 阶段的“Text Only”部分一致
        batch = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc["token_type_ids"],
            "next_sentence_label": torch.tensor(labels, dtype=torch.long),
            "labels": None, # MLM Label 为空
        }
        perfect_values = torch.tensor(PERFECT_VALUES, dtype=torch.float32).view(1, 1, -1).repeat(bsz, seq_len, 1)
        batch["noise_values"] = perfect_values
        if uses_bucket_noise(self.noise_mode):
            if self._cached_perfect_ids_row is None:
                self._cached_perfect_ids_row = [0] * len(FEATURES)
                if self.noise_processor:
                    self._cached_perfect_ids_row = self.noise_processor.map_batch([PERFECT_VALUES])[0]
            batch["noise_ids"] = torch.tensor(self._cached_perfect_ids_row, dtype=torch.long).view(1, 1, -1).repeat(bsz, seq_len, 1)
        return batch


# ===========================
# 3. 主流程
# ===========================

def main():
    parser = argparse.ArgumentParser()
    # 基础配置
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="/data/ocean/DAPT/workspace/processed_dataset")
    parser.add_argument("--nsp_data_dir", type=str, default="/data/ocean/DAPT/data/pseudo_kv_labels_filtered.json")
    parser.add_argument("--tokenizer_path", type=str, default="/data/ocean/DAPT/my-medical-tokenizer")
    parser.add_argument(
        "--pretrain_use_fast_tokenizer",
        action="store_true",
        help=(
            "Force using a fast tokenizer (use_fast=True) during pretraining. "
            "Default behavior prefers slow (use_fast=False) for stability, with an automatic fallback to fast if slow loading is broken."
        ),
    )
    parser.add_argument("--noise_bins_json", type=str, default="/data/ocean/DAPT/workspace/noise_bins.json")
    parser.add_argument(
        "--noise_mode",
        type=str,
        default="bucket",
        choices=["bucket", "linear", "mlp"],
        help="Noise fusion mode: bucket=离散分桶嵌入, linear=连续值线性投影, mlp=连续值两层 MLP 投影。",
    )
    parser.add_argument(
        "--noise_mlp_hidden_dim",
        type=int,
        default=128,
        help="noise_mode=mlp 时的隐藏层维度。",
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    # 训练超参
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="DataLoader worker 数。0 最稳定但可能 CPU 成为瓶颈；建议先尝试 2~8。",
    )
    parser.add_argument("--num_rounds", type=int, default=3, help="交替训练的总轮数 (MLM -> NSP -> MLM -> NSP ...)")
    parser.add_argument("--mlm_epochs_per_round", type=int, default=1, help="每轮 MLM 训练的 epoch 数")
    parser.add_argument("--nsp_epochs_per_round", type=int, default=3, help="每轮 NSP 训练的 epoch 数")
    parser.add_argument(
        "--mlm_masking",
        type=str,
        default="kv_wwm",
        choices=["kv_wwm", "token"],
        help="MLM masking mode. kv_wwm=word_ids whole-word masking (KV-aware). token=standard token-level MLM (control ablation).",
    )
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for pretraining.")
    parser.add_argument("--nsp_negative_prob", type=float, default=0.5, help="KV-NSP 中把正样本改造成负样本的总概率。")
    parser.add_argument("--nsp_reverse_negative_ratio", type=float, default=1.0, help="KV-NSP 负样本里 reverse 倒序策略的权重，例如 3:1。")
    parser.add_argument("--nsp_random_negative_ratio", type=float, default=1.0, help="KV-NSP 负样本里 random 随机 value 策略的权重，例如 1:3。")
    parser.add_argument("--nsp_max_easy_retries", type=int, default=10, help="构造 random 负样本时避免真实正例的最大重试次数。")
    parser.add_argument(
        "--export_fast_tokenizer",
        action="store_true",
        help="After saving checkpoints, also generate tokenizer.json (fast backend) in the output dirs for downstream offset_mapping.",
    )
    parser.add_argument(
        "--no_export_fast_tokenizer",
        action="store_true",
        help="Disable fast-tokenizer export even if downstream needs it.",
    )
    
    args = parser.parse_args()

    # Default behavior: export fast tokenizer unless explicitly disabled.
    export_fast_tokenizer = bool(args.export_fast_tokenizer) or not bool(args.no_export_fast_tokenizer)
    set_global_seed(int(args.seed))

    if is_main_process():
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        print(f"CUDA_VISIBLE_DEVICES={cvd}")
        print(f"Global seed={args.seed}")
        if torch.cuda.is_available():
            try:
                n = torch.cuda.device_count()
                print(f"Visible CUDA device_count={n}")
                for i in range(n):
                    p = torch.cuda.get_device_properties(i)
                    total_gib = p.total_memory / (1024 ** 3)
                    print(f"  cuda:{i} {p.name} total_mem={total_gib:.1f}GiB")
            except Exception as e:
                print(f"(warn) failed to query cuda devices: {e}")

    # 1. 资源准备
    if bool(args.pretrain_use_fast_tokenizer):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
        if isinstance(tokenizer, bool) or not _looks_like_tokenizer(tokenizer):
            raise TypeError(
                f"--pretrain_use_fast_tokenizer was set, but failed to load a usable fast tokenizer from: {args.tokenizer_path}"
            )
        print(f"[tokenizer] pretrain_use_fast_tokenizer=True loaded={type(tokenizer).__name__} is_fast={getattr(tokenizer, 'is_fast', None)}")
    else:
        tokenizer = _load_tokenizer_with_fallback(args.tokenizer_path)
        print(f"[tokenizer] pretrain_use_fast_tokenizer=False loaded={type(tokenizer).__name__} is_fast={getattr(tokenizer, 'is_fast', None)}")
    
    noise_processor = NoiseFeatureProcessor.load(args.noise_bins_json) if os.path.exists(args.noise_bins_json) else NoiseFeatureProcessor()
    
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
    print(
        "Train throughput config: "
        f"per_device_train_batch_size={args.per_device_train_batch_size}, "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
        f"dataloader_num_workers={args.dataloader_num_workers}"
    )

    # 3. 初始化模型
    # 尝试加载最新 checkpint 或者 基座
    model_path = "hfl/chinese-macbert-base"
    if args.resume_from_checkpoint:
        model_path = args.resume_from_checkpoint
        print(f"Resuming form checkpoint: {model_path}")

    config = BertConfig.from_pretrained(model_path)
    config.noise_mode = str(args.noise_mode).lower()
    config.noise_mlp_hidden_dim = int(args.noise_mlp_hidden_dim)
    config.noise_bin_edges = getattr(noise_processor, "bin_edges", {})
    model = BertForDaptMTL.from_pretrained(model_path, config=config)
    
    # Resize Token Embeddings (Safe Logic)
    if len(tokenizer) > model.config.vocab_size:
        print(f"Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    # 4. 准备 Collators
    mlm_collator = MLMStageCollator(
        tokenizer,
        noise_processor,
        noise_mode=str(args.noise_mode).lower(),
        mlm_probability=float(args.mlm_probability),
        max_length=int(args.max_length),
        mlm_masking=str(args.mlm_masking),
    )
    nsp_collator = NSPStageCollator(tokenizer, noise_processor, noise_mode=str(args.noise_mode).lower())

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
            per_device_train_batch_size=int(args.per_device_train_batch_size),
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            seed=int(args.seed),
            data_seed=int(args.seed),
            learning_rate=args.learning_rate,
            logging_steps=50,
            save_strategy="epoch", # 每轮只在结束时保存，避免中间文件过多
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            # DDP Settings
            ddp_find_unused_parameters=True, 
            dataloader_num_workers=int(args.dataloader_num_workers),
            dataloader_pin_memory=True,
            dataloader_persistent_workers=bool(int(args.dataloader_num_workers) > 0),
            save_safetensors=False,
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
        if export_fast_tokenizer:
            _export_fast_tokenizer_if_possible(mlm_output_dir)
        
        # --- Phase B: NSP Training ---
        print(f"--- [Round {round_idx}] Phase B: NSP Training ---")
        nsp_output_dir = os.path.join(args.output_dir, f"round_{round_idx}_nsp")
        
        training_args_nsp = TrainingArguments(
            output_dir=nsp_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.nsp_epochs_per_round,
            per_device_train_batch_size=int(args.per_device_train_batch_size),
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            seed=int(args.seed),
            data_seed=int(args.seed),
            learning_rate=args.learning_rate,
            logging_steps=20,
            save_strategy="epoch",
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            ddp_find_unused_parameters=True,
            dataloader_num_workers=int(args.dataloader_num_workers),
            dataloader_pin_memory=True,
            dataloader_persistent_workers=bool(int(args.dataloader_num_workers) > 0),
            save_safetensors=False,
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
        if export_fast_tokenizer:
            _export_fast_tokenizer_if_possible(nsp_output_dir)
        
        print(f"Round {round_idx} completed. Checkpoint saved at {nsp_output_dir}")

    # Final Save
    final_output_dir = os.path.join(args.output_dir, "final_staged_model")
    print(f"All rounds finished. Saving final model to {final_output_dir}...")
    trainer_nsp.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    if export_fast_tokenizer:
        _export_fast_tokenizer_if_possible(final_output_dir)

if __name__ == "__main__":
    main()
