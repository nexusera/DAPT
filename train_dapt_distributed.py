import os
import math
import torch
import sys
import random
import argparse

# ===========================
# 0. 设置环境变量
# ===========================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Disable heavy inductor kernel benchmarking that can spike memory during torch.compile
os.environ.setdefault("TORCHINDUCTOR_BENCHMARK_KERNEL", "0")

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaForMaskedLM,
    RobertaModel,
)

from noise_embeddings import RobertaNoiseEmbeddings
from noise_feature_processor import NoiseFeatureProcessor, FEATURES

# 完美物理值（非 OCR 样本用），由 processor 映射到桶 ID
PERFECT_VALUES = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# ===========================
# 1. 配置路径与参数
# ===========================
WORKSPACE_DIR = "/data/ocean/DAPT/workspace"
# Tokenizer 存放在 /data/ocean/DAPT/my-medical-tokenizer
TOKENIZER_PATH = "/data/ocean/DAPT/my-medical-tokenizer"
DATASET_PATH = os.path.join(WORKSPACE_DIR, "processed_dataset")
MODEL_CHECKPOINT = "hfl/chinese-roberta-wwm-ext"
# 可通过命令行 --output_dir 覆盖，避免多实验互相覆盖
DEFAULT_OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "output_medical_bert_v2_8gpu")

# ===========================
# 8卡 H200 极速配置
# ===========================
# Global Batch Size = 16 * 8(GPUs) * 4(Accum) = 512
PER_DEVICE_BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 4  # 通过更高累积进一步降低单卡占用
LEARNING_RATE = 8e-5  # 默认 LR，可通过 CLI 覆盖
NUM_TRAIN_EPOCHS = 7  # 默认 epoch，可通过 CLI 覆盖
MAX_SEQ_LEN = 512  # 保持基座模型的512，不扩展位置编码

def is_main_process():
    """判断是否为主进程 (Rank 0)"""
    return int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]

# ===========================
# 2. 核心组件 (保持不变)
# ===========================
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
        batch_word_ids = [f["word_ids"] for f in features]
        batch = self.tokenizer.pad(
            {"input_ids": batch_input_ids},
            padding="longest",
            max_length=MAX_SEQ_LEN,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        for i in range(len(features)):
            word_ids = batch_word_ids[i]
            current_ids = input_ids[i]
            mapping = {}
            for idx, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if idx >= len(current_ids):
                    break
                mapping.setdefault(wid, []).append(idx)

            unique_words = list(mapping.keys())
            num_to_mask = max(1, int(len(unique_words) * self.mlm_probability))
            masked_words = set(random.sample(unique_words, num_to_mask))
            mask_indices = torch.zeros(len(current_ids), dtype=torch.bool)

            for wid in masked_words:
                for idx in mapping[wid]:
                    mask_indices[idx] = True

            special_tokens_mask = torch.tensor(
                self.tokenizer.get_special_tokens_mask(current_ids.tolist(), already_has_special_tokens=True),
                dtype=torch.bool,
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


class NoiseAwareCollator(PrecomputedWWMCollator):
    """
    生成离散噪声桶 ID：noise_ids=[B, L, 7]，anchor=0。
    - 若样本无 noise_values，则填满完美噪音（完美语料）。
    - 不修改 input_ids 的 [MASK] 位置对应的 noise_ids。
    """

    def __init__(self, *args, noise_processor: NoiseFeatureProcessor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_processor = noise_processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(features)
        max_len = batch["input_ids"].shape[1]

        noise_ids = []
        for feat in features:
            nv = feat.get("noise_values") or []
            if not nv:
                # 完美语料：使用完美物理值，交给 processor 映射到对应桶
                nv = [PERFECT_VALUES for _ in range(len(feat["input_ids"]))]
            nv = (nv + [[0.0] * len(FEATURES)] * max_len)[:max_len]
            ids = self.noise_processor.map_batch(nv) if self.noise_processor else [[0] * len(FEATURES) for _ in nv]
            noise_ids.append(ids)

        batch["noise_ids"] = torch.tensor(noise_ids, dtype=torch.long)
        return batch


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


class RobertaForMaskedLMWithNoise(RobertaForMaskedLM):
    """
    覆盖 RobertaForMaskedLM 以支持 noise_features/noise_masks。
    """

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModelWithNoise(config, add_pooling_layer=False)

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
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        noise_ids: Optional[torch.Tensor] = None,
    ) -> MaskedLMOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            noise_ids=noise_ids,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # 只有主进程打印 Log，防止刷屏
        if state.is_local_process_zero:
            loss = metrics.get("eval_loss")
            if loss:
                ppl = math.exp(loss)
                print(f"\n[Evaluation] Perplexity (PPL): {ppl:.4f}\n")
                metrics["perplexity"] = ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="训练输出目录，避免覆盖可传入新的路径")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="可选：从已有 checkpoint 继续训练")
    parser.add_argument("--num_train_epochs", type=int, default=None, help=f"训练 epoch 数，默认 {NUM_TRAIN_EPOCHS}")
    parser.add_argument("--learning_rate", type=float, default=None, help=f"学习率，默认 {LEARNING_RATE}")
    parser.add_argument("--noise_bins_json", type=str, required=True, help="预计算的噪声分桶边界 json，供 NoiseFeatureProcessor 使用")
    parser.add_argument(
        "--bf16",
        dest="bf16",
        action="store_true",
        help="启用 bf16（默认开启）",
    )
    parser.add_argument(
        "--no_bf16",
        dest="bf16",
        action="store_false",
        help="关闭 bf16 做短程 sanity check",
    )
    parser.set_defaults(bf16=True)
    args = parser.parse_args()

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"数据集未找到: {DATASET_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    dataset = load_from_disk(DATASET_PATH)
    
    # 打印只在主进程进行
    if is_main_process():
        print(f"Loading base model {MODEL_CHECKPOINT}...")
    
    model = RobertaForMaskedLMWithNoise.from_pretrained(MODEL_CHECKPOINT)
    model.resize_token_embeddings(len(tokenizer))
    # 注释掉权重克隆功能，不扩展位置编码，保持基座模型的512长度
    # model = resize_position_embeddings(model, new_max_len=MAX_SEQ_LEN)
    model.gradient_checkpointing_enable()  # 启用梯度检查点减少显存
    processor = NoiseFeatureProcessor.load(args.noise_bins_json)
    data_collator = NoiseAwareCollator(tokenizer=tokenizer, noise_processor=processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs or NUM_TRAIN_EPOCHS,
        # 分布式相关
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        ddp_find_unused_parameters=False,  # 优化速度
        learning_rate=args.learning_rate or LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.05,  # 缩短预热，加快进入主学习率区间
        bf16=args.bf16,
        tf32=True,
        dataloader_num_workers=8,  # 8卡 x 8 workers = 64线程，再多CPU受不了
        torch_compile=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=100,  # 8卡跑得快，评估频率要调高 (Steps变少了)
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
        print("Starting 8-GPU Distributed Training...")
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if is_main_process():
        print("Saving final model...")
        trainer.save_model(os.path.join(args.output_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))

if __name__ == "__main__":
    main()