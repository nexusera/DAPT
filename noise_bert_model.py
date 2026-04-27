"""
noise_bert_model.py
-------------------
预训练与下游微调共享的 BERT+噪声嵌入模块。

核心类：
  BertNoiseEmbeddings  —— 替换标准 BertEmbeddings，在词嵌入层（early fusion）
                           注入 OCR 噪声特征，与预训练完全对齐。
  BertModelWithNoise   —— 使用上述嵌入层的 BertModel。
  load_bert_with_noise —— 从 checkpoint（预训练或微调）加载 BertModelWithNoise。

设计原则：
  - 预训练和下游微调完全共用同一个类，消除 early/late fusion 不一致。
  - 预训练时保存的噪声嵌入权重（alpha、noise_embeddings.*、concat_embedder.*）
    在下游微调时自动恢复并参与训练。
  - 支持 bucket / linear / mlp / concat_linear 四种噪声模式。
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from noise_fusion import (
    ContinuousNoiseProjector,
    ConcatLinearNoiseEmbedder,
    uses_bucket_noise,
    uses_continuous_noise,
    uses_concat_noise,
    FEATURES,
)

logger = logging.getLogger(__name__)

# 桶数（各维特征），与 noise_embeddings.py / noise_utils.py 保持一致
NUM_BINS: Dict[str, int] = {
    "conf_avg": 64,
    "conf_min": 64,
    "conf_var_log": 32,
    "conf_gap": 32,
    "punct_err_ratio": 16,
    "char_break_ratio": 32,
    "align_score": 64,
}


class BertNoiseEmbeddings(BertEmbeddings):
    """
    BERT 嵌入层 + OCR 噪声 early fusion。

    噪声在 word/position/token_type 嵌入求和之后、LayerNorm 之前注入：
        embeddings += alpha * noise_embed
    支持三种融合模式（由 config.noise_mode 决定）：
      bucket       : 7路各自查 Embedding(n_bins+1, hidden_size)，然后 Add
      linear/mlp   : 7维连续值经 ContinuousNoiseProjector 投影
      concat_linear: 7路各自查 Embedding(n_bins+1, embed_dim)，Concat 后 Linear→768
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.noise_mode = str(getattr(config, "noise_mode", "bucket") or "bucket").lower()
        self.noise_mlp_hidden_dim = getattr(config, "noise_mlp_hidden_dim", None)
        self.noise_concat_embed_dim = int(getattr(config, "noise_concat_embed_dim", 64) or 64)
        self.noise_bin_edges = getattr(config, "noise_bin_edges", None)

        self.noise_embeddings: Optional[nn.ModuleDict] = None
        self.noise_projector: Optional[nn.Module] = None
        self.concat_embedder: Optional[nn.Module] = None

        if uses_bucket_noise(self.noise_mode):
            self.noise_embeddings = nn.ModuleDict()
            for feat in FEATURES:
                n_bins = NUM_BINS.get(feat, 64)
                self.noise_embeddings[feat] = nn.Embedding(n_bins + 1, config.hidden_size)
        elif uses_continuous_noise(self.noise_mode):
            self.noise_projector = ContinuousNoiseProjector(
                config.hidden_size,
                mode=self.noise_mode,
                dropout=getattr(config, "hidden_dropout_prob", 0.1),
                mlp_hidden_dim=self.noise_mlp_hidden_dim,
                feature_ranges=self.noise_bin_edges,
            )
        elif uses_concat_noise(self.noise_mode):
            self.concat_embedder = ConcatLinearNoiseEmbedder(
                hidden_size=config.hidden_size,
                num_bins_per_feat=NUM_BINS,
                embed_dim=self.noise_concat_embed_dim,
                dropout=getattr(config, "hidden_dropout_prob", 0.1),
            )
        else:
            raise ValueError(f"Unsupported noise_mode: {self.noise_mode}")

        # 可学习残差缩放系数，初始值小（0.1）使训练初期噪声影响温和
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self._reset_noise_parameters()

    def _reset_noise_parameters(self) -> None:
        if self.noise_embeddings is not None:
            for emb in self.noise_embeddings.values():
                nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
        noise_ids: Optional[torch.Tensor] = None,    # [B, L, 7]  离散桶 ID
        noise_values: Optional[torch.Tensor] = None, # [B, L, 7]  连续噪声值
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        # ── 噪声 early fusion ─────────────────────────────────────────────
        if uses_bucket_noise(self.noise_mode) and noise_ids is not None:
            if noise_ids.dim() == 2:
                noise_ids = noise_ids.unsqueeze(0)
            # H1: 校验最后一维必须等于 FEATURES 数量，防止 C1 fallback 5 维错配静默污染
            if noise_ids.shape[-1] != len(FEATURES):
                raise ValueError(
                    f"noise_ids.shape[-1]={noise_ids.shape[-1]} != len(FEATURES)={len(FEATURES)}. "
                    "请检查 build_zero_feats 或 OCR fallback 路径是否返回了错误维度。"
                )
            noise_ids = noise_ids.to(device)
            noise_embed = torch.zeros_like(embeddings)
            for i, feat in enumerate(FEATURES):
                emb_layer = self.noise_embeddings[feat]
                ids_i = noise_ids[:, :, i].clamp(0, emb_layer.num_embeddings - 1)
                noise_embed = noise_embed + emb_layer(ids_i)
            embeddings = embeddings + self.alpha * noise_embed

        elif uses_continuous_noise(self.noise_mode) and noise_values is not None:
            if noise_values.dim() == 2:
                noise_values = noise_values.unsqueeze(0)
            # H2: 校验 noise_values 的最后一维与 FEATURES 数量一致，并强制 float32
            if noise_values.shape[-1] != len(FEATURES):
                raise ValueError(
                    f"noise_values.shape[-1]={noise_values.shape[-1]} != len(FEATURES)={len(FEATURES)}. "
                    "noise_values 维度与特征定义不匹配。"
                )
            noise_values = noise_values.to(device, dtype=torch.float32)
            noise_embed = self.noise_projector(noise_values)
            embeddings = embeddings + self.alpha * noise_embed

        elif uses_concat_noise(self.noise_mode) and noise_ids is not None:
            if noise_ids.dim() == 2:
                noise_ids = noise_ids.unsqueeze(0)
            # H1: concat_linear 同样使用 noise_ids，同步校验维度
            if noise_ids.shape[-1] != len(FEATURES):
                raise ValueError(
                    f"noise_ids.shape[-1]={noise_ids.shape[-1]} != len(FEATURES)={len(FEATURES)}. "
                    "concat_linear 模式下 noise_ids 维度与特征定义不匹配。"
                )
            noise_ids = noise_ids.to(device)
            noise_embed = self.concat_embedder(noise_ids)
            embeddings = embeddings + self.alpha * noise_embed
        # ─────────────────────────────────────────────────────────────────

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModelWithNoise(BertModel):
    """
    BertModel，嵌入层替换为 BertNoiseEmbeddings（early fusion）。
    forward 额外接受 noise_ids / noise_values 并透传给嵌入层。
    """

    def __init__(self, config: BertConfig, add_pooling_layer: bool = True) -> None:
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.embeddings = BertNoiseEmbeddings(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        noise_ids: Optional[torch.Tensor] = None,
        noise_values: Optional[torch.Tensor] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds.")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered = self.embeddings.token_type_ids[:, :input_shape[1]]
                token_type_ids = buffered.expand(input_shape[0], input_shape[1])
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


def load_bert_with_noise(
    model_path: str,
    *,
    noise_mode: Optional[str] = None,
    noise_concat_embed_dim: Optional[int] = None,
    noise_mlp_hidden_dim: Optional[int] = None,
    noise_bin_edges: Optional[Dict] = None,
) -> BertModelWithNoise:
    """
    从 checkpoint 目录加载 BertModelWithNoise，自动恢复噪声嵌入权重。

    兼容两种 checkpoint 格式：
      1. 预训练 checkpoint（BertForDaptMTL）：权重 key 带 "bert." 前缀
      2. 微调 checkpoint（BertCrfTokenClassifier）：权重 key 也带 "bert." 前缀

    config.json 中的噪声字段（noise_mode / noise_concat_embed_dim 等）优先级最高；
    若 config.json 中不存在，则使用函数参数作为补充。

    Args:
        model_path: checkpoint 目录路径。
        noise_mode: 覆盖 config 中的 noise_mode（通常无需显式传）。
        noise_concat_embed_dim: 覆盖 concat_linear 的每路嵌入维度。
        noise_mlp_hidden_dim: 覆盖 mlp 模式的隐藏层维度。
        noise_bin_edges: 覆盖分桶边界（通常保存在 config 中）。

    Returns:
        已加载权重的 BertModelWithNoise 实例。
    """
    config = BertConfig.from_pretrained(model_path)

    # 用参数补充 config 中缺失的噪声字段（向后兼容旧 checkpoint）
    if noise_mode is not None:
        config.noise_mode = str(noise_mode).lower()
    elif not hasattr(config, "noise_mode"):
        config.noise_mode = "bucket"

    if noise_concat_embed_dim is not None:
        config.noise_concat_embed_dim = int(noise_concat_embed_dim)
    elif not hasattr(config, "noise_concat_embed_dim"):
        config.noise_concat_embed_dim = 64

    if noise_mlp_hidden_dim is not None:
        config.noise_mlp_hidden_dim = int(noise_mlp_hidden_dim)

    if noise_bin_edges is not None:
        config.noise_bin_edges = noise_bin_edges

    bert = BertModelWithNoise(config)

    # ── 加载权重 ──────────────────────────────────────────────────────
    path = Path(model_path)
    sf_path = path / "model.safetensors"
    pt_path = path / "pytorch_model.bin"

    if sf_path.is_file():
        try:
            from safetensors.torch import load_file
            full_state = load_file(str(sf_path))
        except ImportError:
            logger.warning("safetensors not installed; trying pytorch_model.bin")
            full_state = torch.load(str(pt_path), map_location="cpu") if pt_path.is_file() else {}
    elif pt_path.is_file():
        full_state = torch.load(str(pt_path), map_location="cpu")
    else:
        logger.warning(f"[load_bert_with_noise] No weights at {model_path}. Random init.")
        return bert

    # 剥离 "bert." 前缀（两种 checkpoint 格式均适用）
    bert_state = {k[5:]: v for k, v in full_state.items() if k.startswith("bert.")}
    if not bert_state:
        logger.warning("[load_bert_with_noise] No 'bert.*' keys found; trying direct load.")
        bert_state = full_state

    missing, unexpected = bert.load_state_dict(bert_state, strict=False)

    # 分类汇报缺失/意外键
    noise_keywords = ("noise", "alpha", "concat_embedder")
    noise_missing = [k for k in missing if any(kw in k for kw in noise_keywords)]
    core_missing = [k for k in missing if k not in noise_missing]

    if core_missing:
        logger.warning(
            f"[load_bert_with_noise] Missing BERT core keys ({len(core_missing)}): "
            f"{core_missing[:5]}{'...' if len(core_missing) > 5 else ''}"
        )
    if noise_missing:
        logger.info(
            f"[load_bert_with_noise] Noise weights not in checkpoint (random init) "
            f"({len(noise_missing)}): {noise_missing[:5]}{'...' if len(noise_missing) > 5 else ''}"
        )
    if unexpected:
        logger.debug(
            f"[load_bert_with_noise] Unexpected keys ({len(unexpected)}): {unexpected[:5]}"
        )

    logger.info(
        f"[load_bert_with_noise] Loaded from '{model_path}' "
        f"(noise_mode={config.noise_mode}, "
        f"core_missing={len(core_missing)}, noise_missing={len(noise_missing)})"
    )
    return bert
