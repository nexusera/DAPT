import torch
from torch import nn
from typing import Optional
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings

# 7 维特征（与 processor 对齐）
FEATURES = [
    "conf_avg",
    "conf_min",
    "conf_var_log",
    "conf_gap",
    "punct_err_ratio",
    "char_break_ratio",
    "align_score",
]

# 桶数（不含 anchor=0，embedding 实际建 num_bins+1）
NUM_BINS = {
    "conf_avg": 64,
    "conf_min": 64,
    "conf_var_log": 32,
    "conf_gap": 32,
    "punct_err_ratio": 16,
    "char_break_ratio": 32,
    "align_score": 64,
}


class RobertaNoiseEmbeddings(RobertaEmbeddings):
    """
    离散噪声嵌入版：
      - 输入 noise_ids: [batch, seq_len, 7]，每维为桶 ID，0 为 anchor（完美/零值）。
      - 7 路 embedding 分别 lookup 后求和，再乘以可学习系数 alpha，叠加到 text embedding。
      - 其余与原 RoBERTaEmbedding 相同。
    """

    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.noise_dim = len(FEATURES)
        self.noise_embeddings = nn.ModuleDict()
        for feat in FEATURES:
            n_bins = NUM_BINS[feat]
            self.noise_embeddings[feat] = nn.Embedding(n_bins + 1, config.hidden_size)
        # 残差缩放系数 alpha（可学习）
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self._reset_noise_parameters()

    def _reset_noise_parameters(self):
        for emb in self.noise_embeddings.values():
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        # alpha 已初始化为 0.1

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
        noise_ids: Optional[torch.Tensor] = None,  # [batch, seq, 7]
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 位置与 segment
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        if noise_ids is not None:
            if noise_ids.dim() == 2:
                noise_ids = noise_ids.unsqueeze(0)  # [1, seq, 7]
            noise_ids = noise_ids.to(inputs_embeds.device)
            noise_embed = 0.0
            for i, feat in enumerate(FEATURES):
                emb_layer = self.noise_embeddings[feat]
                ids = noise_ids[:, :, i].clamp(min=0, max=emb_layer.num_embeddings - 1)
                noise_embed = noise_embed + emb_layer(ids)
            inputs_embeds = inputs_embeds + self.alpha * noise_embed

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

