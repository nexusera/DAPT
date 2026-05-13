"""Qwen-style causal LM wrapper with KV-LLM noise embedding and NSP head."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from noise_fusion import ContinuousNoiseProjector, ConcatLinearNoiseEmbedder, needs_bucket_ids, uses_continuous_noise
from noise_feature_processor import FEATURES, NUM_BINS


class BucketNoiseEmbedder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.noise_embeddings = nn.ModuleDict(
            {feat: nn.Embedding(NUM_BINS.get(feat, 64) + 1, hidden_size) for feat in FEATURES}
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in self.noise_embeddings.values():
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, noise_ids: torch.Tensor) -> torch.Tensor:
        if noise_ids.shape[-1] != len(FEATURES):
            raise ValueError(f"noise_ids last dim must be {len(FEATURES)}, got {noise_ids.shape[-1]}")
        out = 0.0
        for i, feat in enumerate(FEATURES):
            emb = self.noise_embeddings[feat]
            ids = noise_ids[:, :, i].clamp(min=0, max=emb.num_embeddings - 1)
            out = out + emb(ids)
        return out


class KvLlmForCausalPreTraining(nn.Module):
    """A light wrapper around `AutoModelForCausalLM`.

    - CLM/span corruption: passes `labels` to the base causal LM.
    - KV-NSP: takes the last non-padding hidden state and applies a 2-way head.
    - Noise-Embedding: adds bucket/linear/mlp/concat noise features to token embeddings.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        *,
        noise_mode: str = "bucket",
        noise_mlp_hidden_dim: Optional[int] = None,
        concat_embed_dim: int = 64,
        noise_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.noise_mode = str(noise_mode or "bucket").lower()
        self.noise_alpha = float(noise_alpha)
        hidden_size = int(getattr(self.config, "hidden_size", 0) or getattr(self.config, "n_embd", 0))
        if hidden_size <= 0:
            raise ValueError("Cannot infer hidden size from base model config")
        if self.noise_mode == "bucket":
            self.noise_module = BucketNoiseEmbedder(hidden_size)
        elif self.noise_mode == "concat_linear":
            self.noise_module = ConcatLinearNoiseEmbedder(hidden_size, NUM_BINS, embed_dim=concat_embed_dim)
        elif self.noise_mode in {"linear", "mlp"}:
            self.noise_module = ContinuousNoiseProjector(
                hidden_size,
                mode=self.noise_mode,
                mlp_hidden_dim=noise_mlp_hidden_dim,
            )
        elif self.noise_mode in {"none", "no_noise"}:
            self.noise_module = None
        else:
            raise ValueError(f"Unsupported noise_mode: {self.noise_mode}")
        self.nsp_head = nn.Linear(hidden_size, 2)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> "KvLlmForCausalPreTraining":
        noise_kwargs = {
            k: kwargs.pop(k)
            for k in list(kwargs.keys())
            if k in {"noise_mode", "noise_mlp_hidden_dim", "concat_embed_dim", "noise_alpha"}
        }
        base = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        return cls(base, **noise_kwargs)

    def _add_noise(
        self,
        input_ids: torch.Tensor,
        noise_ids: Optional[torch.Tensor],
        noise_values: Optional[torch.Tensor],
    ) -> torch.Tensor:
        embeds = self.base_model.get_input_embeddings()(input_ids)
        if self.noise_module is None:
            return embeds
        if needs_bucket_ids(self.noise_mode):
            if noise_ids is None:
                return embeds
            noise = self.noise_module(noise_ids.to(device=input_ids.device, dtype=torch.long))
        elif uses_continuous_noise(self.noise_mode):
            if noise_values is None:
                return embeds
            if noise_values.shape[-1] != len(FEATURES):
                raise ValueError(f"noise_values last dim must be {len(FEATURES)}, got {noise_values.shape[-1]}")
            noise = self.noise_module(noise_values.to(device=input_ids.device, dtype=torch.float32))
        else:
            return embeds
        return embeds + self.noise_alpha * noise.to(dtype=embeds.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        nsp_labels: Optional[torch.Tensor] = None,
        noise_ids: Optional[torch.Tensor] = None,
        noise_values: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        inputs_embeds = self._add_noise(input_ids, noise_ids, noise_values)
        if nsp_labels is not None or task_type == "nsp":
            out = self.base_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                **kwargs,
            )
            hidden = out.hidden_states[-1]
            if attention_mask is None:
                last_idx = torch.full((hidden.shape[0],), hidden.shape[1] - 1, device=hidden.device, dtype=torch.long)
            else:
                last_idx = attention_mask.to(hidden.device).sum(dim=1).clamp(min=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx]
            logits = self.nsp_head(pooled)
            loss = None
            if nsp_labels is not None:
                loss = nn.CrossEntropyLoss()(logits, nsp_labels.to(device=logits.device))
            return SequenceClassifierOutput(loss=loss, logits=logits)

        return self.base_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            **kwargs,
        )

    def save_pretrained(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(output)
        torch.save(self.nsp_head.state_dict(), output / "kv_llm_nsp_head.pt")
        if self.noise_module is not None:
            torch.save(self.noise_module.state_dict(), output / "kv_llm_noise_module.pt")
        meta = {
            "noise_mode": self.noise_mode,
            "noise_alpha": self.noise_alpha,
            "features": FEATURES,
        }
        (output / "kv_llm_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
