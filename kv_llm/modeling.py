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

from kv_llm.ncag import (
    NCAGConfig,
    NCAGGate,
    NCAG_NOISE_DIM,
    build_ncag_mask,
    gate_stats,
    ncag_keeps_additive,
    uses_ncag,
)


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
        ncag_hidden_dim: Optional[int] = None,
        ncag_gate_side: str = "key",
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.noise_mode = str(noise_mode or "bucket").lower()
        self.noise_alpha = float(noise_alpha)
        hidden_size = int(getattr(self.config, "hidden_size", 0) or getattr(self.config, "n_embd", 0))
        if hidden_size <= 0:
            raise ValueError("Cannot infer hidden size from base model config")
        # --- Additive Noise-Embedding branch (input-side, ConfBERT-style) ----
        # Active for legacy noise_modes AND for the NCAG-additive variant (N4).
        if self.noise_mode == "bucket" or self.noise_mode == "ncag_additive":
            self.noise_module = BucketNoiseEmbedder(hidden_size)
        elif self.noise_mode == "concat_linear":
            self.noise_module = ConcatLinearNoiseEmbedder(hidden_size, NUM_BINS, embed_dim=concat_embed_dim)
        elif self.noise_mode in {"linear", "mlp"}:
            self.noise_module = ContinuousNoiseProjector(
                hidden_size,
                mode=self.noise_mode,
                mlp_hidden_dim=noise_mlp_hidden_dim,
            )
        elif self.noise_mode in {"none", "no_noise", "ncag"}:
            # "ncag" (without additive) is N3: gate-only, no input-side noise add.
            self.noise_module = None
        else:
            raise ValueError(f"Unsupported noise_mode: {self.noise_mode}")
        # --- NCAG branch (attention-side gating, the new method) ------------
        if uses_ncag(self.noise_mode):
            self.ncag_module = NCAGGate(in_dim=NCAG_NOISE_DIM, hidden_dim=ncag_hidden_dim)
            self.ncag_config = NCAGConfig(gate_side=ncag_gate_side)
        else:
            self.ncag_module = None
            self.ncag_config = None
        # Tiny telemetry slot, mutated in forward(); read by callbacks / pilot.
        self.last_gate_stats: Optional[dict] = None
        self.nsp_head = nn.Linear(hidden_size, 2)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> "KvLlmForCausalPreTraining":
        noise_kwargs = {
            k: kwargs.pop(k)
            for k in list(kwargs.keys())
            if k in {
                "noise_mode",
                "noise_mlp_hidden_dim",
                "concat_embed_dim",
                "noise_alpha",
                "ncag_hidden_dim",
                "ncag_gate_side",
            }
        }
        # NCAG modes require eager attention so our 4-D custom mask is
        # honoured exactly. SDPA / FlashAttn silently drop additive bias
        # terms in transformers 4.57.
        if uses_ncag(noise_kwargs.get("noise_mode")):
            kwargs.setdefault("attn_implementation", "eager")
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
        # NCAG-additive (N4): the additive arm uses bucket IDs just like
        # plain "bucket" mode.
        mode_for_lookup = "bucket" if self.noise_mode == "ncag_additive" else self.noise_mode
        if needs_bucket_ids(mode_for_lookup):
            if noise_ids is None:
                return embeds
            noise = self.noise_module(noise_ids.to(device=input_ids.device, dtype=torch.long))
        elif uses_continuous_noise(mode_for_lookup):
            if noise_values is None:
                return embeds
            if noise_values.shape[-1] != len(FEATURES):
                raise ValueError(f"noise_values last dim must be {len(FEATURES)}, got {noise_values.shape[-1]}")
            noise = self.noise_module(noise_values.to(device=input_ids.device, dtype=torch.float32))
        else:
            return embeds
        return embeds + self.noise_alpha * noise.to(dtype=embeds.dtype)

    def _ncag_attention_mask(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        noise_values: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """If NCAG is active build a 4-D additive mask; otherwise return None."""
        if self.ncag_module is None:
            return None
        if noise_values is None:
            # No noise → gate would be uninformative; fall back to the
            # standard 2-D causal mask handling by returning None.
            self.last_gate_stats = None
            return None
        if noise_values.shape[-1] != NCAG_NOISE_DIM:
            raise ValueError(
                f"NCAG expects noise_values last dim {NCAG_NOISE_DIM}, got {noise_values.shape[-1]}"
            )
        # NCAGGate runs in fp32 for numerical safety; cast result back later.
        gate = self.ncag_module(
            noise_values.to(device=inputs_embeds.device, dtype=torch.float32)
        )  # [B, L]
        # Truncate/pad gate to match inputs_embeds seq dim (defensive against
        # collator emitting [B, L_full, 7] when the tokenizer truncated).
        seq_len = inputs_embeds.shape[1]
        if gate.shape[1] != seq_len:
            gate = gate[:, :seq_len] if gate.shape[1] > seq_len else nn.functional.pad(
                gate, (0, seq_len - gate.shape[1]), value=1.0
            )
        self.last_gate_stats = gate_stats(gate, attention_mask)
        return build_ncag_mask(
            gate=gate,
            attention_mask=attention_mask,
            dtype=inputs_embeds.dtype,
            config=self.ncag_config,
        )

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
        # NCAG: replace the 2-D padding mask with our 4-D additive mask
        # carrying the gate bias. When NCAG is off, ``ncag_mask`` is None
        # and the base model uses the standard ``attention_mask`` path.
        ncag_mask = self._ncag_attention_mask(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            noise_values=noise_values,
        )
        effective_mask = ncag_mask if ncag_mask is not None else attention_mask

        if nsp_labels is not None or task_type == "nsp":
            out = self.base_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=effective_mask,
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
            attention_mask=effective_mask,
            labels=labels,
            use_cache=False,
            **kwargs,
        )

    def save_pretrained(self, output_dir: str | Path, **kwargs: Any) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("safe_serialization", False)
        self.base_model.save_pretrained(output, **kwargs)
        torch.save(self.nsp_head.state_dict(), output / "kv_llm_nsp_head.pt")
        if self.noise_module is not None:
            torch.save(self.noise_module.state_dict(), output / "kv_llm_noise_module.pt")
        if self.ncag_module is not None:
            torch.save(self.ncag_module.state_dict(), output / "kv_llm_ncag_gate.pt")
        meta = {
            "noise_mode": self.noise_mode,
            "noise_alpha": self.noise_alpha,
            "features": FEATURES,
        }
        if self.ncag_config is not None:
            meta["ncag"] = {
                "gate_side": self.ncag_config.gate_side,
                "eps": self.ncag_config.eps,
            }
        (output / "kv_llm_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
