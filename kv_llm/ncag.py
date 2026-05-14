"""Noise-Conditioned Attention Gating (NCAG) — first implementation.

Implements the additive 4D attention-bias path designated as the preferred
route in plan §A5 lines 720-724 (does NOT touch Qwen3 attention internals;
hooks into the model only via the existing ``attention_mask`` argument).

Formal definition (plan §1.2 lines 41-46):

    g_j  = σ(W · noise_j)                             # 7-dim → scalar gate ∈ (0,1)
    A_ij = softmax( Q_i K_jᵀ / √d  +  log(g_j + ε) )   # additive attention bias

Key-side gating only (the default plan choice; query-side / both are N7
ablations and live behind the ``gate_side`` knob below).

Implementation strategy:

1. ``NCAGGate``       — a tiny ``Linear(7, 1)`` + sigmoid (~8 parameters).
2. ``build_ncag_mask`` — combines the 2-D padding mask, the causal mask,
   and the gate-derived bias into a single 4-D additive mask shaped
   ``[B, 1, L_q, L_k]`` that we pass straight through to the base model.

We require ``attn_implementation="eager"`` on the base Qwen3 model so that
4-D float masks are honoured exactly — SDPA / FlashAttn currently silently
discard custom 4-D bias terms in transformers 4.57.

Reference: see ``scripts/analysis/ncag_pilot.py`` for the pilot judging
script (Spearman ρ(gate, conf) > 0.3 quantitative criterion from plan
line 728) and ``scripts/analysis/ncag_gate_correlation.py`` (N12 mechanism).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

# 7-dim noise feature schema — must stay aligned with noise_fusion.FEATURES
NCAG_NOISE_DIM = 7
# Numerical floor for log(g + eps) — gate ≥ ε guarantees finite bias.
DEFAULT_GATE_EPS = 1e-4


class NCAGGate(nn.Module):
    """Tiny per-token reliability gate.

    Parameters
    ----------
    in_dim
        Dimensionality of the noise feature vector (default 7).
    hidden_dim
        Optional hidden width — when set, the gate is a 2-layer MLP
        ``Linear(in_dim, hidden) → tanh → Linear(hidden, 1)``; when ``None``
        the gate is a single ``Linear(in_dim, 1)``. Default ``None`` to
        match the plan's "minimal additional parameters" framing.
    init_bias
        Initial value for the output bias. Set to ``+4.0`` so that the
        gate starts ≈ σ(4) ≈ 0.982 — i.e. behaves like an identity for
        all tokens at step 0, giving the optimiser a smooth warm-up.
    """

    def __init__(
        self,
        in_dim: int = NCAG_NOISE_DIM,
        hidden_dim: Optional[int] = None,
        init_bias: float = 4.0,
    ) -> None:
        super().__init__()
        if hidden_dim is None or hidden_dim <= 0:
            self.proj: nn.Module = nn.Linear(in_dim, 1)
        else:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        # Initialise so the very first forward pass yields g ≈ 1 for any
        # input — avoids attention collapse during the first few steps.
        with torch.no_grad():
            if isinstance(self.proj, nn.Linear):
                nn.init.zeros_(self.proj.weight)
                nn.init.constant_(self.proj.bias, init_bias)
            else:
                last = self.proj[-1]  # type: ignore[index]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.constant_(last.bias, init_bias)

    def forward(self, noise_values: torch.Tensor) -> torch.Tensor:
        """noise_values: ``[B, L, in_dim]`` → gate ``[B, L]`` in (0, 1)."""
        if noise_values.dim() != 3:
            raise ValueError(
                f"NCAGGate expects noise_values of shape [B, L, {NCAG_NOISE_DIM}], got {tuple(noise_values.shape)}"
            )
        logits = self.proj(noise_values).squeeze(-1)
        return torch.sigmoid(logits)


@dataclass
class NCAGConfig:
    """Bag of switches for the gate. Mostly placeholders for N7/N8 ablations.

    gate_side
        ``"key"`` (default, plan choice) — bias is ``[B, 1, 1, L_k]``,
            i.e. the gate of the *attended* token. Low-confidence tokens
            get down-weighted as keys.
        ``"query"`` — bias is ``[B, 1, L_q, 1]``; low-confidence tokens
            down-weight their *own* outgoing attention.
        ``"both"`` — sum of the two.
    eps
        Floor passed to ``log(g + eps)`` to keep the bias finite.
    """

    gate_side: str = "key"
    eps: float = DEFAULT_GATE_EPS


def build_ncag_mask(
    *,
    gate: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
    config: Optional[NCAGConfig] = None,
) -> torch.Tensor:
    """Compose the 4-D additive attention mask consumed by Qwen3 eager attn.

    Parameters
    ----------
    gate
        ``[B, L]`` gate values in (0, 1) from :class:`NCAGGate`.
    attention_mask
        Optional ``[B, L]`` 0/1 padding mask (1 = real token, 0 = pad).
        If ``None``, all tokens are assumed real.
    dtype
        Target dtype of the returned mask. Should match base-model dtype
        (typically ``torch.bfloat16`` during CPT). The "minus-infinity"
        floor below is replaced with the dtype's most negative finite
        value to stay safe under fp16 / bf16.
    config
        :class:`NCAGConfig` instance. Defaults to ``NCAGConfig()`` =
        key-side gating with eps = 1e-4.

    Returns
    -------
    torch.Tensor
        Shape ``[B, 1, L_q, L_k]``. Add to attention logits before the
        softmax — kept positions get a finite ≤ 0 bias from the gate;
        padded / future positions get a large-negative floor.
    """
    cfg = config or NCAGConfig()
    bsz, seqlen = gate.shape
    device = gate.device

    # log(g + eps) ∈ (-∞, 0]. Cast to high precision first to avoid bf16
    # underflow when gate is very small.
    log_gate = torch.log(gate.float().clamp(min=cfg.eps))

    # Causal mask: position i (query) may attend to positions ≤ i (key).
    # Construct in fp32 then cast at the end.
    neg_inf = torch.finfo(dtype).min
    causal = torch.zeros(seqlen, seqlen, device=device, dtype=torch.float32)
    causal.masked_fill_(
        torch.triu(torch.ones(seqlen, seqlen, device=device, dtype=torch.bool), diagonal=1),
        neg_inf,
    )
    # Broadcast to [B, 1, L_q, L_k]
    mask4d = causal.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seqlen, seqlen).clone()

    # Padding mask — both query-side rows and key-side cols of padded
    # tokens must be -inf. We do key-side (cols) here; query-side rows
    # contribute zero loss because they're labelled -100, so collapsing
    # their softmax is harmless. Still, mask them to keep gradients clean.
    if attention_mask is not None:
        pad = attention_mask.to(device=device).to(torch.bool)  # [B, L], True = real
        # Mask out columns of padded tokens
        key_mask = pad.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L_k]
        mask4d = mask4d.masked_fill(~key_mask, neg_inf)
        # Mask out rows of padded queries
        query_mask = pad.unsqueeze(1).unsqueeze(-1)  # [B, 1, L_q, 1]
        mask4d = mask4d.masked_fill(~query_mask, neg_inf)

    # Inject NCAG gate bias only on positions that are still finite —
    # we don't want to ever turn a -inf into a finite value.
    finite = mask4d > (neg_inf / 2)  # tolerant comparison
    if cfg.gate_side in {"key", "both"}:
        key_bias = log_gate.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L_k]
        mask4d = torch.where(finite, mask4d + key_bias, mask4d)
    if cfg.gate_side in {"query", "both"}:
        query_bias = log_gate.unsqueeze(1).unsqueeze(-1)  # [B, 1, L_q, 1]
        mask4d = torch.where(finite, mask4d + query_bias, mask4d)
    if cfg.gate_side not in {"key", "query", "both"}:
        raise ValueError(f"Unsupported gate_side: {cfg.gate_side!r}")

    return mask4d.to(dtype=dtype)


def gate_stats(gate: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> dict:
    """Cheap scalar telemetry on the gate values — for training logs.

    Returns mean / min / max / std over non-padded tokens. Useful for
    Day-1 pilot: if mean drifts toward 1.0 the gate is collapsing to an
    identity (no effect); if it drifts toward 0 the model is suppressing
    everything (likely diverging).
    """
    if attention_mask is None:
        flat = gate.flatten()
    else:
        mask = attention_mask.to(device=gate.device).to(torch.bool)
        flat = gate[mask]
    if flat.numel() == 0:
        return {"gate_mean": 0.0, "gate_min": 0.0, "gate_max": 0.0, "gate_std": 0.0}
    flat32 = flat.detach().float()
    return {
        "gate_mean": flat32.mean().item(),
        "gate_min": flat32.min().item(),
        "gate_max": flat32.max().item(),
        "gate_std": flat32.std().item() if flat32.numel() > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Mode helpers
# ---------------------------------------------------------------------------

NCAG_MODES = {"ncag", "ncag_additive"}


def uses_ncag(noise_mode: Optional[str]) -> bool:
    return str(noise_mode or "").lower() in NCAG_MODES


def ncag_keeps_additive(noise_mode: Optional[str]) -> bool:
    """True when both NCAG bias AND additive Noise-Embedding are applied (N4)."""
    return str(noise_mode or "").lower() == "ncag_additive"
