import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn


FEATURES = [
    "conf_avg",
    "conf_min",
    "conf_var_log",
    "conf_gap",
    "punct_err_ratio",
    "char_break_ratio",
    "align_score",
]

PERFECT_VALUES = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

DEFAULT_FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "conf_avg": (0.0, 1.0),
    "conf_min": (0.0, 1.0),
    "conf_var_log": (-12.0, 0.0),
    "conf_gap": (0.0, 1.0),
    "punct_err_ratio": (0.0, 1.0),
    "char_break_ratio": (0.0, 0.25),
    "align_score": (0.0, 3500.0),
}


def uses_continuous_noise(noise_mode: Optional[str]) -> bool:
    return str(noise_mode or "bucket").lower() in {"linear", "mlp"}


def uses_bucket_noise(noise_mode: Optional[str]) -> bool:
    return str(noise_mode or "bucket").lower() == "bucket"


def load_noise_bin_edges(path: Optional[str]) -> Dict[str, List[float]]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def build_feature_ranges(bin_edges: Optional[Dict[str, List[float]]] = None) -> Dict[str, Tuple[float, float]]:
    ranges: Dict[str, Tuple[float, float]] = {}
    bin_edges = bin_edges or {}
    for feat in FEATURES:
        lo, hi = DEFAULT_FEATURE_RANGES[feat]
        edges = bin_edges.get(feat) or []
        if edges:
            try:
                edge_lo = float(edges[0])
                edge_hi = float(edges[-1])
                if feat == "conf_var_log":
                    lo = min(lo, edge_lo)
                hi = max(edge_hi, lo + 1e-6)
            except Exception:
                pass
        if hi <= lo:
            hi = lo + 1.0
        ranges[feat] = (float(lo), float(hi))
    return ranges


class ContinuousNoiseProjector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        mode: str = "linear",
        dropout: float = 0.1,
        mlp_hidden_dim: Optional[int] = None,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        super().__init__()
        self.mode = str(mode or "linear").lower()
        if self.mode not in {"linear", "mlp"}:
            raise ValueError(f"Unsupported continuous noise mode: {self.mode}")

        ranges = build_feature_ranges(feature_ranges)
        mins = [ranges[f][0] for f in FEATURES]
        maxs = [ranges[f][1] for f in FEATURES]
        self.register_buffer("range_mins", torch.tensor(mins, dtype=torch.float32), persistent=True)
        self.register_buffer("range_maxs", torch.tensor(maxs, dtype=torch.float32), persistent=True)

        if self.mode == "linear":
            self.proj = nn.Linear(len(FEATURES), hidden_size)
        else:
            hid = int(mlp_hidden_dim or max(32, hidden_size // 4))
            self.proj = nn.Sequential(
                nn.Linear(len(FEATURES), hid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hid, hidden_size),
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def normalize(self, noise_values: torch.Tensor) -> torch.Tensor:
        x = noise_values.to(dtype=torch.float32)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mins = self.range_mins.to(device=x.device)
        maxs = self.range_maxs.to(device=x.device)
        x = torch.max(torch.min(x, maxs), mins)
        denom = torch.clamp(maxs - mins, min=1e-6)
        x = (x - mins) / denom
        x = x * 2.0 - 1.0
        return x

    def forward(self, noise_values: torch.Tensor) -> torch.Tensor:
        return self.proj(self.normalize(noise_values))


def _is_global_noise_vector(noise_values: Any) -> bool:
    return (
        isinstance(noise_values, list)
        and len(noise_values) == len(FEATURES)
        and all(not isinstance(v, (list, tuple)) for v in noise_values)
    )


def aggregate_token_noise_values(
    offset_mapping: Sequence[Sequence[int]],
    noise_values: Any,
    *,
    chunk_char_start: int = 0,
    perfect_values: Optional[List[float]] = None,
) -> List[List[float]]:
    perfect = list(perfect_values or PERFECT_VALUES)
    if _is_global_noise_vector(noise_values):
        return [list(noise_values) for _ in offset_mapping]

    if not isinstance(noise_values, list):
        noise_values = []

    out: List[List[float]] = []
    text_len = len(noise_values)
    for offset in offset_mapping:
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            out.append(list(perfect))
            continue
        s, e = offset
        if s is None or e is None:
            out.append(list(perfect))
            continue
        s = int(s)
        e = int(e)
        if e <= s:
            out.append(list(perfect))
            continue
        abs_s = chunk_char_start + s
        abs_e = chunk_char_start + e
        vecs: List[List[float]] = []
        for ci in range(abs_s, min(abs_e, text_len)):
            row = noise_values[ci]
            if isinstance(row, (list, tuple)) and len(row) == len(FEATURES):
                vecs.append([float(v) for v in row])
        if not vecs:
            out.append(list(perfect))
            continue
        avg = [sum(col) / len(col) for col in zip(*vecs)]
        out.append(avg)
    return out
