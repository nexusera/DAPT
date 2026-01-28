# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional


def default_config_path() -> str:
    """Return the canonical EBQA config path.

    Training and inference must read from pre_struct/ebqa/ebqa_config.json.
    """
    return str(Path(__file__).with_name("ebqa_config.json"))


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the EBQA JSON config strictly.

    - Reads only from the given path or the canonical ebqa_config.json.
    - No fallbacks. Raises if file missing or invalid.
    """
    cfg_path = path or default_config_path()
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"EBQA config not found: {cfg_path}")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("EBQA config must be a JSON object")
        return data
    except Exception as e:
        # surface parse errors explicitly
        raise ValueError(f"Failed to load EBQA config from {cfg_path}: {e}")


def _require_str(d: Dict[str, Any], key: str) -> str:
    if key not in d or not isinstance(d[key], str) or not d[key].strip():
        raise KeyError(f"Missing or invalid '{key}' in EBQA config")
    return d[key].strip()


def resolve_model_dir(cfg: Dict[str, Any]) -> str:
    """Return required model directory used for inference.

    No fallback to training output_dir or hardcoded paths.
    """
    return _require_str(cfg, "model_dir")


def resolve_tokenizer_name(cfg: Dict[str, Any]) -> str:
    """Return tokenizer resource name/path.

    Accepts either 'tokenizer_name_or_path', 'tokenizer_name', or 'model_name_or_path'.
    No hardcoded default path.
    """
    for k in ("tokenizer_name_or_path", "tokenizer_name", "model_name_or_path"):
        v = cfg.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    raise KeyError(
        "Missing tokenizer path: set 'tokenizer_name_or_path' (or 'tokenizer_name'/'model_name_or_path') in EBQA config"
    )


def resolve_report_struct_path(cfg: Dict[str, Any]) -> str:
    return _require_str(cfg, "report_struct_path")


def lengths_from(cfg: Dict[str, Any]) -> Dict[str, int]:
    """Strictly read length-related params from config."""
    try:
        return {
            "max_seq_len": int(cfg["max_seq_len"]),
            "max_tokens_ctx": int(cfg["max_tokens_ctx"]),
            "max_answer_len": int(cfg.get("max_answer_len", 1000)),  # 默认大值，由动态机制控制
        }
    except KeyError as e:
        raise KeyError(f"Missing length config: {e}")


def chunk_mode_from(cfg: Dict[str, Any]) -> str:
    cm = cfg.get("chunk_mode")
    if not isinstance(cm, str) or not cm.strip():
        raise KeyError("Missing 'chunk_mode' in EBQA config")
    return cm.strip()


def predict_block(cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = cfg.get("predict")
    if not isinstance(p, dict):
        raise KeyError("Missing 'predict' block in EBQA config")
    return p


def train_block(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t = cfg.get("train")
    if not isinstance(t, dict):
        raise KeyError("Missing 'train' block in EBQA config")
    return t
