# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Config helpers for the KV-NER pipeline.

Training and inference scripts must load configuration from
`pre_struct/kv_ner/kv_ner_config.json` (or an explicitly provided path).
The schema mirrors the EBQA config but exposes fields that are specific to
sequence labelling (e.g. label map, CRF options, dataset splits).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def default_config_path() -> str:
    """Return the canonical config path for the KV-NER pipeline."""
    return str(Path(__file__).with_name("kv_ner_config.json"))


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the JSON config strictly from disk."""
    cfg_path = path or default_config_path()
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"KV-NER config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("KV-NER config must be a JSON object")
    return data


def ensure_block(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    block = cfg.get(key)
    if not isinstance(block, dict):
        raise KeyError(f"Missing '{key}' block in KV-NER config")
    return block


def model_name_from(cfg: Dict[str, Any]) -> str:
    for key in ("model_name_or_path", "model_dir", "model_path"):
        v = cfg.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    raise KeyError("Missing model path in KV-NER config")


def tokenizer_name_from(cfg: Dict[str, Any]) -> str:
    for key in ("tokenizer_name_or_path", "tokenizer_name", "model_name_or_path"):
        v = cfg.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    raise KeyError("Missing tokenizer path in KV-NER config")


def label_map_from(cfg: Dict[str, Any]) -> Dict[str, str]:
    lm = cfg.get("label_map")
    if not isinstance(lm, dict) or not lm:
        raise KeyError("KV-NER config must provide a non-empty 'label_map'")
    cleaned: Dict[str, str] = {}
    for raw, normalized in lm.items():
        if not isinstance(raw, str) or not isinstance(normalized, str):
            raise ValueError("label_map must map strings to strings")
        cleaned[raw.strip()] = normalized.strip().upper()
    return cleaned


def max_seq_length(cfg: Dict[str, Any]) -> int:
    try:
        return int(cfg.get("max_seq_length", 512))
    except Exception as exc:
        raise ValueError("Invalid 'max_seq_length' in KV-NER config") from exc


def label_all_tokens(cfg: Dict[str, Any]) -> bool:
    return bool(cfg.get("label_all_tokens", True))
