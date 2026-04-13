# -*- coding: utf-8 -*-
"""
后处理：实体列表 → KV 配对 + structured 字典。

直接复用 compare_models.py 中的 _assemble_pairs 和 _postprocess_value_for_key，
不做任何修改以保证与实验脚本行为一致。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 注入 sys.path（与 model_engine.py 保持一致）
_DAPT_ROOT = Path(__file__).resolve().parents[2]
_EVAL_PKG = _DAPT_ROOT / "dapt_eval_package"
for _p in [str(_DAPT_ROOT), str(_EVAL_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pre_struct.kv_ner.compare_models import (
    _assemble_pairs,
    _postprocess_value_for_key,
)


def assemble_kv(
    entities: List[Dict[str, Any]],
    full_text: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    将实体列表转为 KV 配对、structured 字典、hospital 等结果。

    Args:
        entities: 模型输出的实体列表，每项含 {type, text, start, end}
        full_text: OCR 全文
        cfg: 后处理配置，键见 config.py（value_attach_window 等）

    Returns:
        {
            kv_pairs:         List[KVPair]
            structured:       Dict[str, str | List[str]]
            hospital:         str | None
            key_without_value: List[str]
            value_without_key: List[str]
        }
    """
    cfg = cfg or {}
    attach_window = int(cfg.get("value_attach_window", 50))
    same_line = bool(cfg.get("value_same_line_only", True))
    crossline_len = int(cfg.get("value_crossline_fallback_len", 0))

    assembled = _assemble_pairs(
        entities,
        value_attach_window=attach_window,
        value_same_line_only=same_line,
        value_crossline_fallback_len=crossline_len,
        full_text=full_text,
    )

    # ── 构造 kv_pairs（含 value 后处理） ────────────────────────────────────
        kv_pairs: List[Dict] = []
    for entry in assembled["pairs"]:
        key_ent = entry["key"]
        key_text: str = key_ent.get("text", "")
        if not key_text:
            continue

        vals: List[Dict] = entry.get("values", [])
        if not vals:
            continue

        # VALUE span 边界
        v0 = vals[0]
        vN = vals[-1]
        v_start = int(v0.get("start", 0))
        v_end = int(vN.get("end", 0))
        value_text = full_text[v_start:v_end] if 0 <= v_start < v_end <= len(full_text) else ""

        # value 后处理（截断、电话提取等）
        value_text, v_start, v_end = _postprocess_value_for_key(
            key_text, value_text, v_start, v_end, full_text, cfg
        )

        k_start = int(key_ent.get("start", 0))
        k_end = int(key_ent.get("end", 0))

        kv_pairs.append(
            {
                "key": key_text,
                "value": value_text,
                "key_span": [k_start, k_end],
                "value_span": [v_start, v_end],
            }
        )

    # ── hospital ─────────────────────────────────────────────────────────────
    hospital: Optional[str] = None
    hosp_entities = assembled.get("hospital", [])
    if hosp_entities:
        hospital = hosp_entities[0].get("text", "").strip() or None

    # ── key_without_value / value_without_key ────────────────────────────────
    kwv = [
        e.get("text", "").strip()
        for e in assembled.get("key_without_value", [])
        if isinstance(e, dict)
    ]
    vwk = [
        e.get("text", "").strip()
        for e in assembled.get("value_without_key", [])
        if isinstance(e, dict)
    ]

    return {
        "kv_pairs": kv_pairs,
        "structured": assembled.get("structured", {}),
        "hospital": hospital,
        "key_without_value": [k for k in kwv if k],
        "value_without_key": [v for v in vwk if v],
    }
