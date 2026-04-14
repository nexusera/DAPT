# -*- coding: utf-8 -*-
"""
后处理：实体列表 → KV 配对 + structured 字典。

直接复用 compare_models.py 中的 _assemble_pairs 和 _postprocess_value_for_key，
不做任何修改以保证与实验脚本行为一致。

额外增加两个推理层扩展（不修改原始逻辑，作为 serving 层后处理）：
  1. VALUE 边界扩展（adjust_boundaries）：将模型截断的日期/数字型 value 向周围延伸。
  2. 孤儿回链（backlink）：key_without_value 中的 KEY 与 value_without_key 中的 VALUE 按位置匹配。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# 注入 sys.path（与 model_engine.py 保持一致）
_DAPT_ROOT = Path(__file__).resolve().parents[2]
_EVAL_PKG = _DAPT_ROOT / "dapt_eval_package"
for _p in [str(_DAPT_ROOT), str(_EVAL_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pre_struct.kv_ner.predict import (
    _assemble_pairs,
    _postprocess_value_for_key,
)

# 默认边界扩展字符集（数字、日期分隔符、小数点）
_DEFAULT_ADJUST_CHARS: Set[str] = set("0123456789./年月日时分秒-")
# 默认最大扩展步长（字符数）
_DEFAULT_ADJUST_MAX_SHIFT: int = 12
# 孤儿回链最大距离（字符数）：orphan VALUE 的起点距离 KEY 结束不超过此距离才回链
_DEFAULT_BACKLINK_WINDOW: int = 300


def _adjust_entity_boundaries(
    entities: List[Dict[str, Any]],
    full_text: str,
    adjust_chars: Set[str],
    max_shift: int,
) -> List[Dict[str, Any]]:
    """
    对 VALUE 实体进行边界向外扩展：
      - 向左扩展：若 start-1 字符在 adjust_chars 中，则左移
      - 向右扩展：若 end 字符在 adjust_chars 中，则右移
    KEY / HOSPITAL 类实体不做修改。

    典型场景：模型将"2025/12/3"中的末尾"3"标注为 VALUE；
    扩展后可恢复为完整日期串。
    """
    L = len(full_text)
    result: List[Dict[str, Any]] = []
    for ent in entities:
        if ent.get("type") != "VALUE":
            result.append(ent)
            continue

        s = int(ent.get("start", 0))
        e = int(ent.get("end", 0))

        # 向左扩展
        for _ in range(max_shift):
            if s > 0 and full_text[s - 1] in adjust_chars:
                s -= 1
            else:
                break

        # 向右扩展
        for _ in range(max_shift):
            if e < L and full_text[e] in adjust_chars:
                e += 1
            else:
                break

        if s != ent["start"] or e != ent["end"]:
            ent = dict(ent)
            ent["start"] = s
            ent["end"] = e
            ent["text"] = full_text[s:e].strip()

        result.append(ent)
    return result


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
            kv_pairs:          List[dict]
            structured:        Dict[str, str | List[str]]
            hospital:          str | None
            key_without_value: List[str]
            value_without_key: List[str]
        }
    """
    cfg = cfg or {}
    attach_window = int(cfg.get("value_attach_window", 50))
    same_line = bool(cfg.get("value_same_line_only", True))
    crossline_len = int(cfg.get("value_crossline_fallback_len", 0))

    # ── 1. 边界扩展（可选）────────────────────────────────────────────────────
    if cfg.get("adjust_boundaries", True):
        adj_chars_cfg = cfg.get("adjust_chars", None)
        adj_chars: Set[str] = (
            set(str(adj_chars_cfg)) if adj_chars_cfg is not None else _DEFAULT_ADJUST_CHARS
        )
        max_shift = int(cfg.get("adjust_max_shift", _DEFAULT_ADJUST_MAX_SHIFT))
        entities = _adjust_entity_boundaries(entities, full_text, adj_chars, max_shift)

    # ── 2. 主配对逻辑（直接复用 predict.py 的实现） ──────────────────────────
    assembled = _assemble_pairs(
        entities,
        value_attach_window=attach_window,
        value_same_line_only=same_line,
        value_crossline_fallback_len=crossline_len,
        full_text=full_text,
    )

    # ── 3. 构造 kv_pairs（含 value 后处理） ─────────────────────────────────
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

    # ── 4. hospital ───────────────────────────────────────────────────────────
    hospital: Optional[str] = None
    hosp_entities = assembled.get("hospital", [])
    if hosp_entities:
        hospital = hosp_entities[0].get("text", "").strip() or None

    # ── 5. key_without_value / value_without_key ──────────────────────────────
    raw_kwv: List[Dict] = assembled.get("key_without_value", [])
    raw_vwk: List[Dict] = assembled.get("value_without_key", [])

    kwv_texts = [e.get("text", "").strip() for e in raw_kwv if isinstance(e, dict)]
    vwk_texts = [e.get("text", "").strip() for e in raw_vwk if isinstance(e, dict)]

    # ── 6. 孤儿回链：将 key_without_value 与 value_without_key 按位置配对 ──────
    backlink_window = int(cfg.get("backlink_window", _DEFAULT_BACKLINK_WINDOW))
    if cfg.get("enable_backlink", True) and raw_kwv and raw_vwk:
        orphan_values = sorted(
            [e for e in raw_vwk if isinstance(e, dict)],
            key=lambda e: e.get("start", 0),
        )
        used_val_indices: Set[int] = set()

        for key_ent in raw_kwv:
            if not isinstance(key_ent, dict):
                continue
            key_text = key_ent.get("text", "").strip()
            if not key_text:
                continue
            key_end = int(key_ent.get("end", 0))
            k_start = int(key_ent.get("start", 0))
            k_end_pos = int(key_ent.get("end", 0))

            # 找最近的、尚未使用的 orphan VALUE（在 key 结束后 backlink_window 字符内）
            best_idx: Optional[int] = None
            best_dist = float("inf")
            for vi, val_ent in enumerate(orphan_values):
                if vi in used_val_indices:
                    continue
                val_start = int(val_ent.get("start", 0))
                # VALUE 必须在 KEY 之后
                dist = val_start - key_end
                if 0 <= dist <= backlink_window and dist < best_dist:
                    best_dist = dist
                    best_idx = vi

            if best_idx is not None:
                val_ent = orphan_values[best_idx]
                used_val_indices.add(best_idx)
                val_text = val_ent.get("text", "").strip()
                val_start = int(val_ent.get("start", 0))
                val_end = int(val_ent.get("end", 0))

                # value 后处理
                val_text, val_start, val_end = _postprocess_value_for_key(
                    key_text, val_text, val_start, val_end, full_text, cfg
                )

                kv_pairs.append(
                    {
                        "key": key_text,
                        "value": val_text,
                        "key_span": [k_start, k_end_pos],
                        "value_span": [val_start, val_end],
                        "_backlinked": True,
                    }
                )

    # 重建 structured（合并 kv_pairs 与 backlinked）
    structured: Dict[str, Any] = {}
    for pair in kv_pairs:
        kt = pair["key"]
        vt = pair["value"]
        existing = structured.get(kt)
        if existing is None:
            structured[kt] = vt
        elif isinstance(existing, list):
            existing.append(vt)
        else:
            structured[kt] = [existing, vt]

    # 同步 assembled.structured（保留 hospital 等原始字段）
    assembled_struct: Dict[str, Any] = assembled.get("structured", {})
    for kt, vt in assembled_struct.items():
        if kt not in structured:
            structured[kt] = vt

    return {
        "kv_pairs": kv_pairs,
        "structured": structured,
        "hospital": hospital,
        "key_without_value": [k for k in kwv_texts if k],
        "value_without_key": [v for v in vwk_texts if v],
    }
