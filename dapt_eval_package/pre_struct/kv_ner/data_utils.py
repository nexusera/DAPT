# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sklearn.model_selection import train_test_split
from .noise_utils import PERFECT_VALUES


@dataclass
class Entity:
    start: int
    end: int
    label: str
    result_id: Optional[str] = None
    text: Optional[str] = None


@dataclass
class Relation:
    from_id: str
    to_id: str
    direction: str = "right"


@dataclass
class Sample:
    task_id: str
    text: str
    title: str
    entities: List[Entity]
    relations: List[Relation]
    # 可选：与全文字符对齐的OCR噪声特征，长度通常等于 text 的字符数。
    # 每个元素为7维连续值 [conf_avg, conf_min, conf_var_log, conf_gap, punct_err_ratio, char_break_ratio, align_score]
    noise_values: Optional[List[List[float]]] = None

    @property
    def has_labels(self) -> bool:
        return any(self.entities)


def _select_latest_annotation(task: dict) -> List[dict]:
    # Revised: Check for 'annotations' OR 'transferred_annotations'
    # 优先检查标准的 annotations
    annotations = task.get("annotations")
    if annotations:
        valid = [a for a in annotations if not a.get("was_cancelled")]
        pool = valid if valid else annotations
        if pool:
            def _anno_key(anno: dict) -> Tuple[str, str]:
                return (str(anno.get("updated_at") or ""), str(anno.get("created_at") or ""))
            latest = sorted(pool, key=_anno_key)[-1]
            return latest.get("result", [])

    # 回退到 transferred_annotations (通常 MedStruct-S Real 格式直接存储结果列表)
    transferred = task.get("transferred_annotations")
    if transferred and isinstance(transferred, list) and len(transferred) > 0:
        first = transferred[0]

        # Case A: List[Annotation] (standard Label Studio export snippet)
        if isinstance(first, dict) and "result" in first:
             return first.get("result", [])

        # Case B: List[ResultItem] (Label Studio 'result' list directly)
        if isinstance(first, dict) and "value" in first and "type" in first:
             return transferred
        
        # Case C: MedStruct-S Real simplified format (direct entity list)
        converted = []
        # 1. Process Entities from 'transferred_annotations'
        full_text = task.get("ocr_text") or task.get("text") or ""
        for item in transferred:
            if not isinstance(item, dict):
                continue
            labels = item.get("labels")
            if not labels:
                continue
            
            val_text = item.get("text", "")
            
            # Estimate start/end if missing
            start_idx = item.get("start")
            end_idx = item.get("end")
            
            if (start_idx is None or end_idx is None) and val_text and full_text:
                # Find first occurrence of val_text in full_text
                found = full_text.find(val_text)
                if found != -1:
                    start_idx = found
                    end_idx = found + len(val_text)
                else:
                    # Text not found, cannot anchor entity
                    continue
            
            if start_idx is None or end_idx is None:
                continue

            converted.append({
                "type": "labels",
                "value": {
                    "labels": labels,
                    "text": val_text,
                    "start": start_idx,
                    "end": end_idx
                },
                "id": item.get("original_id") # Use original_id for linking relations
            })
            
        # 2. Process Relations from root 'relations' key
        root_relations = task.get("relations")
        if root_relations and isinstance(root_relations, list):
            for rel in root_relations:
                if not isinstance(rel, dict):
                    continue
                # Ensure it has from_id and to_id
                if "from_id" in rel and "to_id" in rel:
                    converted.append({
                        "type": "relation",
                        "from_id": rel["from_id"],
                        "to_id": rel["to_id"],
                        "direction": rel.get("direction", "right")
                    })
            
        return converted
        
    return []


def _normalize_label(raw_label: str, label_map: Dict[str, str]) -> Optional[str]:
    if not raw_label:
        return None
    raw = raw_label.strip()
    return label_map.get(raw)


def _expand_word_noise_to_chars(ocr_raw, noise_values_per_word):
    """Expand per-word 7-d noise to per-character list using ocr_raw.words_result."""
    if not (isinstance(ocr_raw, dict) and isinstance(noise_values_per_word, list)):
        return None
    words_result = ocr_raw.get("words_result")
    if not isinstance(words_result, list):
        return None
    char_noise = []
    for wr, nv in zip(words_result, noise_values_per_word):
        if not (isinstance(wr, dict) and isinstance(nv, (list, tuple)) and len(nv) == 7):
            continue
        w = wr.get("words", "")
        if not isinstance(w, str):
            continue
        repeat = max(1, len(w))
        char_noise.extend([list(nv)] * repeat)
    return char_noise if char_noise else None


def _broadcast_global_noise(noise_values, text_len: int):
    """If noise is a single 7-d vector, broadcast to text length."""
    if (
        isinstance(noise_values, list)
        and len(noise_values) == 7
        and all(not isinstance(v, (list, tuple)) for v in noise_values)
        and text_len > 0
    ):
        return [list(noise_values) for _ in range(text_len)]
    return noise_values


def load_labelstudio_export(
    path: str | Path,
    label_map: Dict[str, str],
    *,
    include_unlabeled: bool = True,
) -> List[Sample]:
    """Load a Label Studio JSON export and convert it into Sample objects."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Label Studio export not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Label Studio export must be a JSON array")

    samples: List[Sample] = []
    for task in data:
        data_block = task.get("data", {}) if isinstance(task, dict) else {}
        try:
            # Revised: check for text in data_block first, then fallback to task root
            text = str(data_block.get("ocr_text") or data_block.get("text") or task.get("ocr_text") or task.get("text") or "")
        except Exception:
            text = ""
        if not text:
            if include_unlabeled:
                # Revised: check for category in data_block first, then fallback to task root
                title_val = str(data_block.get("category") or task.get("category") or "")
                samples.append(
                    Sample(
                        task_id=str(task.get("id")),
                        text="",
                        title=title_val,
                        entities=[],
                        relations=[],
                    )
                )
            continue

        # 读取噪声：优先 per-word 扩展，其次 noise_values / data.noise_values / task.noise_values
        # 优先使用 OCR 的 words_result 与 per-word 噪声，展开到字符级
        ocr_raw = task.get("ocr_raw") or data_block.get("ocr_raw")
        per_word_noise = data_block.get("noise_values_per_word") or task.get("noise_values_per_word")
        noise_values = _expand_word_noise_to_chars(ocr_raw, per_word_noise)
        if noise_values is None:
            # 回退：直接读取已对齐到字符的 noise_values
            noise_values = data_block.get("noise_values") or task.get("noise_values")
        noise_values = _broadcast_global_noise(noise_values, len(text))
        if noise_values is None:
            # 最终回退：用完美值填充，确保下游不会出现 None
            noise_values = [list(PERFECT_VALUES) for _ in range(len(text))] if len(text) > 0 else None

        results = _select_latest_annotation(task)
        entities: List[Entity] = []
        relations: List[Relation] = []

        for res in results:
            r_type = res.get("type")
            if r_type == "labels":
                value = res.get("value") or {}
                raw_labels = value.get("labels") or []
                if not raw_labels:
                    continue
                normalized = _normalize_label(raw_labels[0], label_map)
                if not normalized:
                    continue
                start = int(value.get("start") or 0)
                end = int(value.get("end") or 0)
                if end <= start or start < 0:
                    continue
                entities.append(
                    Entity(
                        start=start,
                        end=end,
                        label=normalized,
                        result_id=res.get("id"),
                        text=value.get("text"),
                    )
                )
            elif r_type == "relation":
                from_id = res.get("from_id")
                to_id = res.get("to_id")
                if isinstance(from_id, str) and isinstance(to_id, str):
                    relations.append(
                        Relation(
                            from_id=from_id,
                            to_id=to_id,
                            direction=str(res.get("direction") or "right"),
                        )
                    )

        sample = Sample(
            task_id=str(task.get("id")),
            text=text,
            title=str(data_block.get("category") or ""),
            entities=sorted(entities, key=lambda e: (e.start, e.end)),
            relations=relations,
            noise_values=noise_values,
        )
        if sample.entities or include_unlabeled:
            samples.append(sample)
    return samples


def split_samples(
    samples: Sequence[Sample],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Split annotated samples into train/val/test subsets."""
    if train_ratio <= 0 or val_ratio < 0 or (train_ratio + val_ratio) >= 1:
        raise ValueError("Train/val ratios must satisfy: train>0, val>=0, train+val<1")
    filtered = [s for s in samples if s.has_labels]
    if not filtered:
        raise ValueError("No annotated samples available for splitting")

    if len(filtered) < 3:
        dup = filtered * (3 // len(filtered) + 1)
        filtered = dup[:3]

    train_samples, tmp = train_test_split(
        filtered,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )
    remaining_ratio = 1.0 - train_ratio
    if val_ratio == 0:
        return train_samples, [], tmp

    val_rel = val_ratio / remaining_ratio
    if val_rel >= 1.0:
        raise ValueError("Invalid val_ratio resulting in empty test split")
    val_samples, test_samples = train_test_split(
        tmp,
        train_size=val_rel,
        random_state=seed + 1,
        shuffle=True,
    )
    return train_samples, val_samples, test_samples


def build_bioe_label_list(label_map: Dict[str, str]) -> List[str]:
    """Return BIOE labels sorted alphabetically for stable id mapping."""
    base_labels = sorted(set(label_map.values()))
    labels: List[str] = []
    for base in base_labels:
        base = base.upper()
        labels.append(f"B-{base}")
        labels.append(f"I-{base}")
        labels.append(f"E-{base}")
    labels.append("O")
    return labels


def build_bio_label_list(label_map: Dict[str, str]) -> List[str]:
    """Backward-compatible alias for BIOE schemas."""
    return build_bioe_label_list(label_map)


def generate_char_labels(length: int, entities: Iterable[Entity]) -> List[str]:
    labels = ["O"] * length
    for ent in entities:
        start = max(0, min(length, ent.start))
        end = max(0, min(length, ent.end))
        if end <= start:
            continue
        base = ent.label.upper()
        span_len = end - start
        if span_len == 1:
            labels[start] = f"E-{base}"
            continue
        labels[start] = f"B-{base}"
        labels[end - 1] = f"E-{base}"
        for idx in range(start + 1, end - 1):
            labels[idx] = f"I-{base}"
    return labels
