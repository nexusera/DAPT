# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


Span = Tuple[str, int, int]
CharSpan = Tuple[str, int, int]


def _label_name(label_id: int, id2label: Dict[int, str]) -> str:
    return id2label.get(int(label_id), "O")


def _token_spans(
    labels: Sequence[int],
    mask: Sequence[bool],
    id2label: Dict[int, str],
) -> List[Span]:
    spans: List[Span] = []
    current = None
    for idx, lab_id in enumerate(labels):
        if not mask[idx]:
            continue
        label = _label_name(lab_id, id2label)
        if label == "O" or "-" not in label:
            if current is not None:
                spans.append((current["type"], current["start"], idx))
                current = None
            continue
        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current is not None:
                spans.append((current["type"], current["start"], idx))
            current = {"type": ent_type, "start": idx}
        elif prefix == "I":
            if current is None or current["type"] != ent_type:
                if current is not None:
                    spans.append((current["type"], current["start"], idx))
                current = {"type": ent_type, "start": idx}
        elif prefix == "E":
            if current is None or current["type"] != ent_type:
                spans.append((ent_type, idx, idx + 1))
            else:
                spans.append((current["type"], current["start"], idx + 1))
            current = None
        else:
            if current is not None:
                spans.append((current["type"], current["start"], idx))
            current = None
    if current is not None:
        spans.append((current["type"], current["start"], len(labels)))
    return spans


def char_spans(
    labels: Sequence[int],
    mask: Sequence[bool],
    offsets: Sequence[Tuple[int, int]],
    id2label: Dict[int, str],
) -> List[CharSpan]:
    spans: List[CharSpan] = []
    current = None
    for idx, lab_id in enumerate(labels):
        if not mask[idx]:
            continue
        start_char, end_char = offsets[idx]
        if end_char <= start_char:
            continue
        label = _label_name(lab_id, id2label)
        if label == "O" or "-" not in label:
            if current is not None:
                spans.append((current["type"], current["start"], current["end"]))
                current = None
            continue
        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current is not None:
                spans.append((current["type"], current["start"], current["end"]))
            current = {"type": ent_type, "start": start_char, "end": end_char}
        elif prefix == "I":
            if current is None or current["type"] != ent_type:
                if current is not None:
                    spans.append((current["type"], current["start"], current["end"]))
                current = {"type": ent_type, "start": start_char, "end": end_char}
            else:
                current["end"] = end_char
        elif prefix == "E":
            if current is None or current["type"] != ent_type:
                spans.append((ent_type, start_char, end_char))
            else:
                current["end"] = end_char
                spans.append((current["type"], current["start"], current["end"]))
            current = None
        else:
            if current is not None:
                spans.append((current["type"], current["start"], current["end"]))
            current = None
    if current is not None:
        spans.append((current["type"], current["start"], current["end"]))
    return spans


def compute_ner_metrics(
    predictions: Sequence[Sequence[int]],
    references: Sequence[Sequence[int]],
    masks: Sequence[Sequence[bool]],
    id2label: Dict[int, str],
    *,
    offsets: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute entity-level precision/recall/F1 metrics."""
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    use_char = offsets is not None
    for i in range(len(predictions)):
        pred = predictions[i]
        gold = references[i]
        mask = masks[i]

        if use_char:
            pred_spans = char_spans(pred, mask, offsets[i], id2label)
            gold_spans = char_spans(gold, mask, offsets[i], id2label)
        else:
            pred_spans = _token_spans(pred, mask, id2label)
            gold_spans = _token_spans(gold, mask, id2label)

        pred_set = set(pred_spans)
        gold_set = set(gold_spans)

        for span in pred_set:
            ent_type = span[0]
            if span in gold_set:
                counts[ent_type]["tp"] += 1
            else:
                counts[ent_type]["fp"] += 1
        for span in gold_set:
            ent_type = span[0]
            if span not in pred_set:
                counts[ent_type]["fn"] += 1

    metrics: Dict[str, Dict[str, float]] = {}
    total_tp = total_fp = total_fn = 0
    for ent_type, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics[ent_type] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall)
        else 0.0
    )
    metrics["overall"] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "support": total_tp + total_fn,
    }
    return metrics
