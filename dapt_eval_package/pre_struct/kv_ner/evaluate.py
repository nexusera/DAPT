#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KV-NER 模型评估脚本

参考 pre_struct/evaluate.py，为 KV-NER 模型提供键值对级别的评估：
1. 读取评估数据（JSONL 格式，包含 spans）
2. 使用模型预测 NER 实体并组装成键值对
3. 使用 evaluation 库计算键值对匹配的 F1 指标
4. 生成评估报告和错误样本分析
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import unicodedata
import re
import string

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

if __package__ in (None, ""):
    _PACKAGE_ROOT = Path(__file__).resolve().parents[2]
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))
    from pre_struct.kv_ner import config_io
    from pre_struct.kv_ner.data_utils import build_bio_label_list
    from pre_struct.kv_ner.modeling import BertCrfTokenClassifier
    from pre_struct.kv_ner.chunking import predict_with_chunking
    # H4: 从公共模块导入共享工具函数，避免与 evaluate_with_dapt_noise.py 重复
    from pre_struct.kv_ner.evaluate_core import (
        set_seed as _set_seed_core,
        _read_jsonl,
        _normalize_text_for_eval as _normalize_text_core,
        _extract_ground_truth,
    )
else:
    from . import config_io
    from .data_utils import build_bio_label_list
    from .modeling import BertCrfTokenClassifier
    from .chunking import predict_with_chunking
    # H4: 从公共模块导入共享工具函数，避免与 evaluate_with_dapt_noise.py 重复
    from .evaluate_core import (
        set_seed as _set_seed_core,
        _read_jsonl,
        _normalize_text_for_eval as _normalize_text_core,
        _extract_ground_truth,
    )

sys.path.append(str(_PACKAGE_ROOT if '_PACKAGE_ROOT' in locals() else Path.cwd()))
from evaluation.src.easy_eval import evaluate_entities  # type: ignore

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# H4: set_seed 和 _read_jsonl 已移至 evaluate_core.py，通过上方 import 引入。
def set_seed(seed: Optional[int]) -> None:
    _set_seed_core(seed)


def strip_trailing_punctuation(text: str, start: int, end: int) -> Tuple[str, int, int]:
    trailing_puncts = set(' \t\u3000\r\n。，、；：？！,.;:?![]【】()（）「」『』""\'\'…—')
    
    original_text = text
    
    while text and text[0] in trailing_puncts:
        text = text[1:]
        start += 1
    
    while text and text[-1] in trailing_puncts:
        text = text[:-1]
        end -= 1
    
    if not text:
        return original_text, start, end
    
    return text, start, end


# H4: 与 evaluate_with_dapt_noise.py 共享的 _normalize_text_for_eval 已提取到 evaluate_core.py；
# 此处保留同名绑定以维持内部调用不变。
_normalize_text_for_eval = _normalize_text_core


EDGE_STRIP_CHARS = set(string.whitespace) | set("。，、；：？！,.!?;:()（）[]【】{}<>「」『』""''\"'…—-_/\\") | {"\u3000"}
BOUNDARY_NOISE_CHARS = set(string.whitespace) | set("：:") | set(string.digits) | set("()（）[]【】") | {"\u3000"}
TOLERANT_TRAILING_CHARS = set("。，、；：？！,.!?;:") | set(string.whitespace) | {"\u3000"}


def _strip_edge_chars_limited(text: str, max_remove: int = 3) -> Tuple[str, int]:
    if not text or max_remove <= 0:
        return text, 0
    left = 0
    right = len(text)
    removed = 0
    while left < right and removed < max_remove and text[left] in EDGE_STRIP_CHARS:
        left += 1
        removed += 1
    while left < right and removed < max_remove and text[right - 1] in EDGE_STRIP_CHARS:
        right -= 1
        removed += 1
    return text[left:right], removed


def _strip_boundary_noise(text: str) -> str:
    if not text:
        return ""
    left = 0
    right = len(text)
    while left < right and text[left] in BOUNDARY_NOISE_CHARS:
        left += 1
    while left < right and text[right - 1] in BOUNDARY_NOISE_CHARS:
        right -= 1
    return text[left:right]


def _strip_boundary_noise_with_offset(text: str, start: int, end: int) -> Tuple[str, int, int]:
    if not text:
        return text, start, end
    
    original_text = text
    left = 0
    right = len(text)
    
    while left < right and text[left] in BOUNDARY_NOISE_CHARS:
        left += 1
    
    while left < right and text[right - 1] in BOUNDARY_NOISE_CHARS:
        right -= 1
    
    new_start = start + left
    new_end = start + right
    new_text = text[left:right]
    
    return new_text, new_start, new_end


def _normalize_key_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = re.sub(r"\s+", "", s)
    while s and s[-1] in ":：;；、。，,.;":
        s = s[:-1]
    return s


def _accumulate_char_freq(freq: Dict[str, int], s: str) -> None:
    for ch in s:
        if not ch:
            continue
        freq[ch] = freq.get(ch, 0) + 1


def _top_k_items(d: Dict[Any, int], k: int = 20) -> Dict[str, int]:
    return {str(k_): int(v) for k_, v in sorted(d.items(), key=lambda x: -x[1])[:k]}


def _char_to_token_index(offsets: List[Tuple[int, int]], pos: int, right: bool = False) -> int:
    if pos <= 0:
        return 0
    n = len(offsets)
    for i, (s, e) in enumerate(offsets):
        if s <= pos < e:
            return i
    if right:
        for i in range(n - 1, -1, -1):
            if offsets[i][1] <= pos:
                return i
        return 0
    else:
        for i in range(n):
            if offsets[i][0] >= pos:
                return i
        return n - 1


def _is_valid_span(v: dict) -> bool:
    try:
        s = int(v.get("start", -1))
        e = int(v.get("end", -1))
        t = str(v.get("text", "")).strip()
        return (s >= 0) and (e > s) and bool(t)
    except Exception:
        return False


def _to_eval_pack(spans_or_preds: Dict[str, Any], exclude_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    exclude_keys = exclude_keys or []
    ents: List[Dict[str, Any]] = []
    
    for k, v in (spans_or_preds or {}).items():
        if k in exclude_keys:
            continue
        
        if not isinstance(v, dict):
            continue
        if not _is_valid_span(v):
            continue
        s = int(v["start"])
        e = int(v["end"])
        t = str(v.get("text", "")).strip()
        
        t_clean, s_clean, e_clean = strip_trailing_punctuation(t, s, e)
        t_norm = _normalize_text_for_eval(t_clean)
        ents.append({"start": s_clean, "end": e_clean, "text": t_norm})
    return {"entities": ents}


def predict_entities(
    model: BertCrfTokenClassifier,
    tokenizer: AutoTokenizer,
    text: str,
    id2label: Dict[int, str],
    device: torch.device,
    max_seq_length: int = 512,
    chunk_size: int = 450,
    chunk_overlap: int = 50,
    merge_adjacent_gap: int = 2,
    adjust_boundaries: bool = False,
    adjust_max_shift: int = 1,
    adjust_chars: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ents = predict_with_chunking(
        text=text,
        model=model,
        tokenizer=tokenizer,
        id2label=id2label,
        device=device,
        max_seq_length=max_seq_length,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        merge_adjacent_gap=merge_adjacent_gap,
    )
    if adjust_boundaries and ents:
        chars = set(adjust_chars or "")
        L = len(text)
        adjusted: List[Dict[str, Any]] = []
        for ent in ents:
            if str(ent.get("type")) == "KEY":
                adjusted.append(ent)
                continue
            s = int(ent.get("start", -1))
            e = int(ent.get("end", -1))
            if 0 <= s < e <= L:
                for _ in range(int(adjust_max_shift)):
                    if s > 0 and text[s-1] in chars:
                        s -= 1
                    else:
                        break
                for _ in range(int(adjust_max_shift)):
                    if e < L and text[e] in chars:
                        e += 1
                    else:
                        break
                ent = dict(ent)
                ent["start"] = s
                ent["end"] = e
                ent["text"] = text[s:e].strip()
            adjusted.append(ent)
        return adjusted
    return ents


def assemble_key_value_pairs(
    entities: List[Dict[str, Any]],
    text: str,
    *,
    value_attach_window: int = 50,
    value_same_line_only: bool = True,
    value_crossline_fallback_len: int = 0,
) -> Dict[str, Dict[str, Any]]:
    seq = [e for e in entities if e["type"] in {"KEY", "VALUE"}]
    hospitals = [e for e in entities if e["type"] == "HOSPITAL"]
    seq.sort(key=lambda x: (x["start"], x["end"]))
    
    pairs: List[Tuple[Dict, List[Dict]]] = []
    pending: Optional[Tuple[Dict, List[Dict]]] = None
    
    for idx, ent in enumerate(seq):
        if ent["type"] == "KEY":
            if pending:
                pairs.append(pending)
            pending = (ent, [])
        elif ent["type"] == "VALUE":
            if pending:
                pending[1].append(ent)
            else:
                attached = False
                for j in range(idx - 1, -1, -1):
                    prev = seq[j]
                    if prev["type"] != "KEY":
                        continue
                    if ent["start"] - prev["end"] <= value_attach_window:
                        if value_same_line_only:
                            middle = text[prev["end"]:ent["start"]]
                            if "\n" in middle and len(ent.get("text", "")) > int(value_crossline_fallback_len):
                                continue
                        pairs.append((prev, [ent]))
                        attached = True
                        break
                if not attached:
                    pass
            
    if pending:
        pairs.append(pending)
    
    result: Dict[str, Dict[str, Any]] = {}
    tol_keys_env = os.environ.get("KVNER_TOL_KEYS")
    tol_set = {"现病史", "体格检查", "病理诊断", "治疗计划", "处理", "注意事项", "既往史", "个人史", "婚育史", "辅助检查"}
    if tol_keys_env:
        for k in tol_keys_env.split(","):
            k = k.strip()
            if k:
                tol_set.add(k)
    tolerant_keys_norm = {_normalize_key_name(k) for k in tol_set}
    
    all_kvs = []
    
    for key_ent, value_ents in pairs:
        key_text = key_ent["text"]
        if not key_text:
            continue
        
        if value_ents:
            first_val = value_ents[0]
            last_val = value_ents[-1]
            span_start = int(first_val["start"]) if isinstance(first_val.get("start"), int) else first_val["start"]
            span_end = int(last_val["end"]) if isinstance(last_val.get("end"), int) else last_val["end"]
            span_start = max(0, min(len(text), span_start))
            span_end = max(span_start, min(len(text), span_end))
            key_norm = _normalize_key_name(key_text)
            expand_to_sentence = key_norm in tolerant_keys_norm and bool(int(os.environ.get("KVNER_EXPAND_SENTENCE", "0")))
            if expand_to_sentence:
                stopset = set("。；;.!？！?\n")
                limit = int(os.environ.get("KVNER_EXPAND_MAX", "120"))
                i = span_end
                L = len(text)
                steps = 0
                while i < L and steps < limit:
                    ch = text[i]
                    i += 1
                    steps += 1
                    if ch in stopset:
                        span_end = i
                        break
            slice_text = text[span_start:span_end]
            clean_text, clean_start, clean_end = strip_trailing_punctuation(
                slice_text, span_start, span_end
            )
            all_kvs.append((
                key_text,
                clean_start,
                clean_end,
                clean_text,
                False,
                key_ent["start"],
            ))
        else:
            all_kvs.append((
                key_text,
                key_ent["end"],
                key_ent["end"],
                "",
                False,
                key_ent["start"],
            ))
    
    for hospital_ent in hospitals:
        hospital_text = hospital_ent.get("text", "").strip()
        if hospital_text:
            all_kvs.append((
                "医院名称",
                hospital_ent["start"],
                hospital_ent["end"],
                hospital_text,
                True,
                hospital_ent["start"],
            ))
    
    all_kvs.sort(key=lambda x: (0 if x[4] else 1, x[5]))
    
    result = {}
    for key_text, start, end, text_content, _, _ in all_kvs:
        result[key_text] = {
            "start": start,
            "end": end,
            "text": text_content,
        }
    
    return result


def evaluate_dataset(
    model: BertCrfTokenClassifier,
    tokenizer: AutoTokenizer,
    test_data: List[Dict[str, Any]],
    id2label: Dict[int, str],
    device: torch.device,
    max_seq_length: int = 512,
    chunk_size: int = 450,
    chunk_overlap: int = 50,
    merge_adjacent_gap: int = 2,
    error_dump_path: Optional[str] = None,
    error_threshold: float = 0.99,
    align_mode: str = "gold",
    exclude_keys: Optional[List[str]] = None,
    report_title_filter: Optional[List[str]] = None,
    value_attach_window: int = 50,
    value_same_line_only: bool = True,
    adjust_boundaries: bool = False,
    adjust_max_shift: int = 1,
    adjust_chars: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    totals_txt_exact: Dict[str, float] = {}
    totals_txt_overlap: Dict[str, float] = {}
    totals_txt_tol: Dict[str, float] = {}
    totals_txt_tolerant: Dict[str, float] = {}
    totals_txt_edge3: Dict[str, float] = {}
    counted_reports = 0
    error_samples = []
    
    if report_title_filter:
        logger.info(f"报告类型过滤: {report_title_filter}")
    else:
        logger.info("报告类型: 所有类型")
    
    logger.info(f"开始评估 {len(test_data)} 个样本（键值对级别）...")
    start_delta_hist: Dict[int, int] = {}
    end_delta_hist: Dict[int, int] = {}
    leading_extra_chars: Dict[str, int] = {}
    trailing_extra_chars: Dict[str, int] = {}
    key_total: Dict[str, int] = defaultdict(int)
    key_miss: Dict[str, int] = defaultdict(int)
    key_substr: Dict[str, int] = defaultdict(int)
    key_edge1: Dict[str, int] = defaultdict(int)
    key_other: Dict[str, int] = defaultdict(int)

    iter_data = test_data if not max_samples else test_data[: int(max_samples)]
    default_tolerant_keys = {
        "现病史", "体格检查", "病理诊断", "治疗计划", "处理", "注意事项",
        "既往史", "个人史", "婚育史", "辅助检查",
    }
    env_tol = os.environ.get("KVNER_TOL_KEYS")
    if env_tol:
        for k in env_tol.split(","):
            k = k.strip()
            if k:
                default_tolerant_keys.add(k)
    tolerant_keys_norm = {_normalize_key_name(k) for k in default_tolerant_keys}
    tol_cov = float(os.environ.get("KVNER_TEXT_COVERAGE", "0.85"))

    def _covered_equal(a: str, b: str) -> bool:
        na, nb = _normalize_text_for_eval(a), _normalize_text_for_eval(b)
        if na == nb:
            return True
        la, lb = len(na), len(nb)
        if la == 0 or lb == 0:
            return False
        if na in nb or nb in na:
            cov = min(la, lb) / max(la, lb)
            return cov >= tol_cov
        return False

    for item in tqdm(iter_data, desc="评估进度"):
        text = str(item.get("report", "") or "")
        if not text.strip():
            continue
        
        report_title = str(item.get("report_title", "") or "")
        
        if report_title_filter and report_title not in report_title_filter:
            continue
        gold_raw = item.get("spans", {}) or {}
        
        entities = predict_entities(
            model, tokenizer, text, id2label, device, max_seq_length,
            chunk_size, chunk_overlap,
            merge_adjacent_gap=merge_adjacent_gap,
            adjust_boundaries=adjust_boundaries,
            adjust_max_shift=adjust_max_shift,
            adjust_chars=adjust_chars,
        )
        
        pred_pairs = assemble_key_value_pairs(
            entities, text,
            value_attach_window=value_attach_window,
            value_same_line_only=value_same_line_only,
            value_crossline_fallback_len=int(os.environ.get('KVNER_CROSSL_FALLBACK_LEN', '2')),
        )
        
        gold_valid = {
            k: {
                "start": int(v.get("start", -1)),
                "end": int(v.get("end", -1)),
                "text": str(v.get("text", "")).strip(),
            }
            for k, v in gold_raw.items()
            if isinstance(v, dict) and _is_valid_span(v)
        }
        pred_valid = {
            k: {
                "start": int(v.get("start", -1)),
                "end": int(v.get("end", -1)),
                "text": str(v.get("text", "")).strip(),
            }
            for k, v in pred_pairs.items()
            if isinstance(v, dict) and _is_valid_span(v)
        }
        
        am = str(align_mode or "gold").lower()
        gold_norm: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for k, v in gold_valid.items():
            nk = _normalize_key_name(k)
            gold_norm.setdefault(nk, (k, v))
        pred_norm: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for k, v in pred_valid.items():
            nk = _normalize_key_name(k)
            pred_norm.setdefault(nk, (k, v))

        if am == "gold":
            ref_keys = set(gold_norm.keys())
        elif am == "pred":
            ref_keys = set(pred_norm.keys())
        else:
            ref_keys = set(gold_norm.keys()) | set(pred_norm.keys())

        true_map = {nk: gold_norm[nk][1] for nk in ref_keys if nk in gold_norm}
        pred_map = {nk: pred_norm[nk][1] for nk in ref_keys if nk in pred_norm}
        
        data_true = _to_eval_pack(true_map, exclude_keys=exclude_keys)
        data_pred = _to_eval_pack(pred_map, exclude_keys=exclude_keys)
        
        if not data_true["entities"] and not data_pred["entities"]:
            continue
        
        res_txt_exact = evaluate_entities(
            [data_true],
            [data_pred],
            mode="semi_structured",
            matching_method="text",
            text_match_mode="exact",
        )
        
        res_txt_overlap = evaluate_entities(
            [data_true],
            [data_pred],
            mode="semi_structured",
            matching_method="text",
            text_match_mode="overlap",
        )

        try:
            encoding = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = encoding.get('offset_mapping')
            if offsets:
                tp = fp = fn = 0
                for nk in ref_keys:
                    pred_has = nk in pred_map
                    gold_has = nk in true_map
                    if pred_has and gold_has:
                        gs, ge = true_map[nk]['start'], true_map[nk]['end']
                        ps, pe = pred_map[nk]['start'], pred_map[nk]['end']
                        gs_tok = _char_to_token_index(offsets, gs)
                        ge_tok = _char_to_token_index(offsets, ge, right=True)
                        ps_tok = _char_to_token_index(offsets, ps)
                        pe_tok = _char_to_token_index(offsets, pe, right=True)
                        delta = max(abs(ps_tok - gs_tok), abs(pe_tok - ge_tok))
                        if delta <= int(os.environ.get('KVNER_TOL_TOKENS', '3')):
                            tp += 1
                        else:
                            fp += 1
                            fn += 1
                    elif pred_has and not gold_has:
                        fp += 1
                    elif (not pred_has) and gold_has:
                        fn += 1
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
                for k,v in {"precision":prec,"recall":rec,"f1_score":f1}.items():
                    totals_txt_tol[k] = totals_txt_tol.get(k, 0.0) + float(v)
        except Exception:
            pass

        try:
            tp = fp = fn = 0
            for nk in ref_keys:
                pred_has = nk in pred_map
                gold_has = nk in true_map
                if pred_has and gold_has:
                    gtxt = str(true_map[nk].get("text", ""))
                    ptxt = str(pred_map[nk].get("text", ""))
                    if nk in tolerant_keys_norm:
                        ok = _covered_equal(gtxt, ptxt)
                    else:
                        ok = _normalize_text_for_eval(gtxt) == _normalize_text_for_eval(ptxt)
                    if ok:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                elif pred_has and not gold_has:
                    fp += 1
                elif (not pred_has) and gold_has:
                    fn += 1
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            for k, v in {"precision": prec, "recall": rec, "f1_score": f1}.items():
                totals_txt_tolerant[k] = totals_txt_tolerant.get(k, 0.0) + float(v)
        except Exception:
            pass

        try:
            tp = fp = fn = 0
            for nk in ref_keys:
                pred_has = nk in pred_map
                gold_has = nk in true_map
                if pred_has and gold_has:
                    gtxt = str(true_map[nk].get("text", ""))
                    ptxt = str(pred_map[nk].get("text", ""))
                    g_trim, _ = _strip_edge_chars_limited(gtxt, 3)
                    p_trim, _ = _strip_edge_chars_limited(ptxt, 3)
                    if _normalize_text_for_eval(g_trim) == _normalize_text_for_eval(p_trim):
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                elif pred_has and not gold_has:
                    fp += 1
                elif (not pred_has) and gold_has:
                    fn += 1
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            for k, v in {"precision": prec, "recall": rec, "f1_score": f1}.items():
                totals_txt_edge3[k] = totals_txt_edge3.get(k, 0.0) + float(v)
        except Exception:
            pass

        inter_keys = set(true_map.keys()) & set(pred_map.keys())
        for k in inter_keys:
            gs, ge = int(true_map[k]["start"]), int(true_map[k]["end"])
            ps, pe = int(pred_map[k]["start"]), int(pred_map[k]["end"])
            
            try:
                gtxt = str(true_map[k].get("text", ""))
                ptxt = str(pred_map[k].get("text", ""))
                
                gtxt_clean, gs_clean, ge_clean = _strip_boundary_noise_with_offset(gtxt, gs, ge)
                ptxt_clean, ps_clean, pe_clean = _strip_boundary_noise_with_offset(ptxt, ps, pe)
                
                if gtxt_clean and ptxt_clean:
                    sd = ps_clean - gs_clean
                    ed = pe_clean - ge_clean
                    
                    is_tolerant_start = False
                    is_tolerant_end = False
                    
                    if abs(sd) == 1:
                        try:
                            if sd > 0 and gs_clean < len(text):
                                if text[gs_clean] in TOLERANT_TRAILING_CHARS:
                                    is_tolerant_start = True
                            elif sd < 0 and ps_clean < len(text):
                                if text[ps_clean] in TOLERANT_TRAILING_CHARS:
                                    is_tolerant_start = True
                        except Exception:
                            pass
                    
                    if abs(ed) == 1:
                        try:
                            if ed < 0 and pe_clean < len(text):
                                if text[pe_clean] in TOLERANT_TRAILING_CHARS:
                                    is_tolerant_end = True
                            elif ed > 0 and ge_clean < len(text):
                                if text[ge_clean] in TOLERANT_TRAILING_CHARS:
                                    is_tolerant_end = True
                        except Exception:
                            pass
                    
                    should_count_start = (sd != 0 and not is_tolerant_start)
                    should_count_end = (ed != 0 and not is_tolerant_end)
                    
                    if should_count_start or should_count_end:
                        if should_count_start:
                            start_delta_hist[sd] = start_delta_hist.get(sd, 0) + 1
                        if should_count_end:
                            end_delta_hist[ed] = end_delta_hist.get(ed, 0) + 1
                        
                        if should_count_start:
                            if sd < 0:
                                diff_text = text[ps_clean:gs_clean] if ps_clean < gs_clean else ""
                                _accumulate_char_freq(leading_extra_chars, diff_text)
                            elif sd > 0:
                                diff_text = text[gs_clean:ps_clean] if gs_clean < ps_clean else ""
                                _accumulate_char_freq(leading_extra_chars, diff_text)
                        
                        if should_count_end:
                            if ed > 0:
                                diff_text = text[ge_clean:pe_clean] if ge_clean < pe_clean else ""
                                _accumulate_char_freq(trailing_extra_chars, diff_text)
                            elif ed < 0:
                                diff_text = text[pe_clean:ge_clean] if pe_clean < ge_clean else ""
                                _accumulate_char_freq(trailing_extra_chars, diff_text)
            except Exception:
                pass

        all_keys = set(true_map.keys()) | set(pred_map.keys())
        for nk in all_keys:
            key_total[nk] += 1
            gt = _normalize_text_for_eval(str(true_map.get(nk, {}).get("text", ""))) if nk in true_map else ""
            pt = _normalize_text_for_eval(str(pred_map.get(nk, {}).get("text", ""))) if nk in pred_map else ""
            if not gt or not pt:
                key_miss[nk] += 1
                continue
            if gt == pt:
                continue
            if gt in pt or pt in gt:
                key_substr[nk] += 1
            elif (gt[:-1] == pt or pt[:-1] == gt or (len(gt) > 1 and gt[1:] == pt) or (len(pt) > 1 and pt[1:] == gt)):
                key_edge1[nk] += 1
            else:
                key_other[nk] += 1
        
        if error_dump_path and res_txt_exact.get("f1_score", 1.0) < error_threshold:
            true_entities = data_true.get("entities", [])
            pred_entities = data_pred.get("entities", [])
            
            true_full_map = {(e["start"], e["end"], e["text"]): e for e in true_entities}
            pred_full_map = {(e["start"], e["end"], e["text"]): e for e in pred_entities}
            
            error_true_ents = []
            error_pred_ents = []
            matched_count = 0
            
            all_full_keys = set(true_full_map.keys()) | set(pred_full_map.keys())
            
            for key in all_full_keys:
                true_ent = true_full_map.get(key)
                pred_ent = pred_full_map.get(key)
                
                if true_ent is not None and pred_ent is not None:
                    matched_count += 1
                    continue
                
                if true_ent is not None:
                    error_true_ents.append(true_ent)
                if pred_ent is not None:
                    error_pred_ents.append(pred_ent)
            
            if error_true_ents or error_pred_ents:
                true_kv_map = {}
                for nk in ref_keys:
                    if nk in true_map:
                        v = true_map[nk]
                        true_kv_map[nk] = {
                            "start": v["start"],
                            "end": v["end"],
                            "text": v["text"]
                        }
                
                pred_kv_map = {}
                for nk in ref_keys:
                    if nk in pred_map:
                        v = pred_map[nk]
                        pred_kv_map[nk] = {
                            "start": v["start"],
                            "end": v["end"],
                            "text": v["text"]
                        }
                
                error_samples.append({
                    "metrics": res_txt_exact,
                    "ground_truth": {
                        "entities": error_true_ents,
                        "key_value_pairs": true_kv_map
                    },
                    "predict": {
                        "entities": error_pred_ents,
                        "key_value_pairs": pred_kv_map
                    },
                    "report_title": report_title,
                    "report": text,
                    "total_true": len(true_entities),
                    "total_pred": len(pred_entities),
                    "matched": matched_count,
                    "error_count": len(error_true_ents) + len(error_pred_ents),
                    "all_keys": list(ref_keys),
                })
        
        for k, v in res_txt_exact.items():
            if isinstance(v, (int, float)):
                totals_txt_exact[k] = totals_txt_exact.get(k, 0.0) + float(v)
        for k, v in res_txt_overlap.items():
            if isinstance(v, (int, float)):
                totals_txt_overlap[k] = totals_txt_overlap.get(k, 0.0) + float(v)
        
        counted_reports += 1
    
    def _avg_inplace(d: Dict[str, float], n: int):
        if n <= 0:
            return
        for m in ("precision", "recall", "f1_score"):
            if m in d:
                d[m] = d[m] / n
    
    _avg_inplace(totals_txt_exact, counted_reports)
    _avg_inplace(totals_txt_overlap, counted_reports)
    _avg_inplace(totals_txt_tol, counted_reports)
    _avg_inplace(totals_txt_tolerant, counted_reports)
    _avg_inplace(totals_txt_edge3, counted_reports)
    
    if error_dump_path and error_samples:
        error_path = Path(error_dump_path)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        with error_path.open("w", encoding="utf-8") as f:
            for err in error_samples:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")
        logger.info(f"错误样本已保存到: {error_dump_path} ({len(error_samples)} 条)")
    
    logger.info(
        f"[INFO] 计入平均的样本数 = {counted_reports}\n"
        f" - text (exact)          : {totals_txt_exact}\n"
        f" - text (overlap)        : {totals_txt_overlap}\n"
        f" - text exact in ≤K tok.: {totals_txt_tol}\n"
        f" - text exact edge≤3     : {totals_txt_edge3}\n"
        f"注意：boundary_stats 已过滤：\n"
        f"  1. 边界噪声字符（空格、冒号、数字、括号）\n"
        f"  2. 可容忍的±1字符差异（标点符号)"
    )
    
    boundary_stats = {
        "start_delta_top": _top_k_items({int(k): int(v) for k, v in start_delta_hist.items()}),
        "end_delta_top": _top_k_items({int(k): int(v) for k, v in end_delta_hist.items()}),
        "leading_diff_top_chars": _top_k_items(leading_extra_chars),
        "trailing_diff_top_chars": _top_k_items(trailing_extra_chars),
    }
    logger.info("\nBoundary delta (start->pred-start, end->pred-end):")
    logger.info(f"  start_delta_top: {boundary_stats['start_delta_top']}")
    logger.info(f"  end_delta_top  : {boundary_stats['end_delta_top']}")
    logger.info("  leading_diff_top_chars: %s", boundary_stats['leading_diff_top_chars'])
    logger.info("  trailing_diff_top_chars: %s", boundary_stats['trailing_diff_top_chars'])

    try:
        items = []
        for nk in key_total:
            t = key_total[nk]
            m = key_miss[nk] + key_substr[nk] + key_edge1[nk] + key_other[nk]
            if t >= 50:
                rate = m / t if t else 0.0
                items.append((rate, nk, t, key_substr[nk], key_edge1[nk], key_other[nk], key_miss[nk]))
        items.sort(reverse=True)
        logger.info("Top key mismatches (rate,total,substr,±1,other,missing):")
        for rate, nk, t, ss, e1, ot, ms in items[:10]:
            logger.info("  %s  rate=%.2f%% total=%d substr=%d ±1=%d other=%d miss=%d", nk, rate*100, t, ss, e1, ot, ms)
    except Exception:
        pass
    
    return {
        "text_exact": totals_txt_exact,
        "text_overlap": totals_txt_overlap,
        "text_exact_in_k": totals_txt_tol,
        "text_exact_tolerant": totals_txt_tolerant,
        "text_exact_edge3": totals_txt_edge3,
        "num_samples": counted_reports,
        "num_errors": len(error_samples),
        "boundary_stats": boundary_stats,
        "key_mismatch_stats": {
            "total": key_total,
            "substr": key_substr,
            "edge1": key_edge1,
            "other": key_other,
            "missing": key_miss,
        },
    }


def run_evaluation(
    config_path: str,
    model_dir: Optional[str] = None,
    test_data_path: Optional[str] = None,
    output_dir: str = "data/kv_ner_eval",
    seed: Optional[int] = 42,
    error_threshold: float = 0.99,
    align_mode: str = "gold",
    exclude_keys: Optional[List[str]] = None,
    report_title_filter: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    override_chunk_size: Optional[int] = None,
    override_chunk_overlap: Optional[int] = None,
) -> Dict[str, Any]:
    set_seed(seed)
    
    cfg = config_io.load_config(config_path)
    train_block = cfg.get("train", {})
    
    if model_dir is None:
        model_dir = str(Path(train_block.get("output_dir", "runs/kv_ner")) / "best")
    
    if test_data_path is None:
        eval_block = cfg.get("evaluate", {})
        test_data_path = eval_block.get("eval_data_path")
        
    if test_data_path is None:
        raise ValueError("test_data_path is required")
    
    logger.info(f"配置文件: {config_path}")
    logger.info(f"模型目录: {model_dir}")
    logger.info(f"测试数据: {test_data_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"对齐模式: {align_mode}")
    
    logger.info("加载测试数据（JSONL 格式）...")
    test_data = _read_jsonl(Path(test_data_path))
    logger.info(f"测试样本数: {len(test_data)}")
    
    label_map = config_io.label_map_from(cfg)
    labels = build_bio_label_list(label_map)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    logger.info(f"标签: {labels}")
    
    logger.info("加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertCrfTokenClassifier.from_pretrained(model_dir).to(device)
    
    tokenizer_path = Path(model_dir) / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(config_io.model_name_from(cfg))
    logger.info(f"Tokenizer: {tokenizer.name_or_path}")
    
    max_seq_length = config_io.max_seq_length(cfg)
    chunk_size = int(cfg.get("chunk_size", 450))
    chunk_overlap = int(cfg.get("chunk_overlap", 50))
    if isinstance(override_chunk_size, int):
        chunk_size = int(override_chunk_size)
    if isinstance(override_chunk_overlap, int):
        chunk_overlap = int(override_chunk_overlap)
    error_dump_path = str(Path(output_dir) / "error_samples.jsonl")
    
    default_adjust_chars = " \t\u3000:：,，.;；。()（）[]【】{}<>%％-—–/\\"
    results = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        id2label=id2label,
        device=device,
        max_seq_length=max_seq_length,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        merge_adjacent_gap=int(cfg.get("merge_adjacent_gap", 2)),
        error_dump_path=error_dump_path,
        error_threshold=error_threshold,
        align_mode=align_mode,
        exclude_keys=exclude_keys,
        report_title_filter=report_title_filter,
        value_attach_window=int(cfg.get("value_attach_window", 50)),
        value_same_line_only=bool(cfg.get("value_same_line_only", True)),
        adjust_boundaries=bool(cfg.get("adjust_boundaries", True)),
        adjust_max_shift=int(cfg.get("adjust_max_shift", 1)),
        adjust_chars=str(cfg.get("adjust_chars", default_adjust_chars)),
        max_samples=max_samples,
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "config_path": config_path,
        "model_dir": model_dir,
        "test_data_path": test_data_path,
        "num_samples": results["num_samples"],
        "num_errors": results["num_errors"],
        "max_seq_length": max_seq_length,
        "seed": seed,
        "error_threshold": error_threshold,
        "align_mode": align_mode,
        "exclude_keys": exclude_keys,
        "boundary_stats_note": "已过滤：1)边界噪声字符（空格、冒号、数字、括号） 2)可容忍的±1字符差异（标点符号）",
        "text_exact_metrics": results["text_exact"],
        "text_overlap_metrics": results["text_overlap"],
        "text_exact_in_k_metrics": results.get("text_exact_in_k", {}),
        "text_exact_tolerant_metrics": results.get("text_exact_tolerant", {}),
        "text_exact_edge3_metrics": results.get("text_exact_edge3", {}),
        "boundary_stats": results.get("boundary_stats", {}),
    }
    
    summary_path = output_path / "eval_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    
    logger.info(f"\n{'='*80}")
    logger.info("评估结果（键值对级别）")
    logger.info(f"{'='*80}")
    logger.info(f"样本数: {results['num_samples']}")
    logger.info(f"错误样本数: {results['num_errors']}")
    logger.info(f"\nText Exact Matching:")
    logger.info(f"  Precision: {results['text_exact'].get('precision', 0):.4f}")
    logger.info(f"  Recall:    {results['text_exact'].get('recall', 0):.4f}")
    logger.info(f"  F1 Score:  {results['text_exact'].get('f1_score', 0):.4f}")
    logger.info(f"\nText Overlap Matching:")
    logger.info(f"  Precision: {results['text_overlap'].get('precision', 0):.4f}")
    logger.info(f"  Recall:    {results['text_overlap'].get('recall', 0):.4f}")
    logger.info(f"  F1 Score:  {results['text_overlap'].get('f1_score', 0):.4f}")
    logger.info(f"\nText Exact (≤K tokens) Matching:")
    logger.info(f"  Precision: {results['text_exact_in_k'].get('precision', 0):.4f}")
    logger.info(f"  Recall:    {results['text_exact_in_k'].get('recall', 0):.4f}")
    logger.info(f"  F1 Score:  {results['text_exact_in_k'].get('f1_score', 0):.4f}")
    logger.info(f"\nText Exact (tolerant) Matching:")
    logger.info(f"  Precision: {results['text_exact_tolerant'].get('precision', 0):.4f}")
    logger.info(f"  Recall:    {results['text_exact_tolerant'].get('recall', 0):.4f}")
    logger.info(f"  F1 Score:  {results['text_exact_tolerant'].get('f1_score', 0):.4f}")
    logger.info(f"\nText Exact (edge≤3) Matching:")
    logger.info(f"  Precision: {results['text_exact_edge3'].get('precision', 0):.4f}")
    logger.info(f"  Recall:    {results['text_exact_edge3'].get('recall', 0):.4f}")
    logger.info(f"  F1 Score:  {results['text_exact_edge3'].get('f1_score', 0):.4f}")
    logger.info(f"\n注意：boundary_stats 已过滤边界噪声和可容忍差异，只统计实质性问题")
    logger.info(f"\n结果已保存到: {summary_path}")
    logger.info(f"{'='*80}\n")
    
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 KV-NER 模型（键值对级别）")
    parser.add_argument(
        "--config",
        type=str,
        default="pre_struct/kv_ner/kv_ner_config.json",
        help="配置文件路径",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="模型目录（默认从配置读取）",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="测试数据路径（JSONL 格式，默认 data/ground_truth.jsonl）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/kv_ner_eval",
        help="输出目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--error_threshold",
        type=float,
        default=0.99,
        help="错误样本 F1 阈值",
    )
    parser.add_argument(
        "--align_mode",
        type=str,
        default="gold",
        choices=["gold", "pred", "union"],
        help="键对齐模式",
    )
    parser.add_argument(
        "--exclude_keys",
        type=str,
        nargs="*",
        default=None,
        help="排除的键列表",
    )
    parser.add_argument(
        "--report_titles",
        type=str,
        nargs="*",
        default=None,
        help="评估的报告类型过滤（不指定=评估所有类型，指定=只评估指定类型）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多评估多少条样本（用于快速调试）",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="覆盖配置的 chunk_size（推理/评估）",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=None,
        help="覆盖配置的 chunk_overlap（推理/评估）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        config_path=args.config,
        model_dir=args.model_dir,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        seed=args.seed,
        error_threshold=args.error_threshold,
        align_mode=args.align_mode,
        exclude_keys=args.exclude_keys,
        report_title_filter=args.report_titles,
        max_samples=args.max_samples,
        override_chunk_size=args.chunk_size,
        override_chunk_overlap=args.chunk_overlap,
    )
