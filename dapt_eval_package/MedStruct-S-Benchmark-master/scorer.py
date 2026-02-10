
import argparse
import json
import sys
import os
import logging
import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.getcwd())

# Modified to import from local metrics which handles 4-arg calculate_task1_stats
from metrics import (
    calculate_task1_stats,
    calculate_task2_stats,
    calculate_task3_stats,
    calc_micro_f1
)
from pre_struct.kv_ner.schema_utils import load_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

    return keys, pairs, spans_map

def extract_spans_from_item(item, allowed_keys=None, key_normalizer=None):
    """
    Extract spans from item for alignment.
    Returns:
        keys: list of key texts
        pairs: list of (key, value) tuples
        spans_map: dict {key_text: (start, end)} or {key_text: None}
        
    Note: The input format might vary (BERT vs GPT).
    We expect a standard: `_kv_spans` or `spans` (GT) or `pred_pairs` with structure.
    Fallback: If none above, and allowed_keys provided, treat as Flat Dict (GT).
    """
    keys = []
    pairs = []
    spans_map = {}
    
    # 1. Check for `_kv_spans` (Produced by convert_da_with_spans.py)
    if '_kv_spans' in item:
        for k, v_info in item['_kv_spans'].items():
            norm_k = key_normalizer(k) if key_normalizer else k
            keys.append(norm_k)
            pairs.append((norm_k, v_info.get('value', '')))
            spans_map[norm_k] = None

    # 2. Check for `spans` (GT format in val_eval.jsonl)
    elif 'spans' in item: # GT style
        for k, v in item['spans'].items():
            norm_k = key_normalizer(k) if key_normalizer else k
            keys.append(norm_k)
            v_text = v.get('text', '')
            pairs.append((norm_k, v_text))
            spans_map[norm_k] = None 
            
    # 3. Check for `pred_pairs` (Pred style)
    elif 'pred_pairs' in item: # Pred style
        for p in item['pred_pairs']:
            k = p.get('key')
            v = p.get('value', '')  # 空值也保留
            
            norm_k = key_normalizer(k) if key_normalizer else k
            keys.append(norm_k)
            pairs.append((norm_k, v if v else ''))
            
            # Check if spans exist
            if 'key_span' in p and p['key_span'] is not None:
                spans_map[norm_k] = tuple(p['key_span'])
            else:
                spans_map[norm_k] = None
                
    # 4. Fallback: Flat Dict (GT Raw)
    else:
        # 元数据字段列表（不是真正的 Key-Value 对）
        META_KEYS = {'id', 'report_title', 'text', 'ocr_text', '_kv_spans', 'spans'}
        
        for k, v in item.items():
            # 跳过元数据字段
            if k in META_KEYS:
                continue
            # If allowed_keys is provided, filter by it
            if allowed_keys is not None and k not in allowed_keys:
                continue
            
            norm_k = key_normalizer(k) if key_normalizer else k
            keys.append(norm_k)
            # Ensure value is string
            pairs.append((norm_k, str(v) if v is not None else ""))
            spans_map[norm_k] = None

    return keys, pairs, spans_map

def main():
    parser = argparse.ArgumentParser(description="Unified Scorer for Intermediate Data Protocol")
    parser.add_argument("--pred_file", required=True, help="Prediction JSONL")
    parser.add_argument("--gt_file", required=True, help="Ground Truth JSONL")
    parser.add_argument("--schema_file", default="data/kv_ner_prepared_comparison/keys_merged_1027_cleaned.json", help="Schema for Task 3")
    parser.add_argument("--output_file", default=None, help="Save metrics to JSON")
    parser.add_argument("--model_name", default=None, help="Model Name (optional)")
    parser.add_argument("--dataset_type", default="Original", help="DA or Original")
    parser.add_argument("--train_mode", default="Unknown", help="LoRA/ZeroShot")
    parser.add_argument("--language", default="Unknown", help="en/zh")
    args = parser.parse_args()

    # Load Data
    logger.info(f"Loading predictions: {args.pred_file}")
    preds = load_jsonl(args.pred_file)
    logger.info(f"Loading ground truth: {args.gt_file}")
    gts = load_jsonl(args.gt_file)
    
    # Load Schema
    keys_dict = load_schema(args.schema_file) or {}
    
    # Build Alias Map (Alias -> Canonical)
    alias_map = {}
    flat_keys = set()
    
    # helper to add to map
    def add_map(k, v):
        alias_map[k] = v
        # Also normalize spaces/lower? For now exact.
    
    for cat, items in keys_dict.items():
        if isinstance(items, dict):
            for key, info in items.items():
                # Canonical
                flat_keys.add(key)
                add_map(key, key)
                
                # Aliases
                if isinstance(info, dict) and "别名" in info:
                    for alias in info["别名"]:
                        add_map(alias, key)
        else:
            # Flat list or unexpected structure
            flat_keys.add(items) # items is likely the key name if list?
            # But the file structure seen is Dict[Cat, Dict[Key, Info]]
            pass

    # Hardcoded Fallbacks for Original Data issues (GT vs Schema)
    fallback_map = {
        "医院名称": "医院",
        "病史叙述者": "病史陈述者",
        "常规病理号": "病理号",  # Hypothesis, or map to '病案号'? No, '病理号' usually.
        "送检医院": "医院",       # Map to generic hospital if specific not found? 
                                  # Or keeps as '送检单位' (from file). 
                                  # Note: File maps '送检医院' -> '送检单位'. 
                                  # If Model predicts '医院', we want GT '送检医院' to match '医院'? 
                                  # No, '送检医院' is Sending Hospital. '医院' is usually Treating Hospital.
                                  # They are different. 
                                  # But if Model output '医院' for '送检医院' text -> Mismatch is correct.
                                  # Unless text is same.
    }
    for k, v in fallback_map.items():
        if k not in alias_map:
            alias_map[k] = v

    logger.info(f"Loaded Schema with {len(flat_keys)} canonical keys and {len(alias_map)} total mappings.")
    keys_search_set = flat_keys
    
    def normalize_key(k):
        # 1. Direct Lookup
        if k in alias_map: return alias_map[k]
        # 2. Hardcoded specific string ops (optional)
        return k

    # Truncate to min length
    num_samples = min(len(preds), len(gts))
    logger.info(f"Evaluating {num_samples} samples...")
    
    # Accumulators
    t1_strict = {"tp": 0, "fp": 0, "fn": 0}
    t1_robust = {"tp": 0, "fp": 0, "fn": 0}
    
    t2_global = {"matched_keys_count": 0, "target_gt_count": 0, "tp_exact": 0, "tp_approx": 0}
    t2_pos = {"matched_keys_count": 0, "target_gt_count": 0, "tp_exact": 0, "tp_approx": 0}
    
    t3_counts = {"total_p": 0, "total_g": 0, "ss_tp": 0, "sr_tp": 0, "rr_tp": 0}

    for i in range(num_samples):
        # Extract Preds
        # Extract Raw (for Task 1: Key Discovery - Strict Raw Match)
        p_keys_raw, p_pairs_raw, p_spans_raw = extract_spans_from_item(preds[i], allowed_keys=None, key_normalizer=None)
        g_keys_raw, g_pairs_raw, g_spans_raw = extract_spans_from_item(gts[i], allowed_keys=None, key_normalizer=None)

        # Extract Normalized (for Task 2/3: Value Extraction - Schema Aligned)
        p_keys_norm, p_pairs_norm, p_spans_norm = extract_spans_from_item(preds[i], allowed_keys=keys_search_set, key_normalizer=normalize_key)
        g_keys_norm, g_pairs_norm, g_spans_norm = extract_spans_from_item(gts[i], allowed_keys=keys_search_set, key_normalizer=normalize_key)
        
        # --- Task 1 (Use Raw Keys) ---
        # 根据用户要求，Task 1 不进行键名映射，直接使用原文/GT中的原始键名
        # Filter GT keys to only include those with actual values (Positive Discovery)
        def is_empty(v): return not v or str(v).strip() == "" or str(v).lower() == "null"
        g_keys_pos_raw = [k for k, v in g_pairs_raw if not is_empty(v)]
        
        p_span_list = [p_spans_raw.get(k) for k in p_keys_raw]
        g_span_list = [g_spans_raw.get(k) for k in g_keys_pos_raw]
        
        t1_e, t1_a = calculate_task1_stats(p_keys_raw, g_keys_pos_raw, p_span_list, g_span_list)
        
        for k in t1_strict: t1_strict[k] += t1_e[k]
        for k in t1_robust: t1_robust[k] += t1_a[k]
        
        # --- Task 2 (Use Normalized Pairs with Schema Densification) ---
        # 动态稠密化：根据标题加载对应的 Schema 字段
        title = preds[i].get('report_title', "") or gts[i].get('report_title', "通用病历")
        local_schema = keys_dict.get(title, {})
        
        # Fallback: 如果按标题找不到 Schema，则使用全量 Keys (flat_keys)
        # 这对于 keys_v2.json 这种可能按类别而非标题组织的 Schema 很重要
        if not local_schema and keys_search_set:
            # logger.debug(f"Title '{title}' not found in schema, using full schema.")
            local_schema = {k: k for k in keys_search_set}

        if isinstance(local_schema, list):
            local_schema = {k: k for k in local_schema}
        
        # 构建当前标题下的稠密 GT 字典
        # 1. 获取归一化后的 Schema 键集
        schema_keys_norm = set(normalize_key(sk) for sk in local_schema.keys())
        # 2. 将原始归一化后的 GT 对转为字典
        g_dict_norm = dict(g_pairs_norm)
        # 3. 构建稠密 GT：包含 Schema 里的所有键，GT 里没有的填空
        g_pairs_densed = []
        for sk_norm in schema_keys_norm:
            gv = g_dict_norm.get(sk_norm, "")
            g_pairs_densed.append((sk_norm, gv))
        
        # 使用稠密 GT 进行 Task 2 评测
        t2 = calculate_task2_stats(p_pairs_norm, g_pairs_densed)
        for k in t2_global: t2_global[k] += t2['global'][k]
        for k in t2_pos: t2_pos[k] += t2['pos'][k]
        
        # --- Task 3 (Use Raw Pairs - No Schema) ---
        t3 = calculate_task3_stats(p_pairs_raw, g_pairs_raw, p_spans_raw, g_spans_raw)
        for k in t3_counts: t3_counts[k] += t3[k]

    # --- Results ---
    results = {}
    
    # Task 1
    t1_s_res = calc_micro_f1(t1_strict['tp'], t1_strict['tp']+t1_strict['fp'], t1_strict['tp']+t1_strict['fn'])
    t1_r_res = calc_micro_f1(t1_robust['tp'], t1_robust['tp']+t1_robust['fp'], t1_robust['tp']+t1_robust['fn'])
    results["Task 1 (Key Discovery)"] = {
        "Strict (K_E)": t1_s_res,
        "Robust (K_A)": t1_r_res
    }
    
    # Task 2
    def calc_t2_res(stats):
        # P = TP / Matched Keys
        # R = TP / GT Target Count
        denom_p = stats['matched_keys_count']
        denom_r = stats['target_gt_count']
        
        pe = stats['tp_exact'] / denom_p if denom_p else 0
        re = stats['tp_exact'] / denom_r if denom_r else 0
        f1e = 2*pe*re/(pe+re) if pe+re else 0
        
        pa = stats['tp_approx'] / denom_p if denom_p else 0
        ra = stats['tp_approx'] / denom_r if denom_r else 0
        f1a = 2*pa*ra/(pa+ra) if pa+ra else 0
        
        return {
            "Exact": {"p": pe, "r": re, "f1": f1e},
            "Approx": {"p": pa, "r": ra, "f1": f1a}
        }

    res_global = calc_t2_res(t2_global)
    res_pos = calc_t2_res(t2_pos)

    results["Task 2 (Value Extraction)"] = {
        "Exact (QA_E)": res_global["Exact"],
        "Approx (QA_A)": res_global["Approx"],
        "Exact (QA_Pos-E)": res_pos["Exact"],
        "Approx (QA_Pos-A)": res_pos["Approx"]
    }
    
    # Task 3
    # SS
    res_ss = calc_micro_f1(t3_counts['ss_tp'], t3_counts['total_p'], t3_counts['total_g'])
    res_sr = calc_micro_f1(t3_counts['sr_tp'], t3_counts['total_p'], t3_counts['total_g'])
    res_rr = calc_micro_f1(t3_counts['rr_tp'], t3_counts['total_p'], t3_counts['total_g'])
    
    results["Task 3 (E2E Extraction)"] = {
        "Strict-Strict (K_E V_E)": res_ss,
        "Strict-Robust (K_E V_A)": res_sr,
        "Robust-Robust (K_A V_A)": res_rr
    }

    # --- Metadata ---
    model_name = args.model_name
    if not model_name:
        # Infer from filename: remove extension and path
        base = os.path.basename(args.pred_file)
        model_name = os.path.splitext(base)[0]
    
    results["metadata"] = {
        "model_name": model_name,
        "dataset_type": args.dataset_type,
        "train_mode": args.train_mode,
        "language": args.language,
        "eval_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": args.gt_file,
        "num_samples": num_samples
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Metrics saved to {args.output_file}")

if __name__ == "__main__":
    main()
