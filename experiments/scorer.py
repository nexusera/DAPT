
import argparse
import json
import sys
import os
import logging
import datetime
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.getcwd())
# Also include local core/ under repo root
repo_root = Path(__file__).resolve().parents[1]
core_dir = repo_root / "core"
if core_dir.is_dir() and str(core_dir) not in sys.path:
    sys.path.insert(0, str(core_dir))

try:
    from core.metrics import (
        calculate_task1_stats,
        calculate_task2_stats,
        calculate_task3_stats,
        calc_micro_f1
    )
except ImportError:
    # Fallback: locate core.metrics relative to repo root
    here = Path(__file__).resolve()
    alt = here.parents[1] / "core"
    if str(alt) not in sys.path:
        sys.path.insert(0, str(alt))
    from core.metrics import (
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

def main():
    parser = argparse.ArgumentParser(description="Unified Scorer for Intermediate Data Protocol")
    parser.add_argument("--pred_file", required=True, help="Prediction JSONL (must have 'pred_pairs')")
    parser.add_argument("--gt_file", required=True, help="Ground Truth JSONL (val_eval.jsonl)")
    parser.add_argument("--schema_file", default="data/kv_ner_prepared_comparison/keys_merged_1027_cleaned.json", help="Schema for Task 3")
    parser.add_argument("--output_file", default=None, help="Save metrics to JSON")
    args = parser.parse_args()

    # Load Data
    logger.info(f"Loading predictions: {args.pred_file}")
    preds = load_jsonl(args.pred_file)
    
    logger.info(f"Loading ground truth: {args.gt_file}")
    gts = load_jsonl(args.gt_file)
    
    # Load Schema
    keys_dict = load_schema(args.schema_file)
    if not keys_dict:
        logger.warning(f"Schema file {args.schema_file} load failed, T3 might vary.")
        keys_dict = {}

    # Map GT by some ID? 
    # Current protocol: Assume 1-to-1 mapping by order? Or by 'report_title' + content hash?
    # BERT compare_models.py relies on strict ordering (zip).
    # Unified scorer should also probably rely on order if ID is missing.
    # Let's try to match by index first.
    
    if len(preds) != len(gts):
        logger.warning(f"Count Mismatch! Preds: {len(preds)}, GTs: {len(gts)}. Using min length.")
        
    num_samples = min(len(preds), len(gts))
    
    # Stats Accumulators
    t1_strict_stats = {'tp':0, 'fp':0, 'fn':0}
    t1_loose_stats = {'ps':0, 'pc':0, 'rs':0, 'rc':0}
    
    t2_ss_stats = {'tp':0, 'fp':0, 'fn':0}
    t2_sl_stats = {'ps':0, 'pc':0, 'rs':0, 'rc':0}
    t2_ll_stats = {'ps':0, 'pc':0, 'rs':0, 'rc':0}
    
    t3_stats_all = {"all_match_sum": 0, "all_count": 0, "all_f1_sum": 0}
    t3_stats_pos = {"pos_match_sum": 0, "pos_count": 0, "pos_f1_sum": 0}
    
    for i in range(num_samples):
        # GT Extract
        gt_item = gts[i]
        
        # Build GT Sets (Similar to get_ground_truth in compare_models.py)
        gt_keys = set()
        gt_pairs = set()
        gt_qa_map = {}
        
        # Prioritize 'spans' for consistency with Task 3 definition
        if 'spans' in gt_item:
            for k, v in gt_item['spans'].items():
                v_text = v.get('text', '')
                gt_qa_map[k] = v_text
                gt_keys.add(k)
                if v_text: # Only non-empty pairs count for Task 2? usually yes.
                     gt_pairs.add((k, v_text))
        
        # Pred Extract
        pred_item = preds[i]
        # Protocol: 'pred_pairs' is list of {key:..., value:...}
        pred_pairs_list = pred_item.get('pred_pairs', [])
        
        pred_keys = []
        pred_pairs_tuples = []
        pred_dict = {}
        
        for p in pred_pairs_list:
            k = p['key']
            v = p['value']
            pred_keys.append(k)
            pred_pairs_tuples.append((k, v))
            pred_dict[k] = v
            
        # Title for Schema Lookup
        # Prefer GT title (key is report_title), fallback to Pred title
        title = gt_item.get('report_title', "") or gt_item.get('title', "")
        if not title:
            # Try to grab from pred if available (protocol says report_title in output)
            title = pred_item.get('report_title', "")
            
        # --- Calc Metrics ---
        # T1
        s1, l1 = calculate_task1_stats(pred_keys, list(gt_keys))
        for k in t1_strict_stats: t1_strict_stats[k] += s1[k]
        for k in t1_loose_stats: t1_loose_stats[k] += l1[k]
        
        # T2
        s2, sl2, ll2 = calculate_task2_stats(pred_pairs_tuples, list(gt_pairs))
        for k in t2_ss_stats: t2_ss_stats[k] += s2[k]
        for k in t2_sl_stats: t2_sl_stats[k] += sl2[k]
        for k in t2_ll_stats: t2_ll_stats[k] += ll2[k]
        
        # T3
        s3 = calculate_task3_stats(pred_dict, title, keys_dict, gt_qa_map)
        t3_stats_all['all_match_sum'] += s3['all_match_sum']
        t3_stats_all['all_count'] += s3['all_count']
        t3_stats_all['all_f1_sum'] += s3['all_f1_sum']
        
        t3_stats_pos['pos_match_sum'] += s3['pos_match_sum']
        t3_stats_pos['pos_count'] += s3['pos_count']
        t3_stats_pos['pos_f1_sum'] += s3['pos_f1_sum']

    # --- Report ---
    results = {}
    
    # Add Metadata for Traceability
    results["metadata"] = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": os.path.basename(__file__),
        "pred_file": args.pred_file,
        "gt_file": args.gt_file,
        "schema_file": args.schema_file
    }

    results["Task 1 (Key Discovery)"] = {
        "Strict (K_E)": calc_micro_f1(t1_strict_stats),
        "Loose (K_A)": calc_micro_f1(t1_loose_stats)
    }
    
    results["Task 2 (KV Pairing)"] = {
        "Strict (K_E V_E)": calc_micro_f1(t2_ss_stats),
        "Mixed (K_E V_A)": calc_micro_f1(t2_sl_stats),
        "Loose (K_A V_A)": calc_micro_f1(t2_ll_stats)
    }
    
    t3_all_cnt = t3_stats_all['all_count']
    t3_pos_cnt = t3_stats_pos['pos_count']
    
    results["Task 3 (QA)"] = {
        "Global_EM (QA_E)": t3_stats_all['all_match_sum'] / t3_all_cnt if t3_all_cnt else 0.0,
        "Global_AM (QA_A)": t3_stats_all['all_f1_sum'] / t3_all_cnt if t3_all_cnt else 0.0,
        "Pos_EM (Pos_E)": t3_stats_pos['pos_match_sum'] / t3_pos_cnt if t3_pos_cnt else 0.0,
        "Pos_AM (Pos_A)": t3_stats_pos['pos_f1_sum'] / t3_pos_cnt if t3_pos_cnt else 0.0
    }
    
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Metrics saved to {args.output_file}")

if __name__ == "__main__":
    main()
