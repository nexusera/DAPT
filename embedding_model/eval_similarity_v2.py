import argparse
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def compute_metrics(gt_embs, pred_embs, threshold=0.8):
    """
    Compute raw matches for Micro-average calculation.
    """
    if len(gt_embs) == 0:
        # tp, fn, fp, n_gt, n_pred
        return 0, 0, len(pred_embs), 0, len(pred_embs)
        
    if len(pred_embs) == 0:
        return 0, len(gt_embs), 0, len(gt_embs), 0

    # Similarity Matrix: [n_gt, n_pred]
    sim_matrix = cosine_similarity(gt_embs, pred_embs)
    
    # Recall perspective: For each GT, find best Pred
    gt_max_sims = np.max(sim_matrix, axis=1) # [n_gt]
    
    # Precision perspective: For each Pred, find best GT
    pred_max_sims = np.max(sim_matrix, axis=0) # [n_pred]
    
    # Counts
    matched_gt = (gt_max_sims >= threshold).sum()      # TP for Recall perspective
    matched_pred = (pred_max_sims >= threshold).sum()  # TP for Precision perspective
    
    # For semantic matching, TP definition can be tricky because it's many-to-many soft match.
    # Standard Micro-F1 usually defines TP, FP, FN globally.
    # TP (Global): Total number of GT items that were successfully retrieved.
    # FN (Global): Total number of GT items that were MISSED.
    # FP (Global): Total number of Pred items that were WRONG (did not match any GT).
    
    # IMPORTANT:
    # matched_gt = TP (how many GTs found a match)
    # n_gt - matched_gt = FN
    # n_pred - matched_pred = FP (Number of preds that didn't find a GT)
    # Note: matched_pred should ideally be close to matched_gt, but in many-to-one cases they differ.
    # For strict Micro F1, we usually use the Recall-side TP as the true TP count.
    
    tp = matched_gt
    fn = len(gt_embs) - matched_gt
    fp = len(pred_embs) - matched_pred 
    
    return tp, fn, fp, len(gt_embs), len(pred_embs)

def main():
    parser = argparse.ArgumentParser(description="Eval similarity with Micro & Macro metrics.")
    parser.add_argument("--gt_pkl", type=str, required=True, help="Path to the GT pickle file.")
    parser.add_argument("--pred_pkl", type=str, required=True, help="Path to the Prediction pickle file.")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold.")
    
    args = parser.parse_args()
    
    print(f"Loading GT from {args.gt_pkl}...")
    with open(args.gt_pkl, 'rb') as f:
        gt_data = pickle.load(f)
        
    print(f"Loading Pred from {args.pred_pkl}...")
    with open(args.pred_pkl, 'rb') as f:
        pred_data = pickle.load(f)
        
    # --- Alignment Strategy (Fingerprint -> Line -> ID) ---
    common_ids = []
    
    # 1. Try Fingerprint (Most Robust)
    print("Attempting to align by Content Fingerprint...")
    gt_hash_map = {v['fingerprint']: k for k, v in gt_data.items() if v.get('fingerprint')}
    
    aligned_pred_data = {}
    valid_fingerprint_count = 0
    
    for k, v in pred_data.items():
        if v.get('fingerprint') and v['fingerprint'] in gt_hash_map:
            gt_id = gt_hash_map[v['fingerprint']]
            aligned_pred_data[gt_id] = v
            valid_fingerprint_count += 1
            
    if valid_fingerprint_count > 0:
        print(f"Align: Found {valid_fingerprint_count} matches via Fingerprint.")
        common_ids = sorted(list(aligned_pred_data.keys()))
        pred_data = aligned_pred_data # Replay pred_data with aligned keys
    else:
        # 2. Fallback to Line Index
        print("Align: Fingerprint failed. Trying Line Index...")
        gt_line_map = {v['line_idx']: v for k, v in gt_data.items() if 'line_idx' in v}
        pred_line_map = {v['line_idx']: v for k, v in pred_data.items() if 'line_idx' in v}
        
        common_lines = sorted(list(set(gt_line_map.keys()).intersection(pred_line_map.keys())))
        if common_lines:
            print(f"Align: Found {len(common_lines)} matches via Line Index.")
            common_ids = common_lines
            gt_data = gt_line_map
            pred_data = pred_line_map
        else:
             # 3. Fallback to raw ID
             print("Align: Line Index failed. Trying Raw IDs...")
             common_ids = sorted(list(set(gt_data.keys()).intersection(set(pred_data.keys()))))

    print(f"Final Eval Items: {len(common_ids)}")

    # --- Micro Aggregation Variables ---
    # Keys
    k_tp_total = 0
    k_fn_total = 0
    k_fp_total = 0
    k_gt_total = 0
    k_pred_total = 0
    
    # Values
    v_tp_total = 0
    v_fn_total = 0
    v_fp_total = 0
    v_gt_total = 0
    v_pred_total = 0
    
    # Macro storage
    macro_k_f1s = []
    
    for item_id in tqdm(common_ids):
        gt_item = gt_data[item_id]
        pred_item = pred_data[item_id]
        
        # --- Keys Eval ---
        tp, fn, fp, n_gt, n_pred = compute_metrics(gt_item['keys_emb'], pred_item['keys_emb'], args.threshold)
        k_tp_total += tp
        k_fn_total += fn
        k_fp_total += fp
        k_gt_total += n_gt
        k_pred_total += n_pred
        
        # Macro F1 per item
        prec = tp / n_pred if n_pred > 0 else 0
        rec = tp / n_gt if n_gt > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        macro_k_f1s.append(f1)

        # --- Values Eval ---
        tp, fn, fp, n_gt, n_pred = compute_metrics(gt_item['values_emb'], pred_item['values_emb'], args.threshold)
        v_tp_total += tp
        v_fn_total += fn
        v_fp_total += fp
        v_gt_total += n_gt
        v_pred_total += n_pred

    # --- Calculation ---
    def calc_stats(tp, fn, fp):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    micro_k_p, micro_k_r, micro_k_f1 = calc_stats(k_tp_total, k_fn_total, k_fp_total)
    micro_v_p, micro_v_r, micro_v_f1 = calc_stats(v_tp_total, v_fn_total, v_fp_total)
    
    macro_k_avg = np.mean(macro_k_f1s) if macro_k_f1s else 0.0

    print("\n" + "="*60)
    print(f"EVALUATION REPORT (Micro-Averaged & Macro)")
    print("="*60)
    print(f"Threshold: {args.threshold} | Items: {len(common_ids)}")
    print("-" * 60)
    print(f"{'METRIC':<20} | {'PRECISION':<10} | {'RECALL':<10} | {'F1 SCORE':<10}")
    print("-" * 60)
    print(f"{'Keys (Micro)':<20} | {micro_k_p:.4f}     | {micro_k_r:.4f}     | {micro_k_f1:.4f}")
    print(f"{'Keys (Macro Avg)':<20} | {'-':<10} | {'-':<10} | {macro_k_avg:.4f}")
    print("-" * 60)
    print(f"{'Values (Micro)':<20} | {micro_v_p:.4f}     | {micro_v_r:.4f}     | {micro_v_f1:.4f}")
    print("="*60)
    print(f"Total Keys GT: {k_gt_total} | Total Keys Pred: {k_pred_total}")
    print(f"TP: {k_tp_total} | FP: {k_fp_total} | FN: {k_fn_total}")
    print("="*60)

if __name__ == "__main__":
    main()
