
from collections import Counter
import copy

def compute_char_f1(pred_str, gt_str):
    """
    Compute Character-level F1 Score between two strings.
    Used for Approximate Match (AM) in Task 2 and Task 3.
    """
    pred_str = str(pred_str)
    gt_str = str(gt_str)
    
    # If both empty, perfect match (1.0), but usually means nothing to predict.
    # In standard SQuAD, if GT is empty and Pred is empty -> EM=1, F1=1
    if not pred_str and not gt_str:
        return 1.0, 1.0, 1.0
        
    pred_chars = list(pred_str)
    gt_chars = list(gt_str)
    
    common = Counter(pred_chars) & Counter(gt_chars)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0, 0.0, 0.0
        
    precision = 1.0 * num_same / len(pred_chars)
    recall = 1.0 * num_same / len(gt_chars)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return precision, recall, f1

def compute_ned_similarity(s1, s2):
    """
    Compute Normalized Edit Distance similarity (1 - NED).
    """
    m = len(s1)
    n = len(s2)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0
        
    # Standard levenshtein
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # deletion
                           dp[i][j - 1] + 1,      # insertion
                           dp[i - 1][j - 1] + cost) # substitution
                           
    dist = dp[m][n]
    max_len = max(m, n)
    return 1.0 - (dist / max_len)

# --- Micro-Avg Aggregation Helper ---
def calc_micro_f1(stats):
    """
    Generic Micro-Avg P/R/F1 calculator.
    Input stats dict must have keys: 'tp', 'fp', 'fn' 
    OR 'p_sum', 'p_count', 'r_sum', 'r_count' for loose metrics.
    """
    # Strict Case (Count based)
    if 'tp' in stats:
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"p": p, "r": r, "f1": f1}
        
    # Loose Case (Sum based)
    if 'ps' in stats:
        # ps: precision_score_sum, pc: precision_count
        # rs: recall_score_sum, rc: recall_count
        p = stats['ps'] / stats['pc'] if stats.get('pc', 0) > 0 else 0.0
        r = stats['rs'] / stats['rc'] if stats.get('rc', 0) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"p": p, "r": r, "f1": f1}
        
    return {"p": 0.0, "r": 0.0, "f1": 0.0}

# --- Task 1 Metrics ---
def calculate_task1_stats(pred_keys, gt_keys):
    """
    Calculate T1 stats for a SINGLE sample.
    Returns: strict_counts (tp,fp,fn), loose_counts (ps,pc,rs,rc)
    """
    # Strict (Set Operation) - DEDUPLICATED
    pred_set = set(pred_keys)
    gt_set = set(gt_keys)
    
    tp = len(pred_set.intersection(gt_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    t1_strict = {"tp": tp, "fp": fp, "fn": fn}
    
    # Loose (NED) - DEDUPLICATED
    # Precision Path: For each unique pred, find max sim in gt
    p_sim_sum = 0
    unique_preds = list(pred_set)
    if unique_preds:
        for pk in unique_preds:
            max_sim = 0
            for gk in gt_keys:
                max_sim = max(max_sim, compute_ned_similarity(pk, gk))
            p_sim_sum += max_sim
            
    # Recall Path: For each unique gt, find max sim in unique pred
    # (Note: Logic remains similar, comparing list(set) vs list(set) is fine)
    r_sim_sum = 0
    unique_gts = list(gt_set)
    if unique_gts:
        for gk in unique_gts:
            max_sim = 0
            for pk in unique_preds:
                max_sim = max(max_sim, compute_ned_similarity(gk, pk))
            r_sim_sum += max_sim
            
    t1_loose = {
        "ps": p_sim_sum, "pc": len(unique_preds),
        "rs": r_sim_sum, "rc": len(unique_gts)
    }
    
    return t1_strict, t1_loose

# --- Task 2 Metrics ---
def calculate_task2_stats(pred_pairs, gt_pairs):
    """
    Calculate T2 stats for a SINGLE sample.
    Pairs are tuples of (key, value).
    Returns: strict_strict, strict_loose, loose_loose stats.
    """
    # 1. Strict-Strict (Exact Match on K and V)
    # Strict (Set Operation) - DEDUPLICATED
    # Ensure all elements are strings to be hashable
    pred_set = set((str(k), str(v)) for k, v in pred_pairs)
    gt_set = set((str(k), str(v)) for k, v in gt_pairs)
    
    tp = len(pred_set.intersection(gt_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    t2_ss = {"tp": tp, "fp": fp, "fn": fn}
    
    # helper maps
    pred_dict = {k: v for k, v in pred_pairs}
    gt_dict = {k: v for k, v in gt_pairs}
    
    # 2. Strict-Loose (Exact Key, Fuzzy Value)
    # Precision: For each pred pair (pk, pv), if pk in gt, measure val similarity
    # Note: If pk not in gt, similarity is 0. 
    # But wait, T2 S-L definition usually implies if Key is missing, it's 0.
    # Logic: Sum(CharF1(pv, gv) if pk==gk) / len(pred)
    
    sl_p_sum = 0
    for pk, pv in pred_pairs:
        if pk in gt_dict:
            _, _, val_f1 = compute_char_f1(pv, gt_dict[pk])
            sl_p_sum += val_f1
    
    sl_r_sum = 0
    for gk, gv in gt_pairs:
        if gk in pred_dict:
            _, _, val_f1 = compute_char_f1(pred_dict[gk], gv)
            sl_r_sum += val_f1
            
    t2_sl = {
        "ps": sl_p_sum, "pc": len(pred_pairs),
        "rs": sl_r_sum, "rc": len(gt_pairs)
    }
    
    # 3. Loose-Loose (Fuzzy Key, Fuzzy Value)
    # Combined Score = Sim(k1, k2) * CharF1(v1, v2)
    # Match Logic: For each pred, find max match in gt
    
    ll_p_sum = 0
    for pk, pv in pred_pairs:
        max_score = 0
        for gk, gv in gt_pairs:
            k_sim = compute_ned_similarity(pk, gk)
            if k_sim > 0: # Optimize
                _, _, v_f1 = compute_char_f1(pv, gv)
                max_score = max(max_score, k_sim * v_f1)
        ll_p_sum += max_score
        
    ll_r_sum = 0
    for gk, gv in gt_pairs:
        max_score = 0
        for pk, pv in pred_pairs:
            k_sim = compute_ned_similarity(gk, pk)
            if k_sim > 0:
                _, _, v_f1 = compute_char_f1(pv, gv)
                max_score = max(max_score, k_sim * v_f1)
        ll_r_sum += max_score
        
    t2_ll = {
        "ps": ll_p_sum, "pc": len(pred_pairs),
        "rs": ll_r_sum, "rc": len(gt_pairs)
    }
    
    return t2_ss, t2_sl, t2_ll

# --- Task 3 Metrics ---
def calculate_task3_stats(pred_dict, title, schema_map, gt_qa_map):
    """
    Calculate T3 stats for a SINGLE sample.
    """
    # Get Schema Keys for this title
    # (Caller should handle schema fallback logic if needed, but we can do simple here)
    keys_def = schema_map.get(title, {})
    if not keys_def and '通用病历' in schema_map:
        keys_def = schema_map['通用病历']
        
    stats = {
        "all_match_sum": 0, "all_count": 0,
        "all_f1_sum": 0,
        "pos_match_sum": 0, "pos_count": 0,
        "pos_f1_sum": 0
    }
    
    for key_name in keys_def:
        gt_val = gt_qa_map.get(key_name, "")
        pred_val = pred_dict.get(key_name, "")
        
        # Strict EM
        is_correct = 1.0 if pred_val == gt_val else 0.0
        
        # Loose F1
        _, _, f1 = compute_char_f1(pred_val, gt_val)
        
        # All
        stats["all_count"] += 1
        stats["all_match_sum"] += is_correct
        stats["all_f1_sum"] += f1
        
        # Pos Only (GT Not Empty)
        if gt_val != "":
            stats["pos_count"] += 1
            stats["pos_match_sum"] += is_correct
            stats["pos_f1_sum"] += f1
            
    return stats
