import re
from collections import Counter, defaultdict
import copy

# =============================================================================
# Infrastructure Layer
# =============================================================================

def normalize_text(text):
    """
    Standardize text for comparison.
    1. Strip whitespace.
    2. Convert to lowercase.
    3. Remove punctuation/control characters (optional, but good for robust match).
    """
    if not text:
        return ""
    text = str(text).lower().strip()
    # Remove common punctuation if needed, for now standardizing whitespace is key.
    # text = re.sub(r'[^\w\s]', '', text) 
    return text

def compute_ned_similarity(s1, s2):
    """
    Compute Normalized Edit Distance similarity (1 - NED).
    Formula: 1 - (Levenshtein(s1, s2) / max(|s1|, |s2|))
    """
    s1 = normalize_text(s1)
    s2 = normalize_text(s2)
    
    m = len(s1)
    n = len(s2)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0

    # Standard Levenshtein
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

def get_dynamic_threshold(text_len):
    """
    Length-Adaptive Threshold Function (Tau).
    - Len < 10: 0.8 (Short text needs strict match)
    - 10 <= Len <= 20: Linear transition
    - Len > 20: 0.9 (Long text allows small noise but requires high structure match)
    """
    if text_len < 10:
        return 0.8
    elif text_len > 20:
        return 0.9
    else:
        # Linear interpolation: y = 0.01 * x + 0.7
        # x=10 -> 0.8, x=20 -> 0.9
        return 0.8 + (text_len - 10) * 0.01

def compute_iou(span1, span2):
    """
    Compute Intersection over Union between two spans (start, end).
    """
    # Defensive check
    if span1 is None or span2 is None:
        return 0.0
        
    s1, e1 = span1
    s2, e2 = span2
    
    start_max = max(s1, s2)
    end_min = min(e1, e2)
    
    if end_min <= start_max:
        return 0.0
        
    intersection = end_min - start_max
    union = (e1 - s1) + (e2 - s2) - intersection
    
    if union <= 0:
        return 0.0
        
    return intersection / union

# =============================================================================
# Core Alignment Mechanism (Uniqueness-First Locking)
# =============================================================================

def align_instances(pred_items, gt_items):
    """
    Execute 3-Phase Alignment Strategy.
    
    Args:
        pred_items: list of dict {'text': str, 'span': (s, e), 'id': any}
        gt_items: list of dict {'text': str, 'span': (s, e), 'id': any}
        
    Returns:
        matches: list of (pred_idx, gt_idx, similarity_score, stage)
        unmatched_preds: list of pred_idx
        unmatched_gts: list of gt_idx
    """
    matches = []
    
    # Track indices
    remaining_preds = {i for i in range(len(pred_items))}
    remaining_gts = {j for j in range(len(gt_items))}
    
    # Pre-compute normalized texts
    p_norm = [normalize_text(p['text']) for p in pred_items]
    g_norm = [normalize_text(g['text']) for g in gt_items]
    
    # --- Phase 1: Global Unique Exact Match ---
    # Find texts that appear exactly ONCE in both P and G, and are identical
    
    p_counts = Counter(p_norm)
    g_counts = Counter(g_norm)
    
    # Identify unique texts
    unique_texts = set()
    for t in p_counts:
        if p_counts[t] == 1 and g_counts[t] == 1 and t != "":
            unique_texts.add(t)
            
    # Lock them
    # Since they are unique, we can map directly by text
    text_to_pidx = {p_norm[i]: i for i in remaining_preds}
    text_to_gidx = {g_norm[j]: j for j in remaining_gts}
    
    for t in unique_texts:
        if t in text_to_pidx and t in text_to_gidx:
            pidx = text_to_pidx[t]
            gidx = text_to_gidx[t]
            
            matches.append((pidx, gidx, 1.0, "P1_Unique"))
            remaining_preds.remove(pidx)
            remaining_gts.remove(gidx)

    # --- Phase 2: Ambiguous Exact Match (IoU Guided) ---
    # Handle texts that match exactly but have duplicates (Multi-to-Multi)
    # Group by text
    p_groups = defaultdict(list)
    g_groups = defaultdict(list)
    
    for i in remaining_preds:
        p_groups[p_norm[i]].append(i)
    for j in remaining_gts:
        g_groups[g_norm[j]].append(j)
        
    common_texts = set(p_groups.keys()) & set(g_groups.keys())
    
    for t in common_texts:
        if t == "": continue 
        
        # Greedy Match by IoU
        # Calculate all pair IoUs
        pairs = []
        for pidx in p_groups[t]:
            for gidx in g_groups[t]:
                score = compute_iou(pred_items[pidx].get('span'), gt_items[gidx].get('span'))
                pairs.append((score, pidx, gidx))
                
        # Sort by IoU desc
        pairs.sort(key=lambda x: x[0], reverse=True)
        
        local_used_p = set()
        local_used_g = set()
        
        for score, pidx, gidx in pairs:
            if pidx in local_used_p or gidx in local_used_g:
                continue
            
            # Must have some overlap or be forced? 
            # If Strict Text Match, we trust it even if IoU is low? 
            # Proposal says: "Greedy match by IoU". 
            # If IoU is 0, it might be a wrong match (Page Header vs Footer). 
            # Let's enforce small IoU > 0 or fallback?
            # Proposal: "If IoU all 0, match fail".
            if score > 0:
                matches.append((pidx, gidx, 1.0, "P2_Ambiguous"))
                local_used_p.add(pidx)
                local_used_g.add(gidx)
                if pidx in remaining_preds: remaining_preds.remove(pidx)
                if gidx in remaining_gts: remaining_gts.remove(gidx)

    # --- Phase 3: Robust Approximate Match (IoU + NED) ---
    # For remaining items, try to match by Position first, then check Text Similarity
    
    pairs = []
    for i in remaining_preds:
        for j in remaining_gts:
            # Physical Anchor
            iou = compute_iou(pred_items[i].get('span'), gt_items[j].get('span'))
            if iou > 0: # Candidate
                pairs.append((iou, i, j))
                
    pairs.sort(key=lambda x: x[0], reverse=True)
    
    for score, i, j in pairs:
        if i not in remaining_preds or j not in remaining_gts:
            continue
            
        # Check Text Similarity
        sim = compute_ned_similarity(pred_items[i]['text'], gt_items[j]['text'])
        threshold = get_dynamic_threshold(len(gt_items[j]['text']))
        
        if sim >= threshold:
            matches.append((i, j, sim, "P3_Robust"))
            remaining_preds.remove(i)
            remaining_gts.remove(j)
            
    return matches, list(remaining_preds), list(remaining_gts)


# =============================================================================
# Task Specific Logic
# =============================================================================

def calculate_task1_stats(pred_keys, gt_keys, pred_spans=None, gt_spans=None):
    """
    Task 1: Key Discovery.
    
    设计：
    - K_E: 纯文本精确匹配（集合级别）
    - K_A: 先用 IoU 找位置相近的候选，再用 NED 判定是否匹配
    
    Returns:
        K_E (Set-level Exact): TP/FP/FN counts
        K_A (Instance-level Approximate): TP/FP/FN counts (K_A.TP >= K_E.TP)
    """
    # 归一化处理
    p_norm = [normalize_text(k) for k in pred_keys]
    g_norm = [normalize_text(k) for k in gt_keys]
    
    p_norm_set = set(p_norm)
    g_norm_set = set(g_norm)
    p_norm_set.discard("")
    g_norm_set.discard("")
    
    # ========================================
    # K_E: 纯文本精确匹配
    # ========================================
    intersection = p_norm_set.intersection(g_norm_set)
    tp_e = len(intersection)
    fp_e = len(p_norm_set - g_norm_set)
    fn_e = len(g_norm_set - p_norm_set)
    
    stats_e = {"tp": tp_e, "fp": fp_e, "fn": fn_e}
    
    # ========================================
    # K_A: IoU + NED 近似匹配
    # ========================================
    # 如果没有 Span 信息，退化为纯 NED 匹配
    if pred_spans is None or gt_spans is None:
        # Fallback: 纯 NED 匹配（与之前逻辑一致）
        remaining_p = list(p_norm_set - intersection)
        remaining_g = list(g_norm_set - intersection)
        
        tp_fuzzy = 0
        candidates = []
        for p in remaining_p:
            for g in remaining_g:
                sim = compute_ned_similarity(p, g)
                thresh = get_dynamic_threshold(len(g))
                if sim >= thresh:
                    candidates.append((sim, p, g))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        used_p, used_g = set(), set()
        
        for score, p, g in candidates:
            if p not in used_p and g not in used_g:
                tp_fuzzy += 1
                used_p.add(p)
                used_g.add(g)
        
        tp_a = tp_e + tp_fuzzy
        stats_a = {
            "tp": tp_a, 
            "fp": len(p_norm_set) - tp_a, 
            "fn": len(g_norm_set) - tp_a
        }
        return stats_e, stats_a
    
    # 构建实例列表
    p_items = []
    for i, (k, span) in enumerate(zip(pred_keys, pred_spans)):
        norm_k = normalize_text(k)
        if norm_k:
            p_items.append({"idx": i, "text": norm_k, "span": span})
    
    g_items = []
    for i, (k, span) in enumerate(zip(gt_keys, gt_spans)):
        norm_k = normalize_text(k)
        if norm_k:
            g_items.append({"idx": i, "text": norm_k, "span": span})
            
    if not p_items and not g_items:
        return {"tp": 0, "fp": 0, "fn": 0}, {"tp": 0, "fp": 0, "fn": 0}

    # ========================================
    # 统一计算：实例级别对齐
    # ========================================
    # 辅助函数：判断位置是否匹配（有 Span 查 IoU，无 Span 默认匹配）
    def is_pos_match(s1, s2):
        if s1 is None or s2 is None: return True
        return compute_iou(s1, s2) > 0

    # 1. Strict TP: text_match=True AND pos_match=True
    matched_g_idx_e = set()
    tp_instance_e = 0
    for pi, p_item in enumerate(p_items):
        for gi, g_item in enumerate(g_items):
            if gi in matched_g_idx_e: continue
            if p_item["text"] == g_item["text"] and is_pos_match(p_item["span"], g_item["span"]):
                tp_instance_e += 1
                matched_g_idx_e.add(gi)
                break
                
    # 2. Robust TP: (text_match OR fuzzy_match) AND pos_match=True
    candidates = []
    for pi, p_item in enumerate(p_items):
        for gi, g_item in enumerate(g_items):
            if is_pos_match(p_item["span"], g_item["span"]):
                sim = compute_ned_similarity(p_item["text"], g_item["text"])
                thresh = get_dynamic_threshold(len(g_item["text"]))
                if sim >= thresh:
                    # 我们希望 Robust 包含所有 Strict，所以即便 sim < thresh 但 text 相等也要考虑 (虽然 sim >= thresh 通常涵盖 EM)
                    candidates.append((max(sim, 1.0 if p_item["text"] == g_item["text"] else 0.0), pi, gi))
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_g_idx_a = set()
    matched_p_idx_a = set()
    tp_instance_a = 0
    for score, pi, gi in candidates:
        if pi not in matched_p_idx_a and gi not in matched_g_idx_a:
            tp_instance_a += 1
            matched_p_idx_a.add(pi)
            matched_g_idx_a.add(gi)
            
    stats_e = {
        "tp": tp_instance_e,
        "fp": len(p_items) - tp_instance_e,
        "fn": len(g_items) - tp_instance_e
    }
    
    stats_a = {
        "tp": tp_instance_a,
        "fp": len(p_items) - tp_instance_a,
        "fn": len(g_items) - tp_instance_a
    }
    
    return stats_e, stats_a

def calculate_task2_stats(pred_pairs, gt_pairs, pred_spans=None, gt_spans=None):
    """
    Task 2: Value Extraction (Conditioned on Key).
    Returns metrics for both Global (All Keys) and Positive (Non-empty Values).
    """
    # Normalize Pairs to dicts
    # Handle potentially multiple values? Task 2 dedups.
    # We use normalize_text for keys to align.
    
    # 1. Global Set (All GTs)
    gt_dict_all = {normalize_text(k): v for k, v in gt_pairs}
    pred_dict = {normalize_text(k): v for k, v in pred_pairs}
    
    # 2. Positive Set (GT Value != Empty)
    # We treat "NULL", "None", "" as empty
    def is_empty(v): return not v or str(v).strip() == "" or str(v).lower() == "null"
    gt_dict_pos = {k: v for k, v in gt_dict_all.items() if not is_empty(v)}
    
    # Helper to calculate stats for a specific GT Scope
    def calc_scope_stats(target_gt_dict):
        # Conditioning: We only evaluate on Keys that exist in BOTH Pred and GT (or just GT?)
        # Standard "Value Extraction" usually implies: 
        # For the keys in GT, what did we predict?
        # If Pred misses the key, it counts as empty value? matches as False?
        # If we use Intersection (Matched Keys), it's Accuracy "Given Key is Found".
        # But User matched "QA_E" to F1. F1 implies P/R.
        # P = TP / Pred_Count (of value predictions for these scope keys?)
        # R = TP / GT_Count (size of scope)
        
        # Let's align by Keys.
        # Scope Keys = target_gt_dict.keys()
        
        tp_exact = 0
        tp_approx = 0
        
        # Intersection for calculation
        common_keys = set(target_gt_dict.keys()) & set(pred_dict.keys())
        
        for k in common_keys:
            gv = target_gt_dict[k]
            pv = pred_dict[k]
            
            # Exact
            if normalize_text(pv) == normalize_text(gv):
                tp_exact += 1
            
            # Approx
            sim = compute_ned_similarity(pv, gv)
            if sim >= get_dynamic_threshold(len(str(gv))):
                tp_approx += 1
                
        # Denominators
        # For 'QA', usually we care about "Did we extract the right value?".
        # Pred_Count: Number of predictions made for keys in the scope (or total preds?)
        # Usually precision is against Total Preds. But here it's "Task 2".
        # Let's stick to the previous implementation spirit:
        # P = TP / Matched_Key_Count (Accuracy on retrieved keys)
        # R = TP / GT_Count (Recall on target scope)
        # But User used standard P/R logic.
        
        return {
            "matched_keys_count": len(common_keys),
            "target_gt_count": len(target_gt_dict),
            "tp_exact": tp_exact,
            "tp_approx": tp_approx
        }

    stats_global = calc_scope_stats(gt_dict_all)
    stats_pos = calc_scope_stats(gt_dict_pos)
    
    return {
        "global": stats_global,
        "pos": stats_pos
    }

def calculate_task3_stats(pred_pairs, gt_pairs, pred_spans_map=None, gt_spans_map=None):
    """
    Task 3: E2E Key-Value Extraction.
    Returns:
        strict_strict: K_E + V_E
        strict_robust: K_E + V_A (Core Metric)
        robust_robust: K_A + V_A
    """
    # We need to align PAIRS. 
    # Best way: Align Keys first (Primary Index).
    
    # Extract Keys and Spans for alignment
    p_keys = [p[0] for p in pred_pairs]
    p_vals = [p[1] for p in pred_pairs]
    g_keys = [p[0] for p in gt_pairs]
    g_vals = [p[1] for p in gt_pairs]
    
    if pred_spans_map and gt_spans_map:
        p_k_spans = [pred_spans_map.get(k, None) for k in p_keys]
        g_k_spans = [gt_spans_map.get(k, None) for k in g_keys]
    else:
        # Mock spans if missing (fallback to text-only align inside?)
        p_k_spans = [None] * len(p_keys)
        g_k_spans = [None] * len(g_keys)
        
    # Align Keys (Instances)
    # Using the Robust Aligner
    p_items = [{"text": k, "span": s} for k, s in zip(p_keys, p_k_spans)]
    g_items = [{"text": k, "span": s} for k, s in zip(g_keys, g_k_spans)]
    
    matches, rem_p, rem_g = align_instances(p_items, g_items)
    
    # Initialize counts
    ss_tp = 0 # Strict Key, Strict Val
    sr_tp = 0 # Strict Key, Robust Val
    rr_tp = 0 # Robust Key, Robust Val
    
    for pidx, gidx, k_sim, stage in matches:
        pv = p_vals[pidx]
        gv = g_vals[gidx]
        
        # Check Value Similarity
        v_sim = compute_ned_similarity(pv, gv)
        v_thresh = get_dynamic_threshold(len(str(gv)))
        v_is_exact = (normalize_text(pv) == normalize_text(gv))
        v_is_robust = (v_sim >= v_thresh)
        
        # Check Key Exactness (for Strict-* metrics)
        k_is_exact = (normalize_text(p_keys[pidx]) == normalize_text(g_keys[gidx]))
        
        # 1. Strict-Strict
        if k_is_exact and v_is_exact:
            ss_tp += 1
            
        # 2. Strict-Robust
        if k_is_exact and v_is_robust:
            sr_tp += 1
            
        # 3. Robust-Robust
        # Key is already "Robustly Aligned" if it's in matches (sim >= thresh or exact)
        # But wait, Phase 3 allows Approx Match. Phase 1/2 are Exact.
        # So "matches" contains K_A valid pairs.
        if v_is_robust:
            rr_tp += 1
            
    # FP / FN
    # For F1, FP = Total Preds - TP
    # FN = Total GTs - TP
    total_p = len(pred_pairs)
    total_g = len(gt_pairs)
    
    return {
        "total_p": total_p,
        "total_g": total_g,
        "ss_tp": ss_tp,
        "sr_tp": sr_tp,
        "rr_tp": rr_tp
    }

def calc_micro_f1(tp, total_p, total_g):
    """
    Standard F1 from counts.
    Precision = TP / Total_Pred
    Recall = TP / Total_GT
    """
    p = tp / total_p if total_p > 0 else 0.0
    r = tp / total_g if total_g > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"p": p, "r": r, "f1": f1}
