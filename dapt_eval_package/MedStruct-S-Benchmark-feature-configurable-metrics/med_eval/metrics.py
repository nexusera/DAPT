import re
from collections import Counter, defaultdict

# =============================================================================
# Infrastructure Layer: Core Algorithms
# =============================================================================

def normalize_text(text, use_normalization=True):
    """
    Standardize text for comparison.
    - If use_normalization is False, returns str(text) as is.
    - Otherwise:
      1. Strip whitespace.
      2. Convert to lowercase.
      3. Map 'null', 'none', 'nan', '-' to empty string.
    """
    if text is None:
        return ""
    
    s = str(text).strip()
    if not use_normalization:
        return s

    return s.lower()

def get_threshold(text_len, t_min=0.8, t_max=0.9, l_min=10, l_max=20):
    """
    Length-Adaptive Threshold Function (Tau logic).
    - Len < l_min: returns t_min
    - Len > l_max: returns t_max
    - Otherwise: linear interpolation between t_min and t_max.
    """
    if text_len < l_min:
        return t_min
    elif text_len > l_max:
        return t_max
    else:
        # Linear interpolation
        return t_min + (text_len - l_min) * (t_max - t_min) / (l_max - l_min)

def compute_similarity(s1, s2, use_normalization=True):
    """
    Compute Normalized Edit Distance similarity (1 - NED).
    Formula: 1 - (Levenshtein(s1, s2) / max(|s1|, |s2|))
    """
    s1 = normalize_text(s1, use_normalization)
    s2 = normalize_text(s2, use_normalization)
    
    m, n = len(s1), len(s2)
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
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # deletion
                           dp[i][j - 1] + 1,      # insertion
                           dp[i - 1][j - 1] + cost) # substitution
                           
    dist = dp[m][n]
    max_len = max(m, n)
    return 1.0 - (dist / max_len)

def compute_iou(span1, span2):
    """
    Compute Intersection over Union between two spans (start, end).
    """
    if span1 is None and span2 is None:
        return 1.0  # 双方均无位置信息，视为兼容
    if span1 is None or span2 is None:
        return 0.0  # 仅一方有位置信息，视为不兼容
        
    s1, e1 = span1
    s2, e2 = span2
    
    start_max = max(s1, s2)
    end_min = min(e1, e2)
    
    if end_min <= start_max:
        return 0.0
        
    intersection = end_min - start_max
    union = (e1 - s1) + (e2 - s2) - intersection
    
    return intersection / union if union > 0 else 0.0

def align_instances(pred_items, gt_items, config):
    """
    Execute 3-Phase Alignment Strategy.
    
    Args:
        pred_items: list of dict {'text': str, 'span': (s, e)}
        gt_items: list of dict {'text': str, 'span': (s, e)}
        config: Dict containing thresholds and normalization toggles
        
    Returns:
        matches: list of (pred_idx, gt_idx, similarity_score, stage)
        unmatched_preds: list of pred_idx
        unmatched_gts: list of gt_idx
    """
    matches = []
    remaining_preds = {i for i in range(len(pred_items))}
    remaining_gts = {j for j in range(len(gt_items))}
    
    use_norm = config.get("normalize", True)
    sim_threshold = config.get("similarity_threshold", 0.8)
    tau_dynamic = config.get("tau_dynamic", True)
    overlap_threshold = config.get("overlap_threshold", 0.0)

    # Pre-compute segments for comparison
    p_norm = [normalize_text(p['text'], use_norm) for p in pred_items]
    g_norm = [normalize_text(g['text'], use_norm) for g in gt_items]
    
    # --- Phase 1: Global Unique Exact Match ---
    p_counts = Counter(p_norm)
    g_counts = Counter(g_norm)
    unique_texts = {t for t in p_counts if p_counts[t] == 1 and g_counts.get(t) == 1 and t != ""}
    
    text_to_pidx = {p_norm[i]: i for i in remaining_preds}
    text_to_gidx = {g_norm[j]: j for j in remaining_gts}
    
    for t in unique_texts:
        pidx, gidx = text_to_pidx[t], text_to_gidx[t]
        matches.append((pidx, gidx, 1.0, "P1_Unique"))
        remaining_preds.remove(pidx)
        remaining_gts.remove(gidx)

    # --- Phase 2: Ambiguous Exact Match (IoU Guided) ---
    p_groups = defaultdict(list)
    g_groups = defaultdict(list)
    for i in remaining_preds: p_groups[p_norm[i]].append(i)
    for j in remaining_gts: g_groups[g_norm[j]].append(j)
        
    common_texts = set(p_groups.keys()) & set(g_groups.keys())
    for t in common_texts:
        if t == "": continue 
        pairs = []
        for pidx in p_groups[t]:
            for gidx in g_groups[t]:
                iou = compute_iou(pred_items[pidx].get('span'), gt_items[gidx].get('span'))
                pairs.append((iou, pidx, gidx))
        
        pairs.sort(key=lambda x: x[0], reverse=True)
        local_used_p, local_used_g = set(), set()
        for iou, pidx, gidx in pairs:
            if pidx in local_used_p or gidx in local_used_g: continue
            if iou > overlap_threshold:
                matches.append((pidx, gidx, 1.0, "P2_Ambiguous"))
                local_used_p.add(pidx)
                local_used_g.add(gidx)
                remaining_preds.remove(pidx)
                remaining_gts.remove(gidx)

    # --- Phase 3: Robust Approximate Match (IoU + NED) ---
    pairs = []
    for i in remaining_preds:
        for j in remaining_gts:
            iou = compute_iou(pred_items[i].get('span'), gt_items[j].get('span'))
            if iou > overlap_threshold:
                pairs.append((iou, i, j))
                
    pairs.sort(key=lambda x: x[0], reverse=True)
    for iou, i, j in pairs:
        if i not in remaining_preds or j not in remaining_gts: continue
        sim = compute_similarity(pred_items[i]['text'], gt_items[j]['text'], use_norm)
        
        # Determine effective threshold
        threshold = get_threshold(len(gt_items[j]['text'])) if tau_dynamic else sim_threshold
        
        if sim >= threshold:
            matches.append((i, j, sim, "P3_Robust"))
            remaining_preds.remove(i)
            remaining_gts.remove(j)
            
    return matches, list(remaining_preds), list(remaining_gts)
