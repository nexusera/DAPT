from med_eval.metrics import (
    normalize_text, 
    compute_similarity, 
    get_threshold, 
    align_instances
)

def evaluate_task3_pairing(predictions, ground_truth, normalize=True, tau_dynamic=True, similarity_threshold=0.8, overlap_threshold=0.0):
    """
    Task 3: E2E Key-Value Extraction (Pairing Mode).
    Evaluates the accuracy of (Key, Value) pairs.
    
    Input format (standardized):
        Each sample's pairs are [{"key": str, "value": str, "key_span": [int,int]|null}, ...]
    
    Args:
        predictions: List of predicted samples.
        ground_truth: List of ground truth samples.
        normalize: Text normalization toggle.
        tau_dynamic: Dynamic threshold toggle.
        similarity_threshold: Base NED threshold.
        overlap_threshold: Span IoU threshold.
        
    Returns:
        Evaluation results dictionary containing three-level metrics.
    """
    total_ee_tp = 0 # Exact Key, Exact Val
    total_ea_tp = 0 # Exact Key, Approximate Val
    total_aa_tp = 0 # Approximate Key, Approximate Val
    
    total_p_count = 0
    total_g_count = 0
    
    # Internal config for align_instances
    align_config = {
        "normalize": normalize,
        "tau_dynamic": tau_dynamic,
        "similarity_threshold": similarity_threshold,
        "overlap_threshold": overlap_threshold,
        "use_em": True,
        "use_am": True,
        "use_span": True
    }
    use_norm = normalize
    sim_threshold = similarity_threshold

    for p_sample, g_sample in zip(predictions, ground_truth):
        p_pairs = p_sample.get('pairs', [])
        g_pairs = g_sample.get('pairs', [])

        total_p_count += len(p_pairs)
        total_g_count += len(g_pairs)

        # Build items required for alignment from dict-style pairs (based on key text and key_span)
        p_items = [{"text": p["key"], "span": p.get("key_span")} for p in p_pairs]
        g_items = [{"text": p["key"], "span": p.get("key_span")} for p in g_pairs]

        # Component-level alignment based on Key
        matches, _, _ = align_instances(p_items, g_items, align_config)

        for pidx, gidx, k_sim, stage in matches:
            pk, pv = p_pairs[pidx]["key"], p_pairs[pidx]["value"]
            gk, gv = g_pairs[gidx]["key"], g_pairs[gidx]["value"]

            # Value similarity
            v_sim = compute_similarity(pv, gv, use_norm)
            v_threshold = get_threshold(len(str(gv))) if tau_dynamic else sim_threshold
            
            v_is_exact = (normalize_text(pv, use_norm) == normalize_text(gv, use_norm))
            v_is_approx = (v_sim >= v_threshold)
            
            # Key exactness
            k_is_exact = (normalize_text(pk, use_norm) == normalize_text(gk, use_norm))
            # Key approximation (implicitly satisfied after alignment, but explicit check is more rigorous)
            k_sim_val = compute_similarity(pk, gk, use_norm)
            k_thresh = get_threshold(len(str(gk))) if tau_dynamic else sim_threshold
            k_is_approx = (k_sim_val >= k_thresh)
            
            # Three-level TP determination
            if k_is_exact and v_is_exact:
                total_ee_tp += 1
            if k_is_exact and v_is_approx:
                total_ea_tp += 1
            if k_is_approx and v_is_approx:
                total_aa_tp += 1

    def calc(tp):
        p = tp / total_p_count if total_p_count > 0 else 0.0
        r = tp / total_g_count if total_g_count > 0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        return {"p": round(p, 4), "r": round(r, 4), "f1": round(f1, 4), "tp": tp}

    return {
        "stats": {
            "ee_tp": total_ee_tp, 
            "ea_tp": total_ea_tp, 
            "aa_tp": total_aa_tp,
            "total_p": total_p_count,
            "total_g": total_g_count
        },
        "metrics": {
            "exact_exact": calc(total_ee_tp),
            "exact_approximate": calc(total_ea_tp),
            "approximate_approximate": calc(total_aa_tp)
        }
    }
