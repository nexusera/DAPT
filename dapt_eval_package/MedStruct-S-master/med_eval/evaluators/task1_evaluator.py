from med_eval.metrics import (
    normalize_text, 
    compute_similarity, 
    get_threshold, 
    compute_iou
)

def evaluate_task1_discovery(predictions, ground_truth, normalize=True, tau_dynamic=True, similarity_threshold=0.8, overlap_threshold=0.0):
    """
    Task 1: Key Discovery.
    Evaluates whether the model correctly identifies keys in the document.
    
    Input format (standardized):
        Each sample's pairs are [{"key": str, "value": str, "key_span": [int,int]|null}, ...]
        For GT, scorer.py pre-filters to keep only pairs with values (Positive Discovery).
    
    Args:
        predictions: List of predicted samples.
        ground_truth: List of ground truth samples (pre-filtered for Positive Discovery).
        normalize: Text normalization toggle.
        tau_dynamic: Dynamic threshold toggle.
        similarity_threshold: Base NED threshold.
        overlap_threshold: Span IoU threshold.
        
    Returns:
        Evaluation results dictionary containing stats and metrics.
    """
    total_tp_e = 0
    total_tp_a = 0
    total_p_count = 0
    total_g_count = 0
    
    # Internal behavior flags (forced enabled)
    use_em = True
    use_am = True
    use_span = True
    
    use_norm = normalize
    sim_threshold = similarity_threshold

    for p_sample, g_sample in zip(predictions, ground_truth):
        # Extract keys and spans from dict-style pairs
        p_pairs = p_sample.get('pairs', [])
        g_pairs = g_sample.get('pairs', [])

        p_items = []
        for p in p_pairs:
            nk = normalize_text(p["key"], use_norm)
            if nk:
                p_items.append({"text": nk, "span": p.get("key_span")})
            
        g_items = []
        for p in g_pairs:
            nk = normalize_text(p["key"], use_norm)
            if nk:
                g_items.append({"text": nk, "span": p.get("key_span")})

        total_p_count += len(p_items)
        total_g_count += len(g_items)

        # Track matches for this sample
        matched_g_e = set()
        matched_g_a = set()

        # Phase 1: Exact Match (EM)
        for pi, p_item in enumerate(p_items):
            for gi, g_item in enumerate(g_items):
                if gi in matched_g_e: continue
                # Span position verification
                if use_span and compute_iou(p_item["span"], g_item["span"]) <= overlap_threshold:
                    continue
                # Text exact match
                if p_item["text"] == g_item["text"]:
                    matched_g_e.add(gi)
                    break
        
        # Phase 2: Approx Match (AM) - includes EM
        for pi, p_item in enumerate(p_items):
            best_gi = -1
            max_sim = -1
            for gi, g_item in enumerate(g_items):
                if gi in matched_g_a: continue
                # Span position verification
                if use_span and compute_iou(p_item["span"], g_item["span"]) <= overlap_threshold:
                    continue
                # Similarity verification
                sim = compute_similarity(p_item["text"], g_item["text"], use_normalization=use_norm)
                thresh = get_threshold(len(g_item["text"])) if tau_dynamic else sim_threshold
                if sim >= thresh:
                    if sim > max_sim:
                        max_sim = sim
                        best_gi = gi
            if best_gi != -1:
                matched_g_a.add(best_gi)

        total_tp_e += len(matched_g_e)
        total_tp_a += len(matched_g_a)

    def calc(tp):
        p = tp / total_p_count if total_p_count > 0 else 0.0
        r = tp / total_g_count if total_g_count > 0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
        return {"p": round(p, 4), "r": round(r, 4), "f1": round(f1, 4), "tp": tp}

    return {
        "stats": {"tp_e": total_tp_e, "tp_a": total_tp_a, "total_p": total_p_count, "total_g": total_g_count},
        "metrics": {
            "exact": calc(total_tp_e),
            "approx": calc(total_tp_a)
        }
    }
