from med_eval.metrics import (
    normalize_text, 
    compute_similarity, 
    get_threshold
)

def is_empty(v):
    return not v or str(v).strip() == "" or str(v).lower() == "null"

def evaluate_task2_qa(predictions, ground_truth, key_alias_map, normalize=True, tau_dynamic=True, similarity_threshold=0.8):
    """
    Task 2: Value Extraction (Schema-Driven).
    
    Args:
        predictions: Normalized list of samples.
        ground_truth: Normalized list of samples.
        key_alias_map: Schema mapping.
        normalize: Text normalization flag.
        tau_dynamic: Dynamic threshold flag.
        similarity_threshold: Base NED threshold.
        
    Returns:
        Global and Pos-Only metrics.
    """
    stats = {
        "global": {"tp_exact": 0, "tp_approx": 0, "matched_keys": 0, "total_gt": 0},
        "pos_only": {"tp_exact": 0, "tp_approx": 0, "matched_keys": 0, "total_gt": 0}
    }

    # These flags are now directly from function arguments
    use_norm = normalize
    sim_threshold = similarity_threshold
    # Assuming use_am is always True for primary metrics if not specified otherwise
    use_am = True 

    # Build a flat alias lookup for convenience: {alias: canonical}
    # and also collect all canonical keys per category
    schema_by_category = {}
    alias_to_canonical = {}
    
    for category, fields in key_alias_map.items():
        if not isinstance(fields, dict): continue
        schema_by_category[category] = set(fields.keys())
        for canonical, info in fields.items():
            alias_to_canonical[canonical] = canonical
            if isinstance(info, dict) and "别名" in info:
                for alias in info["别名"]:
                    alias_to_canonical[alias] = canonical

    for p_sample, g_sample in zip(predictions, ground_truth):
        # Determine current schema based on report title
        title = p_sample.get('report_title') or g_sample.get('report_title', '通用病历')
        current_fields = schema_by_category.get(title, set())
        
        # If title not found, fallback to all canonical keys or empty? 
        # Scorer code used keys_dict.get(title, {})
        if not current_fields:
            # Maybe it's a category not in schema, or generic
            pass

        # 从 dict-style pairs 构建规范化字典
        p_pairs = p_sample.get('pairs', [])
        g_pairs = g_sample.get('pairs', [])
        
        p_dict = {}
        for p in p_pairs:
            can_k = alias_to_canonical.get(p["key"], p["key"])
            p_dict[can_k] = p["value"]
            
        g_dict = {}
        for p in g_pairs:
            can_k = alias_to_canonical.get(p["key"], p["key"])
            g_dict[can_k] = p["value"]

        # Schema-driven loop
        for field in current_fields:
            pv = p_dict.get(field)
            gv = g_dict.get(field, "") # Default to empty if not in GT

            is_pos = not is_empty(gv)
            has_pred = field in p_dict
            
            # Comparison
            t1_match, t2_match = False, False
            if has_pred:
                # Exact Match
                if normalize_text(pv, use_norm) == normalize_text(gv, use_norm):
                    t1_match = True
                
                # Approx Match
                sim = compute_similarity(pv, gv, use_norm)
                threshold = get_threshold(len(str(gv))) if tau_dynamic else sim_threshold
                if sim >= threshold:
                    t2_match = True

            # Update Global
            stats["global"]["total_gt"] += 1
            if has_pred: stats["global"]["matched_keys"] += 1
            if t1_match: stats["global"]["tp_exact"] += 1
            if t2_match: stats["global"]["tp_approx"] += 1
            
            # Update Pos-only
            if is_pos:
                stats["pos_only"]["total_gt"] += 1
                if has_pred: stats["pos_only"]["matched_keys"] += 1
                if t1_match: stats["pos_only"]["tp_exact"] += 1
                if t2_match: stats["pos_only"]["tp_approx"] += 1

    def calculate_metrics(s):
        denom_p = s['matched_keys']
        denom_r = s['total_gt']
        
        # Exact stats
        pe = s['tp_exact'] / denom_p if denom_p > 0 else 0.0
        re = s['tp_exact'] / denom_r if denom_r > 0 else 0.0
        f1e = 2*pe*re/(pe+re) if (pe+re) > 0 else 0.0
        
        # Approx stats
        pa = s['tp_approx'] / denom_p if denom_p > 0 else 0.0
        ra = s['tp_approx'] / denom_r if denom_r > 0 else 0.0
        f1a = 2*pa*ra/(pa+ra) if (pa+ra) > 0 else 0.0
        
        # Primary metrics based on config (Approx is standard for T2 unless EM-only specified)
        metrics = {"p": pa, "r": ra, "f1": f1a} if use_am else {"p": pe, "r": re, "f1": f1e}

        return {
            "metrics": metrics,
            "exact": {"p": pe, "r": re, "f1": f1e, "tp": s['tp_exact']},
            "approx": {"p": pa, "r": ra, "f1": f1a, "tp": s['tp_approx']}
        }

    res_global = calculate_metrics(stats["global"])
    res_pos = calculate_metrics(stats["pos_only"])

    return {
        "global": {
            "stats": {"tp_e": stats["global"]["tp_exact"], "tp_a": stats["global"]["tp_approx"], "total": stats["global"]["total_gt"]},
            "metrics": {
                "exact": res_global["exact"],
                "approx": res_global["approx"]
            }
        },
        "pos_only": {
            "stats": {"tp_e": stats["pos_only"]["tp_exact"], "tp_a": stats["pos_only"]["tp_approx"], "total": stats["pos_only"]["total_gt"]},
            "metrics": {
                "exact": res_pos["exact"],
                "approx": res_pos["approx"]
            }
        }
    }
