from med_eval.metrics import (
    normalize_text, 
    compute_similarity, 
    get_threshold, 
    align_instances
)

def evaluate_task3_pairing(predictions, ground_truth, normalize=True, tau_dynamic=True, similarity_threshold=0.8, overlap_threshold=0.0):
    """
    Task 3: E2E Key-Value Extraction (Pairing Mode).
    评估 (Key, Value) 配对的准确性。
    
    输入格式（新标准）:
        每个 sample 的 pairs 为 [{"key": str, "value": str, "key_span": [int,int]|null}, ...]
    
    Args:
        predictions: 预测样本列表。
        ground_truth: 真值样本列表。
        normalize: 文本归一化开关。
        tau_dynamic: 动态阈值开关。
        similarity_threshold: NED 基准阈值。
        overlap_threshold: Span IoU 阈值。
        
    Returns:
        包含三阶指标的评测结果字典。
    """
    total_ee_tp = 0 # Exact Key, Exact Val
    total_ea_tp = 0 # Exact Key, Approximate Val
    total_aa_tp = 0 # Approximate Key, Approximate Val
    
    total_p_count = 0
    total_g_count = 0
    
    # align_instances 所需的内部配置
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

        # 从 dict-style pairs 构建对齐所需的 items（基于 key 文本和 key_span）
        p_items = [{"text": p["key"], "span": p.get("key_span")} for p in p_pairs]
        g_items = [{"text": p["key"], "span": p.get("key_span")} for p in g_pairs]

        # 基于 Key 的组件级对齐
        matches, _, _ = align_instances(p_items, g_items, align_config)

        for pidx, gidx, k_sim, stage in matches:
            pk, pv = p_pairs[pidx]["key"], p_pairs[pidx]["value"]
            gk, gv = g_pairs[gidx]["key"], g_pairs[gidx]["value"]

            # Value 相似度
            v_sim = compute_similarity(pv, gv, use_norm)
            v_threshold = get_threshold(len(str(gv))) if tau_dynamic else sim_threshold
            
            v_is_exact = (normalize_text(pv, use_norm) == normalize_text(gv, use_norm))
            v_is_approx = (v_sim >= v_threshold)
            
            # Key 精确性
            k_is_exact = (normalize_text(pk, use_norm) == normalize_text(gk, use_norm))
            # Key 近似性（对齐后隐含满足，但显式检查更严谨）
            k_sim_val = compute_similarity(pk, gk, use_norm)
            k_thresh = get_threshold(len(str(gk))) if tau_dynamic else sim_threshold
            k_is_approx = (k_sim_val >= k_thresh)
            
            # 三阶 TP 判定
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
