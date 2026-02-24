from med_eval.metrics import (
    normalize_text, 
    compute_similarity, 
    get_threshold, 
    compute_iou
)

def evaluate_task1_discovery(predictions, ground_truth, normalize=True, tau_dynamic=True, similarity_threshold=0.8, overlap_threshold=0.0):
    """
    Task 1: Key Discovery.
    评估模型是否正确发现了文档中的键名。
    
    输入格式（新标准）:
        每个 sample 的 pairs 为 [{"key": str, "value": str, "key_span": [int,int]|null}, ...]
        对于 GT，scorer.py 会预先过滤只保留有值的 pair（Positive Discovery）。
    
    Args:
        predictions: 预测样本列表。
        ground_truth: 真值样本列表（已过滤为 Positive Discovery）。
        normalize: 文本归一化开关。
        tau_dynamic: 动态阈值开关。
        similarity_threshold: NED 基准阈值。
        overlap_threshold: Span IoU 阈值。
        
    Returns:
        包含 stats 和 metrics 的评测结果字典。
    """
    total_tp_e = 0
    total_tp_a = 0
    total_p_count = 0
    total_g_count = 0
    
    # 内部行为标志（强制启用）
    use_em = True
    use_am = True
    use_span = True
    
    use_norm = normalize
    sim_threshold = similarity_threshold

    for p_sample, g_sample in zip(predictions, ground_truth):
        # 从 dict-style pairs 中提取 keys 和 spans
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
                # Span 位置校验
                if use_span and compute_iou(p_item["span"], g_item["span"]) <= overlap_threshold:
                    continue
                # 文本精确匹配
                if p_item["text"] == g_item["text"]:
                    matched_g_e.add(gi)
                    break
        
        # Phase 2: Approx Match (AM) - 包含 EM
        for pi, p_item in enumerate(p_items):
            best_gi = -1
            max_sim = -1
            for gi, g_item in enumerate(g_items):
                if gi in matched_g_a: continue
                # Span 位置校验
                if use_span and compute_iou(p_item["span"], g_item["span"]) <= overlap_threshold:
                    continue
                # 相似度校验
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
