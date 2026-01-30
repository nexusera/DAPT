import argparse
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def compute_metrics(gt_embs, pred_embs, threshold=0.8):
    """
    Compute Recall, Precision, and F1 based on Embedding Similarity using RAG-like matching.
    
    For each GT item, we find the closest Pred item (max similarity).
    If max_sim >= threshold, we consider it a 'match'.
    
    Recall = (Matched GT items) / (Total GT items)
    Precision = (Matched Pred items) / (Total Pred items)
    
    Also returns average similarity scores.
    """
    if len(gt_embs) == 0:
        return {
            "recall": 0.0, "precision": 0.0, "f1": 0.0,
            "avg_max_sim_recall": 0.0, "avg_max_sim_precision": 0.0
        }
        
    if len(pred_embs) == 0:
        return {
            "recall": 0.0, "precision": 0.0, "f1": 0.0,
            "avg_max_sim_recall": 0.0, "avg_max_sim_precision": 0.0
        }

    # Similarity Matrix: [n_gt, n_pred]
    sim_matrix = cosine_similarity(gt_embs, pred_embs)
    
    # 1. Recall perspective: For each GT, what is the best Pred?
    # max along axis 1 (columns/predictions)
    gt_max_sims = np.max(sim_matrix, axis=1)
    
    # 2. Precision perspective: For each Pred, what is the best GT?
    # max along axis 0 (rows/gt)
    pred_max_sims = np.max(sim_matrix, axis=0)
    
    # Metrics based on threshold
    matched_gt = (gt_max_sims >= threshold).sum()
    matched_pred = (pred_max_sims >= threshold).sum()
    
    recall = matched_gt / len(gt_embs)
    precision = matched_pred / len(pred_embs)
    
    f1 = 0.0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "avg_max_sim_recall": np.mean(gt_max_sims),
        "avg_max_sim_precision": np.mean(pred_max_sims)
    }

def main():
    parser = argparse.ArgumentParser(description="Compare GT and Pred embeddings to calculate similarity.")
    parser.add_argument("--gt_pkl", type=str, required=True, help="Path to the GT pickle file.")
    parser.add_argument("--pred_pkl", type=str, required=True, help="Path to the Prediction pickle file.")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold for considering a match.")
    
    args = parser.parse_args()
    
    print(f"Loading GT from {args.gt_pkl}...")
    with open(args.gt_pkl, 'rb') as f:
        gt_data = pickle.load(f)
        
    print(f"Loading Pred from {args.pred_pkl}...")
    with open(args.pred_pkl, 'rb') as f:
        pred_data = pickle.load(f)
        
    # Find common IDs
    common_ids = set(gt_data.keys()).intersection(set(pred_data.keys()))
    print(f"Found {len(common_ids)} common IDs to evaluate.")
    
    key_metrics_list = []
    value_metrics_list = []
    
    for item_id in tqdm(common_ids):
        gt_item = gt_data[item_id]
        pred_item = pred_data[item_id]
        
        # Compare Keys
        k_metrics = compute_metrics(gt_item['keys_emb'], pred_item['keys_emb'], args.threshold)
        key_metrics_list.append(k_metrics)
        
        # Compare Values
        v_metrics = compute_metrics(gt_item['values_emb'], pred_item['values_emb'], args.threshold)
        value_metrics_list.append(v_metrics)
        
    # Aggregate results
    def aggregate(metrics_list):
        if not metrics_list:
            return {}
        avg_res = {}
        for k in metrics_list[0].keys():
            avg_res[k] = np.mean([m[k] for m in metrics_list])
        return avg_res
        
    avg_key_metrics = aggregate(key_metrics_list)
    avg_value_metrics = aggregate(value_metrics_list)
    
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(f"Total Items Evaluated: {len(common_ids)}")
    print(f"Similarity Threshold: {args.threshold}")
    print("-" * 30)
    print("KEY EXTRACTION METRICS:")
    print(f"  Recall (Semantic):    {avg_key_metrics.get('recall', 0):.4f}")
    print(f"  Precision (Semantic): {avg_key_metrics.get('precision', 0):.4f}")
    print(f"  F1 Score (Semantic):  {avg_key_metrics.get('f1', 0):.4f}")
    print(f"  Avg Max Sim (Recall side):    {avg_key_metrics.get('avg_max_sim_recall', 0):.4f}")
    print(f"  Avg Max Sim (Precision side): {avg_key_metrics.get('avg_max_sim_precision', 0):.4f}")
    
    print("-" * 30)
    print("VALUE EXTRACTION METRICS:")
    print(f"  Recall (Semantic):    {avg_value_metrics.get('recall', 0):.4f}")
    print(f"  Precision (Semantic): {avg_value_metrics.get('precision', 0):.4f}")
    print(f"  F1 Score (Semantic):  {avg_value_metrics.get('f1', 0):.4f}")
    print(f"  Avg Max Sim (Recall side):    {avg_value_metrics.get('avg_max_sim_recall', 0):.4f}")
    print(f"  Avg Max Sim (Precision side): {avg_value_metrics.get('avg_max_sim_precision', 0):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
