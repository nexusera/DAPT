import json
import os
import sys
import re
import argparse

def normalize_text(text):
    if not text: return ""
    # 简单的清理：去除冒号等
    return str(text).replace("：", "").replace(":", "").strip()

def process_gt(input_path, output_path):
    print(f"正在转换 GT 文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f) # 假设是一整个列表
    
    new_data = []
    for item in data:
        # 优先使用 record_id 即使是字符串
        rid = str(item.get("record_id") or item.get("id", "N/A"))
        pairs = []
        
        # 1. 提取所有标注节点
        raw_annos = item.get("transferred_annotations", []) or item.get("annotations", [])
        # 构建 id -> node 映射
        annos = {}
        for a in raw_annos:
            if isinstance(a, dict):
                aid = a.get("original_id") or a.get("id")
                if aid:
                    annos[aid] = a
        
        matched_ids = set()
        
        # 2. 处理 Relations (明确的 Key -> Value 关系)
        # 对应同事脚本中处理 relations 的部分
        for rel in item.get("relations", []):
            f_id = rel.get("from_id")
            t_id = rel.get("to_id")
            f_node = annos.get(f_id)
            t_node = annos.get(t_id)
            
            if f_node and t_node:
                # from 通常是 Key, to 是 Value
                k = normalize_text(f_node.get("text", ""))
                v = normalize_text(t_node.get("text", ""))
                
                # 只有当 Key 存在时才添加
                if k: 
                    pairs.append({"key": k, "value": v})
                
                matched_ids.add(f_id)
                matched_ids.add(t_id)
        
        # 3. 处理独立的 Annotations (未在 Relation 中的)
        # 对应同事脚本中处理剩余 label 的部分
        for aid, node in annos.items():
            if aid not in matched_ids:
                labels = node.get("labels", [])
                if not labels: continue
                label = labels[0] # 取第一个标签作为 Key
                val = normalize_text(node.get("text", ""))
                
                # 排除通用标签，只保留特定实体的标签作为 Key
                if label not in ["键名", "值", "KEY", "VALUE", "Unknown"] and val:
                    pairs.append({"key": label, "value": val})

        # 输出格式：Task 1/3 需要的格式
        # 注意：这里我们故意不包含 key_span，迫使 scorer 使用纯文本匹配（配合参数）
        new_data.append({
            "id": rid,
            "pairs": pairs,
            "ocr_text": item.get("ocr_text", "") # 保留 OCR 原文以防万一
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in new_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"GT 转换完成，样本数: {len(new_data)}")

def process_pred(input_path, output_path):
    print(f"正在转换预测文件: {input_path}")
    # 我们的 compare_models.py 输出已经是 jsonl 格式
    # 但里面的结构可能需要微调以匹配上面 GT 的结构 (主要是 ID 对齐)
    
    new_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            # 兼容：compare_models.py 输出的 pairs 已经是 [{"key":..., "value":...}]
            # 只需要确保 ID 是字符串且一致
            rid = str(item.get("id", "N/A"))
            
            processed_pairs = []
            for p in item.get("pairs", []):
                # 清理 Key/Value
                k = normalize_text(p.get("key"))
                v = normalize_text(p.get("value"))
                if k and v:
                    processed_pairs.append({"key": k, "value": v})
            
            new_data.append({
                "id": rid,
                "pairs": processed_pairs,
                "report_title": item.get("report_title", "")
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in new_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"预测文件转换完成，样本数: {len(new_data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_in", required=True, help="原始 GT JSON 文件路径 (real_test.json)")
    parser.add_argument("--pred_in", required=True, help="原始预测 JSONL 文件路径 (_preds.jsonl)")
    parser.add_argument("--gt_out", required=True, help="对齐后的 GT 输出路径")
    parser.add_argument("--pred_out", required=True, help="对齐后的预测输出路径")
    args = parser.parse_args()

    # 1. 处理 GT
    process_gt(args.gt_in, args.gt_out)
    
    # 2. 处理 Pred
    process_pred(args.pred_in, args.pred_out)
