import json, os, hashlib, re
import argparse

def get_text_hash(text):
    if not text: return ""
    # 只保留中文、英文和数字进行哈希，忽略标点和空格差异
    clean = "".join(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", str(text)))[:100]
    return hashlib.md5(clean.encode()).hexdigest()

def process_gt(input_path, output_path):
    print(f"正在转换 Real EBQA 真值 (GT): {os.path.basename(input_path)}")
    with open(input_path, 'r', encoding='utf-8') as f: data = json.load(f)
    new_data = []
    for item in data:
        # 优先使用 record_id
        rid = str(item.get("record_id") or item.get("id", "N/A"))
        pairs = []
        
        # 处理标注：兼容 transferred_annotations 和 annotations
        raw_annos = item.get("transferred_annotations", []) or item.get("annotations", [])
        # 建立 id -> entity 映射
        annos = {str(a.get("original_id") or a.get("id")): a for a in raw_annos}
        
        # 从 relations 中提取 Key-Value 对
        for rel in item.get("relations", []):
            f_node = annos.get(str(rel.get("from_id")))
            t_node = annos.get(str(rel.get("to_id")))
            if f_node and t_node:
                pairs.append({
                    "key": f_node.get("text", "").strip(), 
                    "value": t_node.get("text", "").strip()
                })
        
        new_data.append({
            "id": rid, 
            "report_title": item.get("category") or item.get("report_title", "通用病历"), 
            "text": item.get("ocr_text") or item.get("text", ""), # 保留文本以备查
            "pairs": pairs
        })
        
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in new_data: 
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"GT 转换完成，共 {len(new_data)} 条，保存至 {output_path}")
    return new_data

def process_pred(p_in, p_out, h_meta):
    if not os.path.exists(p_in): 
        print(f"警告：找不到预测文件 {p_in}，跳过。")
        return

    print(f"正在对齐预测文件: {os.path.basename(p_in)}")
    results = []
    matched_count = 0
    
    with open(p_in, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            it = json.loads(line)
            
            # 计算哈希以查找对应的 GT ID
            txt = it.get("text", "") or it.get("ocr_text", "")
            txt_hash = get_text_hash(txt)
            
            # 从 h_meta 中查找元数据
            m = h_meta.get(txt_hash)
            
            if m:
                matched_count += 1
                final_id = m["id"]
                final_title = m["title"]
            else:
                # 没匹配上也保留，生成一个临时ID
                final_id = str(it.get("id", f"UNKNOWN_{txt_hash[:8]}"))
                final_title = it.get("report_title", "通用病历")
            
            # 提取预测对，兼容 pred_pairs 或 pairs 字段
            pairs = it.get("pred_pairs", it.get("pairs", []))
            
            results.append({
                "id": final_id, 
                "report_title": final_title, 
                "pairs": pairs
            })
            
    with open(p_out, 'w', encoding='utf-8') as f:
        for it in results: 
            f.write(json.dumps(it, ensure_ascii=False) + '\n')
    
    print(f"预测文件对齐完成：{matched_count}/{len(results)} 条成功匹配 GT。保存至 {p_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align GT and Pred data for Scorer using Content Hashing")
    parser.add_argument("--gt_raw", required=True, help="Path to raw GT JSON file (e.g. real_test_with_ocr.json)")
    parser.add_argument("--pred_file", required=True, help="Path to Prediction JSONL file (e.g. ebqa_macbert_preds.jsonl)")
    parser.add_argument("--output_dir", default="aligned_data", help="Directory to save aligned files")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 处理 GT 并建立哈希索引
    gt_aligned_path = os.path.join(args.output_dir, "gt_aligned.jsonl")
    gt_data = process_gt(args.gt_raw, gt_aligned_path)
    
    # 构建哈希表：Hash -> {id, title}
    # 注意：使用 ocr_text 或 text 字段
    h_meta = {}
    for i in gt_data:
        h = get_text_hash(i.get("text", ""))
        h_meta[h] = {"id": i["id"], "title": i["report_title"]}
        
    # 2. 处理预测文件
    base_name = os.path.basename(args.pred_file)
    pred_aligned_path = os.path.join(args.output_dir, f"aligned_{base_name}")
    process_pred(args.pred_file, pred_aligned_path, h_meta)
    
    print("\n[完成] 现在可以使用以下命令运行 scorer:")
    print(f"python DAPT/MedStruct-S-Benchmark-master/scorer.py --gt_file {gt_aligned_path} --pred_file {pred_aligned_path}")
