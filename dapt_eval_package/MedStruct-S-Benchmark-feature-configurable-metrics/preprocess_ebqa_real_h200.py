import json, os, hashlib, re

def get_text_hash(text):
    if not text: return ""
    clean = "".join(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", str(text)))[:100]
    return hashlib.md5(clean.encode()).hexdigest()

def _clean_key(k):
    # Key matching is schema-driven; remove whitespace and trailing separators.
    s = str(k or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", "", s)
    s = s.replace(":", "").replace("：", "")
    s = s.strip()
    return s

def _clean_val(v):
    # Value comparison uses NED; strip only common wrappers and trailing punctuation.
    s = str(v or "").strip()
    if not s:
        return ""
    # remove surrounding brackets repeatedly
    while True:
        ns = s
        for l, r in [("[", "]"), ("（", "）"), ("(", ")"), ("【", "】"), ("<", ">"), ("《", "》")]:
            if ns.startswith(l) and ns.endswith(r) and len(ns) >= 2:
                ns = ns[1:-1].strip()
        if ns == s:
            break
        s = ns
    # trim trailing separators commonly attached in OCR text
    s = re.sub(r"[，。,；;、:：\]\)）】>》]+$", "", s).strip()
    return s

def process_gt(input_path, output_path):
    print(f"正在转换 Real EBQA 真值 (GT): {os.path.basename(input_path)}")
    with open(input_path, 'r', encoding='utf-8') as f: data = json.load(f)
    new_data = []
    for item in data:
        rid = str(item.get("record_id") or item.get("id", "N/A"))
        pairs = []
        raw_annos = item.get("transferred_annotations", []) or item.get("annotations", [])
        annos = {a.get("original_id") or a.get("id"): a for a in raw_annos}
        for rel in item.get("relations", []):
            f_node, t_node = annos.get(rel.get("from_id")), annos.get(rel.get("to_id"))
            if f_node and t_node:
                pairs.append({
                    "key": _clean_key(f_node.get("text", "")),
                    "value": _clean_val(t_node.get("text", "")),
                })
        new_data.append({"id": rid, "report_title": item.get("category", "通用病历"), "pairs": pairs})
    with open(output_path, 'w') as f:
        for item in new_data: f.write(json.dumps(item, ensure_ascii=False) + '\n')

def process_pred(p_in, p_out, h_meta):
    if not os.path.exists(p_in): return
    results = []
    with open(p_in, 'r') as f:
        for line in f:
            if not line.strip(): continue
            it = json.loads(line)
            m = h_meta.get(get_text_hash(it.get("text", "")), {"id": str(it.get("id", "N/A")), "title": "通用病历"})
            raw_pairs = it.get("pred_pairs", it.get("pairs", [])) or []
            cleaned_pairs = []
            for p in raw_pairs:
                if not isinstance(p, dict):
                    continue
                ck = _clean_key(p.get("key"))
                cv = _clean_val(p.get("value"))
                if ck == "":
                    continue
                cleaned_pairs.append({"key": ck, "value": cv})
            results.append({"id": m["id"], "report_title": m["title"], "pairs": cleaned_pairs})
    with open(p_out, 'w') as f:
        for it in results: f.write(json.dumps(it, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Align Prediction and GT using Text Hash (Teammate's Logic)")
    parser.add_argument("--gt_file", required=True, help="Path to Ground Truth JSON file (original format with annotations)")
    parser.add_argument("--pred_file", required=True, help="Path to Prediction JSONL file")
    parser.add_argument("--output_dir", default="./aligned_data", help="Directory to save aligned files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load GT and Build Hash Map
    print(f"Loading GT from {args.gt_file}...")
    with open(args.gt_file, 'r', encoding='utf-8') as f: 
        gt_raw = json.load(f)
    
    # Critical: Use teammate's hashing logic to map content -> ID/Title
    h_meta = {}
    for i in gt_raw:
        # Try both 'ocr_text' (original) and 'text' as fallback
        text_content = i.get("ocr_text") or i.get("text", "")
        h = get_text_hash(text_content)
        
        rid = str(i.get("record_id") or i.get("id", ""))
        title = i.get("category", "通用病历")
        h_meta[h] = {"id": rid, "title": title}
        
    print(f"Built hash map for {len(h_meta)} documents.")

    # 2. Process GT (Convert relationships to pairs)
    gt_out = os.path.join(args.output_dir, "gt_ebqa_aligned.jsonl")
    process_gt(args.gt_file, gt_out)
    print(f"Saved aligned GT to {gt_out}")

    # 3. Process Prediction (Align ID via Hash)
    base_name = os.path.basename(args.pred_file)
    pred_out = os.path.join(args.output_dir, f"aligned_{base_name}")
    process_pred(args.pred_file, pred_out, h_meta)
    print(f"Saved aligned Prediction to {pred_out}")