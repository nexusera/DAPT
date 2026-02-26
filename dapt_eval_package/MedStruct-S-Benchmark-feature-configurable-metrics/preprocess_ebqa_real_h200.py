import json, os, hashlib, re

def get_text_hash(text):
    if not text: return ""
    clean = "".join(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", str(text)))[:100]
    return hashlib.md5(clean.encode()).hexdigest()

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
                pairs.append({"key": f_node.get("text", "").strip(), "value": t_node.get("text", "").strip()})
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
            results.append({"id": m["id"], "report_title": m["title"], "pairs": it.get("pred_pairs", it.get("pairs", []))})
    with open(p_out, 'w') as f:
        for it in results: f.write(json.dumps(it, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    GT = "/data/ocean/medstruct_s/medstruct_s_real/sft_data/MedStruct_S_Real_test.json"
    TAG = "real_0221"; OUT_DIR = f"./aligned_data/{TAG}"
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(GT, 'r') as f: gt_raw = json.load(f)
    h_meta = {get_text_hash(i.get("ocr_text", "")): {"id": str(i.get("record_id") or i.get("id", "")), "title": i.get("category", "通用病历")} for i in gt_raw}
    process_gt(GT, f"{OUT_DIR}/gt_ebqa_aligned.jsonl")
    PRED_DIR = "/home/ocean/semi_label/bert/predictions/medstruct-s-real/ebqa/"
    for m in ["macbert", "mbert", "mcbert", "roberta"]:
        process_pred(os.path.join(PRED_DIR, f"MedStruct_S_Real_{m}_test_preds.jsonl"), f"{OUT_DIR}/pred_{m}_ebqa_aligned.jsonl", h_meta)