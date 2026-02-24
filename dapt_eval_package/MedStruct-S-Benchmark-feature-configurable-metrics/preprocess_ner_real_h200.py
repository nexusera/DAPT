import json, os, hashlib, re

def get_text_hash(text):
    if not text: return ""
    clean = "".join(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", str(text)))[:100]
    return hashlib.md5(clean.encode()).hexdigest()

def process_gt(input_path, output_path):
    print(f"正在转换 Real NER 真值 (GT): {os.path.basename(input_path)}")
    with open(input_path, 'r', encoding='utf-8') as f: data = json.load(f)
    new_data = []
    for item in data:
        rid = str(item.get("record_id") or item.get("id", "N/A"))
        pairs = []
        raw_annos = item.get("transferred_annotations", []) or item.get("annotations", [])
        annos = {a.get("original_id") or a.get("id"): a for a in raw_annos}
        matched_ids = set()
        for rel in item.get("relations", []):
            f_node, t_node = annos.get(rel.get("from_id")), annos.get(rel.get("to_id"))
            if f_node and t_node:
                k = f_node.get("text", "").replace("：", "").replace(":", "").strip()
                v = t_node.get("text", "").strip()
                if k: pairs.append({"key": k, "value": v})
                matched_ids.add(rel.get("from_id")); matched_ids.add(rel.get("to_id"))
        for aid, node in annos.items():
            if aid not in matched_ids:
                label = node.get("labels", ["Unknown"])[0]
                val = node.get("text", "").strip()
                if label not in ["键名", "值", "KEY", "VALUE", "Unknown"] and val:
                    pairs.append({"key": label, "value": val})
        new_data.append({"id": rid, "pairs": pairs})
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in new_data: f.write(json.dumps(item, ensure_ascii=False) + '\n')

def process_pred(input_path, output_path, hash_to_id):
    if not os.path.exists(input_path): return
    with open(input_path, 'r') as f: data = json.load(f)
    if isinstance(data, dict): data = data.get("results", [])
    label_map = {"HOSPITAL": "医院", "ORG": "医院", "DEPARTMENT": "科室", "DOCTOR": "医生", "PATIENT": "姓名"}
    results = []
    for item in data:
        real_id = hash_to_id.get(get_text_hash(item.get("text", "")), "N/A")
        pairs = []
        ents = item.get("entities", [])
        ents.sort(key=lambda x: x.get("start", 0))
        cur_k = None
        for e in ents:
            lbl, txt = (e.get("type") or e.get("label", "UK")).upper(), (e.get("text") or "").replace("：", "").replace(":", "").strip()
            if not txt: continue
            if lbl in label_map: pairs.append({"key": label_map[lbl], "value": txt})
            elif lbl in ["KEY", "键名"]: cur_k = txt
            elif lbl in ["VALUE", "值"] and cur_k: pairs.append({"key": cur_k, "value": txt}); cur_k = None
            else: pairs.append({"key": lbl, "value": txt})
        results.append({"id": real_id, "report_title": None, "pairs": pairs})
    with open(output_path, 'w') as f:
        for it in results: f.write(json.dumps(it, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    GT = "/data/ocean/medstruct_s/medstruct_s_real/sft_data/MedStruct_S_Real_test.json"
    TAG = "real_0221"; OUT_DIR = f"./aligned_data/{TAG}"
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(GT, 'r') as f: gt_raw = json.load(f)
    h2id = {get_text_hash(i.get("ocr_text", "")): str(i.get("record_id") or i.get("id", "")) for i in gt_raw}
    process_gt(GT, os.path.join(OUT_DIR, "gt_ner_aligned.jsonl"))
    PRED_DIR = "/home/ocean/semi_label/bert/predictions/medstruct-s-real/ner/"
    for m in ["macbert", "mbert", "mcbert", "roberta"]:
        process_pred(os.path.join(PRED_DIR, f"MedStruct_S_Real_{m}_test_preds.json"), f"{OUT_DIR}/pred_{m}_ner_aligned.jsonl", h2id)