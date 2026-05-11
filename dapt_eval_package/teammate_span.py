import json, os, hashlib, re

def get_text_hash(text):
    if not text: return ""
    clean = "".join(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", str(text)))[:100]
    return hashlib.md5(clean.encode()).hexdigest()

def process_gt(input_path, output_path):
    print(f"正在转换 Real NER 真值 (GT) 并提取 Span: {os.path.basename(input_path)}")
    with open(input_path, 'r', encoding='utf-8') as f: data = json.load(f)
    new_data = []
    for item in data:
        ocr_text = item.get("ocr_text", "")
        anno_spans = {}
        last_pos = 0
        raw_annos = item.get("transferred_annotations", []) or item.get("annotations", [])
        for a in raw_annos:
            aid = a.get("original_id") or a.get("id")
            val = a.get("text", "")
            if val:
                # 1. 尝试原始文本顺序搜索
                start = ocr_text.find(val, last_pos)
                v_len = len(val)
                # 2. 如果失败，尝试去掉前后空格
                if start == -1:
                    v_strip = val.strip()
                    start = ocr_text.find(v_strip, last_pos)
                    v_len = len(v_strip)
                # 3. 如果还是失败，深入处理内部空格 (正则搜索)
                if start == -1:
                    pattern_str = "".join([re.escape(c) + r"\s*" for c in val.strip() if not c.isspace()])
                    if pattern_str:
                        match = re.search(pattern_str, ocr_text[last_pos:])
                        if match:
                            start = last_pos + match.start()
                            v_len = match.end() - match.start()
                # 4. 如果还是失败，回退到全局带正则搜索
                if start == -1:
                    pattern_str = "".join([re.escape(c) + r"\s*" for c in val.strip() if not c.isspace()])
                    if pattern_str:
                        match = re.search(pattern_str, ocr_text)
                        if match:
                            start = match.start()
                            v_len = match.end() - match.start()

                if start != -1:
                    end = start + v_len
                    anno_spans[aid] = [start, end]
                    last_pos = end
                else:
                    anno_spans[aid] = None
            else:
                anno_spans[aid] = None

        rid = str(item.get("record_id") or item.get("id", "N/A"))
        pairs = []
        annos = {a.get("original_id") or a.get("id"): a for a in raw_annos}
        matched_ids = set()
        for rel in item.get("relations", []):
            f_node, t_node = annos.get(rel.get("from_id")), annos.get(rel.get("to_id"))
            if f_node and t_node:
                fid, tid = rel.get("from_id"), rel.get("to_id")
                k = f_node.get("text", "").replace("：", "").replace(":", "").strip()
                v = t_node.get("text", "").strip()
                if k:
                    pairs.append({
                        "key": k, "value": v, 
                        "key_span": anno_spans.get(fid), 
                        "value_span": anno_spans.get(tid)
                    })
                matched_ids.add(fid); matched_ids.add(tid)
        
        for aid, node in annos.items():
            if aid not in matched_ids:
                label = node.get("labels", ["Unknown"])[0]
                val = node.get("text", "").strip()
                if label not in ["键名", "值", "KEY", "VALUE", "Unknown"] and val:
                    pairs.append({
                        "key": label, "value": val, 
                        "key_span": None, 
                        "value_span": anno_spans.get(aid)
                    })
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
        cur_k, cur_k_span = None, None
        for e in ents:
            lbl = (e.get("type") or e.get("label", "UK")).upper()
            txt = (e.get("text") or "").replace("：", "").replace(":", "").strip()
            span = [e.get("start"), e.get("end")] if e.get("start") is not None else None
            if not txt: continue
            if lbl in label_map:
                pairs.append({"key": label_map[lbl], "value": txt, "key_span": None, "value_span": span})
            elif lbl in ["KEY", "键名"]:
                cur_k, cur_k_span = txt, span
            elif lbl in ["VALUE", "值"] and cur_k:
                pairs.append({"key": cur_k, "value": txt, "key_span": cur_k_span, "value_span": span})
                cur_k, cur_k_span = None, None
            else:
                pairs.append({"key": lbl, "value": txt, "key_span": None, "value_span": span})
        results.append({"id": real_id, "pairs": pairs})
    with open(output_path, 'w') as f:
        for it in results: f.write(json.dumps(it, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    GT = "/data/ocean/medstruct_s/medstruct_s_real/MedStruct_S_Real_test.json"
    TAG = "real_ner_span_v3"; OUT_DIR = f"./aligned_data/{TAG}"
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(GT, 'r') as f: gt_raw = json.load(f)
    h2id = {get_text_hash(i.get("ocr_text", "")): str(i.get("record_id") or i.get("id", "")) for i in gt_raw}
    process_gt(GT, os.path.join(OUT_DIR, "gt_ner_aligned.jsonl"))
    PRED_DIR = "/data/ocean/medstruct_s/runs-bert/ner/"
    for m in ["macbert", "mbert", "mcbert", "roberta"]:
        process_pred(os.path.join(PRED_DIR, m, f"MedStruct_S_Real_{m}_test_preds.json"), f"{OUT_DIR}/pred_{m}_ner_aligned.jsonl", h2id)

