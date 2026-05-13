
import json
import argparse
import sys
import os
import re
from collections import defaultdict

# Add current directory to path for query_set_utils
sys.path.insert(0, os.getcwd())
try:
    from pre_struct.kv_ner.query_set_utils import load_query_set
except ImportError:
    # Fallback if structure varies
    def load_query_set(p): 
        with open(p, 'r', encoding='utf-8') as f: return json.load(f)

def robust_parse_json_list(text):
    """
    Attempt to parse a JSON list of objects robustly.
    1. Try standard json.loads
    2. Try fixing common issues (unescaped newlines)
    3. Fallback to regex extraction of individual objects
    """
    text = text.strip()
    
    # 1. Clean Markdown
    text = re.sub(r'```+json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```+', '', text).strip()
    
    if not text: return []

    # 2. Try Standard Parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    # 3. Try to sanitize control characters (newlines in strings are common culprets)
    # Replace real newlines with \n, but this is dangerous for structure. 
    # Better to just try extracting objects.

    objects = []
    
    # Regex to find {...} patterns. 
    # Assuming no nested braces for Task 2 (it's flat k-v pairs).
    # We use a non-greedy match for content inside braces.
    # We allow multiline matches.
    matches = re.finditer(r'\{([^{}]*)\}', text, re.DOTALL)
    
    for match in matches:
        obj_str = match.group(0)
        try:
            # Try parsing the individual object
            obj = json.loads(obj_str)
            objects.append(obj)
        except json.JSONDecodeError:
            # If that fails, it might be unescaped content. 
            # Try to fix newlines inside the string
            try:
                # Replace unescaped newlines with space or \n
                # This is a heuristic: replace \n with \\n
                clean_obj_str = obj_str.replace('\n', '\\n').replace('\r', '')
                obj = json.loads(clean_obj_str)
                objects.append(obj)
            except:
                # Last resort: regex extract key and value from the object string
                # Look for "key": "..." and "value": "..."
                k_match = re.search(r'"key"\s*:\s*"(.*?)(?<!\\)"', obj_str, re.DOTALL)
                v_match = re.search(r'"value"\s*:\s*"(.*?)(?<!\\)"', obj_str, re.DOTALL)
                
                if k_match and v_match:
                    k = k_match.group(1)
                    v = v_match.group(1)
                    # We might have raw escaped chars now, but it's better than nothing
                    objects.append({"key": k, "value": v})
    
    return objects

def get_fingerprint(text, length=100):
    """Generate a stable fingerprint from the beginning of a medical report."""
    if not text: return ""
    # Standard cleanup: remove non-alphanumeric/Chinese, remove whitespace
    # We want something that survives minor OCR variations or manual pruning.
    clean = re.sub(r'[^\w\u4e00-\u9fa5]', '', text)
    return clean[:length]

def normalize_question_text(text):
    if not text:
        return ""
    return re.sub(r'[\s\W_]+', '', str(text))

def extract_key_from_question(question, title, query_set, q_to_key_map):
    q_str = question.strip()
    normalized_q = normalize_question_text(q_str)
    
    # 1. Exact map
    if q_str in q_to_key_map:
        return q_to_key_map[q_str]
    if normalized_q in q_to_key_map:
        return q_to_key_map[normalized_q]
        
    # 2. Query-Set-based containment
    if title and query_set:
        candidates = []
        if title in query_set:
            candidates.extend(query_set[title].keys())
        if 'General Medical Record' in query_set or '通用病历' in query_set:
            gen_key = 'General Medical Record' if 'General Medical Record' in query_set else '通用病历'
            candidates.extend(query_set[gen_key].keys())
        
        # Sort by length desc to match specific keys first
        candidates.sort(key=len, reverse=True)
        
        for k in candidates:
            if k in q_str or normalize_question_text(k) in normalized_q:
                return k

    # 2.5 Fallback: search all titles if the title-specific lookup failed
    if query_set:
        global_candidates = []
        for fields in query_set.values():
            if isinstance(fields, dict):
                global_candidates.extend(fields.keys())
        global_candidates = sorted(set(global_candidates), key=len, reverse=True)
        for k in global_candidates:
            if k in q_str or normalize_question_text(k) in normalized_q:
                return k

    # 3. Heuristic fallback (Matches specific Chinese patterns in questions)
    match = re.search(r"患者?(.+?)是什么", q_str)
    if match: return match.group(1)
    match = re.search(r"此.+?的(.+?)是什么", q_str)
    if match: return match.group(1)
    match = re.search(r"(.+?)是什么", q_str)
    if match: return match.group(1)
    
    return "UNKNOWN_KEY"

def clean_cot(text):
    """Clean Chain-of-Thought artifacts and JSON leftovers."""
    if not text: return ""
    text = text.strip()
    # Remove trailing brace if present (common artifact)
    if text.endswith("}"):
        text = text[:-1].strip()
    # If </think> exists, usually the real answer is after the last one
    if "</think>" in text:
        parts = text.split("</think>")
        # Take the last non-empty part
        for p in reversed(parts):
            if p.strip():
                text = p.strip()
                break
    # Remove prefixes (Matches "Answer:" in Chinese)
    text = text.replace("答案：", "").replace("### 答案：", "").strip()
    return text

def normalize_gold_value(value):
    """Normalize GT payloads from native Python objects into a string form."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()

def normalize_pair_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def append_pairs_from_payload(payload, out_pairs):
    """Append standardized {key, value} pairs from parsed JSON payloads."""
    if isinstance(payload, list):
        for item in payload:
            append_pairs_from_payload(item, out_pairs)
    elif isinstance(payload, dict):
        explicit_key = payload.get('key') if 'key' in payload else payload.get('k')
        has_explicit_value = 'value' in payload or 'v' in payload
        explicit_value = payload.get('value') if 'value' in payload else payload.get('v')
        if explicit_key is not None and has_explicit_value:
            out_pairs.append({"key": str(explicit_key), "value": normalize_pair_value(explicit_value)})
            return
        if len(payload) == 1:
            for k, v in payload.items():
                out_pairs.append({"key": str(k), "value": normalize_pair_value(v)})
            return
        for k, v in payload.items():
            out_pairs.append({"key": str(k), "value": normalize_pair_value(v)})

def main():
    parser = argparse.ArgumentParser(description="Convert LLM Outputs and recover Titles for Scoring")
    parser.add_argument("--llm_file", required=True, help="GPT output JSONL")
    parser.add_argument("--gt_master", default="data/kv_ner_prepared_comparison/val_eval.jsonl")
    parser.add_argument("--query_set", dest="query_set_file", default="data/kv_ner_prepared_comparison/keys_merged_1027_cleaned.json")
    parser.add_argument("--task_type", choices=['task1', 'task2', 'task3'], required=True)
    parser.add_argument("--output_pred", required=True)
    parser.add_argument("--output_gt", required=True)
    args = parser.parse_args()

    # 1. Build Fingerprint Database from Master GT
    print(f"Building fingerprint index from {args.gt_master}...")
    fingerprint_to_rec = defaultdict(list) # Changed to list to handle 1-to-Many (Text -> Questions)
    gt_fingerprints = []  # Store for fuzzy matching
    
    if os.path.exists(args.gt_master):
        with open(args.gt_master, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                txt = rec.get('text') or rec.get('ocr_text') or rec.get('report') or rec.get('input') or ""
                # Handle <text> tag if present in GT master (e.g. SFT data)
                if "<text>" in txt:
                    match = re.search(r'<text>(.*?)</text>', txt, re.DOTALL)
                    if match:
                        txt = match.group(1).strip()
                
                fp = get_fingerprint(txt)
                if fp:
                    # Map to title and full record if needed
                    item = {
                        "report_title": rec.get('report_title', "Unknown"),
                        "full_gt": rec
                    }
                    fingerprint_to_rec[fp].append(item)
                    gt_fingerprints.append(fp)
                    
    print(f"Indexed {len(fingerprint_to_rec)} fingerprints.")

    # 2. Load Query Set for Task 3 Key discovery
    query_set = {}
    q_to_key = {}
    if args.query_set_file and os.path.exists(args.query_set_file):
        print(f"Loading Query Set from {args.query_set_file}...")
        query_set = load_query_set(args.query_set_file)
        for title, key_map in query_set.items():
            for key, info in key_map.items():
                q = info.get("Q", "").strip()
                if q:
                    q_to_key[q] = key
                    q_to_key[normalize_question_text(q)] = key

    # Pre-compute GT tokens for fuzzy matching to avoid O(N*M) re-tokenization
    print("Pre-computing GT tokens for fuzzy matching...")
    def get_tokens(t): return set(re.findall(r'[a-zA-Z0-9\u4e00-\u9fa5]{2,}', t))
    
    gt_token_cache = []
    # For fuzzy text matching, we just need one representative text per fingerprint group
    for gt_fp, items in fingerprint_to_rec.items():
        if not items: continue
        gt_item = items[0] # Take first one for text matching
        gt_txt = gt_item['full_gt'].get('input') or gt_item['full_gt'].get('text') or ""
        if "<text>" in gt_txt:
            m = re.search(r'<text>(.*?)</text>', gt_txt, re.DOTALL)
            if m: gt_txt = m.group(1).strip()
        
        tokens = get_tokens(gt_txt)
        if len(tokens) > 2:
            gt_token_cache.append((gt_fp, tokens))

    # 3. Process LLM File
    print(f"Processing LLM file: {args.llm_file}...")
    final_preds = []
    final_gts = []
    
    # Task 3 Aggegation
    task3_buffer = defaultdict(lambda: {"raw_text": "", "pred_pairs": [], "gt_pairs": [], "title": "Unknown"})

    with open(args.llm_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx % 50 == 0: print(f"Processing record {idx}...", end='\r')
            try:
                item = json.loads(line)
            except: continue
            
            full_input = item.get('input', "")
            raw_pred = normalize_gold_value(item.get('prediction', ""))
            
            # Extract "clean" report text for matching
            if args.task_type == 'task2' and "问题：" in full_input:
                parts = full_input.split("问题：")
                report_part = parts[0].replace("文档内容：", "").strip()
                q_part = parts[1].strip().split('\n')[0].strip()
            else:
                if "<text>" in full_input:
                    if "</text>" in full_input:
                        report_part = full_input.split("<text>")[-1].split("</text>")[0].strip()
                    else:
                        report_part = full_input.replace("<text>", "").strip()
                else:
                    # Remove common prefixes in LLM input
                    report_part = full_input.replace("医疗报告文本：", "").replace("文档内容：", "").replace("---\n", "").strip()
                
                # Extract question if present
                if "<question>" in full_input:
                    match = re.search(r'<question>(.*?)</question>', full_input, re.DOTALL)
                    q_part = match.group(1).strip() if match else ""
                else:
                    q_part = ""

            # Match Title & GT Record
            
            # 1. Text Matching
            txt_content = report_part
            if "<text>" in full_input:
                match = re.search(r'<text>(.*?)</text>', full_input, re.DOTALL)
                if match:
                    txt_content = match.group(1).strip()
            
            fp = get_fingerprint(txt_content)
            candidates = fingerprint_to_rec.get(fp)
            
            if not candidates:
                # Text Jaccard Fallback
                llm_tokens = get_tokens(txt_content)
                if len(llm_tokens) > 2: 
                    best_score = 0
                    best_fp = None
                    for gt_fp, gt_tokens in gt_token_cache:
                        intersection = len(llm_tokens & gt_tokens)
                        union = len(llm_tokens | gt_tokens)
                        if union == 0: continue
                        score = intersection / union
                        if score > best_score:
                            best_score = score
                            best_fp = gt_fp
                    
                    if best_score > 0.5:
                        candidates = fingerprint_to_rec[best_fp]

            match_rec = None
            if candidates:
                if len(candidates) == 1:
                     match_rec = candidates[0]
                elif q_part:
                    # 2. Question Matching (Disambiguate duplicates)
                    # Try to find which candidate record has a matching question
                    for cand in candidates:
                        gt_input = cand['full_gt'].get('input', '')
                        # Extract question from GT input
                        gt_q = ""
                        if "<question>" in gt_input:
                            m = re.search(r'<question>(.*?)</question>', gt_input, re.DOTALL)
                            if m: gt_q = m.group(1).strip()
                        
                        # Fuzzy match questions (cleaning whitespace)
                        clean_llm_q = re.sub(r'\s+', '', q_part)
                        clean_gt_q = re.sub(r'\s+', '', gt_q)
                        
                        if clean_llm_q in clean_gt_q or clean_gt_q in clean_llm_q:
                            match_rec = cand
                            # Use GT's question to be safe, as user requested
                            q_part = gt_q 
                            break
                    
                    # Fallback: if no question match, pick first (risk of misalignment)
                    if not match_rec:
                        match_rec = candidates[0]
                else:
                    # No question in LLM input to disambiguate? Pick first
                    match_rec = candidates[0]

            title = "Unknown"
            existing_title = item.get('report_title')
            matched_title = match_rec['report_title'] if match_rec else None
            query_titles = set(query_set.keys()) if isinstance(query_set, dict) else set()
            if matched_title:
                # Prefer the title recovered from gt_master unless the existing one is already
                # a valid query-set title. This keeps QA scoring aligned to scorer taxonomy.
                if existing_title in query_titles:
                    title = existing_title
                else:
                    title = matched_title
            elif existing_title and existing_title not in ["Unknown", "", None]:
                title = existing_title
            
            # --- Per Task Logic ---
            if args.task_type == 'task2':
                raw_pred = clean_cot(raw_pred)

                # Pass title and query_set to help with key extraction from SFT-style questions
                key = extract_key_from_question(q_part, title, query_set if args.task_type == 'task2' else {}, q_to_key)
                
                # Use FP as key for aggregation
                task_key = fp if fp else str(idx) # Fallback to index if empty text
                task3_buffer[task_key]['raw_text'] = report_part
                task3_buffer[task_key]['title'] = title
                task3_buffer[task_key]['pred_pairs'].append({"key": key, "value": raw_pred.strip()})
                # For QA-style tasks, gold is the answer text for the current question.
                task3_buffer[task_key]['gt_pairs'].append({"key": key, "value": normalize_gold_value(item.get('gold', "")).strip()})
                
            else:
                # Task 1 & 3
                pred_pairs = []
                if args.task_type == 'task1':
                    raw_pred = clean_cot(raw_pred)
                    for k in raw_pred.split('\n'):
                        if k.strip(): pred_pairs.append({"key": k.strip(), "value": ""})
                else: # Task 3
                    try:
                        # Handle potential CoT artifacts even if grep didn't find them (defensive)
                        if "</think>" in raw_pred:
                            raw_pred = raw_pred.split("</think>")[-1].strip()
                        
                        # Use Robust Parse
                        d = robust_parse_json_list(raw_pred)
                        append_pairs_from_payload(d, pred_pairs)

                    except Exception as e:
                        print(f"DEBUG: Task 3 Parse Error: {e} | Content: {raw_pred[:100]}...")
                        pass
                
                if not pred_pairs and raw_pred and len(raw_pred) > 20:
                     print(f"DEBUG: Empty pred_pairs. Content: {raw_pred[:100]}... Parsed Type: {type(d) if 'd' in locals() else 'N/A'}")

                final_preds.append({
                    "report_title": title,
                    "text": report_part,
                    "pairs": pred_pairs  # scorer reads "pairs", not "pred_pairs"
                })
                
                # GT from 'gold' field
                gold_val = normalize_gold_value(item.get('gold', ""))
                gt_pairs = []
                # ... (keep parsing logic for fallback) ...
                if args.task_type == 'task1':
                    for k in gold_val.split('\n'):
                        if k.strip(): gt_pairs.append({"key": k.strip(), "value": ""})
                else:
                    try:
                        clean_gold = gold_val.replace("```json", "").replace("```", "").strip()
                        d = robust_parse_json_list(clean_gold)
                        append_pairs_from_payload(d, gt_pairs)
                    except Exception as e:
                        pass

                # Construct Spans
                # If we have a match_rec with _kv_spans (Created by our new file), use it!
                final_spans = {}
                used_master_spans = False
                
                if (
                    match_rec
                    and 'full_gt' in match_rec
                    and '_kv_spans' in match_rec['full_gt']
                    and match_rec['full_gt']['_kv_spans']
                ):
                     # Map keys from _kv_spans
                     # Structure: Key -> {value, start, end}
                     # We need: Key -> {text, start, end}
                     for k, v_info in match_rec['full_gt']['_kv_spans'].items():
                         # Ensure keys match standard
                         final_spans[k] = {
                             "text": v_info.get("value", ""),
                             "start": v_info.get("start", 0),
                             "end": v_info.get("end", 0)
                         }
                     used_master_spans = True
                
                # If no master spans or fallback needed:
                if not used_master_spans:
                    final_spans = {p['key']: {"text": p['value'], "start": 0, "end": 0} for p in gt_pairs}

                gt_pairs_out = [
                    {"key": k, "value": v.get("text", ""),
                     "key_span": [v["start"], v["end"]] if v.get("end", 0) > 0 else None}
                    for k, v in final_spans.items()
                ]
                final_gts.append({
                    "report_title": title,
                    "text": report_part,
                    "pairs": gt_pairs_out
                })

    # Finalize QA task aggregation
    if args.task_type == 'task2':
        for key_fp, data in task3_buffer.items():
            final_preds.append({
                "report_title": data['title'],
                "text": data['raw_text'],
                "pairs": data['pred_pairs']  # scorer reads "pairs", not "pred_pairs"
            })
            final_gts.append({
                "report_title": data['title'],
                "text": data['raw_text'],
                "pairs": [
                    {"key": p['key'], "value": p['value'], "key_span": None}
                    for p in data['gt_pairs']
                ]
            })

    # Save
    print(f"Saving {len(final_preds)} records to {args.output_pred}...")
    with open(args.output_pred, 'w', encoding='utf-8') as f:
        for p in final_preds: f.write(json.dumps(p, ensure_ascii=False) + '\n')
    
    with open(args.output_gt, 'w', encoding='utf-8') as f:
        for g in final_gts: f.write(json.dumps(g, ensure_ascii=False) + '\n')
    print("Done.")

if __name__ == "__main__":
    main()
