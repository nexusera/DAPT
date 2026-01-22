import os
import sys
import json
import logging
import collections
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Add repo root (dapt_eval_package parent) to path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pre_struct.kv_ner.modeling import BertCrfTokenClassifier
from pre_struct.kv_ner.config_io import load_config
# from pre_struct.kv_ner.predict import predict_batch_with_model, load_ner_model # Removed unused
from pre_struct.kv_ner.data_utils import build_bio_label_list
from pre_struct.kv_ner.schema_utils import load_schema # Imported Schema Utils
from transformers import AutoTokenizer
import torch
# Noise support
from pre_struct.kv_ner.noise_utils import NoiseFeatureProcessor, PERFECT_VALUES
# Removed EBQA imports

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_ground_truth(item):
    """
    Extract GT Keys and Pairs from item.
    Returns:
        gt_keys: set of key texts
        gt_pairs: set of (key_text, value_text)
        schema_keys: list of keys to query for QA (from spans)
    """
    gt_keys = set()
    gt_pairs = set()
    
    # Check if 'spans' exists (preferred for QA schema source)
    schema_keys = list(item.get('spans', {}).keys())
    
    # If not spans, try to infer from key_value_pairs or return empty
    if not schema_keys and 'key_value_pairs' in item:
        schema_keys = [p['key']['text'] for p in item['key_value_pairs']]

    # If key_value_pairs exists, use it (it might contain richer info like key positions if valid)
    if 'key_value_pairs' in item:
        for p in item['key_value_pairs']:
            k_text = p['key']['text']
            v_text = p.get('value_text', '') 
            # ... (omitted detail logic, but actually spans is cleaner for this dataset)
            pass

    # ALWAYS build GT from 'spans' because val_eval.jsonl relies on it
    # This ensures Task 1 and 2 have data
    if 'spans' in item:
        gt_qa_map = {}
        for k, v in item['spans'].items():
            if v and 'text' in v:
                v_text = v['text']
                gt_qa_map[k] = v_text
                
                # Also add to Keys and Pairs for Task 1 & 2
                gt_keys.add(k)
                if v_text:
                    gt_pairs.add((k, v_text))
            else:
                gt_qa_map[k] = ""
                # Empty value means we expect key but no value? 
                # Or just no key? In NER usually implies we don't extract it.
                # But if it's in spans keys, it's a schema key.
                pass
                
    return gt_keys, gt_pairs, schema_keys, gt_qa_map


# --- Helper Functions Adapted from predict.py ---
def _entity_records(label_ids, mask, offsets, id2label, text):
    # Minimal version of char_spans logic
    # We need pre_struct.kv_ner.metrics.char_spans or implement it
    from pre_struct.kv_ner.metrics import char_spans
    spans = char_spans(label_ids, mask, offsets, id2label)
    entities = []
    for ent_type, start, end in spans:
        snippet = text[start:end] if 0 <= start < end <= len(text) else ""
        entities.append({
            "type": ent_type,
            "start": start,
            "end": end,
            "text": snippet,
        })
    entities.sort(key=lambda x: (x["start"], x["end"]))
    return entities

def _assemble_pairs(entities, full_text):
    # Simplified version of predict.py's _assemble_pairs
    # Assuming simple nearest neighbor: Key -> Next Value
    # This is "Task 2" logic
    seq = [e for e in entities if e["type"] in {"KEY", "VALUE"}]
    seq.sort(key=lambda x: (x["start"], x["end"]))
    pairs = []
    
    # Logic: iterate, if KEY, look ahead for VALUE
    # Default pairing heuristic from predict.py is slightly disjointed (it iterates and tracks 'pending' key)
    
    pending_key = None
    structured_pairs = []
    
    for ent in seq:
        if ent["type"] == "KEY":
            pending_key = ent
        elif ent["type"] == "VALUE":
            if pending_key:
                # Pair it
                pairs.append({
                    "key": pending_key,
                    "value_text": ent["text"]
                })
                # Reset key? Strict Nearest Neighbor usually implies 1 Key -> 1 Value
                # But sometimes 1 Key -> Multiple Values. 
                # Let's simple reset for now or keep if we support multi-value. 
                # The repo supports multi-value (pending["values"].append). 
                # Let's mimic repo logic simplified:
                pass
            else:
                # Value without key
                pass
    
    return pairs

def predict_ner_text(text, model, tokenizer, max_len, device, id2label, o_id):
    # 1. Tokenize (Simple truncation for now, or use sliding window if enabled)
    # For comparison, let's assume valid data fits in max_seq_len or we just take first chunk
    # Proper way: sliding window. 
    # Use dataset.TokenClassificationDataset logic? Too complex to instantiate for 1 item.
    
    encoding = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)
    offset_mapping = encoding["offset_mapping"].cpu().tolist()[0]
    
    with torch.no_grad():
        decoded = model.predict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
    
    seq_list = list(decoded[0])
    # Pad or trim
    if len(seq_list) < input_ids.size(1):
        seq_list = seq_list + [o_id] * (input_ids.size(1) - len(seq_list))
    
    # Decode entities
    mask_list = [bool(v) for v in attention_mask.cpu().tolist()[0]]
    entities = _entity_records(seq_list, mask_list, offset_mapping, id2label, text)
    
    # Pair
    pairs_list = _assemble_pairs(entities, text)
    return entities, pairs_list

def _build_noise_ids(offset_mapping, noise_values, processor):
    """根据offset_mapping与noise_values生成每个token的noise_ids（长度7）。"""
    if noise_values is None:
        noise_values = []
    ids = []
    for s, e in offset_mapping:
        s = int(s); e = int(e)
        if e <= s:
            ids.append(processor.values_to_bin_ids(PERFECT_VALUES))
            continue
        vecs = []
        for ci in range(s, e):
            if 0 <= ci < len(noise_values):
                v = noise_values[ci]
                if isinstance(v, (list, tuple)) and len(v) == 7:
                    vecs.append(v)
        if vecs:
            avg = [sum(col) / len(col) for col in zip(*vecs)]
            ids.append(processor.values_to_bin_ids(avg))
        else:
            ids.append(processor.values_to_bin_ids(PERFECT_VALUES))
    return torch.tensor(ids, dtype=torch.long)


def predict_ner_sliding_window(text, model, tokenizer, max_len, device, id2label, o_id, stride=400, noise_values=None, noise_processor=None):
    if len(text) == 0:
        return [], []
        
    all_entities = []
    
    # Naive text chunking is dangerous because tokenization is not 1:1 with chars.
    # We must tokenize FIRST, then sliding window on TOKENS.
    # But current predict_ner_text takes TEXT.
    # Refactor: predict_ner_tokens taking input_ids.
    
    # Alternative: Overlapping Text Windows?
    # Text length != Token length. 
    # Let's try simple text chunking (heuristic: 1 char ~ 1 token for Chinese, but safety margin needed).
    # 512 tokens ~ 512 chars (safely). Let's use 450 chars window, 100 overlap.
    
    # Better approach given existing API:
    # Use tokenizer to encode entire text with overflow_to_sample logic?
    # Or just simple text slicing.
    
    encoding = tokenizer(
        text,
        max_length=max_len,
        truncation=True, 
        stride=128,
        padding="max_length", # Fix: Ensure all sliding windows are same length
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    
    # Iterate over batches
    num_batches = len(encoding["input_ids"])
    merged_entities = {} # (type, start, end) -> entity dict
    
    for i in range(num_batches):
        input_ids = encoding["input_ids"][i].unsqueeze(0).to(device)
        attention_mask = encoding["attention_mask"][i].unsqueeze(0).to(device)
        token_type_ids = encoding["token_type_ids"][i].unsqueeze(0).to(device)
        offset_mapping = encoding["offset_mapping"][i].cpu().tolist() # shape [seq_len, 2]
        
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        if noise_processor is not None:
            noise_ids = _build_noise_ids(offset_mapping, noise_values, noise_processor).unsqueeze(0).to(device)
            kwargs["noise_ids"] = noise_ids
            if getattr(model, "use_noise", False) and getattr(model, "noise_embeddings", None):
                with torch.no_grad():
                    for fi, emb in enumerate(model.noise_embeddings):
                        max_id = int(torch.max(noise_ids[:, :, fi]).item())
                        if max_id >= emb.num_embeddings:
                            raise ValueError(
                                f"noise_ids feature {fi} has max id {max_id} >= num_embeddings {emb.num_embeddings}"
                            )

        with torch.no_grad():
            decoded = model.predict(**kwargs)
        
        seq_list = list(decoded[0])
        # Decode
        mask_list = [bool(v) for v in attention_mask.cpu().tolist()[0]]
        
        # _entity_records expects offsets relative to 'text' passed to it.
        # But 'text' here is full text. 
        # offset_mapping returned by transformers with stride is RELATIVE TO FULL TEXT?
        # Yes, "offset_mapping" contains char indices into original string.
        
        # HOWEVER, we need to pass strict truncated text to _entity_records? 
        # No, _entity_records uses text[start:end]. If start/end are indices into full text, we need full text.
        # Let's verify _entity_records.
        # It takes `text` and uses `text[start:end]`. 
        # `offset_mapping` from tokenizer gives indices into `text`.
        # So we can pass full `text` and the `offset_mapping` for this batch window.
        
        batch_entities = _entity_records(seq_list, mask_list, offset_mapping, id2label, text)
        
        for ent in batch_entities:
            key = (ent['type'], ent['start'], ent['end'])
            # Simple merge: if exists, skip (or keep longer? usually identical).
            if key not in merged_entities:
                merged_entities[key] = ent
                
    # Sort by start
    final_entities = sorted(merged_entities.values(), key=lambda x: (x['start'], x['end']))
    
    # Pair
    final_pairs = _assemble_pairs(final_entities, text)
    
    return final_entities, final_pairs



    return final_entities, final_pairs

# --- Metric Helpers ---

def levenshtein_distance(s1, s2):
    """Simple DP implementation to avoid external dependency."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def compute_ned_similarity(s1, s2):
    """Task 1 Loose: 1 - Normalized Edit Distance"""
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (dist / max_len)

def compute_char_f1(pred_text, gt_text):
    """Task 2 & 3 Loose: Character-level F1"""
    pred_chars = list(pred_text)
    gt_chars = list(gt_text)
    if len(pred_chars) == 0 and len(gt_chars) == 0:
        return 1.0, 1.0, 1.0
        
    common = collections.Counter(pred_chars) & collections.Counter(gt_chars)
    num_same = sum(common.values())
    
    if num_same == 0: return 0.0, 0.0, 0.0
    
    p = num_same / len(pred_chars) if len(pred_chars) > 0 else 0.0
    r = num_same / len(gt_chars) if len(gt_chars) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def calculate_set_metrics_strict_counts(pred_set, gt_set):
    """Return raw TP, FP, FN for Micro-Avg"""
    tp = len(pred_set.intersection(gt_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    return tp, fp, fn

def calculate_set_metrics_strict(pred_set, gt_set):
    """Document-level Strict P/R/F1 (Macro helper if needed)"""
    tp, fp, fn = calculate_set_metrics_strict_counts(pred_set, gt_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def calculate_task1_loose_scores(pred_keys, gt_keys):
    """Return raw sum of Max N.E.D and counts for Micro-Avg"""
    if not pred_keys and not gt_keys: return 0, 0, 0, 0
    
    p_sim_sum = 0
    for pk in pred_keys:
        max_sim = 0
        for gk in gt_keys:
            max_sim = max(max_sim, compute_ned_similarity(pk, gk))
        p_sim_sum += max_sim
        
    r_sim_sum = 0
    for gk in gt_keys:
        max_sim = 0
        for pk in pred_keys:
            max_sim = max(max_sim, compute_ned_similarity(gk, pk))
        r_sim_sum += max_sim
        
    return p_sim_sum, len(pred_keys), r_sim_sum, len(gt_keys)

def calculate_task1_loose(pred_keys, gt_keys):
    ps, pc, rs, rc = calculate_task1_loose_scores(pred_keys, gt_keys)
    p = ps / pc if pc > 0 else (1.0 if rc == 0 else 0.0)
    r = rs / rc if rc > 0 else (1.0 if pc == 0 else 0.0)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def calculate_task2_loose_scores(pred_pairs, gt_pairs):
    """Key Strict + Value Fuzzy counts"""
    pred_dict = {k: v for k, v in pred_pairs}
    gt_dict = {k: v for k, v in gt_pairs}
    
    p_score_sum = 0
    for pk, pv in pred_pairs:
        if pk in gt_dict:
            _, _, val_f1 = compute_char_f1(pv, gt_dict[pk])
            p_score_sum += val_f1
            
    r_score_sum = 0
    for gk, gv in gt_pairs:
        if gk in pred_dict:
            _, _, val_f1 = compute_char_f1(pred_dict[gk], gv)
            r_score_sum += val_f1
            
    return p_score_sum, len(pred_pairs), r_score_sum, len(gt_pairs)

def calculate_task2_loose_loose_scores(pred_pairs, gt_pairs):
    """Fuzzy Key + Fuzzy Value counts"""
    p_score_sum = 0
    for pk, pv in pred_pairs:
        max_pair_score = 0
        for gk, gv in gt_pairs:
            k_sim = compute_ned_similarity(pk, gk)
            _, _, v_f1 = compute_char_f1(pv, gv)
            max_pair_score = max(max_pair_score, k_sim * v_f1)
        p_score_sum += max_pair_score
        
    r_score_sum = 0
    for gk, gv in gt_pairs:
        max_pair_score = 0
        for pk, pv in pred_pairs:
            k_sim = compute_ned_similarity(gk, pk)
            _, _, v_f1 = compute_char_f1(pv, gv)
            max_pair_score = max(max_pair_score, k_sim * v_f1)
        r_score_sum += max_pair_score
        
    return p_score_sum, len(pred_pairs), r_score_sum, len(gt_pairs)

def calculate_task2_loose(pred_pairs, gt_pairs):
    ps, pc, rs, rc = calculate_task2_loose_scores(pred_pairs, gt_pairs)
    p = ps / pc if pc > 0 else 0.0
    r = rs / rc if rc > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_config', default='pre_struct/kv_ner/kv_ner_config_comparison.json')
    parser.add_argument('--keys_file', default='data/kv_ner_comparison_prepared/keys_merged_1027_cleaned.json') 
    parser.add_argument('--test_data', default='data/kv_ner_prepared_comparison/val_eval_titled.jsonl')
    parser.add_argument('--output_summary', default='runs/comparison_results.json')
    parser.add_argument('--noise_bins', default=None, help='Path to noise_bins.json; if provided and data has noise_values, fuse noise embeddings during inference')
    args = parser.parse_args()

    # 1. Load Data
    logger.info(f"Loading data from {args.test_data}")
    dataset = load_data(args.test_data)
    logger.info(f"Loaded {len(dataset)} items.")
    
    # 2. Load Keys Definition
    logger.info(f"Loading Keys Definition from {args.keys_file}")
    keys_dict = load_schema(args.keys_file)

    # 3. Setup NER Model
    logger.info("Subjecting NER Model...")
    ner_cfg = load_config(args.ner_config)
    
    if "model_dir" in ner_cfg:
        ner_model_dir = ner_cfg["model_dir"]
    elif "train" in ner_cfg and "output_dir" in ner_cfg["train"]:
        ner_model_dir = os.path.join(ner_cfg["train"]["output_dir"], "best")
    else:
        ner_model_dir = "runs/kv_ner_comparison/best"
        
    logger.info(f"Loading model from: {ner_model_dir}")
    
    from pre_struct.kv_ner import config_io as ner_config_io
    label_map = ner_config_io.label_map_from(ner_cfg)
    label_list = build_bio_label_list(label_map)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    o_id = next((idx for idx, name in id2label.items() if name == "O"), 0)
    
    tokenizer_name = ner_config_io.tokenizer_name_from(ner_cfg)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner_model = BertCrfTokenClassifier.from_pretrained(ner_model_dir).to(device)
    ner_model.eval()

    noise_processor = None
    if args.noise_bins:
        try:
            noise_processor = NoiseFeatureProcessor.load(args.noise_bins)
            logger.info(f"Loaded noise processor from {args.noise_bins}")
        except Exception as e:
            logger.warning(f"Failed to load noise bins ({args.noise_bins}): {e}; continue without noise")

    max_len = ner_config_io.max_seq_length(ner_cfg)

    # Metrics Accumulators (Micro-Avg Counts)
    t1_strict_counts = {'tp': 0, 'fp': 0, 'fn': 0}
    t1_loose_counts = {'ps': 0.0, 'pc': 0, 'rs': 0.0, 'rc': 0}
    
    t2_strict_counts = {'tp': 0, 'fp': 0, 'fn': 0}
    t2_loose_counts = {'ps': 0.0, 'pc': 0, 'rs': 0.0, 'rc': 0} # Strict-Loose
    t2_fuzzy_counts = {'ps': 0.0, 'pc': 0, 'rs': 0.0, 'rc': 0} # Loose-Loose
    
    results_list = [] # ADDED: For unified output record saving

    # Task 3 (QA)
    t3_strict_counts_all = {'acc_sum': 0.0, 'count': 0}
    t3_strict_counts_pos = {'acc_sum': 0.0, 'count': 0}
    t3_loose_counts_all = {'ps': 0.0, 'pc': 0, 'rs': 0.0, 'rc': 0}
    t3_loose_counts_pos = {'ps': 0.0, 'pc': 0, 'rs': 0.0, 'rc': 0}

    # Helper: F1 Calculation
    def _compute_qa_f1(pred, gt):
        # Character-level F1
        pred_chars = list(pred)
        gt_chars = list(gt)
        common = 0
        import collections
        gt_counter = collections.Counter(gt_chars)
        pred_counter = collections.Counter(pred_chars)
        
        for k in gt_counter:
            common += min(gt_counter[k], pred_counter[k])
            
        if len(pred_chars) == 0: precision = 0.0
        else: precision = common / len(pred_chars)
        
        if len(gt_chars) == 0: recall = 0.0
        else: recall = common / len(gt_chars)
        
        if precision + recall == 0: return 0.0, 0.0, 0.0 # p, r, f1
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    # Iterate
    logger.info("Running comparison (Unified Task 1/2/3) with Strict/Loose Metrics...")
    for idx, item in enumerate(tqdm(dataset)):
        report_text = item['report']
        # Fix: handle empty string title
        title = item.get('report_title')
        if not title: title = '通用病历'
        
        gt_keys, gt_pairs, _, gt_qa_map = get_ground_truth(item) 
        
        # --- NER Prediction ---
        pred_entities, pred_pairs_list = predict_ner_sliding_window(
            report_text,
            ner_model,
            tokenizer,
            max_len,
            device,
            id2label,
            o_id,
            noise_values=item.get('noise_values'),
            noise_processor=noise_processor,
        )
        
        # --- Task 1 Metrics ---
        pred_keys = set([e['text'] for e in pred_entities if e['type'] == 'KEY'])
        
        # Strict
        tp, fp, fn = calculate_set_metrics_strict_counts(pred_keys, gt_keys)
        t1_strict_counts['tp'] += tp
        t1_strict_counts['fp'] += fp
        t1_strict_counts['fn'] += fn
        
        # Loose
        ps, pc, rs, rc = calculate_task1_loose_scores(pred_keys, gt_keys)
        t1_loose_counts['ps'] += ps
        t1_loose_counts['pc'] += pc
        t1_loose_counts['rs'] += rs
        t1_loose_counts['rc'] += rc
        
        # Task 2 Metrics
        pred_pairs_tuples = []
        for p in pred_pairs_list:
            k_text = p['key']['text']
            v_text = p['value_text']
            if k_text and v_text:
                pred_pairs_tuples.append((k_text, v_text))
                
        # 1. Strict-Strict
        tp, fp, fn = calculate_set_metrics_strict_counts(set(pred_pairs_tuples), gt_pairs)
        t2_strict_counts['tp'] += tp
        t2_strict_counts['fp'] += fp
        t2_strict_counts['fn'] += fn
        
        # 2. Strict-Loose
        ps, pc, rs, rc = calculate_task2_loose_scores(pred_pairs_tuples, gt_pairs)
        t2_loose_counts['ps'] += ps
        t2_loose_counts['pc'] += pc
        t2_loose_counts['rs'] += rs
        t2_loose_counts['rc'] += rc

        # 3. Loose-Loose
        ps, pc, rs, rc = calculate_task2_loose_loose_scores(pred_pairs_tuples, gt_pairs)
        t2_fuzzy_counts['ps'] += ps
        t2_fuzzy_counts['pc'] += pc
        t2_fuzzy_counts['rs'] += rs
        t2_fuzzy_counts['rc'] += rc
        
        # --- Task 3 (QA) Metrics ---
        # Iterate over Schema (gt_qa_map)
        schema_keys = keys_dict.get(title, {})
        # If title not found, fallback
        if not schema_keys and '通用病历' in keys_dict:
            schema_keys = keys_dict['通用病历']
            
        # Pred Dictionary for fast lookup
        pred_dict = {k: v for k, v in pred_pairs_tuples}
            
        if idx < 3:
             logger.info(f"Report: {title}, Schema Keys: {len(schema_keys)}")
             
        for key_name, info in schema_keys.items():
            # Ground Truth Value
            gt_val = gt_qa_map.get(key_name, "")
            
            # Predicted Value (Conditioned on Key)
            pred_val = pred_dict.get(key_name, "")
            
            # --- Metrics ---
            
            # Strict (EM)
            # Raw comparison (no strip)
            is_strict_correct = 1.0 if pred_val == gt_val else 0.0
            
            # Loose (Char-F1)
            p_char, r_char, f1_char = compute_char_f1(pred_val, gt_val)
            
            # Accumulate All
            t3_strict_counts_all['count'] += 1
            t3_strict_counts_all['acc_sum'] += is_strict_correct
            
            t3_loose_counts_all['pc'] += 1
            t3_loose_counts_all['rc'] += 1
            t3_loose_counts_all['ps'] += p_char
            t3_loose_counts_all['rs'] += r_char
            
            # Accumulate Positive Only (GT is not empty)
            if gt_val != "":
                t3_strict_counts_pos['count'] += 1
                t3_strict_counts_pos['acc_sum'] += is_strict_correct
                
                t3_loose_counts_pos['pc'] += 1
                t3_loose_counts_pos['rc'] += 1
                t3_loose_counts_pos['ps'] += p_char
                t3_loose_counts_pos['rs'] += r_char

        # --- Record for saving ---
        results_list.append({
            "title": title,
            "text": report_text,
            "pred_pairs": pred_pairs_tuples,
            "gt_keys": gt_keys,
            "gt_pairs": gt_pairs,
            "gt_qa_map": gt_qa_map
        })
                
    # --- Final Aggregation (Micro-Avg) ---
    # --- Standardize Output for Unified Scorer ---
    output_records = []
    
    for item in results_list:
        # Standard format: {"report_title": "...", "text": "...", "pred_pairs": [{"key": "...", "value": "..."}, ...]}
        record = {
            "report_title": item['title'],
            "text": item['text'],
            "pred_pairs": []
        }
        
        # Convert tuple list back to dictionary list for JSON serialization
        for k, v in item['pred_pairs']:
            record['pred_pairs'].append({"key": k, "value": v})
            
        output_records.append(record)
    
    output_file = args.output_summary.replace('.json', '_preds.jsonl') if args.output_summary else 'bert_preds.jsonl'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    logger.info(f"Standardized predictions saved to {output_file} (Ready for Unified Scorer)")
    
    # Still calculate metrics for backward compatibility / quick check? 
    # Yes, using the new core/metrics.py if possible, or keep existing logic temporarily but warn
    # User said "BERT code don't throw away", "Modify to save jsonl"
    # User also said "Action 3: Build Unified Scorer"
    # So ideally we should STOP calculating here and rely on scorer.py?
    # Or import scorer here?
    # Let's import the new metrics to calculate ON THE FLY for immediate feedback, 
    # but ALSO save the file.
    
    from core.metrics import (
        calculate_task1_stats, 
        calculate_task2_stats, 
        calculate_task3_stats,
        calc_micro_f1
    )
    
    # Re-implement aggregation using NEW core metrics
    t1_strict_stats = {'tp':0, 'fp':0, 'fn':0}
    t1_loose_stats = {'ps':0, 'pc':0, 'rs':0, 'rc':0}
    
    t2_ss_stats = {'tp':0, 'fp':0, 'fn':0}
    t2_sl_stats = {'ps':0, 'pc':0, 'rs':0, 'rc':0}
    t2_ll_stats = {'ps':0, 'pc':0, 'rs':0, 'rc':0}
    
    t3_stats_all = {"all_match_sum": 0, "all_count": 0, "all_f1_sum": 0}
    t3_stats_pos = {"pos_match_sum": 0, "pos_count": 0, "pos_f1_sum": 0}

    # Re-iterate to calculate stats
    logger.info("Computing metrics using CORE/METRICS.PY ...")
    
    for item in results_list:
        pred_pairs_tuples = item['pred_pairs']
        gt_keys = item['gt_keys']
        gt_pairs = item['gt_pairs']
        gt_qa_map = item['gt_qa_map']
        title = item['title']
        
        pred_keys = [p[0] for p in pred_pairs_tuples]
        
        # T1
        s1, l1 = calculate_task1_stats(pred_keys, gt_keys)
        for k in t1_strict_stats: t1_strict_stats[k] += s1[k]
        for k in t1_loose_stats: t1_loose_stats[k] += l1[k]
        
        # T2
        s2, sl2, ll2 = calculate_task2_stats(pred_pairs_tuples, gt_pairs)
        for k in t2_ss_stats: t2_ss_stats[k] += s2[k]
        for k in t2_sl_stats: t2_sl_stats[k] += sl2[k]
        for k in t2_ll_stats: t2_ll_stats[k] += ll2[k]
        
        # T3
        pred_dict = {k:v for k,v in pred_pairs_tuples}
        s3 = calculate_task3_stats(pred_dict, title, keys_dict, gt_qa_map)
        
        t3_stats_all['all_match_sum'] += s3['all_match_sum']
        t3_stats_all['all_count'] += s3['all_count']
        t3_stats_all['all_f1_sum'] += s3['all_f1_sum']
        
        t3_stats_pos['pos_match_sum'] += s3['pos_match_sum']
        t3_stats_pos['pos_count'] += s3['pos_count']
        t3_stats_pos['pos_f1_sum'] += s3['pos_f1_sum']

    # Final Results
    final_res = {}
    final_res["Task 1 (Strict)"] = calc_micro_f1(t1_strict_stats)
    final_res["Task 1 (Loose)"] = calc_micro_f1(t1_loose_stats)
    
    final_res["Task 2 (Strict-Strict)"] = calc_micro_f1(t2_ss_stats)
    final_res["Task 2 (Strict-Loose)"] = calc_micro_f1(t2_sl_stats)
    final_res["Task 2 (Loose-Loose)"] = calc_micro_f1(t2_ll_stats)
    
    # Task 3 Formatting
    t3_all_cnt = t3_stats_all['all_count']
    t3_pos_cnt = t3_stats_pos['pos_count']
    
    final_res["Task 3 (All)"] = {
        "strict_em": t3_stats_all['all_match_sum'] / t3_all_cnt if t3_all_cnt else 0,
        "loose_f1": t3_stats_all['all_f1_sum'] / t3_all_cnt if t3_all_cnt else 0,
        "count": t3_all_cnt
    }
    final_res["Task 3 (Pos Only)"] = {
        "strict_em": t3_stats_pos['pos_match_sum'] / t3_pos_cnt if t3_pos_cnt else 0,
        "loose_f1": t3_stats_pos['pos_f1_sum'] / t3_pos_cnt if t3_pos_cnt else 0,
        "count": t3_pos_cnt
    }
    
    print(json.dumps(final_res, indent=2, ensure_ascii=False))
    
if __name__ == "__main__":
    main()
