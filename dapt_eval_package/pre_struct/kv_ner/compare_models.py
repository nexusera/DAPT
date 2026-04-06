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
try:
    from noise_fusion import aggregate_token_noise_values, uses_bucket_noise, uses_continuous_noise, needs_bucket_ids
except Exception:  # pragma: no cover
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from noise_fusion import aggregate_token_noise_values, uses_bucket_noise, uses_continuous_noise, needs_bucket_ids
# Removed EBQA imports

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(filepath):
    # Try reading as JSON list first (standard JSON array)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if isinstance(content, list):
                return content
    except json.JSONDecodeError:
        pass # Not a standard JSON file, try JSONL

    # Fallback to JSONL (Newline Delimited JSON)
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def _extract_ocr_text(ocr_raw):
    """Best-effort OCR text extraction from ocr_raw payload."""
    if not ocr_raw:
        return ""
    # If already a string, use directly
    if isinstance(ocr_raw, str):
        return ocr_raw
    # Common Baidu OCR format: {'words_result': [{'words': '...'}, ...]}
    if isinstance(ocr_raw, dict):
        words_result = ocr_raw.get("words_result")
        if isinstance(words_result, list):
            words = []
            for w in words_result:
                if isinstance(w, dict) and "words" in w:
                    words.append(str(w["words"]))
            if words:
                return "".join(words)
    # Fallback to string representation
    return str(ocr_raw)


def _expand_word_noise_to_chars(ocr_raw, noise_values_per_word):
    """Expand per-word 7-d noise to per-char list using ocr_raw.words_result."""
    if not (isinstance(ocr_raw, dict) and isinstance(noise_values_per_word, list)):
        return None
    words_result = ocr_raw.get("words_result")
    if not isinstance(words_result, list):
        return None
    char_noise = []
    for wr, nv in zip(words_result, noise_values_per_word):
        if not (isinstance(wr, dict) and isinstance(nv, (list, tuple)) and len(nv) == 7):
            continue
        w = wr.get("words", "")
        if not isinstance(w, str):
            continue
        repeat = max(1, len(w))
        char_noise.extend([list(nv)] * repeat)
    return char_noise if char_noise else None

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
    gt_qa_map = {}
    
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
                
    # Support for MedStruct-S Real (transferred_annotations)
    if not gt_keys and 'transferred_annotations' in item:
        transferred = item['transferred_annotations']
        if isinstance(transferred, list):
            for anno in transferred:
                if not isinstance(anno, dict): continue
                
                labels = anno.get('labels', [])
                text = anno.get('text', '')
                
                if labels and text:
                    key = labels[0]
                    gt_keys.add(key)
                    gt_pairs.add((key, text))
                    gt_qa_map[key] = text
                    if key not in schema_keys:
                        schema_keys.append(key)

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
    # Reuse canonical pairing logic from predict.py to keep evaluation consistent.
    try:
        from pre_struct.kv_ner.predict import _assemble_pairs as _assemble_pairs_predict
        packed = _assemble_pairs_predict(entities, full_text=full_text)
        if isinstance(packed, dict) and isinstance(packed.get("pairs"), list):
            return packed["pairs"]
    except Exception:
        pass

    # Conservative fallback to avoid breaking old environments.
    seq = [e for e in entities if e["type"] in {"KEY", "VALUE"}]
    seq.sort(key=lambda x: (x["start"], x["end"]))
    pairs = []
    pending_key = None
    for ent in seq:
        if ent["type"] == "KEY":
            pending_key = ent
        elif ent["type"] == "VALUE" and pending_key:
            pairs.append({"key": pending_key, "value_text": ent["text"]})
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

def _build_noise_features(offset_mapping, noise_values, processor, noise_mode):
    token_noise_values = aggregate_token_noise_values(
        offset_mapping,
        noise_values,
        chunk_char_start=0,
        perfect_values=PERFECT_VALUES,
    )
    if needs_bucket_ids(noise_mode):
        ids = [processor.values_to_bin_ids(row) for row in token_noise_values]
        return {"noise_ids": torch.tensor(ids, dtype=torch.long)}
    if uses_continuous_noise(noise_mode):
        return {"noise_values": torch.tensor(token_noise_values, dtype=torch.float32)}
    return {}


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
        noise_mode = str(getattr(model, "noise_mode", "bucket") or "bucket").lower()
        if noise_processor is not None or uses_continuous_noise(noise_mode):
            noise_kwargs = _build_noise_features(offset_mapping, noise_values, noise_processor, noise_mode)
            for k, v in noise_kwargs.items():
                kwargs[k] = v.unsqueeze(0).to(device)
            if "noise_ids" in noise_kwargs and getattr(model, "use_noise", False) and getattr(model, "noise_embeddings", None):
                noise_ids = kwargs["noise_ids"]
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
    parser.add_argument('--test_data', default='data/kv_ner_prepared_comparison/val_eval_titled.jsonl', help='Path to test data file')
    parser.add_argument('--output_summary', default='runs/comparison_results.json')
    parser.add_argument('--noise_bins', default=None, help='Path to noise_bins.json; if provided and data has noise_values, fuse noise embeddings during inference')
    args = parser.parse_args()

    # 1. Load Data
    logger.info(f"Loading data from {args.test_data}")
    dataset = load_data(args.test_data)
    logger.info(f"Loaded {len(dataset)} items.")
    
    # 2. Load Keys Definition
    logger.info(f"Loading Keys Definition from {args.keys_file}")
    try:
        keys_dict = load_schema(args.keys_file)
    except Exception as e:
        logger.error(f"Failed to load schema from {args.keys_file}: {e}; continue without QA schema")
        keys_dict = {}

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

    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "Expected a fast tokenizer (is_fast=False). "
            "compare_models relies on offset_mapping for alignment."
        )
    _probe = "肿瘤标志物"
    _pieces = tokenizer.tokenize(_probe)
    if len(_pieces) == 1 and _pieces[0] == tokenizer.unk_token:
        raise RuntimeError(
            "Fast tokenizer appears misconfigured (probe tokenizes to a single [UNK]). "
            "Regenerate tokenizer.json: python DAPT/repair_fast_tokenizer.py --tokenizer_dir <TOKENIZER_DIR>"
        )
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
        # 兼容不同字段名，优先 report/text/ocr_text，再尝试 data 字段与 ocr_raw
        ocr_raw = item.get('ocr_raw')
        report_text = (
            item.get('report')
            or item.get('text')
            or item.get('ocr_text')
            or item.get('data', {}).get('ocr_text')
            or _extract_ocr_text(ocr_raw)
        )
        if not report_text:
            logger.warning(f"Skip sample without text/report: {item.get('id', '')}")
            continue

        # 兼容标题字段
        title = (
            item.get('report_title')
            or item.get('title')
            or item.get('category')
            or item.get('data', {}).get('category')
            or '通用病历'
        )

        # 优先使用 per-word 噪声，退化到全局或逐字符
        char_noise = None
        per_word_noise = item.get('noise_values_per_word') or item.get('data', {}).get('noise_values_per_word')
        if per_word_noise:
            char_noise = _expand_word_noise_to_chars(ocr_raw, per_word_noise)
        
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
            noise_values=char_noise or item.get('noise_values') or item.get('data', {}).get('noise_values'),
            noise_processor=noise_processor,
        )
        
        # --- Task 1 Metrics ---
        # Modified to include all predicted entities as potential keys if the GT implies it?
        # Or purely stick to 'KEY' type?
        # If the model predicts specific labels (e.g. HOSPITAL), we should count them as keys found IF GT has them.
        # But wait, Task 1 is Key Extraction. In "Hospital: ABC", "Hospital" is Key.
        # If model predicts "ABC" as HOSPITAL, it found the Value. Where is the Key?
        # If the task is purely V extraction for implicit keys, then 'key' is the label name.
        # Let's assume for now we only evaluate explicit KEY/VALUE pairs or just count valid entities.
        # Let's broaden the filter: include any entity that is NOT 'VALUE' (or 'O') as a key candidate?
        # No, 'VALUE' is a value. 'KEY' is a key. 'HOSPITAL' is a value (implicit key).
        # Evaluating implicit keys is harder.
        # Let's keep it simple: collect ALL entity texts as "Keys" if they are not explicitly VALUE?
        # No, that's dangerous.
        
        # Let's stick to the original logic but adding a fallback:
        # If the entity type is in our schema keys, treat it as a key?
        # No, let's just make it robust to crash first.
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
            "id": item.get('id') or item.get('record_id') or f"sample_{idx}",
            "title": title,
            "text": report_text, # Used for inference input
            "ocr_text": item.get('ocr_text') or report_text, # Preserves original OCR text
            "pred_pairs": pred_pairs_tuples,
            "full_pred_pairs": pred_pairs_list, # Contains span info
            "gt_keys": gt_keys,
            "gt_pairs": gt_pairs,
            "item_original": item, # Keep original item for GT span extraction if needed
            "gt_qa_map": gt_qa_map
        })
                
    # --- Final Aggregation (Micro-Avg) ---
    # --- Standardize Output for Unified Scorer ---
    output_records = []
    gt_records = []
    
    for item in results_list:
        # 1. Prediction Record
        # Standard format: {"id": "...", "report_title": "...", "ocr_text": "...", "pairs": [{"key": "...", "value": "...", "key_span": [s, e]}, ...]}
        pred_record = {
            "id": str(item['id']),
            "report_title": item['title'],
            "ocr_text": item['ocr_text'],
            "pairs": []
        }
        
        for p in item['full_pred_pairs']:
            # Key span from entity
            k_span = [p['key']['start'], p['key']['end']]
            pred_record['pairs'].append({
                "key": p['key']['text'], 
                "value": p['value_text'],
                "key_span": k_span
            })
        output_records.append(pred_record)

        # 2. Ground Truth Record
        # Construct pairs from GT data (requires spans if available)
        gt_record = {
            "id": str(item['id']),
            "report_title": item['title'],
            "ocr_text": item['ocr_text'],
            "pairs": []
        }
        
        # Extract pairs with spans from original item if possible
        # Logic for MedStruct-S Real (transferred_annotations)
        original = item['item_original']
        
        # Case A: transferred_annotations (Real format)
        if 'transferred_annotations' in original:
            transferred = original['transferred_annotations']
            if isinstance(transferred, list):
                for anno in transferred:
                    if isinstance(anno, dict):
                        labels = anno.get('labels', [])
                        text = anno.get('text', '')
                        # Try to get span or infer it
                        # For GT file generation, we need 'key_span'. 
                        # In the new data, we might not have 'key' span separate from 'value' span?
                        # Wait, the annotations ARE the values usually. 
                        # {"labels": ["KeyName"], "text": "ValueText", "box": ...}
                        # The "text" IS the value. The "key" is the label class.
                        # The span of the KEY itself is usually NOT in the annotation for KV extraction tasks 
                        # unless it's a "Key-Value Pair" annotation.
                        # But MedStruct Benchmark usually expects key_span to be the span of the key string in text.
                        # If the key is implicit (e.g. "Name: Alice"), "Name" is key.
                        # If "Alice" is labeled as "Name", we know where "Alice" is, but where is "Name"?
                        # If key_span is null, scorer might skipping key-span check? 
                        # "key_span在ocr_text中的字符位置，无则为null" -> If null, allows validation without span.
                        
                        # So we can set key_span: null
                        if labels and text:
                            gt_record['pairs'].append({
                                "key": labels[0],
                                "value": text,
                                "key_span": None  # Set to None/null as we don't know where the key word is
                            })
                            
        # Case B: Standard pairs/spans format
        elif 'spans' in original:
             # Spans usually map key -> {text, start, end} of the VALUE
             # Pairs format requires key span. 
             # If not available, use null.
             for k, v in original['spans'].items():
                 gt_record['pairs'].append({
                     "key": k,
                     "value": v.get('text', ''),
                     "key_span": None # Value span is v['start'],v['end']. Key span unknown.
                 })
                 
        gt_records.append(gt_record)
    
    output_file = args.output_summary.replace('.json', '_preds.jsonl') if args.output_summary else 'bert_preds.jsonl'
    gt_file = args.output_summary.replace('.json', '_gt.jsonl') if args.output_summary else 'bert_gt.jsonl'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    with open(gt_file, 'w', encoding='utf-8') as f:
        for rec in gt_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    logger.info(f"Standardized predictions saved to {output_file}")
    logger.info(f"Standardized GT saved to {gt_file}")
    logger.info(f"Ready for Unified Scorer: python dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py --pred_file {output_file} --gt_file {gt_file}")
            
    logger.info(f"Standardized predictions saved to {output_file} (Ready for Unified Scorer)")
    
    # Optional inline metrics; if core.metrics is unavailable, still write summary for pipeline compatibility.
    final_res = {}
    try:
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

        logger.info("Computing metrics using CORE/METRICS.PY ...")

        for item in results_list:
            pred_pairs_tuples = item['pred_pairs']
            gt_keys = item['gt_keys']
            gt_pairs = item['gt_pairs']
            gt_qa_map = item['gt_qa_map']
            title = item['title']

            pred_keys = [p[0] for p in pred_pairs_tuples]

            s1, l1 = calculate_task1_stats(pred_keys, gt_keys)
            for k in t1_strict_stats:
                t1_strict_stats[k] += s1[k]
            for k in t1_loose_stats:
                t1_loose_stats[k] += l1[k]

            s2, sl2, ll2 = calculate_task2_stats(pred_pairs_tuples, gt_pairs)
            for k in t2_ss_stats:
                t2_ss_stats[k] += s2[k]
            for k in t2_sl_stats:
                t2_sl_stats[k] += sl2[k]
            for k in t2_ll_stats:
                t2_ll_stats[k] += ll2[k]

            pred_dict = {k: v for k, v in pred_pairs_tuples}
            s3 = calculate_task3_stats(pred_dict, title, keys_dict, gt_qa_map)

            t3_stats_all['all_match_sum'] += s3['all_match_sum']
            t3_stats_all['all_count'] += s3['all_count']
            t3_stats_all['all_f1_sum'] += s3['all_f1_sum']

            t3_stats_pos['pos_match_sum'] += s3['pos_match_sum']
            t3_stats_pos['pos_count'] += s3['pos_count']
            t3_stats_pos['pos_f1_sum'] += s3['pos_f1_sum']

        final_res["Task 1 (Strict)"] = calc_micro_f1(t1_strict_stats)
        final_res["Task 1 (Loose)"] = calc_micro_f1(t1_loose_stats)
        final_res["Task 2 (Strict-Strict)"] = calc_micro_f1(t2_ss_stats)
        final_res["Task 2 (Strict-Loose)"] = calc_micro_f1(t2_sl_stats)
        final_res["Task 2 (Loose-Loose)"] = calc_micro_f1(t2_ll_stats)

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
    except ImportError:
        logger.warning("core.metrics not found; skip inline metrics. Use scorer.py on preds/gt files.")
        final_res = {
            "status": "metrics_skipped",
            "reason": "core.metrics not found",
            "pred_file": output_file,
            "gt_file": gt_file
        }

    if args.output_summary:
        out_dir = os.path.dirname(args.output_summary)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_summary, 'w', encoding='utf-8') as f:
            json.dump(final_res, f, ensure_ascii=False, indent=2)
        logger.info(f"Summary saved to {args.output_summary}")

    print(json.dumps(final_res, indent=2, ensure_ascii=False))
    
if __name__ == "__main__":
    main()
