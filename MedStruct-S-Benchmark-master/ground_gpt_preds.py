import argparse
import json
import os
import sys
import re

# Add project root to path
sys.path.insert(0, os.getcwd())

from core.metrics import normalize_text

def find_best_span(text, subtext, start_hint=0):
    """
    Find substring in text.
    Returns (start, end) or None.
    Simple exact match search.
    """
    if not subtext:
        return None
        
    # strip whitespace for search? No, exact search first.
    # text is raw text.
    # subtext might be hallucinatory or slightly different?
    # Strategy 1: Exact search
    idx = text.find(subtext, start_hint)
    if idx != -1:
        return (idx, idx + len(subtext))
        
    # Strategy 2: If fail, try finding stripped version?
    # This might be dangerous if text is complex. 
    # For now, strict exact match for grounding. 
    # If GPT generated text not in source, it's abstractive/hallucinated -> No Span -> No Robust Match.
    # This is correct behavior.
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_pred_file", required=True, help="Input GPT JSONL")
    parser.add_argument("--output_file", required=True, help="Output Standardized JSONL")
    args = parser.parse_args()

    data = []
    with open(args.gpt_pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                
    output_records = []
    
    for item in data:
        text = item.get('text', "")
        pred_pairs = item.get('pred_pairs', [])
        
        grounded_pairs = []
        
        # Cursor to help with sequential search (heuristic: pairs appear in text order)
        # But extracted keys might be in any order.
        # Safe bet: Search from 0, but maybe keep track of used spans to avoid overlap?
        # Let's keep it simple: Search from 0 for every key.
        
        for p in pred_pairs:
            k = p.get('key', "")
            v = p.get('value', "")
            
            # Ground Key
            k_span = find_best_span(text, k)
            
            # Ground Value
            # Heuristic: Value usually follows Key?
            # Or just search 0?
            # Let's search from 0.
            v_span = find_best_span(text, v)
            
            # Refinement: If Key found, search Value AFTER Key?
            if k_span and v:
                v_candidate = find_best_span(text, v, k_span[1])
                if v_candidate:
                    v_span = v_candidate
            
            new_p = {
                "key": k,
                "value": v,
                "key_span": k_span,
                "val_span": v_span
            }
            grounded_pairs.append(new_p)
            
        record = {
            "report_title": item.get("report_title", "Unknown"),
            "text": text,
            "pred_pairs": grounded_pairs
        }
        output_records.append(record)
        
    # Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"Grounded predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()
