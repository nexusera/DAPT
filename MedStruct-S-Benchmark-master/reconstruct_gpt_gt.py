import json
import os
import argparse

def reconstruct_aligned_gt(raw_file, aligned_gt_file):
    print(f"Reconstructing {aligned_gt_file} from {raw_file}...")
    
    # Load raw records (contain the 'gold' field)
    raw_records = []
    with open(raw_file, 'r', encoding='utf-8') as f:
        for line in f:
            raw_records.append(json.loads(line))
            
    # Load aligned records
    aligned_records = []
    with open(aligned_gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            aligned_records.append(json.loads(line))
            
    if len(raw_records) != len(aligned_records):
        print(f"Warning: Record count mismatch! Raw: {len(raw_records)}, Aligned: {len(aligned_records)}")
        # We assume they match 1:1 by index based on previous observations
        
    updated_records = []
    for raw, aligned in zip(raw_records, aligned_records):
        gold_str = raw.get('gold', '{}')
        
        # 'gold' is sometimes a JSON string directly or a fragment.
        # It usually looks like '"Key": "Value", ... }' or '{"Key": "Value", ...}'
        # If it doesn't start with '{', we wrap it.
        gold_str_clean = gold_str.strip()
        if not gold_str_clean.startswith('{'):
            gold_str_clean = '{' + gold_str_clean
        if not gold_str_clean.endswith('}'):
            gold_str_clean = gold_str_clean + '}'
            
        try:
            gold_json = json.loads(gold_str_clean)
        except json.JSONDecodeError:
            print(f"Error decoding gold JSON for record {raw.get('report_index', 'unknown')}")
            # Try a more aggressive fix for trailing commas or other common GPT output issues
            try:
                # Remove possible trailing comma before closing brace
                import re
                gold_str_fixed = re.sub(r',\s*}', '}', gold_str_clean)
                gold_json = json.loads(gold_str_fixed)
            except:
                gold_json = {}

        # Construct spans from gold_json
        # scorer.py expects: "spans": {"Key": {"text": "Value", "start": 0, "end": 0}}
        # We set start/end to 0 as scorer.py seems to primarily use 'text' for Task 2
        spans = {}
        for k, v in gold_json.items():
            spans[k] = {
                "text": str(v),
                "start": 0,
                "end": 0
            }
            
        aligned['spans'] = spans
        updated_records.append(aligned)
        
    # Write back to aligned_gt_file
    with open(aligned_gt_file, 'w', encoding='utf-8') as f:
        for rec in updated_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            
    print(f"Successfully updated {len(updated_records)} records in {aligned_gt_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_file", required=True)
    parser.add_argument("--aligned_gt_file", required=True)
    args = parser.parse_args()
    
    reconstruct_aligned_gt(args.raw_file, args.aligned_gt_file)
