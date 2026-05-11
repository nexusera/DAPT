import json
import os

file_path = "/data/ocean/medstruct_s_llm/results/inference_DA/Qwen2.5-32B-Instruct_task1_zeroshot_en_vllm_results.jsonl"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"Reading first line from {file_path}...")
        line = f.readline()
        if line:
            data = json.loads(line)
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print("File is empty.")
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"Error reading file: {e}")
