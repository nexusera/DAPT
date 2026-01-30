import json
import argparse
import os
import sys
import numpy as np
import pickle
from tqdm import tqdm

# Compatibility: Check for sentence_transformers, else fallback to transformers/torch
try:
    from sentence_transformers import SentenceTransformer
    USE_ST = True
except ImportError:
    USE_ST = False
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("Error: Requirements missing. Please install 'sentence-transformers' OR 'transformers' + 'torch'.")
        sys.exit(1)

# Set default model path for the remote server
DEFAULT_MODEL_PATH = "/data/ocean/embedding_model/BAAI/bge-m3"

if not USE_ST:
    class HFModelWrapper:
        """Minimal wrapper to mimic SentenceTransformer using HuggingFace Transformers"""
        def __init__(self, model_path):
            print(f"Loading HF model from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()

        def encode(self, sentences, batch_size=32, show_progress_bar=False):
            if not sentences: return np.array([])
            all_embeddings = []
            iterator = range(0, len(sentences), batch_size)
            if show_progress_bar: iterator = tqdm(iterator, desc="Embedding")
            
            for i in iterator:
                batch = sentences[i : i + batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use CLS token for BGE models
                    embeddings = outputs.last_hidden_state[:, 0]
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
            
            return np.vstack(all_embeddings) if all_embeddings else np.array([])

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            # Handle jsonl if necessary, but starting with json
            f.seek(0)
            return [json.loads(line) for line in f]

def extract_gt(item):
    """
    Extract keys and values from meta.raw (GT).
    """
    keys = []
    values = []
    
    if 'meta' not in item or 'raw' not in item['meta']:
        return keys, values
    
    raw_data = item['meta']['raw']
    
    ignore_keys = {'__idx', 'added_keys', 'report_title'} 
    
    for k, v in raw_data.items():
        if k in ignore_keys:
            continue
        if not isinstance(v, str):
            continue
            
        keys.append(str(k))
        values.append(str(v))
        
    return keys, values

def extract_pred(item):
    """
    Extract keys and values from predictions (Label Studio format).
    """
    keys = []
    values = []
    
    if 'predictions' not in item:
        return keys, values
    
    preds = item['predictions']
    if not preds:
        return keys, values
        
    # Assuming we take the first prediction if multiple exist, or iterate all
    # Usually valid prediction is just one.
    
    # Process all predictions or just the first one? 
    # Let's take the first one that has a 'result'.
    
    pred_results = []
    for p in preds:
        if 'result' in p:
            pred_results = p['result']
            break
            
    for res in pred_results:
        text = res.get('value', {}).get('text', "")
        labels = res.get('value', {}).get('labels', [])
        
        if not text:
            continue
            
        if "键名" in labels:
            keys.append(text)
        if "值" in labels:
            values.append(text)
            
    return keys, values

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for keys and values from JSON data.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output pickle file.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the embedding model.")
    parser.add_argument("--mode", type=str, choices=['gt', 'pred'], required=True, help="Extraction mode: 'gt' for meta.raw, 'pred' for predictions field.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding generation.")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    
    model = None
    if USE_ST:
        try:
            model = SentenceTransformer(args.model_path)
        except Exception as e:
            print(f"Warning: SentenceTransformer failed ({e}). Trying HF fallback...")
            if 'HFModelWrapper' in globals():
                model = HFModelWrapper(args.model_path)
            else:
                # Need to define it if we are here but USE_ST was True initially
                # This case is rare (import success, load fail), user likely has transformers if they have ST
                import torch
                from transformers import AutoModel, AutoTokenizer
                # We would need to duplicate the class or import it. 
                # Simplification: If ST exists but fails, just crash for now or ask user to check path.
                print("Failed to load model with SentenceTransformer.")
                sys.exit(1)
    else:
        print("Using Transformers library (sentence-transformers not found)...")
        model = HFModelWrapper(args.model_path)
        
    print(f"Loading data from {args.input_file}...")
    data = load_data(args.input_file)
    
    results = {}
    
    # Pre-collect all texts to batch embed
    all_keys = []
    all_values = []
    item_indices = [] # Tuple of (id, type, start_idx, count) to reconstruct
    
    print(f"Extracting texts (Mode: {args.mode})...")
    
    valid_count = 0
    for item in tqdm(data):
        item_id = item.get('id')
        if item_id is None:
            continue
            
        if args.mode == 'gt':
            keys, values = extract_gt(item)
        else:
            keys, values = extract_pred(item)
            
        if not keys and not values:
            continue # Skip empty items
            
        results[item_id] = {
            "keys_text": keys,
            "values_text": values,
            # Placeholders
            "keys_emb": None,
            "values_emb": None
        }
        
    print(f"Generating embeddings for {len(results)} items...")
    
    # We will embed strictly item by item to simplify mapping back, 
    # or batch across everything? Batch across everything is faster.
    
    # Prepare flat lists
    flat_keys = []
    flat_values = []
    
    # Store mapping: item_id -> (start_k, len_k, start_v, len_v)
    mapping = {}
    
    dict_keys = list(results.keys())
    
    current_k_idx = 0
    current_v_idx = 0
    
    for item_id in dict_keys:
        k_list = results[item_id]["keys_text"]
        v_list = results[item_id]["values_text"]
        
        flat_keys.extend(k_list)
        flat_values.extend(v_list)
        
        mapping[item_id] = {
            "k_start": current_k_idx,
            "k_len": len(k_list),
            "v_start": current_v_idx,
            "v_len": len(v_list)
        }
        
        current_k_idx += len(k_list)
        current_v_idx += len(v_list)
        
    # Embed keys
    print(f"Embedding {len(flat_keys)} keys...")
    if flat_keys:
        keys_embeddings = model.encode(flat_keys, batch_size=args.batch_size, show_progress_bar=True)
    else:
        keys_embeddings = np.array([])
        
    # Embed values
    print(f"Embedding {len(flat_values)} values...")
    if flat_values:
        values_embeddings = model.encode(flat_values, batch_size=args.batch_size, show_progress_bar=True)
    else:
        values_embeddings = np.array([])
        
    # Reconstruct structure
    final_output = {}
    
    for item_id, idx_map in mapping.items():
        k_start = idx_map['k_start']
        k_end = k_start + idx_map['k_len']
        
        v_start = idx_map['v_start']
        v_end = v_start + idx_map['v_len']
        
        k_emb = keys_embeddings[k_start:k_end] if idx_map['k_len'] > 0 else []
        v_emb = values_embeddings[v_start:v_end] if idx_map['v_len'] > 0 else []
        
        final_output[item_id] = {
            "keys": results[item_id]["keys_text"],
            "values": results[item_id]["values_text"],
            "keys_emb": k_emb,
            "values_emb": v_emb
        }
        
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'wb') as f:
        pickle.dump(final_output, f)
        
    print("Done.")

if __name__ == "__main__":
    main()
