# Embedding Evaluation Scripts

This folder contains scripts to evaluate Key/Value extraction performance using Semantic Similarity (Embedding-based).

## Scripts

1. `gen_embeddings.py`: Generates embeddings for Keys and Values from a JSON file.
2. `eval_similarity.py`: Compares two embedding pickle files (GT and Pred) and calculates Recall, Precision, and F1.

## Usage

### 1. Generate Embeddings for Ground Truth (GT)

Use your `raw_test.json` (or any file containing the ground truth in `meta.raw`):

```bash
python gen_embeddings.py \
  --input_file raw_test.json \
  --output_file gt_embeddings.pkl \
  --mode gt \
  --model_path /data/ocean/embedding_model/BAAI/bge-m3
```

This extracts keys and values from `item['meta']['raw']`.

### 2. Generate Embeddings for Predictions

Use your prediction file (or `raw_test.json` if it contains predictions in Label Studio format):

```bash
python gen_embeddings.py \
  --input_file raw_test.json \
  --output_file pred_embeddings.pkl \
  --mode pred \
  --model_path /data/ocean/embedding_model/BAAI/bge-m3
```

This extracts keys and values from `item['predictions'][...]['result']` (Label Studio format), looking for labels `"键名"` (Key) and `"值"` (Value).

### 3. Evaluate Similarity

Compare the generated pickle files:

```bash
python eval_similarity.py \
  --gt_pkl gt_embeddings.pkl \
  --pred_pkl pred_embeddings.pkl \
  --threshold 0.85
```

This will output:
- **Recall (Semantic):** Metric on how many GT items are covered by Predictions.
- **Precision (Semantic):** Metric on how many Predicted items are valid (exist in GT).
- **F1 Score:** Harmonic mean.
- **Avg Max Sim:** Average raw cosine similarity scores.
