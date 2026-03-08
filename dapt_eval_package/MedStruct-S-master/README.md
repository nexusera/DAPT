# MedStruct-S: A benchmark for Key Discovery, Key-Conditioned QA and Semi-Structured Extraction from OCR-Derived Clinical Reports

## What is MedStruct-S?

MedStruct-S is a standardized evaluation framework designed to assess the performance of models (such as Encoder-only NER/QA and Decoder-only Models) on medical text semi-structuring tasks.

MedStruct-S provides a robust evaluation engine based on core medical semantic alignment, moving beyond simple string-matching to evaluate the true structural integrity and accuracy of information extraction. It includes task-specific logic to evaluate models fairly on real-world clinical datasets.

## ✅ Key Features

- **📊 Comprehensive Evaluation Engine** An evaluation pipeline that handles OCR errors and diverse clinical terminologies using length-adaptive semantic thresholds, ensuring fair assessment across models.
- **⚕️ Multi-level Clinical Tasks** Tasks are structured around real-world medical extraction requirements:

  - Task 1 (Key Discovery): Identifying relevant fields within the document
  - Task 2 (QA): Query-set driven population of predefined medical schema
  - Task 3 (KV Pairing): Open-World Key-Value Extraction and Pairing
- **🧠 Advanced Matching Logic** Includes dynamic thresholding, text normalization, and span verification (Intersection over Union) to ensure accurate evaluation across different formatting styles.
- **🤖 Model Compatibility** Compatible with any model capable of generating standardized JSONL predictions, including:

  - BERT-based NER/QA models
  - LLaMA, Qwen, and other open-source LLMs
  - GPT-4, Claude, and commercial APIs
- **🧪 Detailed Metrics**
  Calculates comprehensive Precision, Recall, and F1 scores across Exact and Approximate matching criteria, supporting both global and positive-only extraction scopes. All reported Precision, Recall, and F1 metrics are Micro-level across all instances to accurately reflect the model's capability.

## 🚀 Getting Started

### **Step 1: Clone the repository**

```bash
git clone https://github.com/Anonymous/MedStruct-S.git
cd MedStruct-S
```

### **Step 2: Prepare your data**

`scorer.py` requires `--pred_file` and `--gt_file` to be in **standardized JSONL** format. Each line must be a JSON object with the following structure:

```json
{
    "id": "sample_001",
    "report_title": "Discharge Summary",
    "ocr_text": "Name: Zhang San. Diagnosis: Influenza ...",
    "pairs": [
        {"key": "Name", "value": "Zhang San", "key_span": [0, 2]},
        {"key": "Diagnosis", "value": "Influenza", "key_span": [8, 10]}
    ]
}
```

| Field              | Type                   | Required | Description                                                                             |
| :----------------- | :--------------------- | :------- | :-------------------------------------------------------------------------------------- |
| `id`               | `string`               | Optional | Unique identifier for the sample, used for traceability and error positioning.          |
| `report_title`     | `string`               | Optional | Medical record type (e.g., "Discharge Summary"), used for Query Set matching in Task 2. |
| `ocr_text`         | `string`               | Optional | Original OCR text, used for span verification and debugging.                            |
| `pairs`            | `list[dict]`           | ✅        | List of key-value pairs, each being a self-contained dictionary.                        |
| `pairs[].key`      | `string`               | ✅        | Key name.                                                                               |
| `pairs[].value`    | `string`               | ✅        | Extracted value.                                                                        |
| `pairs[].key_span` | `[int, int]` or `null` | ✅        | Character-level position of the key name in `ocr_text`. Use `null` if unavailable.      |

> **Note**: Multiple pairs with the same key name (e.g., two "Dates") are supported. Each pair carries its own `key_span` independently.

### **Step 3: Download the dataset**

Download the MedStruct-S dataset from (https://doi.org/10.5281/zenodo.18814410)

> **Note:** Access requires registration and data use agreement approval (Placeholder).

### **Step 4: Run Evaluation**

**Full Evaluation (Task 1 + 2 + 3)**
Calculate P/R/F1 for all tasks. This is the recommended approach for standard benchmark reporting:

```bash
python scorer.py \
    --pred_file <path_to_predictions.jsonl> \
    --gt_file <path_to_ground_truth.jsonl> \
    --query_set <path_to_query_set.json> \
    --task_type all \
    --output_file <path_to_output.json>
```

**Task 2 (Value Extraction) Only**
Suitable for evaluating the accuracy of value extraction on a predefined set of fields:

```bash
python scorer.py \
    --task_type task2 \
    --pred_file <path_to_predictions.jsonl> \
    --gt_file <path_to_ground_truth.jsonl> \
    --query_set <path_to_query_set.json> \
    --model_name "MacBERT-QA"
```

### Advanced Configuration

The behavior of the evaluation engine can be dynamically controlled via CLI arguments (corresponding to the algorithms in `med_eval/metrics.py`):

| Parameter                | Description                                                                                  | Default Behavior                     |
| :----------------------- | :------------------------------------------------------------------------------------------- | :----------------------------------- |
| `--no_normalize`         | Disable text preprocessing (e.g., lowercase conversion, whitespace removal).                 | Normalization**enabled** by default. |
| `--similarity_threshold` | Set the base NED similarity threshold when length-adaptive dynamic thresholding is disabled. | Tau**enabled** by default.           |
| `--disable_tau`          | Disable length-adaptive dynamic thresholding (Tau Logic).                                    | Tau**enabled** by default.           |
| `--overlap_threshold`    | Set the IoU threshold for position verification.                                             | Defaults to `0.0`.                   |

## 📊 Output Results

The output JSON contains a `summary` (metadata) section and a `tasks` (task-specific metrics) section:

```json
{
  "summary": {
    "model": "model_name",
    "dataset": "MedStruct-S",
    "samples": 100,
    "eval_time": "202X-XX-XX XX:00:00",
    "config": {
      "normalize": true,
      "similarity_threshold": 0.8,
      "overlap_threshold": 0.0,
      "tau_dynamic": true,
      "use_em": true,
      "use_am": true,
      "use_span": true
    }
  },
  "tasks": {
    "task1": {
      "stats": {"tp_e": 90, "tp_a": 95, "total_p": 100, "total_g": 100},
      "metrics": {
        "exact":  {"p": 0.90, "r": 0.90, "f1": 0.90, "tp": 90},
        "approx": {"p": 0.95, "r": 0.95, "f1": 0.95, "tp": 95}
      }
    },
    "task2_global": {
      "stats": {"tp_e": 70, "tp_a": 80, "total": 100},
      "metrics": {
        "exact":  {"p": 0.70, "r": 0.70, "f1": 0.70, "tp": 70},
        "approx": {"p": 0.80, "r": 0.80, "f1": 0.80, "tp": 80}
      }
    },
    "task2_pos_only": {
      "stats": {"tp_e": 65, "tp_a": 75, "total": 85},
      "metrics": {
        "exact":  {"p": 0.76, "r": 0.76, "f1": 0.76, "tp": 65},
        "approx": {"p": 0.88, "r": 0.88, "f1": 0.88, "tp": 75}
      }
    },
    "task3": {
      "stats": {"ee_tp": 60, "ea_tp": 70, "aa_tp": 75, "total_p": 100, "total_g": 100},
      "metrics": {
        "exact_exact":              {"p": 0.60, "r": 0.60, "f1": 0.60, "tp": 60},
        "exact_approximate":        {"p": 0.70, "r": 0.70, "f1": 0.70, "tp": 70},
        "approximate_approximate":  {"p": 0.75, "r": 0.75, "f1": 0.75, "tp": 75}
      }
    }
  }
}
```

### Evaluation Metrics Overview

- **Task 1: Key Discovery**
  Evaluates the model's ability to identify keys (fields) within the document, independent of their associated values. Output: Precision / Recall / F1 via Exact and Approximate matching.
- **Task 2: QA**
  Evaluates the model's accuracy in extracting values for a predefined set of fields. Output: Exact / Approx P/R/F1 across **Global** (all schema fields) and **Positive-only** (existing GT values) scopes.
- **Task 3: KV Pairing**
  Evaluates how accurately the model pairs keys with their correct values. Output splits into Exact-Exact, Exact-Approx, and Approx-Approx matches.

## 📄 Citation

If you find this benchmark or code useful in your research, please cite the following (to be updated):

```bibtex
@inproceedings{med_struct_s,
  title={MedStruct-S: A benchmark for Key Discovery, Key-Conditioned QA and Semi-Structured Extraction from OCR-Derived Clinical Reports},
  author={Anonymous Authors},
  year={202X},
  booktitle={Under Review}
}
```
