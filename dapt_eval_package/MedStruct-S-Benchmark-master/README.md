# 评测工具包 (Evaluation Toolkit)

本目录包含了用于评测 BERT (NER/EBQA) 和 GPT 系列模型在医疗文本信息抽取任务上的所有核心脚本。

---

## 📁 文件结构

| 脚本名 | 功能 | 适用场景 |
| :--- | :--- | :--- |
| `scorer.py` | **核心评分器**：计算 Task 1/2/3 的 P/R/F1 | BERT 和 GPT 通用 |
| `convert_llm_outputs.py` | **GPT 输出对齐**：将 vLLM 原始输出转换为标准评测格式 | GPT 系列专用 |
| `ground_gpt_preds.py` | **预测落地**：将 GPT 抽取结果落回原文 Span | GPT 系列可选 |
| `reconstruct_gpt_gt.py` | **GT 重建**：从 vLLM 原始文件重建对齐的 GT | GPT 系列专用 |
| `metrics.py` | **指标计算核心**：包含 NED、Jaccard、Span IoU 等计算函数 | 被 scorer.py 依赖 |
| `report_keys_alias.py` | **键名映射 Schema**：定义标准键名与别名的映射关系 | 被 scorer.py 依赖 |

---

## 🚀 快速开始

### 1. 评测 BERT 预测结果 (NER / EBQA)

```bash
python scorer.py \
    --pred_file <predictions.jsonl> \
    --gt_file <ground_truth.jsonl> \
    --output_file <output_scores.json> \
    --model_name "MacBERT" \
    --dataset_type "Original"
```

**示例：**
```bash
# 评测 NER MacBERT 在 Original (Real) 数据集上的表现
python scorer.py \
    --pred_file ../predictions/bert/NER/ner_original_macbert.jsonl \
    --gt_file ../data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
    --output_file ../predictions/bert/NER/score_v2_macbert.json \
    --model_name "MacBERT" --dataset_type "Original"
```

---

### 2. 评测 GPT (vLLM) 预测结果

GPT 评测需要先进行**数据对齐**，将 vLLM 的原始输出转换为评分器可识别的格式。

#### Step 1: 数据对齐 (使用 `convert_llm_outputs.py`)

```bash
python convert_llm_outputs.py \
    --llm_file <vLLM_raw_results.jsonl> \
    --gt_master <ground_truth_master.jsonl> \
    --task_type <task1|task2|task3> \
    --output_pred <aligned_pred.jsonl> \
    --output_gt <aligned_gt.jsonl>
```

**示例 (Task 2 KV Pairing)：**
```bash
python convert_llm_outputs.py \
    --llm_file ../predictions/gpt/task2_kv_pairing_da/Baichuan-M2-32B_task2_zeroshot_en_vllm_results.jsonl \
    --gt_master ../data/sft_data_da_clean260123/test_flattened_with_spans_v2.jsonl \
    --task_type task2 \
    --output_pred ../predictions/gpt/task2_kv_pairing_da_re/aligned_pred_Baichuan.jsonl \
    --output_gt ../predictions/gpt/task2_kv_pairing_da_re/aligned_gt_Baichuan.jsonl
```

#### Step 2: 运行评分器

```bash
python scorer.py \
    --pred_file <aligned_pred.jsonl> \
    --gt_file <aligned_gt.jsonl> \
    --output_file <output_scores.json> \
    --model_name "Baichuan-M2-32B" --dataset_type "GPT_DA_v2"
```

---

## 📊 输出格式说明

评分结果为 JSON 格式，包含三大任务的指标：

```json
{
  "Task 1 (Key Discovery)": {
    "Strict (K_E)": {"p": 0.xx, "r": 0.xx, "f1": 0.xx},
    "Robust (K_A)": {"p": 0.xx, "r": 0.xx, "f1": 0.xx}
  },
  "Task 2 (Value Extraction)": {
    "Exact (QA_E)": {"p": 0.xx, "r": 0.xx, "f1": 0.xx},
    "Approx (QA_A)": {"p": 0.xx, "r": 0.xx, "f1": 0.xx},
    "Exact (QA_Pos-E)": {"p": 0.xx, "r": 0.xx, "f1": 0.xx},
    "Approx (QA_Pos-A)": {"p": 0.xx, "r": 0.xx, "f1": 0.xx}
  },
  "Task 3 (KV Pairing)": {
    "Exact-Exact (K_E V_E)": {"p": 0.xx, "r": 0.xx, "f1": 0.xx},
    "Approximate-Approximate (K_A V_A)": {"p": 0.xx, "r": 0.xx, "f1": 0.xx}
  },
  "metadata": { ... }
}
```

### 指标说明

| 指标 | 含义 |
| :--- | :--- |
| **K_E (Strict)** | 键名严格匹配 |
| **K_A (Robust)** | 键名容错匹配 (通过 Schema 别名) |
| **QA_E (Exact)** | 值精确匹配 |
| **QA_A (Approx)** | 值模糊匹配 (基于字符级编辑距离的归一化相似度 ≥ 0.8) |
| **QA_Pos-E/A** | **正向值匹配** (仅统计 GT 非空的样本) |

---

## ⚠️ 注意事项

1. **依赖关系**：`scorer.py` 依赖 `metrics.py` 和 `report_keys_alias.py`，请确保三者在同一目录或正确设置 Python Path。
2. **文件名空格**：如果文件名包含空格，请用引号包裹路径。

---

## 📂 相关数据路径

| 数据类型 | 路径 |
| :--- | :--- |
| Original GT | `data/kv_ner_prepared_comparison/val_eval_titled.jsonl` |
| DA GT| `data/sft_data_da_clean260123/test_flattened_with_spans_v2.jsonl` |
| BERT NER 预测 | `predictions/bert/NER/` |
| BERT EBQA 预测 | `predictions/bert/EBQA/` |
| GPT Task 1/2/3 预测 | `predictions/gpt/task*_da*/` |

---

## 📝 版本历史

- **2026-02-05**: 初始版本，整合 BERT 和 GPT 评测全流程。
