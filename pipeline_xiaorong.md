需要对这 4 个不同的预训练模型分别进行 **Task 1/2 (KV-NER)** 的全流程实验（微调 -> 推理 -> 评测），目的是对比它们在同一下游任务上的表现。

为了保证实验清晰且互不干扰，我将采取以下策略：
1.  **为每个模型创建独立的 Config 文件**：确保输出路径（`output_dir`）不同，避免权重覆盖。
2.  **生成一个“一键运行”脚本**：汇总所有命令，方便你在服务器挂起运行。

### 第一步：检查并创建配置文件

首先，我需要读取现有的基础配置文件 `pre_struct/kv_ner/kv_ner_config.json`，以此为模板生成 4 个新的专用配置。

 

读取 [](file:///Users/shanqi/Documents/BERT_DAPT/DAPT)

读取 [](file:///Users/shanqi/Documents/BERT_DAPT/DAPT/dapt_eval_package/pre_struct/kv_ner)

读取 [](file:///Users/shanqi/Documents/BERT_DAPT/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config.json)

已创建 [](file:///Users/shanqi/Documents/BERT_DAPT/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_hybrid.json)

已创建 [](file:///Users/shanqi/Documents/BERT_DAPT/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_staged.json)

已创建 [](file:///Users/shanqi/Documents/BERT_DAPT/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_macbert.json)

已创建 [](file:///Users/shanqi/Documents/BERT_DAPT/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_mtl.json)

根据你的需求，我已经为你生成了 4 个新的配置文件，并整理好了对应的「训练 -> 推理 -> 评测」全流程脚本。

### 1. 配置文件生成
我已经为你创建了以下 4 个配置文件，分别对应不同的预训练模型和独立的输出目录（防止覆盖）：

| 模型简称 | 配置文件路径 (dapt_eval_package/pre_struct/kv_ner/) | 输出目录 (runs/) |
| :--- | :--- | :--- |
| **Hybrid** | `kv_ner_config_hybrid.json` | `kv_ner_finetuned_hybrid` |
| **Staged** | `kv_ner_config_staged.json` | `kv_ner_finetuned_staged` |
| **MacBERT** | `kv_ner_config_macbert.json` | `kv_ner_finetuned_macbert` |
| **MTL** | `kv_ner_config_mtl.json` | `kv_ner_finetuned_mtl` |

### 2. 运行命令脚本
请在服务器上的 `/data/ocean/DAPT` 目录下执行以下命令。为了方便管理，你可以将它们保存为一个 shell 脚本（如 `run_all_experiments.sh`）运行，或者分块复制执行。

#### 环境变量与通用设置
```bash
cd /data/ocean/DAPT
conda activate medical_bert
export PYTHONPATH=$PYTHONPATH:$(pwd)
# 确保使用正确的 Tokenizer 和 噪声配置
NOISE_BINS="/data/ocean/DAPT/workspace/noise_bins.json"
# 使用新的 "Real" 测试集
TEST_DATA="/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json"

# 【重要】
# 新版 compare_models.py 会自动在 output_summary 路径下生成：
# 1. _preds.jsonl (预测结果)
# 2. _gt.jsonl (从测试集转换的标准答案)
# 后续 scorer.py 直接使用这两个生成的文件即可。
```

---

#### 实验 1: Hybrid Span Model
```bash
# 1.1 训练
python /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/train_with_noise.py \
  --config /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_hybrid.json \
  --noise_bins "/data/ocean/DAPT/workspace/noise_bins.json"

# 1.2 推理 (生成 predictions 和 gt)
python /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/compare_models.py \
  --ner_config /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_hybrid.json \
  --test_data "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --noise_bins "/data/ocean/DAPT/workspace/noise_bins.json" \
  --output_summary /data/ocean/DAPT/runs/hybrid_eval_summary.json

# 1.3 评估阶段
# (a) 对齐数据 (Task 1 关键步骤：解决 Span 不匹配)
python /data/ocean/DAPT/scripts/align_for_scorer.py \
  --gt_in "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --pred_in "/data/ocean/DAPT/runs/hybrid_eval_summary_preds.jsonl" \
  --gt_out "/data/ocean/DAPT/runs/hybrid_eval_aligned_gt.jsonl" \
  --pred_out "/data/ocean/DAPT/runs/hybrid_eval_aligned_preds.jsonl"


# (b) 运行 scorer.py (使用对齐后的文件，添加 --overlap_threshold -1 忽略 span 位置)
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py \
  --pred_file "/data/ocean/DAPT/runs/hybrid_eval_aligned_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/runs/hybrid_eval_aligned_gt.jsonl" \
  --task_type task1 \
  --overlap_threshold -1 \
  --output_file "/data/ocean/DAPT/runs/hybrid_eval_report_task1.json"

# Task 3 Optional (Task 3 通常不需要严格对齐 Span)
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py \
  --pred_file "/data/ocean/DAPT/runs/hybrid_eval_aligned_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/runs/hybrid_eval_aligned_gt.jsonl" \
  --task_type task3 \
  --output_file "/data/ocean/DAPT/runs/hybrid_eval_report_task3.json"
```

---

#### 实验 2: Staged RoBERTa
```bash
# 2.1 训练
python /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/train_with_noise.py \
  --config /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_staged.json \
  --noise_bins "/data/ocean/DAPT/workspace/noise_bins.json"

# 2.2 推理
python /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/compare_models.py \
  --ner_config /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_staged.json \
  --test_data "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --noise_bins "/data/ocean/DAPT/workspace/noise_bins.json" \
  --output_summary /data/ocean/DAPT/runs/staged_eval_summary.json

# 2.3 评估阶段
# (a) 对齐数据 (Task 1 关键步骤：解决 Span 不匹配)
python /data/ocean/DAPT/scripts/align_for_scorer.py \
  --gt_in "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --pred_in "/data/ocean/DAPT/runs/staged_eval_summary_preds.jsonl" \
  --gt_out "/data/ocean/DAPT/runs/staged_eval_aligned_gt.jsonl" \
  --pred_out "/data/ocean/DAPT/runs/staged_eval_aligned_preds.jsonl"

# (b) 运行 scorer.py (使用对齐后的文件，添加 --overlap_threshold -1 忽略 span 位置)
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py \
  --pred_file "/data/ocean/DAPT/runs/staged_eval_aligned_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/runs/staged_eval_aligned_gt.jsonl" \
  --task_type task1 \
  --overlap_threshold -1 \
  --output_file "/data/ocean/DAPT/runs/staged_eval_report_task1.json"

# Task 3 Optional
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py \
  --pred_file "/data/ocean/DAPT/runs/staged_eval_aligned_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/runs/staged_eval_aligned_gt.jsonl" \
  --task_type task3 \
  --output_file "/data/ocean/DAPT/runs/staged_eval_report_task3.json"
```

---

#### 实验 3: MacBERT Staged
```bash
# 3.1 训练
python /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/train_with_noise.py \
  --config /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_macbert.json \
  --noise_bins "/data/ocean/DAPT/workspace/noise_bins.json"

# 3.2 推理
python /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/compare_models.py \
  --ner_config /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_macbert.json \
  --test_data "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --noise_bins "/data/ocean/DAPT/workspace/noise_bins.json" \
  --output_summary /data/ocean/DAPT/runs/macbert_eval_summary.json

# 3.3 评估阶段
# (a) 对齐数据 (Task 1 关键步骤：解决 Span 不匹配)
python /data/ocean/DAPT/scripts/align_for_scorer.py \
  --gt_in "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --pred_in "/data/ocean/DAPT/runs/macbert_eval_summary_preds.jsonl" \
  --gt_out "/data/ocean/DAPT/runs/macbert_eval_aligned_gt.jsonl" \
  --pred_out "/data/ocean/DAPT/runs/macbert_eval_aligned_preds.jsonl"

# (b) 运行 scorer.py (使用对齐后的文件，添加 --overlap_threshold -1 忽略 span 位置)
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py \
  --pred_file "/data/ocean/DAPT/runs/macbert_eval_aligned_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/runs/macbert_eval_aligned_gt.jsonl" \
  --task_type task1 \
  --overlap_threshold -1 \
  --output_file "/data/ocean/DAPT/runs/macbert_eval_report_task1.json"

# Task 3 Optional
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py \
  --pred_file "/data/ocean/DAPT/runs/macbert_eval_aligned_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/runs/macbert_eval_aligned_gt.jsonl" \
  --task_type task3 \
  --output_file "/data/ocean/DAPT/runs/macbert_eval_report_task3.json"
```

---

#### 实验 4: Multi-Task Learning (MTL)
```bash
# 4.1 训练
python /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/train_with_noise.py \
  --config /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_mtl.json \
  --noise_bins "/data/ocean/DAPT/workspace/noise_bins.json"

# 4.2 推理
python /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/compare_models.py \
  --ner_config /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_mtl.json \
  --test_data "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --noise_bins "/data/ocean/DAPT/workspace/noise_bins.json" \
  --output_summary /data/ocean/DAPT/runs/mtl_eval_summary.json

# 4.3 评估阶段
# (a) 对齐数据 (Task 1 关键步骤：解决 Span 不匹配)
python /data/ocean/DAPT/scripts/align_for_scorer.py \
  --gt_in "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --pred_in "/data/ocean/DAPT/runs/mtl_eval_summary_preds.jsonl" \
  --gt_out "/data/ocean/DAPT/runs/mtl_eval_aligned_gt.jsonl" \
  --pred_out "/data/ocean/DAPT/runs/mtl_eval_aligned_preds.jsonl"

# (b) 运行 scorer.py (使用对齐后的文件，添加 --overlap_threshold -1 忽略 span 位置)
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py \
  --pred_file "/data/ocean/DAPT/runs/mtl_eval_aligned_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/runs/mtl_eval_aligned_gt.jsonl" \
  --task_type task1 \
  --overlap_threshold -1 \
  --output_file "/data/ocean/DAPT/runs/mtl_eval_report_task1.json"

# Task 3 Optional
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py \
  --pred_file "/data/ocean/DAPT/runs/mtl_eval_aligned_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/runs/mtl_eval_aligned_gt.jsonl" \
  --task_type task3 \
  --output_file "/data/ocean/DAPT/runs/mtl_eval_report_task3.json"

---

### [新增] 消融实验配置生成
为了运行实验 5 和 实验 6，首先确保对应的 KV-NER 配置文件已上传至服务器：
- `dapt_eval_package/pre_struct/kv_ner/kv_ner_config_no_nsp.json`
- `dapt_eval_package/pre_struct/kv_ner/kv_ner_config_no_mlm.json`

这两个文件已在本地准备好，分别指向：
- No NSP Model: `/data/ocean/DAPT/workspace/output_ablation_no_nsp/final_no_nsp_model`
- No MLM Model: `/data/ocean/DAPT/workspace/output_ablation_no_mlm/final_no_mlm_model`

#### 实验 5: Ablation No NSP (MLM Only)
```bash
# 5.1 训练
python /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/train_with_noise.py \
  --config /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_no_nsp.json \
  --noise_bins "/data/ocean/DAPT/workspace/noise_bins.json"

# 5.2 推理
python /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/compare_models.py \
  --ner_config /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_no_nsp.json \
  --test_data "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --noise_bins "/data/ocean/DAPT/workspace/noise_bins.json" \
  --output_summary /data/ocean/DAPT/runs/no_nsp_eval_summary.json

# 5.3 评估阶段
# (a) 对齐数据
python /data/ocean/DAPT/scripts/align_for_scorer.py \
  --gt_in "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --pred_in "/data/ocean/DAPT/runs/no_nsp_eval_summary_preds.jsonl" \
  --gt_out "/data/ocean/DAPT/runs/no_nsp_eval_aligned_gt.jsonl" \
  --pred_out "/data/ocean/DAPT/runs/no_nsp_eval_aligned_preds.jsonl"

# (b) 运行 scorer.py
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py \
  --pred_file "/data/ocean/DAPT/runs/no_nsp_eval_aligned_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/runs/no_nsp_eval_aligned_gt.jsonl" \
  --task_type task1 \
  --overlap_threshold -1 \
  --output_file "/data/ocean/DAPT/runs/no_nsp_eval_report_task1.json"

# Task 3 Optional
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py \
  --pred_file "/data/ocean/DAPT/runs/no_nsp_eval_aligned_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/runs/no_nsp_eval_aligned_gt.jsonl" \
  --task_type task3 \
  --output_file "/data/ocean/DAPT/runs/no_nsp_eval_report_task3.json"
```

#### 实验 6: Ablation No MLM (NSP Only)
```bash
# 6.1 训练
python /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/train_with_noise.py \
  --config /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_no_mlm.json \
  --noise_bins "/data/ocean/DAPT/workspace/noise_bins.json"

# 6.2 推理
python /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/compare_models.py \
  --ner_config /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_no_mlm.json \
  --test_data "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --noise_bins "/data/ocean/DAPT/workspace/noise_bins.json" \
  --output_summary /data/ocean/DAPT/runs/no_mlm_eval_summary.json

# 6.3 评估阶段
# (a) 对齐数据
python /data/ocean/DAPT/scripts/align_for_scorer.py \
  --gt_in "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --pred_in "/data/ocean/DAPT/runs/no_mlm_eval_summary_preds.jsonl" \
  --gt_out "/data/ocean/DAPT/runs/no_mlm_eval_aligned_gt.jsonl" \
  --pred_out "/data/ocean/DAPT/runs/no_mlm_eval_aligned_preds.jsonl"

# (b) 运行 scorer.py
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py \
  --pred_file "/data/ocean/DAPT/runs/no_mlm_eval_aligned_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/runs/no_mlm_eval_aligned_gt.jsonl" \
  --task_type task1 \
  --overlap_threshold -1 \
  --output_file "/data/ocean/DAPT/runs/no_mlm_eval_report_task1.json"

# Task 3 Optional
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py \
  --pred_file "/data/ocean/DAPT/runs/no_mlm_eval_aligned_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/runs/no_mlm_eval_aligned_gt.jsonl" \
  --task_type task3 \
  --output_file "/data/ocean/DAPT/runs/no_mlm_eval_report_task3.json"
```
```

Made changes.