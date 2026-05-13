以下是task2完整的操作流程，以 MacBERT (实验 3) 为例：

第一步：准备数据 (Format Conversion)
我们需要先将 KV-NER 格式的数据转换为 QA 格式。
# 环境变量
export PYTHONPATH=$PYTHONPATH:/data/ocean/DAPT
export QUERY_SET="/data/ocean/DAPT/dapt_eval_package/MedStruct-S-master/keys_merged_1027_cleaned.json"
NOISE_BINS="/data/ocean/DAPT/workspace/noise_bins.json"
TOKENIZER_PATH="/data/ocean/DAPT/macbert_staged_output/final_staged_model"

# 1.1 转换训练集
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py \
  --input_file "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json" \
  --output_file "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_train_real.jsonl" \
  --struct_path $QUERY_SET \
  --tokenizer_name $TOKENIZER_PATH \
  --noise_bins $NOISE_BINS

# 1.2 转换测试集 (用于推理)
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py \
  --input_file "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --output_file "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_eval_real.jsonl" \
  --struct_path $QUERY_SET \
  --tokenizer_name $TOKENIZER_PATH \
  --noise_bins $NOISE_BINS

第二步：微调 MacBERT EBQA 模型
配置文件 dapt_eval_package/pre_struct/ebqa/ebqa_config_macbert.json 已经创建好了，直接运行训练脚本：
# 2.1 启动训练
# 输出目录: /data/ocean/DAPT/runs/ebqa_macbert
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/train_ebqa.py \
  --config /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/ebqa_config_macbert.json

第三步：推理与评测
训练完成后，使用最优权重 (best 目录) 进行推理，并调用 scorer.py 进行打分。

# 3.1 推理 (生成预测结果)
# 产出文件: runs/ebqa_macbert_preds.jsonl
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/predict_ebqa.py \
  --model_dir "/data/ocean/DAPT/runs/ebqa_macbert/best" \
  --tokenizer $TOKENIZER_PATH \
  --data_path "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_eval_real.jsonl" \
  --output_preds "/data/ocean/DAPT/runs/ebqa_macbert_preds.jsonl"

# 3.2 将 QA 级预测聚合回文档级 (关键！否则后续同事对齐脚本无法工作)
# 输入: runs/ebqa_macbert_preds.jsonl (QA级，包含 report_index/question_key/pred_text)
# 输出: runs/ebqa_macbert_doc_preds.jsonl (文档级，包含 text + pred_pairs)
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/aggregate_qa_preds_to_doc.py \
  --raw_file "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --qa_pred_file "/data/ocean/DAPT/runs/ebqa_macbert_preds.jsonl" \
  --output_file "/data/ocean/DAPT/runs/ebqa_macbert_doc_preds.jsonl" \
  --prefer score

# (可选) 快速自检：QA 级一般是几千行；doc 级应约等于测试集文档数(例如 355)
wc -l /data/ocean/DAPT/runs/ebqa_macbert_preds.jsonl /data/ocean/DAPT/runs/ebqa_macbert_doc_preds.jsonl

# 3.3 评测前转换数据 (Task 2, 与同事对齐：用 text_hash 将预测对齐回 GT 的 id)
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-master/utils/preprocess_ebqa_real_h200.py \
  --gt_file /data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \
  --pred_file /data/ocean/DAPT/runs/ebqa_macbert_doc_preds.jsonl \
  --output_dir aligned_data

# (可选) 自检：aligned_data 下 pred/gt 行数必须一致
wc -l aligned_data/gt_ebqa_aligned.jsonl aligned_data/aligned_ebqa_macbert_doc_preds.jsonl

# 3.4 运行官方 MedStruct-S Scorer (Task 2)
python /data/ocean/DAPT/scripts/run_medstruct_scorer.py \
  --pred_file "/data/ocean/DAPT/aligned_data/aligned_ebqa_macbert_doc_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/aligned_data/gt_ebqa_aligned.jsonl" \
  --query_set $QUERY_SET \
  --task_type task2 \
  --output_file "/data/ocean/DAPT/runs/ebqa_macbert_report_task2.json"

---

#### 实验 5: Ablation No NSP (MLM Only)

# 5.1 转换训练集
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py \
  --input_file "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json" \
  --output_file "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_train_real.jsonl" \
  --struct_path $QUERY_SET \
  --tokenizer_name $TOKENIZER_PATH \
  --noise_bins $NOISE_BINS

# 5.2 转换测试集
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py \
  --input_file "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --output_file "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_eval_real.jsonl" \
  --struct_path $QUERY_SET \
  --tokenizer_name $TOKENIZER_PATH \
  --noise_bins $NOISE_BINS

# 5.3 微调 No NSP EBQA 模型
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/train_ebqa.py \
  --config /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/ebqa_config_no_nsp.json

# 5.4 推理 (生成预测结果)
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/predict_ebqa.py \
  --model_dir "/data/ocean/DAPT/runs/ebqa_no_nsp/best" \
  --tokenizer $TOKENIZER_PATH \
  --data_path "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_eval_real.jsonl" \
  --output_preds "/data/ocean/DAPT/runs/ebqa_no_nsp_preds.jsonl"

# 5.5 聚合 QA 级预测为文档级
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/aggregate_qa_preds_to_doc.py \
  --raw_file "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --qa_pred_file "/data/ocean/DAPT/runs/ebqa_no_nsp_preds.jsonl" \
  --output_file "/data/ocean/DAPT/runs/ebqa_no_nsp_doc_preds.jsonl" \
  --prefer score

# 5.6 评测前对齐
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-master/utils/preprocess_ebqa_real_h200.py \
  --gt_file /data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \
  --pred_file /data/ocean/DAPT/runs/ebqa_no_nsp_doc_preds.jsonl \
  --output_dir aligned_data

# 5.7 运行 Scorer (Task 2)
python /data/ocean/DAPT/scripts/run_medstruct_scorer.py \
  --pred_file "/data/ocean/DAPT/aligned_data/aligned_ebqa_no_nsp_doc_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/aligned_data/gt_ebqa_aligned.jsonl" \
  --query_set $QUERY_SET \
  --task_type task2 \
  --output_file "/data/ocean/DAPT/runs/ebqa_no_nsp_report_task2.json"

---

#### 实验 6: Ablation No MLM (NSP Only)

# 6.1 转换训练集
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py \
  --input_file "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json" \
  --output_file "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_train_real.jsonl" \
  --struct_path $QUERY_SET \
  --tokenizer_name $TOKENIZER_PATH \
  --noise_bins $NOISE_BINS

# 6.2 转换测试集
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py \
  --input_file "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --output_file "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_eval_real.jsonl" \
  --struct_path $QUERY_SET \
  --tokenizer_name $TOKENIZER_PATH \
  --noise_bins $NOISE_BINS

# 6.3 微调 No MLM EBQA 模型
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/train_ebqa.py \
  --config /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/ebqa_config_no_mlm.json

# 6.4 推理 (生成预测结果)
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/predict_ebqa.py \
  --model_dir "/data/ocean/DAPT/runs/ebqa_no_mlm/best" \
  --tokenizer $TOKENIZER_PATH \
  --data_path "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_eval_real.jsonl" \
  --output_preds "/data/ocean/DAPT/runs/ebqa_no_mlm_preds.jsonl"

# 6.5 聚合 QA 级预测为文档级
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/aggregate_qa_preds_to_doc.py \
  --raw_file "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --qa_pred_file "/data/ocean/DAPT/runs/ebqa_no_mlm_preds.jsonl" \
  --output_file "/data/ocean/DAPT/runs/ebqa_no_mlm_doc_preds.jsonl" \
  --prefer score

# 6.6 评测前对齐
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-master/utils/preprocess_ebqa_real_h200.py \
  --gt_file /data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \
  --pred_file /data/ocean/DAPT/runs/ebqa_no_mlm_doc_preds.jsonl \
  --output_dir aligned_data

# 6.7 运行 Scorer (Task 2)
python /data/ocean/DAPT/scripts/run_medstruct_scorer.py \
  --pred_file "/data/ocean/DAPT/aligned_data/aligned_ebqa_no_mlm_doc_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/aligned_data/gt_ebqa_aligned.jsonl" \
  --query_set $QUERY_SET \
  --task_type task2 \
  --output_file "/data/ocean/DAPT/runs/ebqa_no_mlm_report_task2.json"

---

#### 实验 7: No-Noise Baseline (标准 DAPT，无噪声嵌入)

# 7.1 转换训练集
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py \
  --input_file "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json" \
  --output_file "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_train_real.jsonl" \
  --struct_path $QUERY_SET \
  --tokenizer_name $TOKENIZER_PATH \
  --noise_bins $NOISE_BINS

# 7.2 转换测试集
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py \
  --input_file "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --output_file "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_eval_real.jsonl" \
  --struct_path $QUERY_SET \
  --tokenizer_name $TOKENIZER_PATH \
  --noise_bins $NOISE_BINS

# 7.3 微调 No Noise EBQA 模型
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/train_ebqa.py \
  --config /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/ebqa_config_no_noise.json

# 7.4 推理 (生成预测结果)
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/predict_ebqa.py \
  --model_dir "/data/ocean/DAPT/runs/ebqa_no_noise/best" \
  --tokenizer $TOKENIZER_PATH \
  --data_path "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_eval_real.jsonl" \
  --output_preds "/data/ocean/DAPT/runs/ebqa_no_noise_preds.jsonl"

# 7.5 聚合 QA 级预测为文档级
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/aggregate_qa_preds_to_doc.py \
  --raw_file "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --qa_pred_file "/data/ocean/DAPT/runs/ebqa_no_noise_preds.jsonl" \
  --output_file "/data/ocean/DAPT/runs/ebqa_no_noise_doc_preds.jsonl" \
  --prefer score

# 7.6 评测前对齐
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-master/utils/preprocess_ebqa_real_h200.py \
  --gt_file /data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \
  --pred_file /data/ocean/DAPT/runs/ebqa_no_noise_doc_preds.jsonl \
  --output_dir aligned_data

# 7.7 运行 Scorer (Task 2)
python /data/ocean/DAPT/scripts/run_medstruct_scorer.py \
  --pred_file "/data/ocean/DAPT/aligned_data/aligned_ebqa_no_noise_doc_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/aligned_data/gt_ebqa_aligned.jsonl" \
  --query_set $QUERY_SET \
  --task_type task2 \
  --output_file "/data/ocean/DAPT/runs/ebqa_no_noise_report_task2.json"

---

#### 实验 8: Noise Ablation - Bucket / Linear / MLP

新增配置文件：
- `dapt_eval_package/pre_struct/ebqa/ebqa_config_noise_bucket.json`
- `dapt_eval_package/pre_struct/ebqa/ebqa_config_noise_linear.json`
- `dapt_eval_package/pre_struct/ebqa/ebqa_config_noise_mlp.json`

三组配置分别指向：
- Bucket: `/data/ocean/DAPT/workspace/output_ablation_noise_bucket/final_staged_model`
- Linear: `/data/ocean/DAPT/workspace/output_ablation_noise_linear/final_staged_model`
- MLP: `/data/ocean/DAPT/workspace/output_ablation_noise_mlp/final_staged_model`

训练前的数据转换命令不变；`convert_ebqa.py` 现在会同时写出 `noise_ids` 和 `noise_values`，因此三种模式可复用同一份预处理数据。

```bash
# 8.1 Bucket
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/train_ebqa.py \
  --config /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/ebqa_config_noise_bucket.json

python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/predict_ebqa.py \
  --model_dir "/data/ocean/DAPT/runs/ebqa_noise_bucket/best" \
  --tokenizer $TOKENIZER_PATH \
  --data_path "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_eval_real.jsonl" \
  --output_preds "/data/ocean/DAPT/runs/ebqa_noise_bucket_preds.jsonl"

# 8.2 Linear
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/train_ebqa.py \
  --config /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/ebqa_config_noise_linear.json

python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/predict_ebqa.py \
  --model_dir "/data/ocean/DAPT/runs/ebqa_noise_linear/best" \
  --tokenizer $TOKENIZER_PATH \
  --data_path "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_eval_real.jsonl" \
  --output_preds "/data/ocean/DAPT/runs/ebqa_noise_linear_preds.jsonl"

# 8.3 MLP
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/train_ebqa.py \
  --config /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/ebqa_config_noise_mlp.json

python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/predict_ebqa.py \
  --model_dir "/data/ocean/DAPT/runs/ebqa_noise_mlp/best" \
  --tokenizer $TOKENIZER_PATH \
  --data_path "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_eval_real.jsonl" \
  --output_preds "/data/ocean/DAPT/runs/ebqa_noise_mlp_preds.jsonl"
```

后续 `aggregate_qa_preds_to_doc.py`、`preprocess_ebqa_real_h200.py` 与 `scorer.py` 的流程保持不变，只需把输出前缀替换为 `ebqa_noise_bucket` / `ebqa_noise_linear` / `ebqa_noise_mlp`。

---

#### 实验 9: KV-NSP 负样本比例消融（reverse/random）

为了评估 KV-NSP 负样本策略比例对 Task2 的影响，新增三组预训练模型：

- ratio 1:1: `/data/ocean/DAPT/workspace/output_ablation_nsp_ratio_1_1/final_staged_model`
- ratio 3:1: `/data/ocean/DAPT/workspace/output_ablation_nsp_ratio_3_1/final_staged_model`
- ratio 1:3: `/data/ocean/DAPT/workspace/output_ablation_nsp_ratio_1_3/final_staged_model`

建议使用统一脚本直接完成 Task2 的数据转换+微调+推理+评测：

```bash
cd /data/ocean/DAPT
GPU_LIST=0,1,2 PARALLEL=1 RESUME=1 \
bash experiments/nsp_ratio_ablation/run_nsp_ratio_ebqa_all.sh
```

该脚本会自动：
- 按 ratio 生成 EBQA 运行配置并指向对应预训练模型。
- 转换 train/eval 为 EBQA JSONL。
- 训练、推理、doc 聚合、对齐、Task2 打分。

输出报告：
- `/data/ocean/DAPT/runs/nsp_ratio_1_1_report_task2.json`
- `/data/ocean/DAPT/runs/nsp_ratio_3_1_report_task2.json`
- `/data/ocean/DAPT/runs/nsp_ratio_1_3_report_task2.json`

若只想跑单组（例如 3:1）：

```bash
cd /data/ocean/DAPT
VARIANTS=r31 GPU_LIST=0 PARALLEL=0 RESUME=1 \
bash experiments/nsp_ratio_ablation/run_nsp_ratio_ebqa_all.sh
```
