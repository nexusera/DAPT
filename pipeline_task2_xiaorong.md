以下是task2完整的操作流程，以 MacBERT (实验 3) 为例：

第一步：准备数据 (Format Conversion)
我们需要先将 KV-NER 格式的数据转换为 QA 格式。
# 环境变量
export PYTHONPATH=$PYTHONPATH:/data/ocean/DAPT
QUERY_SET="/data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/keys_merged_1027_cleaned.json"
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
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/preprocess_ebqa_real_h200.py \
  --gt_file /data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \
  --pred_file /data/ocean/DAPT/runs/ebqa_macbert_doc_preds.jsonl \
  --output_dir aligned_data

# (可选) 自检：aligned_data 下 pred/gt 行数必须一致
wc -l aligned_data/gt_ebqa_aligned.jsonl aligned_data/aligned_ebqa_macbert_doc_preds.jsonl

# 3.4 运行新版 MedStruct-S-master Scorer (Task 2)
# 注意：新版 scorer.py 依赖当前工作目录导入 med_eval，务必先 cd 到工具包目录。
cd /data/ocean/DAPT/dapt_eval_package/MedStruct-S-master
python scorer.py \
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
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/preprocess_ebqa_real_h200.py \
  --gt_file /data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \
  --pred_file /data/ocean/DAPT/runs/ebqa_no_nsp_doc_preds.jsonl \
  --output_dir aligned_data

# 5.7 运行 Scorer (Task 2)
cd /data/ocean/DAPT/dapt_eval_package/MedStruct-S-master
python scorer.py \
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
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/preprocess_ebqa_real_h200.py \
  --gt_file /data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \
  --pred_file /data/ocean/DAPT/runs/ebqa_no_mlm_doc_preds.jsonl \
  --output_dir aligned_data

# 6.7 运行 Scorer (Task 2)
cd /data/ocean/DAPT/dapt_eval_package/MedStruct-S-master
python scorer.py \
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
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/preprocess_ebqa_real_h200.py \
  --gt_file /data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \
  --pred_file /data/ocean/DAPT/runs/ebqa_no_noise_doc_preds.jsonl \
  --output_dir aligned_data

# 7.7 运行 Scorer (Task 2)
cd /data/ocean/DAPT/dapt_eval_package/MedStruct-S-master
python scorer.py \
  --pred_file "/data/ocean/DAPT/aligned_data/aligned_ebqa_no_noise_doc_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/aligned_data/gt_ebqa_aligned.jsonl" \
  --query_set $QUERY_SET \
  --task_type task2 \
  --output_file "/data/ocean/DAPT/runs/ebqa_no_noise_report_task2.json"