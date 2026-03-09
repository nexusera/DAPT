# 下游微调 + 评测（一键跑 4 个 tokenizer 变体，覆盖 Task1/3 与 Task2）

适用前提：你已经完成 tokenizer ablation 的 4 个 full 预训练（T1~T4），并且每个 run 目录下存在：

- `${OUT_ROOT}/runs/t1_full_seed42/final_staged_model/`
- `${OUT_ROOT}/runs/t2_full_seed42/final_staged_model/`
- `${OUT_ROOT}/runs/t3_full_seed42/final_staged_model/`
- `${OUT_ROOT}/runs/t4_full_seed42/final_staged_model/`

本目录提供：
- 一键脚本：`run_downstream_all.sh`（自动产出 8 个微调模型 + 评测报告）
- 配置生成器：`gen_downstream_configs.py`（自动生成 8 份独立 config，避免覆盖）

## 输出是什么（你会得到哪些模型/报告）

### 1) Task1/3：KV-NER 微调（每个变体 1 个微调模型）
- 微调模型（best）：`/data/ocean/DAPT/runs/kv_ner_finetuned_t{n}/best/`
- 评测报告：
  - Task1：`/data/ocean/DAPT/runs/t{n}_report_task1.json`
  - Task3：`/data/ocean/DAPT/runs/t{n}_report_task3.json`

### 2) Task2：EBQA 微调（每个变体 1 个微调模型）
- 微调模型（best）：`/data/ocean/DAPT/runs/ebqa_t{n}/best/`
- 评测报告：`/data/ocean/DAPT/runs/t{n}_report_task2.json`

> 说明：Task1/3 使用同一个 KV-NER 微调模型；Task2 使用 EBQA 微调模型。所以总计 4×2=8 个微调模型目录（best）。

## 一键运行（推荐）

在服务器上执行：

```bash
cd /data/ocean/DAPT
conda activate medical_bert

# 说明：KV-NER 的推理/对比脚本依赖 torchcrf（一般已在 medical_bert 环境里）。
# 如果你没激活该环境，可能会报：ModuleNotFoundError: No module named 'torchcrf'

# 你 tokenizer ablation 的输出根目录（与 tokenizer_ablation/README.md 一致）
export OUT_ROOT=/data/ocean/DAPT/ablation/tokenizer

# 可选：如果你不想用默认值，可覆盖这些路径
export NOISE_BINS=/data/ocean/DAPT/workspace/noise_bins.json
export QUERY_SET=/data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/keys_merged_1027_cleaned.json
export REAL_TRAIN_JSON=/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json
export REAL_TEST_JSON=/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json

# 可选：指定 seed（用于拼出 t1_full_seed42 这类目录名）
export SEED=42

# 可选：并行跑 4 个变体时，每个变体绑定一张卡（默认：0,1,4,5）
# 你现在正好是 0/1/4/5 空闲，所以可以不设；想改就覆盖这个变量。
export GPU_LIST=0,1,4,5

# 可选：断点续跑（默认开启）。如果之前已经产出了某些中间文件/报告，会自动跳过这些步骤。
# 想强制从头重跑就设为 0。
export RESUME=1

bash /data/ocean/DAPT/experiments/downstream_finetune/run_downstream_all.sh
```

运行过程中所有日志在：`/data/ocean/DAPT/runs/downstream_logs/`。

## 一键汇总 12 个 JSON 指标（生成表格）

当你已经生成了 `t1~t4` 的 task1/task2/task3 报告后，可以用下面命令把关键 F1 抽出来，直接输出 Markdown 表格（可粘贴到论文/报告）：

```bash
python /data/ocean/DAPT/experiments/downstream_finetune/summarize_ablation_reports.py \
  --runs_dir /data/ocean/DAPT/runs \
  --variants t1 t2 t3 t4
```

## 如果你只想看“命令清单”（不用一键脚本）

一键脚本内部实际执行的命令就是：

### A. Task1/3（KV-NER）每个变体 v∈{t1,t2,t3,t4}

1) 生成 config（一次即可）：
```bash
python /data/ocean/DAPT/experiments/downstream_finetune/gen_downstream_configs.py \
  --dapt_root /data/ocean/DAPT \
  --out_root ${OUT_ROOT} \
  --seed 42 \
  --query_set ${QUERY_SET} \
  --output_dir /data/ocean/DAPT/experiments/downstream_finetune/generated_configs
```

2) 训练：
```bash
python /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/train_with_noise.py \
  --config /data/ocean/DAPT/experiments/downstream_finetune/generated_configs/kv_ner_config_${v}.json \
  --noise_bins ${NOISE_BINS}
```

3) 推理（产出统一格式 preds/gt）：
```bash
python /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/compare_models.py \
  --ner_config /data/ocean/DAPT/experiments/downstream_finetune/generated_configs/kv_ner_config_${v}.json \
  --keys_file ${QUERY_SET} \
  --test_data ${REAL_TEST_JSON} \
  --noise_bins ${NOISE_BINS} \
  --output_summary /data/ocean/DAPT/runs/${v}_kvner_eval_summary.json
```

4) 对齐 span（用于 scorer）：
```bash
python /data/ocean/DAPT/scripts/align_for_scorer_span.py \
  --gt_in ${REAL_TEST_JSON} \
  --pred_in /data/ocean/DAPT/runs/${v}_kvner_eval_summary_preds.jsonl \
  --gt_out /data/ocean/DAPT/runs/${v}_task13_aligned_gt.jsonl \
  --pred_out /data/ocean/DAPT/runs/${v}_task13_aligned_preds.jsonl
```

5) Task1 评测：
```bash
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py \
  --pred_file /data/ocean/DAPT/runs/${v}_task13_aligned_preds.jsonl \
  --gt_file /data/ocean/DAPT/runs/${v}_task13_aligned_gt.jsonl \
  --schema_file ${QUERY_SET} \
  --task_type task1 \
  --overlap_threshold -1 \
  --output_file /data/ocean/DAPT/runs/${v}_report_task1.json
```

6) Task3 评测：
```bash
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py \
  --pred_file /data/ocean/DAPT/runs/${v}_task13_aligned_preds.jsonl \
  --gt_file /data/ocean/DAPT/runs/${v}_task13_aligned_gt.jsonl \
  --schema_file ${QUERY_SET} \
  --task_type task3 \
  --overlap_threshold -1 \
  --output_file /data/ocean/DAPT/runs/${v}_report_task3.json
```

### B. Task2（EBQA）每个变体 v∈{t1,t2,t3,t4}

1) 先把 KV-NER 格式转换为 EBQA JSONL（注意：该步骤与 tokenizer 强相关，所以每个变体各做一份）：
```bash
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py \
  --input_file ${REAL_TRAIN_JSON} \
  --output_file /data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_train_real_${v}.jsonl \
  --struct_path ${QUERY_SET} \
  --tokenizer_name ${OUT_ROOT}/runs/${v}_full_seed42/final_staged_model \
  --noise_bins ${NOISE_BINS}

python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py \
  --input_file ${REAL_TEST_JSON} \
  --output_file /data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_eval_real_${v}.jsonl \
  --struct_path ${QUERY_SET} \
  --tokenizer_name ${OUT_ROOT}/runs/${v}_full_seed42/final_staged_model \
  --noise_bins ${NOISE_BINS}
```

2) 微调：
```bash
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/train_ebqa.py \
  --config /data/ocean/DAPT/experiments/downstream_finetune/generated_configs/ebqa_config_${v}.json
```

3) 推理：
```bash
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/predict_ebqa.py \
  --model_dir /data/ocean/DAPT/runs/ebqa_${v}/best \
  --tokenizer ${OUT_ROOT}/runs/${v}_full_seed42/final_staged_model \
  --data_path /data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_eval_real_${v}.jsonl \
  --output_preds /data/ocean/DAPT/runs/ebqa_${v}_preds.jsonl
```

4) QA 级 -> 文档级：
```bash
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/aggregate_qa_preds_to_doc.py \
  --raw_file ${REAL_TEST_JSON} \
  --qa_pred_file /data/ocean/DAPT/runs/ebqa_${v}_preds.jsonl \
  --output_file /data/ocean/DAPT/runs/ebqa_${v}_doc_preds.jsonl \
  --prefer score
```

5) 对齐（task2 scorer 需要 pred/gt 行数一致）：
```bash
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/preprocess_ebqa_real_h200.py \
  --gt_file ${REAL_TEST_JSON} \
  --pred_file /data/ocean/DAPT/runs/ebqa_${v}_doc_preds.jsonl \
  --output_dir /data/ocean/DAPT/runs/ebqa_${v}_aligned
```

6) Task2 评测（注意要先 cd 到 MedStruct-S-master）：
```bash
cd /data/ocean/DAPT/dapt_eval_package/MedStruct-S-master
python scorer.py \
  --pred_file /data/ocean/DAPT/runs/ebqa_${v}_aligned/aligned_ebqa_${v}_doc_preds.jsonl \
  --gt_file /data/ocean/DAPT/runs/ebqa_${v}_aligned/gt_ebqa_aligned.jsonl \
  --query_set ${QUERY_SET} \
  --task_type task2 \
  --output_file /data/ocean/DAPT/runs/${v}_report_task2.json
```
