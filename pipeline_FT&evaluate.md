# Fine-tune & Evaluate Pipeline (Task1/2 KV-NER, Task3 EBQA)

本文件汇总当前使用的两套下游微调与评测流程（基于 DAPT 噪声模型），包含训练脚本、数据准备、评测脚本与示例命令，来自近期对话确认的实际路径与配置。

## 前置条件
- 进入工程根目录：`cd /data/ocean/DAPT`
cd /data/ocean/DAPT
conda activate medical_bert   
- 已安装依赖：`pip install -r requirements.txt`
- 预训练模型与分词器：
  - 模型：`/data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu_noise_v2/final_model`
  - Tokenizer：`/data/ocean/DAPT/my-medical-tokenizer`
  - 噪声分桶：`/data/ocean/DAPT/workspace/noise_bins.json`

## Task1/2：KV-NER（序列标注）
### 训练
- 脚本：`pre_struct/kv_ner/train_with_noise.py`
- 配置：`pre_struct/kv_ner/kv_ner_config.json`（含 train/val 路径、超参）
- 命令示例：
```bash
cd /data/ocean/DAPT
python dapt_eval_package/pre_struct/kv_ner/train_with_noise.py \
  --config dapt_eval_package/pre_struct/kv_ner/kv_ner_config.json \
  --noise_bins /data/ocean/DAPT/workspace/noise_bins.json \
  --pretrained_model /data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu_noise_v2/final_model
```
- 产物：默认输出到配置里的 `output_dir`（例如 `runs/kv_ner_finetuned`），最佳权重在 `.../best`。

### 推理 （Task1/2）
python dapt_eval_package/pre_struct/kv_ner/compare_models.py \
  --ner_config dapt_eval_package/pre_struct/kv_ner/kv_ner_config.json \
  --test_data biaozhu_with_ocr_noise_prepared/test_eval.jsonl \
  --noise_bins workspace/noise_bins.json \
  --output_summary runs/unified_eval_dapt_prepared.json

### 评测 （Task1/2）

python experiments/scorer.py \
  --pred_file runs/unified_eval_dapt_prepared_preds.jsonl \
  --gt_file biaozhu_with_ocr_noise_prepared/test_eval.jsonl \
  --output_file runs/unified_eval_dapt_prepared_scores.json

### 评测（Task1/2 指标）
- 脚本：`pre_struct/kv_ner/evaluate_with_dapt_noise.py`
- 测试集：`data/kv_ner_prepared_comparison/val_eval_titled.jsonl`
- 命令示例（评估 DAPT/微调模型）：
```bash
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
  --model_path runs/kv_ner_finetuned/best \
  --test_data data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
  --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
  --output_summary runs/eval_kvner.json
```
- 产物：`runs/eval_kvner.json`（含 Task1/2 F1），`runs/eval_kvner_preds.jsonl`（预测对）。
- 如需对比多模型可用：`pre_struct/kv_ner/compare_models.py` 或 `experiments/scorer.py`。

## Task3：EBQA（抽取式问答）
### 训练
- 脚本：`pre_struct/ebqa/train_ebqa.py`
- 配置：`pre_struct/ebqa/ebqa_config.json`
  - 关键字段：
    - `model_name_or_path`: `/data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu_noise_v2/final_model`
    - `tokenizer_name_or_path`: `/data/ocean/DAPT/my-medical-tokenizer`
    - `output_dir`: `runs/ebqa_dapt_noise`（最佳在 `best/`）
    - 训练数据：`data/kv_ner_prepared_comparison/ebqa_train.jsonl`
- 命令示例：
```bash
CUDA_VISIBLE_DEVICES=2,3 \
HF_TOKENIZER_NAME=/data/ocean/DAPT/my-medical-tokenizer \
python pre_struct/ebqa/train_ebqa.py \
  --config_file pre_struct/ebqa/ebqa_config.json
```
- 产物：`runs/ebqa_dapt_noise/best`（QA checkpoint），`metrics_history.json` 等。

### 评测 / 推理（Task3 指标）
1) **准备评测集（预计算 JSONL）**
- 脚本：`pre_struct/ebqa/convert_ebqa.py`
- 命令示例：
```bash
python pre_struct/ebqa/convert_ebqa.py \
  --input_file data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
  --struct_path data/kv_ner_prepared_comparison/keys_v2.json \
  --tokenizer_name /data/ocean/DAPT/my-medical-tokenizer \
  --noise_bins /data/ocean/DAPT/workspace/noise_bins.json \
  --output_file data/kv_ner_prepared_comparison/ebqa_eval.jsonl
```

2) **推理与可选打分（EM/F1）**
- 脚本：`pre_struct/ebqa/predict_ebqa.py`
- 命令示例：
```bash
CUDA_VISIBLE_DEVICES=2 \
python pre_struct/ebqa/predict_ebqa.py \
  --model_dir runs/ebqa_dapt_noise/best \
  --tokenizer /data/ocean/DAPT/my-medical-tokenizer \
  --data_path data/kv_ner_prepared_comparison/ebqa_eval.jsonl \
  --output_preds runs/ebqa_eval_preds.jsonl \
  --batch_size 4
```
- 产物：
  - 预测：`runs/ebqa_eval_preds.jsonl`
  - 若评测集含 `answer_text`，将输出 `runs/ebqa_eval_preds.summary.json`（EM/F1 均值）。

## 备注与路径对齐
- 两套任务可共享同一预训练模型与 tokenizer，但需分别微调得到各自下游模型。
- 噪声特征：KV-NER 评测需提供 `--noise_bins_json`；EBQA 转换/训练/推理需保持与训练一致的 tokenizer 与 noise_bins。
- 评测指标：
  - Task1/2：核心计算在 `core/metrics.py`，提供 Strict/Loose；脚本会输出 JSON 与 preds.jsonl。
  - Task3：`predict_ebqa.py` 输出预测；若有 GT 可用 `core/metrics.py` 的 Task3 计算或 `compare_models.py`/`experiments/scorer.py` 汇总。
