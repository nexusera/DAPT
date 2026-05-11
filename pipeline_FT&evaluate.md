# Fine-tune & Evaluate Pipeline (Task1/2 KV-NER, Task3 EBQA)

本文件汇总当前使用的两套下游微调与评测流程（基于 DAPT 噪声模型），包含训练脚本、数据准备、评测脚本与示例命令，来自近期对话确认的实际路径与配置。

kv-masking方法得到的预训练模型：/data/ocean/DAPT/hybrid_dapt_output/final_hybrid_span_model
强行让robaerta塞入segment embedding，进行分阶段训练的模型：./staged_dapt_output_fixed_v2/final_staged_model
macbert:./macbert_staged_output/final_staged_model
多任务模型：/data/ocean/DAPT/workspace/output_medical_mtl_v1/

旧的训练/测试集：
训练集：/data/ocean/DAPT/biaozhu_with_ocr/merged_train_with_ocr.json
测试集：/data/ocean/DAPT/biaozhu_with_ocr/merged_eval_with_ocr.json 

新的训练/测试集：
训练集 (Train):
/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json
测试集 (Test):
/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json

cd /data/ocean/DAPT
conda activate medical_bert
export PYTHONPATH=$PYTHONPATH:$(pwd)
# 确保使用正确的 Tokenizer 和 噪声配置
NOISE_BINS="/data/ocean/DAPT/workspace/noise_bins.json"
TEST_DATA="biaozhu_with_ocr_noise_prepared/test_eval.jsonl"
GT_FILE="biaozhu_with_ocr_noise_prepared/test_eval.jsonl"

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

## 🧪 消融实验方法论 (Ablation Methodology)

当你有一个新的预训练模型（例如 `output_medical_mtl_v1`）并希望进行下游 KV-NER 的全流程测验时，请遵循以下 SOP：

### 1. 配置准备 (Configuration)
1.  **复制配置文件**：不要直接修改现有配置，建议复制一份 `kv_ner_config.json` 或 `kv_ner_multitask_config.json`。
2.  **修改关键参数**：
    *   `model_name_or_path`: 替换为新模型的绝对路径（如 `/data/ocean/DAPT/workspace/output_medical_mtl_v1/`）。
    *   `output_dir`: 修改为你希望保存微调权重的地方（如 `runs/kv_ner_finetuned_mtl`），避免覆盖其他实验。

### 2. 微调 (Fine-tuning)
使用新配置运行 `train_with_noise.py`。
> **注意**：如果新模型使用了不同的 Tokenizer，请确保 `--noise_bins` 如果依赖 Tokenizer 必须重新生成或匹配。如果没变则复用。

### 3. 推理 (Inference)
使用 `compare_models.py` 进行推理生成预测文件。
*   你需要指定 `runs/kv_ner_finetuned_mtl/best/` 作为权重的隐含来源（脚本通常会自动加载 config 里的 output_dir 或通过 `--ner_config` 指定）。
*   但更推荐直接用 `compare_models.py` 显式加载：
    *   它会读取 config 里的 `output_dir` 找到 `best` 模型进行预测。
    *   确保 `--output_summary` 指向一个新的结果文件（如 `runs/unified_eval_mtl.json`），这会自动生成 `runs/unified_eval_mtl_preds.jsonl`。

### 4. 评测 (Evaluation)
使用 `scorer.py` 对生成的 `preds.jsonl` 进行打分。
*   `--pred_file`: 指向第 3 步生成的 `_preds.jsonl`。
*   `--gt_file`: 保持不变 (`test_eval.jsonl`)。
*   `--output_file`: 指定一个新的分数文件（如 `runs/unified_eval_mtl_scores.json`）。

---
### 实战命令速查 (MTL 实验)

```bash
# 1. 训练
python dapt_eval_package/pre_struct/kv_ner/train_with_noise.py \
  --config dapt_eval_package/pre_struct/kv_ner/kv_ner_multitask_config.json \
  --noise_bins /data/ocean/DAPT/workspace/noise_bins.json \
  --pretrained_model /data/ocean/DAPT/workspace/output_medical_mtl_v1/

# 2. 推理 (生成 unified_eval_mtl_preds.jsonl)
python dapt_eval_package/pre_struct/kv_ner/compare_models.py \
  --ner_config dapt_eval_package/pre_struct/kv_ner/kv_ner_multitask_config.json \
  --test_data biaozhu_with_ocr_noise_prepared/test_eval.jsonl \
  --noise_bins workspace/noise_bins.json \
  --output_summary runs/unified_eval_mtl.json

# 3. 评测
PYTHONPATH=/data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-master:/data/ocean/DAPT/dapt_eval_package:/data/ocean/DAPT:$PYTHONPATH \
python dapt_eval_package/MedStruct-S-Benchmark-master/scorer.py \
  --pred_file runs/unified_eval_mtl_preds.jsonl \
  --gt_file biaozhu_with_ocr_noise_prepared/test_eval.jsonl \
  --schema_file data/kv_ner_prepared_comparison/keys_v2.json \
  --output_file runs/unified_eval_mtl_scores.json
```
