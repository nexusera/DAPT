# DAPT 噪声版 KV-NER / EBQA 评估快速指南

面向同事的最简使用说明，帮助直接复现带噪声 DAPT 模型的评估流程。

## 环境与路径
- 进入工程根目录：`cd /Users/shanqi/Documents/BERT_DAPT/DAPT`
- 安装依赖（一次即可）：`pip install -r requirements.txt`
- 关键脚本：`pre_struct/kv_ner/evaluate_with_dapt_noise.py`
- 示例数据（KV-NER）：`data/kv_ner_prepared_comparison/val_eval_titled.jsonl`
- 噪声分桶文件：`/data/ocean/DAPT/workspace/noise_bins.json`（DAPT 模型必需）

## 输入需求
- 模型：
  - 标准 BERT/HF 模型 ID（无需噪声特征）
  - 或 DAPT 噪声模型 checkpoint（需提供 noise_bins）
- 数据：JSONL，至少包含 `text` 和 `spans`；若为 OCR 噪声场景，需有 `noise_values` 字段。
- 设备：默认自动选择 GPU；可用 `--device cuda:0` 指定。

## 最常用命令
### 1) 评估标准 BERT（无噪声）
```bash
action="runs/eval_roberta"
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
  --model_path hfl/chinese-roberta-wwm-ext \
  --test_data data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
  --output_summary ${action}.json
```

### 2) 评估 DAPT 噪声模型
```bash
action="runs/eval_dapt"
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
  --model_path /data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu_noise_v2/final_model \
  --test_data data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
  --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
  --output_summary ${action}.json
```
脚本会自动检测模型是否包含 noise embeddings；未提供 `--noise_bins_json` 时会降级为无噪声推理。

### 3) 批量对比多个模型（示例）
```bash
bash pre_struct/kv_ner/evaluate_dapt_examples.sh
# 自动生成批量评估、结果对比脚本，可按需修改模型列表
```

## 关键参数速查
- `--model_path` (必填) 模型 ID 或本地路径
- `--test_data` (必填) 测试集 JSONL 路径
- `--noise_bins_json` DAPT 噪声分桶 JSON（DAPT 模型必填）
- `--output_summary` 评估结果 JSON 输出路径（同时生成同名 `_preds.jsonl`）
- `--batch_size` 默认 32；显存不足可调小
- `--max_length` 默认 512；与训练时保持一致
- `--device` 默认自动；示例：`cuda:0`、`cpu`
- `--seed` 设定随机种子保证可复现

## 输出文件
- `<output_summary>.json`：主结果，含 Task1/Task2 的 P/R/F1。
- `<output_summary>_preds.jsonl`：逐样本预测的键值对列表，可用于对比或误差分析。

## 常见问题
- **找不到 noise_bins.json**：确认路径 `/data/ocean/DAPT/workspace/noise_bins.json`；若路径含空格请加引号。
- **CUDA OOM**：降低 `--batch_size` 或 `--max_length`，或切换 `--device cpu`。
- **数据缺少 noise_values**：对标准 BERT 无影响；对 DAPT 噪声模型需要提供噪声字段，否则退化为完美值。
- **模型无法加载**：确认 transformers 版本与 checkpoint 匹配，路径可访问。

## 更多资料
- 快速命令与故障排查：`pre_struct/kv_ner/QUICKSTART.md`
- 详细参数与场景说明：`pre_struct/kv_ner/EVALUATE_WITH_DAPT.md`
- 微调/集成噪声的说明：`pre_struct/kv_ner/DAPT_FINETUNING_INTEGRATION.md`
- 项目全貌概览：`DAPT_EVALUATION_SUMMARY.md`
