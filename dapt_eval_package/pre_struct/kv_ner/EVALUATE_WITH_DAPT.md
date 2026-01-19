# DAPT 模型评估流程 (Task 1 & 2)

本文档详细说明如何使用新的evaluate脚本对DAPT预训练模型进行KV-NER评估。

## 背景

DAPT(Domain-Adaptive Pretraining)模型引入了7维分桶噪声特征(noise embedding)，这改变了模型的输入格式：

- **标准BERT**: `input_ids` → `embeddings` → 模型处理
- **DAPT模型**: `input_ids` + `noise_ids` → `embeddings` → 模型处理

其中`noise_ids`是形状为`[batch_size, seq_len, 7]`的张量，每个位置有7维离散桶ID，对应：
1. `conf_avg` - 置信度平均值 (64桶)
2. `conf_min` - 置信度最小值 (64桶)
3. `conf_var_log` - 置信度方差对数 (32桶)
4. `conf_gap` - 置信度差值 (32桶)
5. `punct_err_ratio` - 标点错误比例 (16桶)
6. `char_break_ratio` - 字符断裂比例 (32桶)
7. `align_score` - 对齐分数 (64桶)

新的evaluate脚本自动处理这一差异，使您可以用统一的命令评估两种模型。

---

## 快速开始

### 1. 评估标准BERT模型 (无DAPT特性)

```bash
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path hfl/chinese-roberta-wwm-ext \
    --test_data data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
    --output_summary runs/eval_roberta.json \
    --batch_size 32
```

此时脚本将自动检测模型不含noise embeddings，使用标准推理流程。

### 2. 评估DAPT预训练模型 (含噪声特性)

```bash
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path /path/to/dapt/checkpoint \
    --test_data data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
    --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
    --output_summary runs/eval_dapt.json \
    --batch_size 32
```

关键参数：
- `--noise_bins_json`: 必需，指向DAPT预训练阶段生成的分桶边界JSON文件
- 其他参数与标准BERT评估相同

---

## 完整参数说明

```
usage: evaluate_with_dapt_noise.py [-h] 
    --model_path MODEL_PATH 
    --test_data TEST_DATA 
    [--noise_bins_json NOISE_BINS_JSON]
    [--output_summary OUTPUT_SUMMARY] 
    [--max_length MAX_LENGTH]
    [--batch_size BATCH_SIZE] 
    [--device DEVICE] 
    [--seed SEED]

必需参数:
  --model_path MODEL_PATH
    模型路径，可以是：
    - HuggingFace模型ID (如 hfl/chinese-roberta-wwm-ext)
    - 本地checkpoint路径 (如 /path/to/dapt/checkpoint)
    - 微调后的KV-NER模型路径

  --test_data TEST_DATA
    测试数据JSONL文件路径
    必须包含字段：'text' 和 'spans' (GT标签)
    示例：data/kv_ner_prepared_comparison/val_eval_titled.jsonl

可选参数:
  --noise_bins_json NOISE_BINS_JSON
    噪声分桶边界JSON文件 (必需用于DAPT模型)
    示例：/data/ocean/DAPT/workspace/noise_bins.json
    不提供此参数时，脚本自动使用标准BERT推理

  --output_summary OUTPUT_SUMMARY
    输出评估结果JSON文件路径
    示例：runs/eval_dapt.json
    脚本同时生成 eval_dapt_preds.jsonl (详细预测结果)
    
  --max_length MAX_LENGTH
    最大序列长度 (默认: 512)
    应与模型训练时的max_length一致

  --batch_size BATCH_SIZE
    推理批大小 (默认: 32)
    越大越快，但占用更多显存

  --device DEVICE
    计算设备 (默认: cuda if available else cpu)
    示例：cuda, cuda:0, cpu

  --seed SEED
    随机种子，用于可复现性 (默认: None)
```

---

## 输出格式

### 1. 主结果文件: `eval_dapt.json`

```json
{
  "model": "/path/to/dapt/checkpoint",
  "is_dapt": true,
  "num_samples": 100,
  "task1": {
    "strict": {
      "p": 0.8234,
      "r": 0.7921,
      "f1": 0.8075
    },
    "loose": {
      "p": 0.8456,
      "r": 0.8123,
      "f1": 0.8287
    }
  },
  "task2": {
    "strict_strict": {
      "p": 0.7654,
      "r": 0.7234,
      "f1": 0.7439
    },
    "strict_loose": {
      "p": 0.8123,
      "r": 0.7823,
      "f1": 0.7970
    },
    "loose_loose": {
      "p": 0.8345,
      "r": 0.8045,
      "f1": 0.8192
    }
  }
}
```

指标说明：
- **Task 1 (Keys Discovery)**:
  - Strict: 精确匹配，不包容任何差异
  - Loose: 基于NED(Normalized Edit Distance)的相似度匹配
  
- **Task 2 (Key-Value Pairs)**:
  - Strict-Strict: Key和Value均精确匹配
  - Strict-Loose: Key精确，Value基于相似度
  - Loose-Loose: Key和Value均基于相似度

### 2. 详细预测文件: `eval_dapt_preds.jsonl`

```jsonl
{"pred_pairs": [["患者性别", "女"], ["年龄", "45岁"], ...]}
{"pred_pairs": [["诊断", "肺癌"], ...]}
...
```

每一行对应一个样本的预测结果，`pred_pairs`是`[key, value]`对的列表。

---

## 评估指标解释

### Task 1: Key Discovery (属性发现)

评估模型能否正确识别病历中包含的所有属性名。

- **Precision**: 预测的属性中有多少是正确的
- **Recall**: 实际属性中有多少被正确预测
- **F1**: P和R的调和均值

### Task 2: Key-Value Pair Matching (键值对提取)

评估模型能否正确提取完整的键值对。

三种匹配策略：
1. **Strict-Strict**: 最严格，Key完全匹配且Value完全相同
2. **Strict-Loose**: Key完全匹配，Value相似即可 (基于字符级F1)
3. **Loose-Loose**: Key和Value均允许相似匹配

---

## 常见使用场景

### 场景 A: 对比标准BERT和DAPT的性能

```bash
# 1. 评估标准BERT
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path hfl/chinese-roberta-wwm-ext \
    --test_data data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
    --output_summary runs/eval_roberta.json

# 2. 评估DAPT模型
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path /data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu/final_model \
    --test_data data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
    --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
    --output_summary runs/eval_dapt.json

# 3. 对比结果
python scripts/compare_eval_results.py \
    --baseline runs/eval_roberta.json \
    --dapt runs/eval_dapt.json
```

### 场景 B: 微调DAPT模型后评估

```bash
# 1. 在特定任务上微调DAPT模型
python pre_struct/kv_ner/train.py \
    --model_name_or_path /data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu/final_model \
    --train_config pre_struct/kv_ner/kv_ner_config_dapt.json \
    --output_dir runs/dapt_finetuned_kv_ner \
    --num_train_epochs 3

# 2. 评估微调后的模型
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path runs/dapt_finetuned_kv_ner \
    --test_data data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
    --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
    --output_summary runs/eval_dapt_finetuned.json
```

### 场景 C: 批量评估多个模型

```bash
#!/bin/bash
# eval_all_models.sh

models=(
    "hfl/chinese-roberta-wwm-ext"
    "hfl/chinese-macbert-base"
    "/data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu/final_model"
)

noise_bins="/data/ocean/DAPT/workspace/noise_bins.json"
test_data="data/kv_ner_prepared_comparison/val_eval_titled.jsonl"
output_dir="runs"

for model in "${models[@]}"; do
    model_name=$(basename "$model")
    
    # 判断是否需要噪声特征
    if [[ "$model" == *"DAPT"* ]] || [[ "$model" == *"medical_bert"* ]]; then
        python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
            --model_path "$model" \
            --test_data "$test_data" \
            --noise_bins_json "$noise_bins" \
            --output_summary "${output_dir}/eval_${model_name}.json"
    else
        python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
            --model_path "$model" \
            --test_data "$test_data" \
            --output_summary "${output_dir}/eval_${model_name}.json"
    fi
done
```

---

## 技术细节

### Noise Feature Processing Pipeline

```
数据预处理 (add_noise_features.py)
    ↓
OCR样本: 提取7维连续值 (noise_values)
非OCR样本: 填充完美物理值 [1.0, 1.0, 0, 0, 0, 0, 0]
    ↓
保存到数据集
    ↓
推理时 (evaluate_with_dapt_noise.py)
    ↓
NoiseCollator处理
    ↓
加载noise_values
    ↓
使用NoiseFeatureProcessor映射为离散桶ID
    ↓
生成noise_ids张量 [batch, seq_len, 7]
    ↓
模型接收: input_ids + noise_ids
    ↓
RobertaNoiseEmbeddings处理
    ↓
融合到文本embedding
```

### 模型检测机制

脚本自动检测模型是否为DAPT：

```python
if hasattr(model.bert, 'embeddings'):
    emb_class_name = model.bert.embeddings.__class__.__name__
    is_dapt = 'Noise' in emb_class_name
```

- 如果检测到`RobertaNoiseEmbeddings`，自动启用noise处理
- 如果不提供`--noise_bins_json`，自动降级为标准BERT推理

---

## 故障排查

### 问题1: 模型加载失败

```
Error: Failed to load model: ...
```

**解决方案**:
- 确保模型路径正确
- 确保模型是从HuggingFace或本地checkpoint保存的
- 检查transformers版本是否兼容

### 问题2: Noise bins JSON不存在

```
FileNotFoundError: Bins JSON not found: ...
```

**解决方案**:
- 确保路径正确
- 确保DAPT预训练时已生成了noise_bins.json
- 如果路径中有空格，用引号括起来

### 问题3: CUDA内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案**:
- 减小`--batch_size`（如从32改为16）
- 减小`--max_length`（如从512改为256）
- 使用`--device cpu`在CPU上推理

### 问题4: 预测文件过大

如果生成的`_preds.jsonl`文件过大，可以：
- 减少测试样本数
- 后处理时使用流式读取而非一次性加载

---

## 与原有流程的比较

| 方面 | 原有流程 | 新流程 |
|------|--------|--------|
| 支持模型 | 标准BERT | 标准BERT + DAPT |
| Noise处理 | 不支持 | 自动处理 |
| 脚本复杂度 | 低（仅evaluate.py) | 中（评估 + noise处理） |
| 灵活性 | 固定 | 自动检测 |
| 性能 | - | DAPT有显著提升 |
| 学习成本 | 低 | 低（参数兼容） |

---

## 后续集成

### 与scorer.py的关系

新的evaluate脚本只负责推理和即时计算F1指标。如果需要与其他组件集成：

1. **保存预测结果**: 脚本自动保存`_preds.jsonl`
2. **使用scorer.py**: 可以将预测结果传给原有的`experiments/scorer.py`
3. **对比分析**: 使用预测结果文件进行深度分析

### 扩展到Task 3 (EBQA)

当前版本只支持Task 1&2。若要支持Task 3：
- 需要额外的schema匹配逻辑
- 需要问题生成或检索模块
- 预计需要100+行额外代码

---

## 联系与支持

如有问题或建议，请参考：
- DAPT文档: `DAPT/pipeline_new.md`
- 原有评估文档: `BERT-xxx/README_PIPELINE.md`
- 模型代码: `pre_struct/kv_ner/modeling.py`
