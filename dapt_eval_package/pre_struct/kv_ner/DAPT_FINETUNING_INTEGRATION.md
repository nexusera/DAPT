# 在KV-NER微调中使用DAPT预训练模型

## 概述

本文档说明如何将DAPT预训练模型集成到现有的KV-NER微调流程中。

### 核心要点

1. **兼容性**: DAPT模型与标准BERT完全兼容，只需增加noise处理
2. **最小改动**: 仅需修改DataLoader的collate_fn和模型前向传播
3. **透明集成**: 一旦微调完成，推理和评估的处理方式相同
4. **性能提升**: DAPT预训练提供更好的医学领域特征

---

## 集成步骤

### 步骤1: 准备DAPT模型和配置

```bash
# 从DAPT预训练输出中获取
DAPT_MODEL_PATH=/data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu/final_model
DAPT_NOISE_BINS=/data/ocean/DAPT/workspace/noise_bins.json

# 验证文件存在
ls -la "$DAPT_MODEL_PATH"
ls -la "$DAPT_NOISE_BINS"
```

### 步骤2: 修改KV-NER训练脚本

在现有的`pre_struct/kv_ner/train.py`中，修改DataLoader创建部分：

```python
# 原有代码
from transformers import default_data_collator
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=default_data_collator,
)

# 改为：
if use_dapt:  # 新增参数
    from pre_struct.kv_ner.noise_utils import NoiseFeatureProcessor, NoiseCollator
    
    processor = NoiseFeatureProcessor.load(noise_bins_json)
    collate_fn = NoiseCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
    )
else:
    collate_fn = default_data_collator

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
```

### 步骤3: 修改模型前向传播

在`pre_struct/kv_ner/modeling.py`中的`BertCrfTokenClassifier.forward()`方法中：

```python
# 原有代码
def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
    # ... (existing code)
    bert_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

# 改为：
def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, 
            noise_ids=None, **kwargs):  # 新增参数
    # ... (existing code)
    bert_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        noise_ids=noise_ids,  # DAPT模型自动处理，标准BERT自动忽略
    )
```

### 步骤4: 准备训练数据

如果使用DAPT训练好的预处理数据集（包含noise_values）：

```bash
# 使用DAPT生成的processed_dataset
# 注意：该数据集已包含noise_values字段
TRAIN_DATA=/data/ocean/DAPT/workspace/processed_dataset_ocr9297_with_noise
```

如果使用现有的KV-NER数据集（不含noise_values）：

```python
# 系统会自动填充完美物理值，不需要修改数据格式
# NoiseCollator会为每个样本自动添加完美值：[1.0, 1.0, 0, 0, 0, 0, 0]
```

### 步骤5: 启动微调

```bash
python pre_struct/kv_ner/train.py \
    --model_name_or_path /data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu/final_model \
    --train_file data/kv_ner/train.jsonl \
    --eval_file data/kv_ner/dev.jsonl \
    --output_dir runs/dapt_kv_ner_finetuned \
    --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \  # 新增参数
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --max_seq_length 512
```

---

## 使用微调后的DAPT-KV-NER模型

### 推理和评估

一旦DAPT模型微调完成，使用统一的evaluate脚本：

```bash
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path runs/dapt_kv_ner_finetuned \
    --test_data data/val_eval_titled.jsonl \
    --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
    --output_summary runs/eval_dapt_finetuned.json
```

### 性能对比

对比DAPT-KV-NER vs 标准BERT-KV-NER：

```bash
# 1. 评估标准BERT微调模型
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path runs/roberta_kv_ner_finetuned \
    --test_data data/val_eval_titled.jsonl \
    --output_summary runs/eval_roberta_finetuned.json

# 2. 评估DAPT微调模型
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path runs/dapt_kv_ner_finetuned \
    --test_data data/val_eval_titled.jsonl \
    --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
    --output_summary runs/eval_dapt_finetuned.json

# 3. 对比结果
python pre_struct/kv_ner/compare_eval_results.py \
    runs/eval_roberta_finetuned.json \
    runs/eval_dapt_finetuned.json
```

---

## 完整的微调示例脚本

### 场景A: 从零开始微调DAPT

```bash
#!/bin/bash
set -e

# 配置
DAPT_MODEL="/data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu/final_model"
NOISE_BINS="/data/ocean/DAPT/workspace/noise_bins.json"
TRAIN_DATA="data/kv_ner_prepared_comparison/train_eval_titled.jsonl"
DEV_DATA="data/kv_ner_prepared_comparison/val_eval_titled.jsonl"
OUTPUT_DIR="runs/dapt_kv_ner_full_finetune"

echo "=================================="
echo "从零开始微调DAPT-KV-NER"
echo "=================================="

# Step 1: 微调
echo ""
echo "[1/3] 开始微调..."
python pre_struct/kv_ner/train.py \
    --model_name_or_path "$DAPT_MODEL" \
    --train_file "$TRAIN_DATA" \
    --eval_file "$DEV_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --noise_bins_json "$NOISE_BINS" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --save_total_limit 2 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --logging_steps 50

# Step 2: 评估
echo ""
echo "[2/3] 评估微调后的模型..."
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path "$OUTPUT_DIR/checkpoint-best" \
    --test_data "$DEV_DATA" \
    --noise_bins_json "$NOISE_BINS" \
    --output_summary "runs/eval_dapt_kv_ner.json"

# Step 3: 对比基线
echo ""
echo "[3/3] 对比评估基线..."
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path hfl/chinese-roberta-wwm-ext \
    --test_data "$DEV_DATA" \
    --output_summary "runs/eval_roberta_baseline.json"

echo ""
echo "=================================="
echo "完成！"
echo "结果:"
echo "- 微调模型: $OUTPUT_DIR"
echo "- DAPT评估: runs/eval_dapt_kv_ner.json"
echo "- BERT基线: runs/eval_roberta_baseline.json"
echo "=================================="
```

### 场景B: 在现有KV-NER模型基础上继续微调

```bash
#!/bin/bash
set -e

# 配置
EXISTING_MODEL="runs/roberta_kv_ner_finetuned"  # 现有的KV-NER模型
DAPT_BACKBONE="/data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu/final_model"
NOISE_BINS="/data/ocean/DAPT/workspace/noise_bins.json"
DATA="data/kv_ner_prepared_comparison/train_eval_titled.jsonl"

echo "从现有KV-NER模型迁移到DAPT"
echo ""
echo "方法1: 重新初始化并微调DAPT（推荐）"
# 使用DAPT backbone重新开始微调，获得最好的效果

echo "方法2: 加载现有权重并冻结某些层（高级）"
# 这需要自定义训练脚本，可根据需要实现
```

---

## 常见问题

### Q1: DAPT微调需要noise_bins_json吗？

**A**: 是的。微调时需要通过NoiseCollator处理noise_ids。但一旦模型保存，noise_bins信息已编码在模型权重中，推理时仍需提供bins信息以正确处理输入。

### Q2: 能否使用非DAPT数据集微调DAPT模型？

**A**: 可以。非DAPT数据没有noise_values时，系统自动填充完美物理值。这样做的缺点是无法利用OCR噪声信息，但仍能使用DAPT预训练的医学特征。

### Q3: DAPT-KV-NER和标准KV-NER的计算成本？

**A**: 基本相同。DAPT只是在embedding阶段多了一个乘法操作（alpha系数），不会显著增加计算时间。

### Q4: 如何评估DAPT对微调的帮助？

**A**: 
```bash
# 1. 用标准BERT微调
python train.py --model_name_or_path hfl/chinese-roberta-wwm-ext ...

# 2. 用DAPT微调
python train.py --model_name_or_path /path/to/dapt --noise_bins_json ...

# 3. 对比评估
python compare_eval_results.py eval_roberta.json eval_dapt.json
```

### Q5: 微调DAPT模型时是否应该调整学习率？

**A**: 建议保持或略微降低学习率（如2e-5改为1e-5），因为DAPT模型已在医学数据上预训练，可能需要较小的学习率以防过度调整。

---

## 集成检查清单

- [ ] DAPT模型路径正确
- [ ] noise_bins.json文件存在
- [ ] 修改train.py支持noise_ids参数
- [ ] 修改modeling.py的forward方法接收noise_ids
- [ ] 创建或使用适当的collate_fn
- [ ] 训练数据集准备就绪
- [ ] 验证第一个批次的输出形状无误
- [ ] 微调完成
- [ ] 使用evaluate_with_dapt_noise.py评估
- [ ] 对比性能提升

---

## 参考资源

- **DAPT预训练详情**: `DAPT/pipeline_new.md`
- **评估脚本文档**: `pre_struct/kv_ner/EVALUATE_WITH_DAPT.md`
- **Noise处理代码**: `pre_struct/kv_ner/noise_utils.py`
- **完整评估脚本**: `pre_struct/kv_ner/evaluate_with_dapt_noise.py`
