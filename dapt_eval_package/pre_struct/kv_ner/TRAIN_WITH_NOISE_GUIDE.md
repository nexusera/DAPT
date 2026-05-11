# KV-NER 微调指南（DAPT 噪声特征支持）

## 一、系统架构检查与验证

### ✅ 当前状态
- **数据**: 已准备好 train.jsonl、dev.jsonl、test.jsonl (包含 noise_values 字段)
- **原模型**: `/data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu_noise_v2/final_model` (带 DAPT 噪声嵌入)
- **代码**: 已同步 kv_ner 全套模块到 dapt_eval_package

### ❌ 当前限制
**重要**: 现有的 `BertCrfTokenClassifier` 模型不支持 noise_ids 输入参数

**选项**:
1. **推荐**: 用现有模型进行微调（忽略 noise_ids），输出新模型到 `runs/kv_ner_finetuned/`
2. 后续: 如需完整的noise支持，需要修改modeling.py扩展forward()方法

---

## 二、模型权重保护

✅ **确认**: 微调不会破坏原模型权重

**工作流程**:
```
原模型 (/data/ocean/DAPT/.../final_model)
   ↓ 
   加载权重到内存
   ↓
   微调训练（仅修改内存中的权重）
   ↓
   保存到新目录 (runs/kv_ner_finetuned/best)
   
原模型文件系统级别完全不变 ✓
```

---

## 三、快速开始命令

### 方式 A: 使用新的支持 JSONL 的训练脚本（推荐）

```bash
cd /data/ocean/DAPT

python pre_struct/kv_ner/train_with_noise.py \
  --config pre_struct/kv_ner/kv_ner_config.json \
  --pretrained_model /data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu_noise_v2/final_model \
  --noise_bins /path/to/noise_bins.json
```

**说明**:
- `--config`: kv_ner 配置文件（需保证 train.data_path 指向 train.jsonl）
- `--pretrained_model`: 你的 DAPT 预训练模型路径
- `--noise_bins`: 噪声分桶文件（可选；如不提供，用完美值训练）

**输出**:
```
runs/kv_ner_finetuned/
├── best/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer/
│   └── ...
├── training_summary.json
└── used_config.json
```

---

## 四、配置文件检查清单

确保 `kv_ner_config.json` 包含：

```json
{
  "train": {
    "data_path": "/data/ocean/DAPT/biaozhu_with_ocr_noise/train.jsonl",
    "val_data_path": "/data/ocean/DAPT/biaozhu_with_ocr_noise/dev.jsonl",
    "test_split_ratio": 0.5,
    "num_train_epochs": 3,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "output_dir": "runs/kv_ner_finetuned"
  }
}
```

---

## 五、数据格式验证

JSONL 每行格式示例：
```json
{
  "id": "task_001",
  "text": "患者：李明，性别：男，年龄：45岁...",
  "title": "门诊病历",
  "key_value_pairs": [
    {
      "key": {"start": 2, "end": 4, "text": "患者"},
      "value": {"start": 5, "end": 7, "text": "李明"}
    }
  ],
  "noise_values": [
    [0.95, 0.92, 0.1, 0.05, 0.02, 0.01, 2800],
    [0.94, 0.90, 0.12, 0.06, 0.03, 0.02, 2750],
    ...
  ]
}
```

验证方式：
```bash
head -1 /data/ocean/DAPT/biaozhu_with_ocr_noise/train.jsonl | python3 -m json.tool | head -30
```

---

## 六、训练监控

运行时会输出：
```
Epoch 1/3 - train loss: 0.1234
Validation F1: 0.8567 (KEY=0.8901, VALUE=0.8234, HOSPITAL=0.9012)
Saved new best model to runs/kv_ner_finetuned/best (F1=0.8567)
...
Test F1: 0.8512
```

---

## 七、若要启用完整 noise_ids 支持（可选）

如需让模型真正利用 noise_ids，需要修改 modeling.py:

1. 在 `__init__` 中添加 noise embedding layers（如果模型是 RobertaBertNoiseEmbeddings）
2. 在 `forward()` 中接收 `noise_ids` 参数
3. 在序列编码后融合 noise embeddings

当前暂不实现此功能，因为：
- 原模型已有噪声适应（通过预训练）
- 此功能需要 DAPT 模型的特殊配置

---

## 八、结束后

### 评估微调模型
```bash
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
  --model_path runs/kv_ner_finetuned/best \
  --test_data /data/ocean/DAPT/biaozhu_with_ocr_noise/test.jsonl \
  --output_summary runs/eval_finetuned.json
```

### 比较原模型 vs 微调模型
```bash
# 原模型评估
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
  --model_path /data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu_noise_v2/final_model \
  --test_data /data/ocean/DAPT/biaozhu_with_ocr_noise/test.jsonl \
  --output_summary runs/eval_original.json

# 结果对比
diff runs/eval_original.json runs/eval_finetuned.json
```

---

## 九、故障排查

### 问题 1: JSONL 加载失败
```
FileNotFoundError: File not found: ...
```
→ 检查 `train.data_path` 是否为绝对路径且文件存在

### 问题 2: 内存溢出 OOM
```
RuntimeError: CUDA out of memory
```
→ 降低 `batch_size`（如改为 8）

### 问题 3: 数据格式错误
```
KeyError: 'key_value_pairs'
```
→ 检查 JSONL 格式是否包含 `key_value_pairs` 字段

---

## 十、核心文件清单

| 文件 | 作用 |
|---|---|
| `train_with_noise.py` | 新训练脚本（支持 JSONL + noise_values） |
| `noise_utils.py` | 噪声处理（NoiseCollator, NoiseFeatureProcessor） |
| `modeling.py` | KV-NER 模型（当前不依赖 noise_ids） |
| `dataset.py` | 数据集加载与对齐 |
| `data_utils.py` | 标签处理与样本构造 |
| `evaluate_with_dapt_noise.py` | 支持 DAPT 模型的评估脚本 |

---

## 小结

✅ **立即可做**:
- 用 DAPT 预训练模型进行 KV-NER 微调
- 输出新模型到独立目录（原模型不受影响）
- 支持 JSONL + OCR 7 维噪声特征数据

⏳ **后续改进**（可选）:
- 扩展 modeling.py 以完全利用 noise_ids
- 联合优化噪声嵌入和任务特定头
