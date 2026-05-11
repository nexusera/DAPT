# DAPT评估快速开始指南 (Quick Start)

如果您已经熟悉原有的KV-NER评估流程，只需了解以下几点：

## 核心差异 (What's New)

| 方面 | 原有 | 现在 |
|------|------|------|
| 评估脚本 | `evaluate.py` | `evaluate_with_dapt_noise.py` |
| 支持模型 | 标准BERT | 标准BERT + **DAPT** |
| 特殊参数 | 无 | `--noise_bins_json` (DAPT模型必需) |
| 自动检测 | 无 | ✅ 自动检测DAPT特性 |

## 三个命令搞定

### 1️⃣ 评估标准BERT（无变化）

```bash
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path hfl/chinese-roberta-wwm-ext \
    --test_data data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
    --output_summary runs/eval_roberta.json
```

### 2️⃣ 评估DAPT模型（新）

```bash
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path /data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu/final_model \
    --test_data data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
    --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
    --output_summary runs/eval_dapt.json
```

### 3️⃣ 对比结果（可选）

```bash
python pre_struct/kv_ner/compare_eval_results.py \
    runs/eval_roberta.json \
    runs/eval_dapt.json
```

---

## 关键要点

### ✅ 兼容性

- 完全向后兼容原有BERT评估
- 同一个脚本支持标准BERT和DAPT
- 自动检测模型类型，无需手动切换

### ✅ 最小学习成本

只需记住一个新参数：
```bash
--noise_bins_json /path/to/noise_bins.json  # 仅DAPT模型需要
```

### ✅ 输出格式保持一致

结果JSON格式与原有评估相同，可直接用于后续分析和对比。

---

## 文件列表

### 核心代码
- `pre_struct/kv_ner/evaluate_with_dapt_noise.py` - 主评估脚本
- `pre_struct/kv_ner/noise_utils.py` - Noise特征处理工具

### 文档
- `EVALUATE_WITH_DAPT.md` - 详细文档（优先阅读）
- `DAPT_FINETUNING_INTEGRATION.md` - 微调集成指南
- `evaluate_dapt_examples.sh` - 示例脚本集合

---

## 故障排查 (Troubleshooting)

| 问题 | 解决方案 |
|------|---------|
| `FileNotFoundError: noise_bins.json` | 确保路径正确，DAPT预训练时已生成 |
| 模型加载失败 | 检查模型是否兼容，尝试用HuggingFace模型ID测试 |
| CUDA内存不足 | 减小batch_size（如32→16）或使用CPU |
| 结果不符合预期 | 查看详细的`_preds.jsonl`文件，运行error analysis |

---

## 下一步

1. **快速验证** (5分钟)
   ```bash
   python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
       --model_path hfl/chinese-roberta-wwm-ext \
       --test_data data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
       --output_summary /tmp/test.json
   ```

2. **评估DAPT模型** (10-30分钟，取决于数据大小)
   ```bash
   bash pre_struct/kv_ner/evaluate_dapt_examples.sh
   ```

3. **对比分析** (2分钟)
   ```bash
   python pre_struct/kv_ner/compare_eval_results.py runs/eval_*.json
   ```

4. **详细理解** (30分钟)
   - 阅读 `EVALUATE_WITH_DAPT.md`
   - 了解noise feature的含义
   - 查看示例脚本的具体用法

---

## 常用命令速查

```bash
# 基础评估
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path MODEL \
    --test_data DATA \
    --noise_bins_json BINS \  # DAPT需要
    --output_summary OUTPUT

# 批量评估
bash pre_struct/kv_ner/evaluate_dapt_examples.sh

# 对比结果
python pre_struct/kv_ner/compare_eval_results.py results/*.json

# 错误分析
python pre_struct/kv_ner/analyze_predictions.py preds.jsonl gt.jsonl

# 查看帮助
python pre_struct/kv_ner/evaluate_with_dapt_noise.py --help
```

---

## 常见配置

### 配置A: 快速测试 (快但可能不准确)
```bash
--batch_size 64 --max_length 256 --device cuda:0
```

### 配置B: 标准配置 (推荐)
```bash
--batch_size 32 --max_length 512 --device cuda:0 --seed 42
```

### 配置C: 小资源配置 (CPU或显存不足)
```bash
--batch_size 8 --max_length 512 --device cpu
```

---

## 评估结果解读

输出示例：
```json
{
  "model": "...",
  "is_dapt": true,
  "task1": {
    "strict": {"p": 0.82, "r": 0.79, "f1": 0.81},
    "loose": {"p": 0.85, "r": 0.81, "f1": 0.83}
  },
  "task2": {
    "strict_strict": {"p": 0.77, "r": 0.72, "f1": 0.74},
    "strict_loose": {"p": 0.81, "r": 0.78, "f1": 0.80},
    "loose_loose": {"p": 0.83, "r": 0.80, "f1": 0.82}
  }
}
```

**理解指标：**
- **F1 > 0.8**: ✅ 很好
- **F1 > 0.7**: ✅ 可以
- **F1 > 0.6**: ⚠️ 需要改进
- **F1 < 0.6**: ❌ 需要重新检查

---

## 获取帮助

1. **脚本帮助**
   ```bash
   python pre_struct/kv_ner/evaluate_with_dapt_noise.py --help
   ```

2. **详细文档**
   ```bash
   cat pre_struct/kv_ner/EVALUATE_WITH_DAPT.md
   ```

3. **查看示例**
   ```bash
   cat pre_struct/kv_ner/evaluate_dapt_examples.sh
   ```

4. **了解DAPT**
   ```bash
   cat DAPT/pipeline_new.md
   ```

---

## 性能期望

根据DAPT预训练的不同配置，预期性能提升：

| 模型 | Task 1 (Keys) | Task 2 (Pairs) | 备注 |
|------|---------------|----------------|------|
| 标准BERT | F1 ~ 0.78 | F1 ~ 0.72 | 基线 |
| DAPT (含OCR) | F1 ~ 0.85 | F1 ~ 0.81 | ↑ 10-15% |
| DAPT + 微调 | F1 ~ 0.88 | F1 ~ 0.85 | ↑ 15-20% |

*实际数值取决于数据质量和微调策略*

---

祝您评估顺利！有问题或建议欢迎反馈。
