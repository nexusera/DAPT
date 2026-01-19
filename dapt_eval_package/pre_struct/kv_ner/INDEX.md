# DAPT KV-NER评估框架 - 快速索引

> **这是一个导航文件**，帮助您快速找到需要的文档和代码。

## 🚀 3秒快速开始

```bash
# 评估DAPT模型
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path /path/to/dapt/checkpoint \
    --test_data data/val_eval_titled.jsonl \
    --noise_bins_json /path/to/noise_bins.json \
    --output_summary runs/eval_dapt.json
```

更多命令？→ 查看 `QUICKSTART.md`

---

## 📚 文档导航

### ⭐ 根据您的角色选择

| 我是... | 首先看这个 | 然后看这个 |
|--------|----------|----------|
| **评估工程师** (同事) | [QUICKSTART.md](./QUICKSTART.md) ⚡ | [EVALUATE_WITH_DAPT.md](./EVALUATE_WITH_DAPT.md) 📚 |
| **预训练开发者** (您) | [DAPT_EVALUATION_SUMMARY.md](../DAPT_EVALUATION_SUMMARY.md) | [DAPT_FINETUNING_INTEGRATION.md](./DAPT_FINETUNING_INTEGRATION.md) |
| **微调工程师** | [DAPT_FINETUNING_INTEGRATION.md](./DAPT_FINETUNING_INTEGRATION.md) 🔧 | [EVALUATE_WITH_DAPT.md](./EVALUATE_WITH_DAPT.md) |

---

## 📄 文件清单

### 代码文件

```
✨ 新增文件 ✨

noise_utils.py
├─ 功能: DAPT噪声特征处理
├─ 主要类: NoiseFeatureProcessor, NoiseCollator
├─ 行数: ~240行
└─ 用途: 内部库，被evaluate脚本调用

evaluate_with_dapt_noise.py
├─ 功能: 主评估脚本
├─ 使用: python evaluate_with_dapt_noise.py --model_path ... --test_data ...
├─ 行数: ~600行
└─ 特点: 自动检测DAPT，兼容标准BERT
```

### 文档文件

```
📋 快速参考 (推荐首先阅读)
QUICKSTART.md
├─ 内容: 3个核心命令、快速验证
├─ 长度: 5分钟阅读
└─ 目标: 快速上手

📚 完整指南 (深入理解)
EVALUATE_WITH_DAPT.md
├─ 内容: 参数、输出、场景、故障排查
├─ 长度: 30分钟阅读
└─ 章节: 9个详细章节

🔧 微调集成 (集成DAPT)
DAPT_FINETUNING_INTEGRATION.md
├─ 内容: 集成步骤、代码改动
├─ 长度: 30分钟阅读
└─ 用途: 在KV-NER微调中使用DAPT

🚀 可运行示例
evaluate_dapt_examples.sh
├─ 内容: 7个示例 + 自动生成脚本
├─ 运行: bash evaluate_dapt_examples.sh
└─ 产出: 批量评估、对比、分析脚本

📝 项目总结 (全景了解)
DAPT_EVALUATION_SUMMARY.md
├─ 内容: 背景、目标、交付物
├─ 长度: 15分钟阅读
└─ 用途: 了解完整项目
```

---

## 🎯 按需求快速找文档

### "我要立即评估一个模型"
→ [QUICKSTART.md](./QUICKSTART.md) - 5分钟

### "我要详细了解各个参数"  
→ [EVALUATE_WITH_DAPT.md](./EVALUATE_WITH_DAPT.md) - 完整参数说明部分

### "我要对比多个模型"
→ [evaluate_dapt_examples.sh](./evaluate_dapt_examples.sh) - 运行后自动生成对比脚本

### "我要在微调中使用DAPT"
→ [DAPT_FINETUNING_INTEGRATION.md](./DAPT_FINETUNING_INTEGRATION.md) - 5个集成步骤

### "出了问题该怎么办"
→ [QUICKSTART.md - 故障排查](./QUICKSTART.md#故障排查-troubleshooting)

### "我想了解技术细节"
→ [EVALUATE_WITH_DAPT.md - 技术细节](./EVALUATE_WITH_DAPT.md#技术细节) + 源代码

---

## ⏱️ 推荐学习时间安排

### 快速用户 (15分钟)
```
QUICKSTART.md (5分钟)
→ 复制命令运行 (10分钟)
```

### 标准用户 (60分钟)
```
QUICKSTART.md (5分钟)
→ EVALUATE_WITH_DAPT.md (30分钟)
→ evaluate_dapt_examples.sh + 运行 (15分钟)
→ 查看结果 (10分钟)
```

### 深度用户 (90分钟+)
```
全部文档 (75分钟)
→ 阅读源代码 (30分钟+)
→ 自己写脚本和分析 (开放)
```

---

## 💡 常见任务指南

| 任务 | 文档 | 命令 |
|------|------|------|
| 快速评估DAPT | QUICKSTART | `python evaluate_with_dapt_noise.py --model_path ... --noise_bins_json ...` |
| 评估标准BERT | QUICKSTART | `python evaluate_with_dapt_noise.py --model_path ... --test_data ...` |
| 批量评估多个 | evaluate_dapt_examples.sh | `bash evaluate_dapt_examples.sh` |
| 对比性能结果 | EVALUATE_WITH_DAPT | `python compare_eval_results.py ...` |
| 分析预测错误 | EVALUATE_WITH_DAPT | `python analyze_predictions.py ...` |
| 集成到微调 | DAPT_FINETUNING | 修改train.py + modeling.py |
| 查看详细参数 | EVALUATE_WITH_DAPT | `python evaluate_with_dapt_noise.py --help` |

---

## 🔧 核心参数一览

```bash
python evaluate_with_dapt_noise.py \
    --model_path MODEL                    # ✅ 必需：模型路径
    --test_data TEST_DATA                 # ✅ 必需：测试数据
    --noise_bins_json BINS                # ❌ 可选：仅DAPT需要
    --output_summary OUTPUT               # ❌ 可选：输出文件
    --batch_size 32                       # ❌ 可选：批大小(默认32)
    --max_length 512                      # ❌ 可选：最大长度(默认512)
    --device cuda:0                       # ❌ 可选：设备(默认cuda)
    --seed 42                             # ❌ 可选：随机种子
```

---

## ✅ 新增功能概览

| 功能 | 说明 |
|------|------|
| 自动模型检测 | 自动识别DAPT vs 标准BERT，无需手动切换 |
| Noise处理 | 自动处理7维分桶噪声特征 |
| 向后兼容 | 完全兼容原有BERT评估 |
| 完整文档 | 从快速开始到深度理解 |
| 示例脚本 | 可直接运行或修改的脚本集合 |
| 微调集成 | 详细的集成步骤和代码示例 |

---

## 📊 预期效果

使用新的evaluate框架，您将能够：

✅ **快速评估** - 一行命令评估DAPT或标准BERT模型
✅ **自动对比** - 生成多个模型的性能对比报告
✅ **深度分析** - 分析预测错误，找出改进方向
✅ **集成微调** - 将DAPT用于KV-NER任务微调
✅ **确保复现** - 所有结果可复现（支持seed）

---

## 🎓 学习路径

### 路径A: 只想快速使用
```
QUICKSTART.md → 复制命令 → 运行 → 完成！
```
**时间**: 5-10分钟

### 路径B: 想深入理解
```
QUICKSTART.md → EVALUATE_WITH_DAPT.md → 源代码 → 完整掌握！
```
**时间**: 60-90分钟

### 路径C: 要集成到微调
```
QUICKSTART.md → DAPT_FINETUNING_INTEGRATION.md → 修改代码 → 运行 → 完成！
```
**时间**: 90-120分钟

---

## 🆘 快速排查

| 问题 | 第一步 | 第二步 |
|------|--------|--------|
| 命令怎么用 | QUICKSTART.md | EVALUATE_WITH_DAPT.md参数说明 |
| 模型加载失败 | 检查路径 | 查看错误信息 |
| 显存不足 | 减小batch_size | 改小max_length |
| 结果异常 | 运行analyze_predictions.py | 查看详细预测 |
| 微调怎么用DAPT | DAPT_FINETUNING_INTEGRATION.md | 按步骤修改代码 |

---

## 📞 获取帮助

1. **快速问题** → 查看本索引和QUICKSTART
2. **详细问题** → 查看EVALUATE_WITH_DAPT.md相应章节
3. **技术细节** → 查看源代码和注释
4. **集成问题** → 查看DAPT_FINETUNING_INTEGRATION.md
5. **错误分析** → 运行analyze_predictions.py脚本

---

## 📈 项目成果

✨ **完成交付**:
- ✅ 核心evaluate脚本 (600行)
- ✅ Noise处理库 (240行)
- ✅ 4份详细文档 (~2000行)
- ✅ 可运行示例脚本
- ✅ 完整集成指南
- ✅ 向后兼容设计

📊 **支持能力**:
- 标准BERT模型评估
- DAPT模型评估
- 自动模型检测
- 性能对比分析
- 微调集成支持
- 完全可复现

🎯 **目标达成**:
- ✅ Task 1 (Key Discovery) 评估
- ✅ Task 2 (Pair Extraction) 评估
- ✅ 同事可快速上手
- ✅ 完整的文档体系

---

**版本**: 1.0  
**创建日期**: 2026年1月19日  
**状态**: 生产就绪 ✅

**现在就开始**: [→ QUICKSTART.md](./QUICKSTART.md)
