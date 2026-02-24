# MedStruct-S-Benchmark 评测工具包

本目录包含了用于评测 BERT (NER/EBQA) 和 GPT 系列模型在医疗文本信息抽取任务上的所有核心脚本。
该架构已于 2026-02-10 完成模块化重构。

---

## 文件结构

| 路径 & 脚本 | 功能                                                       | 状态         |
| :---------- | :--------------------------------------------------------- | :----------- |
| `scorer.py` | **核心评估入口**：基于 `med_eval` 引擎执行 Task 1/2/3 评估 | **最新版本** |
| `med_eval/` | **核心引擎包**：包含指标计算及任务评估实现                 | **核心逻辑** |
| `utils/`    | **工具集**：包含 GPT 输出对齐、键名 Schema 构建等辅助脚本  | **工具**     |
| `docs/`     | **文档库**：包含架构设计说明及重构计划                     | **开发文档** |

---

## 输入数据标准格式

`scorer.py` 要求 `--pred_file` 和 `--gt_file` 均为 **标准化 JSONL** 格式。
每行一个 JSON 对象，结构如下：

```json
{
    "id": "sample_001",
    "report_title": "出院小结",
    "ocr_text": "姓名 张三 诊断 流行性感冒 ...",
    "pairs": [
        {"key": "姓名", "value": "张三", "key_span": [0, 2]},
        {"key": "诊断", "value": "流行性感冒", "key_span": [8, 10]}
    ]
}
```

| 字段               | 类型                   | 必需 | 说明                                               |
| :----------------- | :--------------------- | :--- | :------------------------------------------------- |
| `id`               | `string`               | 推荐 | 样本唯一标识，用于溯源和错误定位                   |
| `report_title`     | `string`               | ✅    | 病历类型（如 "出院小结"），Task 2 用于 Schema 匹配 |
| `ocr_text`         | `string`               | 推荐 | OCR 原文，供 span 验证和 debug 使用                |
| `pairs`            | `list[dict]`           | ✅    | 键值对列表，每项为自包含的字典                     |
| `pairs[].key`      | `string`               | ✅    | 键名                                               |
| `pairs[].value`    | `string`               | ✅    | 值                                                 |
| `pairs[].key_span` | `[int, int]` 或 `null` | ✅    | 键名在 `ocr_text` 中的字符位置，无则为 `null`      |

> **重要**：`scorer.py` 不再内置格式转换逻辑。
>
> **重复键支持**：同一样本中可以出现多个同名键（如两个"日期"），每个 pair 独立携带 `key_span`，不存在字典键冲突。

---

## 快速开始

### 1. 全量评测 (Task 1 + 2 + 3)

执行所有任务的 P/R/F1 计算，这是标准 Benchmark 推荐的方式：

```bash
python scorer.py \
    --pred_file predictions.jsonl \
    --gt_file data/val_eval.jsonl \
    --schema_file data/keys_merged_cleaned.json \
    --task_type all \
    --output_file results/eval_output.json
```

### 2. 仅评测 Task 2 (Value Extraction)

适用于评估模型在已知字段上的值提取准确性：

```bash
python scorer.py \
    --task_type task2 \
    --pred_file predictions.jsonl \
    --gt_file data/val_eval.jsonl \
    --schema_file data/keys_merged_cleaned.json \
    --model_name "MacBERT-QA"
```

**关键参数：**

- `--pred_file` / `--gt_file`：**必须**为标准化格式的 JSONL，且样本数量必须一致。
- `--task_type`：可选 `task1`, `task2`, `task3` 或 `all` (默认)。
- `--schema_file`：Task 2 (Schema 驱动) 必须，用于别名映射及全量指标计算。

---

### 高级行为配置

通过命令行参数可以动态控制评测引擎的行为（对应 `med_eval/metrics.py` 的算法）：

| 参数                     | 说明                                                  | 默认行为           |
| :----------------------- | :---------------------------------------------------- | :----------------- |
| `--no_normalize`         | 禁用文本预处理（如转小写、去空格）                    | 默认**启用**归一化 |
| `--similarity_threshold` | 禁用长度自适应动态阈值时设置 NED 相似度判定的基准阈值 | 默认**启用** Tau   |
| `--disable_tau`          | 禁用长度自适应动态阈值 (Tau Logic)                    | 默认**启用** Tau   |
| `--overlap_threshold`    | 设置位置校验的 IoU 阈值                               | `0.0`              |

---

## 📊 输出结果说明

输出 JSON 包含 `summary`（元信息）和 `tasks`（各任务指标）两部分：

```json
{
  "summary": {
    "model": "model_name",
    "dataset": "Original",
    "samples": 100,
    "eval_time": "2026-02-10 18:00:00",
    "config": {
      "normalize": true,
      "similarity_threshold": 0.8,
      "overlap_threshold": 0.0,
      "tau_dynamic": true,
      "use_em": true,
      "use_am": true,
      "use_span": true
    }
  },
  "tasks": {
    "task1": {
      "stats": {"tp_e": 90, "tp_a": 95, "total_p": 100, "total_g": 100},
      "metrics": {
        "exact":  {"p": 0.90, "r": 0.90, "f1": 0.90, "tp": 90},
        "approx": {"p": 0.95, "r": 0.95, "f1": 0.95, "tp": 95}
      }
    },
    "task2_global": {
      "stats": {"tp_e": 70, "tp_a": 80, "total": 100},
      "metrics": {
        "exact":  {"p": 0.70, "r": 0.70, "f1": 0.70, "tp": 70},
        "approx": {"p": 0.80, "r": 0.80, "f1": 0.80, "tp": 80}
      }
    },
    "task2_pos_only": {
      "stats": {"tp_e": 65, "tp_a": 75, "total": 85},
      "metrics": {
        "exact":  {"p": 0.76, "r": 0.76, "f1": 0.76, "tp": 65},
        "approx": {"p": 0.88, "r": 0.88, "f1": 0.88, "tp": 75}
      }
    },
    "task3": {
      "stats": {"ee_tp": 60, "ea_tp": 70, "aa_tp": 75, "total_p": 100, "total_g": 100},
      "metrics": {
        "exact_exact":              {"p": 0.60, "r": 0.60, "f1": 0.60, "tp": 60},
        "exact_approximate":        {"p": 0.70, "r": 0.70, "f1": 0.70, "tp": 70},
        "approximate_approximate":  {"p": 0.75, "r": 0.75, "f1": 0.75, "tp": 75}
      }
    }
  }
}
```

### 指标说明

| 任务       | 指标维度                                                        | 含义                                          |
| :--------- | :-------------------------------------------------------------- | :-------------------------------------------- |
| **Task 1** | `exact` / `approx`                                              | 键名发现的精确匹配与近似匹配                  |
| **Task 2** | `global` + `pos_only` × `exact` / `approx`                      | 全量字段与仅非空字段的值提取准确性            |
| **Task 3** | `exact_exact` / `exact_approximate` / `approximate_approximate` | 键精确+值精确 / 键精确+值近似 / 键近似+值近似 |

---

## 📝 版本历史

- **2026-02-10 (v2.1)**: **输入格式优化 (Format Optimization)**
  - **Dict-style Pairs**：`pairs` 从 `[[key, value]]` 二元组改为 `[{"key", "value", "key_span"}]` 自包含字典。
  - **移除冗余字段**：删除 `keys` 和 `spans_map`，span 信息内嵌于每个 pair 中。
  - **新增 `id` 和 `ocr_text` 字段**：增强样本溯源性和 debug 能力。
  - **重复键支持**：同名键在 pair 列表中独立存在，不再受字典键唯一性约束。
- **2026-02-10 (v2.0)**: **深度模块化架构重构 (Architecture Overhaul)**
  - 核心逻辑迁移至 `med_eval` 包，三个 Task 独立评估函数。
  - **移除内置格式转换**：要求输入文件为标准化格式。
  - **Config 扁平化**：所有评估函数接收独立参数，不再传递 config 字典。
  - **强制 EM+AM 双指标 + Span 校验**：不可关闭，确保评测全面性。
  - **Task 3 三阶指标**：`exact_exact` / `exact_approximate` / `approximate_approximate`。
  - Task 2 改为 **Schema 驱动**，支持 Global/Pos-only 一键输出。
  - 样本数不等时直接报错退出，不再静默截断。
- **2026-02-07**: 新增"配置模式"，支持命令行开启/关闭归一化。
- **2026-02-05**: 初始版本，整合 BERT 和 GPT 评测。
