# 评测系统重构设计方案 (Evaluation System Redesign)

本方案旨在将现有的评测脚本重构为“函数驱动、配置先行、逻辑解耦”的架构。由 **`task_type`** 直接决定评估逻辑与范围。

---

## 一、 核心设计原则
1. **职责分离**：三个任务对应三个独立的评估函数。
2. **任务类型驱动**：通过 `task_type` 决定执行逻辑。
3. **行为参数化**：通过 `config` 字典统一控制逻辑开关与阈值。
4. **Schema 驱动评估 (Task 2)**：舍弃“稠密化数据”的中间过程，直接基于 Schema 定义的字段范围进行引擎级遍历评估。

---

## 二、 评估函数接口定义

| 函数名                     | 业务定义       | `task_type` | 结果维度                         |
| :------------------------- | :------------- | :---------- | :------------------------------- |
| `evaluate_task1_discovery` | **键名发现**   | `task1`     | 标准指标 (P/R/F1)                |
| `evaluate_task2_qa`        | **数值提取**   | `task2`     | **双套指标 (Global + Pos-only)** |
| `evaluate_task3_pairing`   | **端到端配对** | `task3`     | 标准指标 (P/R/F1)                |

---

## 三、 统一入参标准 (Input Arguments)

| 参数名              | Task 1 | Task 2 | Task 3 | 说明                                                |
| :------------------ | :----: | :----: | :----: | :-------------------------------------------------- |
| **`task_type`**     |   ✅    |   ✅    |   ✅    | **[核心]** 指定任务类型 (`task1`/`task2`/`task3`)。 |
| **`predictions`**   |   ✅    |   ✅    |   ✅    | 标准化后的预测列表。                                |
| **`ground_truth`**  |   ✅    |   ✅    |   ✅    | 标准化后的标注列表。                                |
| **`key_alias_map`** |   ❌    |   ✅    |   ❌    | **[核心]** 用于别名映射及 Task 2 确定评测字段范围。 |
| **`config`**        |   ✅    |   ✅    |   ✅    | **[统一配置]** 包含逻辑开关与阈值。                 |

### `config` 参数详情 (Behavior Toggles)：
- **`normalize`** (Bool): **文本预处理开关**。开启后执行转小写、去空格等规范化操作。
- **`use_em`** (Bool): **Exact Match**。是否执行精确全等匹配。
- **`use_am`** (Bool): **Approximate Match**。是否执行模糊匹配。
- **`similarity_threshold`** (Float): **相似度基准阈值**。默认 `0.8`。当 `tau_dynamic` 开启时作为算法基准值。
- **`tau_dynamic`** (Bool): **自适应阈值开关**。开启后根据文本长度动态调节相似度判定阈值（Tau 逻辑）。
- **`use_span`** (Bool): **位置验证**。是否校验物理重叠（Span IoU）。
- **`overlap_threshold`** (Float): **位置重叠判定阈值**。默认 `0.0`（只要有任何重叠即视为通过）。

---

## 四、 评估执行逻辑说明

### Task 2: 数值提取 (QA Mode)
Task 2 采用 **“全量字段循环”** 逻辑：
1. **获取范围**：根据 `key_alias_map` 确定该文档类型对应的标准字段列表 (`schema_fields`)。
2. **核心循环**：遍历每个 `field`，在 Pred 和 GT 中分别查找对应的标准值（执行别名对齐）。
3. **指标产出**：
    - **Global**: 统计 `schema_fields` 中所有字段的表现，包含模型对“应为空”字段的判断能力。
    - **Pos-only**: 仅统计 GT 中实际存在（非空）字段的表现。

---

## 五、 统一出参标准 (Output)

## 五、 统一出参标准 (Output)

### 1. Task 1: 键名发现 (Key Discovery)
同时汇报精确匹配和模糊匹配两套指标：
```json
{
    "stats": {
        "tp_e": int, "tp_a": int, 
        "total_p": int, "total_g": int
    },
    "metrics": {
        "exact": {"p": float, "r": float, "f1": float, "tp": int},
        "approx": {"p": float, "r": float, "f1": float, "tp": int}
    }
}
```

### 2. Task 2: 数值提取 (Value Extraction)
分层汇报全局 (Global) 和仅非空 (Pos-only) 维度的双类指标：
```json
{
    "task2_global": {
        "stats": {"tp_e": int, "tp_a": int, "total": int},
        "metrics": {
            "exact": {...},
            "approx": {...}
        }
    },
    "task2_pos_only": {
        "stats": {...},
        "metrics": {...}
    }
}
```

### 3. Task 3: 端到端配对 (E2E Pairing)
使用三阶指标评估匹配强度：
```json
{
    "stats": {
        "ee_tp": int, "ea_tp": int, "aa_tp": int,
        "total_p": int, "total_g": int
    },
    "metrics": {
        "exact_exact": {...},             # Key精确 + Value精确
        "exact_approximate": {...},       # Key精确 + Value模糊
        "approximate_approximate": {...}  # Key模糊 + Value模糊
    }
}
```
