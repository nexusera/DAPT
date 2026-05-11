# 评测系统重构架构实施计划 (Rearchitecture Implementation Plan) - Final

基于 `evaluation_design.md` 的最终方案，本计划详细描述了重构的实施路径，确保逻辑严谨且 100% 对齐。

---

## 1. 关键逻辑变更说明 (Core Logic Pivot)

### A. 动态阈值 (Tau Logic)
- **不再使用静态 0.8**。重构后的系统将 `similarity_threshold` 与 `tau_dynamic` 解耦：
    - 若 `tau_dynamic=True`，严格执行现有的长度自适应线性插值算法。
    - 若 `tau_dynamic=False`，则退化为使用固定 `similarity_threshold`。

### B. Task 2: Schema 驱动 vs 稠密化数据
- **架构升级**：放弃修改底层的“稠密化 GT”方案，改为在 `evaluate_task2_qa` 函数内执行 **“Schema 驱动对比”**。
- **实现方式**：遍历 `key_alias_map` 定义的字段范围，在 Pred/GT 中进行双向对齐查找。这样逻辑更纯粹，且原生支持并向产出 Global 和 Pos-only 维度。

---

## 2. 模块分层与职责

### A. 基础设施层 (`metrics.py`)
- 提供 `normalize_text`, `compute_similarity`, `compute_iou` 等纯算法函数。
- 引入 `get_dynamic_threshold` 并支持由 `config` 传入基准参数。

### B. 评估逻辑层 (`evaluators/`)
- **Task 2 Evaluator**: 核心逻辑为“遍历 Schema 字段”。
- **Task 1 & 3**: 专注开放域的对齐与计数。

### C. 引擎与接口
- `engine.py`: 分发器。默认按序执行全部任务。
- `scorer.py`: 负责 IO 及 CLI 参数注入。

---

## 3. 实施阶段指南 (Implementation Stages)

### 第一阶段：基础设施升级
- [ ] 重构 `metrics.py`：移除全局单例配置，改为函数传参模式。
- [ ] 确保 `get_dynamic_threshold` 逻辑的可配置化（长度区间和阈值区间）。

### 第二阶段：Task 2 核心重构 (Schema-Driven)
- [ ] 编写 `evaluate_task2_qa`：实现基于 Schema 循环的 TP/FP/FN 统计。
- [ ] 确保函数能一次性正确产出 Global 和 Pos-only 两份指标。

### 第三阶段：Task 1 & 3 移植
- [ ] 整合原有 `calculate_task1_stats` 和 `calculate_task3_stats` 到独立的 Evaluator。
- [ ] 对齐 `config` 字典中的各项开关。

### 第四阶段：全量集成与验证
- [ ] 编写 `engine.py` 调度逻辑。
- [ ] **最终对齐校验**：使用同一份数据集验证重构前后的 JSON 输出，确保每一个 P/R/F1 数值分毫不差。

---

## 4. 核心原则
1. **逻辑等价性**：Schema 驱动的结果必须等价于原有的“稠密化数据结果”。
2. **默认全量**：保持现有的“默认执行全任务”行为。
3. **阈值一致性**：默认值必须精确匹配当前生产环境数值（Tau, IoU=0.0 等）。
