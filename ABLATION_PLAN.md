# KV-BERT 消融实验计划（Ablation Plan）

> 目标：把“哪些组件带来增益、为什么带来增益、在什么条件下会失效（对齐错配）”用最少但关键的实验说清楚。
>
> 使用范围：本文方法（KV-MLM、KV-NSP、Noise-Embedding、Tokenizer/词表、数据配比、长文本切分、对齐策略）。

---

## 0. 统一约定（务必固定）

### 0.1 固定项（所有消融保持一致）
- **骨干模型**：MacBERT-base（或你论文当前选定的 backbone），层数/隐藏维/头数固定。
- **训练数据**：同一份 train/valid/test 切分；除非消融点就是“数据配比/切分”。
- **训练超参**：learning rate、batch size、steps/epochs、warmup、weight decay、max length 固定。
- **随机种子**：至少 3 个 seed（如 42/43/44）；如果算力紧张，核心消融跑 3-seed，次要消融跑 1-seed。
- **评估指标**：沿用主表（Task1/Task2/Task3 的 F1/EM/Acc 等你论文定义）。

### 0.2 实验命名规范（建议）
- 格式：`ABL_{模块}_{变体}_{seed}`
- 例子：
  - `ABL_NOISE_BIN_vs_CONT_seed42`
  - `ABL_ALIGN_SHIFT1_seed42`
  - `ABL_KVMLM_WWM_ENTITY_BOUND_seed43`

### 0.3 结果记录最小字段（每个实验都填）
- 训练：steps/epochs、最终 loss、是否出现 NaN/Inf、训练时长、GPU 型号与数量。
- 数据：OCR 占比、是否 chunk_long_lines、tokenizer 版本（commit/hash 或目录名）。
- 对齐：`verify_noise_alignment` 匹配率、覆盖率（如有）。
- 评估：各任务主指标（与主表一致），以及你最关心的 1-2 个子指标。

---

## 1. 优先级总览（先跑这些，论文说服力最大）

**P0（强烈建议必做，6 个）**
1) Noise：Binning vs 连续投影（Linear/MLP）
2) Noise：7维逐维消融（drop-one / keep-one）
3) 对齐错配反例（shuffle 或 shift）
4) KV-MLM：Random vs WWM vs Entity+Boundary
5) KV-NSP：负样本策略比例（reverse/random）
6) Tokenizer：base vs +keys vs +OCR（过滤前/后）

**P1（有条件就补）**
- Anchor bin / 非OCR完美噪声策略消融
- bins 数量敏感性（更粗/更细）
- OCR 占比敏感性（0/35/60%）
- 长文本滑窗切分收益（chunk vs no-chunk）

---

## 2. 具体消融清单（可直接照着跑）

> 说明：每个实验都应与 **FULL（完整模型）** 以及必要时的 **BASE（无新组件）** 做对比。

### 2.1 FULL / BASE 基线（先确立）
- **EXP-B0: BASE（无新组件）**
  - 配置：不开 Noise-Embedding；不开 KV-NSP；KV-MLM 退化为标准 MLM（或你定义的“最朴素设置”）。
  - 目的：提供“普通预训练”的参照。

- **EXP-B1: FULL（最终方法）**
  - 配置：Noise-Embedding（binning+lookup 或论文最终实现）；KV-MLM（Entity+Boundary）；KV-NSP（reverse+random，默认比例）；Tokenizer（最终合并策略）。
  - 目的：主表结果与后续所有 ablation 的对照。

记录表（先填 B0/B1）：

| ExpID | 设定 | Seeds | Task1 | Task2 | Task3 | 对齐匹配率 | 训练时长 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B0 | BASE | 42/43/44 |  |  |  | - |  |  |
| B1 | FULL | 42/43/44 |  |  |  |  |  |  |

---

## 3. P0 消融（必做）

### 3.1 Noise Embedding：Binning vs 连续投影
- **EXP-N1: BIN（你的离散化实现）**
  - Noise：quantile bins + embedding lookup（7个 embedding 相加）。
- **EXP-N2: CONT（连续映射）**
  - Noise：对 7维连续值做标准化后，`Linear(7→d_model)` 或 2-layer MLP(7→d_model) 映射为噪声向量加到输入。
- 控制变量：同一份 noise_values；其余完全一致。
- 预期：BIN ≥ CONT（尤其在分布偏斜、离群点、多峰时）。

| ExpID | Noise 映射 | Seeds | Task3 | 备注 |
|---|---|---:|---:|---|
| N1 | BIN(lookup) | 42/43/44 |  |  |
| N2 | CONT(linear/mlp) | 42/43/44 |  |  |

> 写法建议：方法段不要同时“只写线性投影”又“实现是 binning”；要么统一成 binning（正文给公式），要么把两者都写成可选并用 ablation 证明选择。

### 3.2 Noise 7维逐维消融（drop-one + keep-one）
- **EXP-N3: DROP-ONE（7 次）**：每次去掉 $f_k$，保留其余 6 维。
- **EXP-N4: KEEP-ONE（7 次）**：每次只保留 $f_k$，其余置为 anchor/perfect。
- 目的：解释“conf 类 vs layout 类”贡献。

| ExpID | 设置 | 维度 | Seeds | Task3 | 备注 |
|---|---|---:|---:|---:|---|
| N3 | drop-one | f1 | 42 |  |  |
| N3 | drop-one | f2 | 42 |  |  |
| ... | ... | ... | ... |  |  |
| N4 | keep-one | f1 | 42 |  |  |
| ... | ... | ... | ... |  |  |

> 算力策略：N3/N4 可以先用 1-seed 跑完画趋势，挑最关键的 2-3 个维度再补 3-seed。

### 3.3 对齐错配反例（Alignment Mismatch）
- **EXP-A1: SHIFT(1)**：把 noise_values 按样本维度整体平移 1（或随机置换），制造轻度错配。
- **EXP-A2: SHUFFLE**：OCR 路开启 shuffle_split 或在合并前打乱 OCR dataset 顺序，再写 noise。
- 目的：把工程经验变成“可复现证据”，证明为什么必须 OCR-only 路保持顺序并 verify。

| ExpID | 错配方式 | Seeds | 对齐匹配率 | Task3 | 备注 |
|---|---|---:|---:|---:|---|
| A1 | shift=1 | 42/43/44 |  |  |  |
| A2 | shuffle | 42/43/44 |  |  |  |

> 建议额外输出：匹配率 vs Task3 的下降曲线（至少 3 个点：100%/90%/70%）。

### 3.4 KV-MLM：mask 策略消融
- **EXP-M1: Random MLM**：标准 15% token-level mask。
- **EXP-M2: WWM**：基于 word_ids 的 whole-word masking，但不区分实体/边界。
- **EXP-M3: Entity+Boundary（FULL）**：你的最终策略。
- 目的：回答“为什么不是普通 WWM 就够了”。

| ExpID | KV-MLM 策略 | Seeds | Task3 | 备注 |
|---|---|---:|---:|---|
| M1 | Random | 42/43/44 |  |  |
| M2 | WWM | 42/43/44 |  |  |
| M3 | Entity+Boundary | 42/43/44 |  |  |

> 可选：mask rate（0.10/0.15/0.20）敏感性作为 P1。

### 3.5 KV-NSP：负样本策略比例
- **EXP-S1: reverse=0%, random=100%**
- **EXP-S2: reverse=50%, random=50%（默认）**
- **EXP-S3: reverse=100%, random=0%**
- 目的：证明 hard negative 的必要性与最佳比例。

| ExpID | Reverse:Random | Seeds | KV-NSP Acc | Task3 | 备注 |
|---|---|---:|---:|---:|---|
| S1 | 0:100 | 42/43/44 |  |  |  |
| S2 | 50:50 | 42/43/44 |  |  |  |
| S3 | 100:0 | 42/43/44 |  |  |  |

### 3.6 Tokenizer / 词表：扩词来源消融
- **EXP-T1: BASE tokenizer**：不加 keys，不加 OCR vocab。
- **EXP-T2: +Keys**：只加 biaozhu_keys_only（或你论文定义的 key 集）。
- **EXP-T3: +OCR vocab（未过滤）**：OCR 词表直接合并。
- **EXP-T4: +OCR vocab（LLM过滤）= FULL**：kept_vocab + keys（最终版）。
- 目的：支撑“盲目扩词会伤性能，过滤有必要”。

| ExpID | Tokenizer 版本 | Seeds | Task3 | OOV/unk(可选) | 备注 |
|---|---|---:|---:|---:|---|
| T1 | base | 42 |  |  |  |
| T2 | +keys | 42 |  |  |  |
| T3 | +ocr_raw | 42 |  |  |  |
| T4 | +ocr_llm | 42 |  |  |  |

> 建议：T1~T4 先 1-seed 快跑，最终只给 FULL/最差/最强 baseline 做 3-seed。

---

## 4. P1 消融（有条件补）

### 4.1 Anchor bin / 非OCR“完美噪声”策略
- **EXP-P1: nonOCR→anchor（当前）**
- **EXP-P2: nonOCR→random bins（反例）**
- **EXP-P3: nonOCR 不加 noise（缺失）**

### 4.2 bins 粒度敏感性
- **EXP-P4: bins×0.5（更粗）**
- **EXP-P5: bins×2（更细）**

### 4.3 OCR 占比敏感性
- **EXP-P6: OCR=0%**
- **EXP-P7: OCR≈35%（当前）**
- **EXP-P8: OCR≈60%**

### 4.4 长文本滑窗切分（chunk_long_lines）
- **EXP-P9: no-chunk（直接截断）**
- **EXP-P10: chunk(window=1000,stride=500)**

---

## 5. 论文呈现建议（怎么写更像“研究结论”）

### 5.1 消融表结构建议
- 主消融表只放 **P0 的关键对比**（N1/N2、A1/A2、M1/M2/M3、S1/S2/S3、T1/T2/T4）。
- “逐维消融（N3/N4）”放到附录或单独图（柱状图/蜘蛛图）。

### 5.2 强叙事点（建议在正文明确写）
- **Alignment 是必要条件**：错配会带来系统性负增益（用 A1/A2 量化）。
- **Binning 的作用不是“凑巧”**：BIN vs CONT 证明离散化优势。
- **KV-NSP 提升的是一致性**：可配合错误类型分析（key对但value错、错配率下降）。

---

## 6. 执行 Checklist（跑之前勾一遍）
- [ ] FULL 与 BASE 都已 3-seed 复现
- [ ] verify_noise_alignment 的指标已稳定输出并记录
- [ ] 训练日志统一保存（含配置、seed、数据版本、tokenizer版本）
- [ ] P0 全部跑完，主表可补齐
- [ ] P1 只补最能解释审稿疑问的 1-2 项

---

## 7. TODO（你填）
- 论文当前主任务指标列表：
  - Task1: ______
  - Task2: ______
  - Task3: ______
- FULL 配置文件/命令入口：______
- 训练预算（每个实验可接受的 GPU-hours）：______
