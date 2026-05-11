# 注意力可视化（Attention Visualization）实验计划（KV-BERT / KV-MLM / KV-NSP）

> 目标：把“注意力”作为**诊断信号**（diagnostic signal）来解释模型是否学到了 KV 结构对齐（Key→Value）以及在噪声条件下是否出现稳定的对齐模式。
>
> 注意：单纯 attention heatmap 很容易被质疑（attention≠解释）。本计划强调：**预先定义假设 + 指标统计 + 对照组**，让可视化具备可检验性。

---

## 1. 总体产出（你最终应得到什么）

- 图 1：KV-NSP 正/负样本的 Key→Value attention 子矩阵热力图对比（同一套可视化规则）。
- 图 2：Attention Rollout 的 Key→Value “注意力流”对比（正样本 vs hard negative）。
- 图 3：指标分布图（箱线图/直方图）：
  - `CSAM`（Cross-Segment Attention Mass）正样本显著高于 hard negative。
  - `Top-k Align Rate`（Top-k 对齐率）正样本显著高于 hard negative。
- 表 1：按噪声等级分桶（高/中/低）后的指标均值±方差，展示在噪声下的稳定性变化。

---

## 2. 研究问题与可检验假设（写论文时建议原封不动用）

### 2.1 KV-NSP（Key-Value Matching）

- H1（结构对齐）：在**正样本**中，Key tokens 更倾向于关注 Value tokens（跨段注意力更集中）。
- H2（hard negative 破坏对齐）：在**hard negative**（如 Value-Key 倒序、随机 Value）中，Key→Value 的跨段注意力对齐被显著削弱。

### 2.2 KV-MLM（KV-aware masking）

- H3（实体重建依赖结构）：当 mask 医学实体或 KV 边界附近 token 时，模型对“同一 KV 块内上下文/边界信号”的依赖增强。

### 2.3 噪声鲁棒性（Noise-Embedding）

- H4（噪声条件下的稳定性）：按 OCR 置信度或你们的 noise bins 分组后，低质量组的注意力更分散；引入 Noise-Embedding 后，Key→Value 对齐指标（如 CSAM）的方差更小/更稳定。

---

## 3. 对照组设计（必须有，否则只是“好看的图”）

最少做以下对照：

1) **同一模型**在 KV-NSP 数据上的：
- 正样本（Match）
- hard negative 1：倒序（Value-Key）
- hard negative 2：随机 Value

2) （可选但很加分）**去组件对照（Ablation）**：
- w/ Noise-Embedding vs w/o Noise-Embedding
- w/ KV-NSP pretrain vs w/o KV-NSP

3) **噪声分组对照**（推荐）：
- 依据 OCR conf_avg（或你们的 bin id）把样本分成 High/Medium/Low 三组。

---

## 4. 数据与样本选择

### 4.1 KV-NSP 样本

- 随机抽取：每类 20–50 条（正 / 倒序负 / 随机负），保证 Key、Value 长度不极端。
- 额外挑选：1–3 条“人类直观看起来很像但其实不匹配”的 hard negative（用于论文展示）。

### 4.2 KV-MLM 样本

- 抽取：被 mask 的医学实体、被 mask 的 KV 边界附近 token 各 20 条。
- 记录：mask span 的字符范围、对应 token indices。

### 4.3 噪声分组

- 如果样本有 OCR 元信息：按 conf_avg 或你们 bins 的某一维（例如 conf_avg 的 bin id）分桶。
- 如果样本无 OCR 元信息：可视为 anchor bin（完美文本），用于“干净文本对照”。

---

## 5. Attention 抽取：实现要点（HuggingFace Transformers）

### 5.1 推理配置

- 推理时设置：`output_attentions=True`。
- 取 attention 张量：形状一般为 `[num_layers, batch, num_heads, seq_len, seq_len]`。

### 5.2 span 定位

KV-NSP 输入：`[CLS] Key [SEP] Value [SEP]`
- 你需要拿到 `key_span = [i_start, i_end)`、`value_span = [j_start, j_end)`。
- 建议：在构造输入时保存这些 indices（最稳），不要靠字符串回查。

---

## 6. 可视化方案 A：跨段注意力热力图（最直观）

### 6.1 聚合规则（避免 cherry-pick）

建议预先固定规则（写论文也要写清楚）：
- 只取最后 4 层（例如 layer 9–12；按实现从 0 开始就取 -4:）。
- 对层做均值；对 head 做均值：
  - `A = mean_{layers,heads}(attn)` 得到 `[seq_len, seq_len]`。

### 6.2 画图内容

- 截取子矩阵：`A_key_to_value = A[key_span, value_span]`。
- 分别画：正样本 / 倒序 hard negative / 随机 value negative。
- 可选：同时画 `A_value_to_key` 作为对称对照。

### 6.3 推荐输出

- 每个样本 3 张图（或 1 张图 3 列），保存为 `png`。
- 论文里放 1 个代表性 case，其余放附录或补充材料。

---

## 7. 可视化方案 B：Attention Rollout（更“整体”，更像方法）

### 7.1 参考方法

- Abnar & Zuidema, *Quantifying Attention Flow in Transformers*（2020）

### 7.2 Rollout 计算（实践版）

1) 每层把 head 做平均得到 `A_l`。
2) 引入残差（常见做法）：`\tilde{A}_l = (A_l + I) / 2` 或重新归一化。
3) 逐层矩阵乘：`R = \tilde{A}_1 \tilde{A}_2 ... \tilde{A}_L`。

然后同样截取 `R[key_span, value_span]` 画热力图，或把 Key→Value 的总流量做成标量指标。

---

## 8. 指标定义（让结果可统计、可做显著性检验）

### 8.1 CSAM：Cross-Segment Attention Mass（推荐主指标）

对聚合后的注意力矩阵 `A`：

- 分子：`mass = sum_{i in key_span, j in value_span} A[i,j]`
- 分母：`norm = sum_{i in key_span, j in all} A[i,j]`
- 指标：`CSAM = mass / norm`

解释：Key token 的注意力有多少比例流向 Value 段。

### 8.2 Top-k Align Rate（辅助指标）

对每个 `i in key_span`：
- 取 `A[i,:]` 最大的 top-k token indices（例如 k=5）。
- 计算其中落在 `value_span` 的比例。
- 对 i 做均值。

### 8.3 噪声分组统计

- 按噪声等级 High/Medium/Low，分别统计 CSAM 的均值、方差。
- 比较 w/ Noise-Embedding vs w/o Noise-Embedding 的“方差变化”。

---

## 9. 显著性检验（写论文更稳）

- 对 CSAM / Top-k Align Rate：
  - 正样本 vs 倒序负样本：做 t-test 或 Mann–Whitney U（分布不正态时）。
  - 正样本 vs 随机负样本：同上。

输出：p-value + effect size（例如 Cohen’s d）。

---

## 10. KV-MLM 专用：mask 位置的 attention “关注对象”分析

### 10.1 选择 token

- 选被 mask 的 token（或 mask span 内所有 token）作为 query。

### 10.2 统计其 attention 分配

将 key/value/boundary/标点/数字 等划分成若干类，统计：
- `mass_in_same_value_block`
- `mass_to_key_tokens`
- `mass_to_sep_and_boundary`

结论写法示例：
- “当 mask 医学实体时，模型更多依赖同 KV 块内上下文；当 mask 边界附近 token 时，模型对分隔符及 Key span 的注意力上升。”

---

## 11. 论文写法模板（可直接改成你们的段落）

### 11.1 严谨性声明（建议 1–2 句）

> 我们将 attention 可视化作为诊断工具而非因果解释，并通过预先定义的跨段对齐指标（CSAM）及正/负样本对照，验证注意力模式与 KV 匹配任务的一致性。

### 11.2 结果描述（建议配图 + 指标）

- 描述正样本的 Key→Value 子矩阵更集中。
- hard negative 破坏对齐，CSAM 显著下降。
- 按噪声等级分组后，Noise-Embedding 使低质量组的对齐指标更稳定。

---

## 12. 引用建议（原始论文名，用于 related/analysis）

- Jain & Wallace. **Attention is not Explanation**. 2019.
- Abnar & Zuidema. **Quantifying Attention Flow in Transformers**. 2020. （Attention Rollout）

> 可选补强：如果你们后续再做梯度归因（Integrated Gradients）来佐证注意力结论：
- Sundararajan et al. **Axiomatic Attribution for Deep Networks**. 2017.（Integrated Gradients）

---

## 13. 最小执行清单（你可以按这个顺序做）

1) 从 KV-NSP 数据集中抽取 20–50 条/类（正、倒序负、随机负），保存为 jsonl（含 span indices）。
2) 推理导出 attention（`output_attentions=True`）。
3) 按固定规则聚合（最后 4 层 + head mean），画 Key→Value 子矩阵热力图。
4) 计算 CSAM + Top-k Align Rate，画分布图并做显著性检验。
5) 做 Attention Rollout，重复 3–4 的统计对比。
6) （可选）按噪声等级分桶，比较 w/ vs w/o Noise-Embedding 的稳定性。
7) 从 KV-MLM 样本中抽取 mask case，做 mask query 的 attention 分配统计。
8) 把“假设—方法—指标—结论—风险声明”写入论文可解释性小节。

---

## 14. 备注（工程细节常见坑）

- 不要只挑某个 head：要么全 head 平均，要么声明统一的 head 选择规则。
- span 索引必须和 tokenizer 输出一致（最好在样本构造时写入）。
- 训练/推理时 dropout 关闭（eval 模式），保证可复现。

---

## 15. 与当前脚本对齐的可执行命令（run_attention_kv_nsp.py）

当前实现脚本：`DAPT/experiments/interpretability/run_attention_kv_nsp.py`  
适用范围：**KV-NSP 的注意力诊断与统计**（含 CSAM / Top-k / Rollout / 显著性检验 / 热力图 / 分布图）。

推荐命令（示例）：

```bash
python DAPT/experiments/interpretability/run_attention_kv_nsp.py \
  --model_dir /path/to/final_staged_model \
  --tokenizer_path /path/to/tokenizer_or_model \
  --input_file /path/to/kv_nsp_pairs.jsonl \
  --output_dir /path/to/attn_vis_out \
  --noise_bins_json /path/to/noise_bins.json \
  --last_n_layers 4 \
  --topk 5 \
  --run_rollout \
  --exclude_special_tokens \
  --max_samples_per_group 200
```

输入数据建议字段（json/jsonl 任一）：  
- `key` / `value`（或 `key_text` / `value_text`）  
- `label`（1/0）  
- `negative_type`（`reverse` / `random`）  
- 可选噪声字段：`noise_level`、`conf_avg`、`noise_values`

输出文件：
- `per_sample_metrics.jsonl`：逐样本指标与子矩阵
- `summary.json`：分组统计 + 显著性检验 + effect size
- `report.md`：简版文字报告
- `cases/*.png`：代表性样本热力图
- `figures/*`：CSAM/Top-k（含 rollout）分布图

> 说明：KV-MLM 的“mask query 注意力去向统计”建议单独脚本实现（输入应是 MLM 样本并显式提供 mask span），与 KV-NSP 的 pair 判别流程分开更稳妥。

