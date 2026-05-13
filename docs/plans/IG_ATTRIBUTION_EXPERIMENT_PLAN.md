# KV-BERT 梯度归因实验计划（Integrated Gradients, IG）

> 目标：围绕论文核心创新（Noise-Embedding、KV-MLM、KV-NSP），建立一套可复现、可量化、可写入论文的梯度归因实验流程，回答“模型为什么有效”。

实验环境（服务器标准流程）：
```bash
cd /data/ocean/DAPT
git pull
conda activate medical_bert
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
---

## 0. 研究问题（对应论文主张）

1. **Noise-Embedding 是否真的被模型使用？**
   - 在 OCR 低质量样本上，模型是否更依赖噪声特征（`noise_ids` / `noise_values`）而非仅文本 token？
2. **KV-MLM 是否强化了 Key/Value 结构感知？**
   - 对 Key 名称、Key-Value 边界附近 token 的归因是否更集中、更稳定？
3. **KV-NSP 是否增强了逻辑匹配能力？**
   - 在容易混淆的键值对中，归因是否更聚焦于“判别性词片段”而非无关上下文？
4. **归因结果是否与性能提升一致（faithfulness）？**
   - 重要 token/noise 被删除后，预测置信度是否显著下降？

---

## 1. 实验范围与版本固定

### 1.1 任务范围
- **Task1/Task3（KV-NER）**：按 `../pipelines/pipeline_xiaorong.md` 的训练/推理/评测流程。
- **Task2（EBQA）**：按 `../pipelines/pipeline_task2_xiaorong.md` 的转换/训练/推理/评测流程。

### 1.2 模型范围（优先级）
- **P0（必做）**
  - Full：`MacBERT Staged + KV-MLM + KV-NSP + Noise`
  - Ablation-NoNoise：`No-Noise Baseline`
  - Ablation-NoNSP：`No NSP`
  - Ablation-NoMLM：`No MLM`
- **P1（可选）**
  - Noise 模式：`bucket` vs `linear` vs `mlp`
  - NSP 负样本比例：`1:1` / `3:1` / `1:3`

### 1.3 固定项（确保可比）
- 相同测试集（`real_test_with_ocr.json`）
- 相同 tokenizer 路径（各实验对应 checkpoint）
- 相同后处理与 scorer 设置
- 至少固定 1 个 seed；关键结论建议 3-seed 复验

---

## 2. 归因方法设计

### 2.1 归因算法
- 主方法：**Integrated Gradients (IG)**（Captum）
- 对照方法（可选）：Gradient×Input、InputXGradient

### 2.2 归因对象（Inputs）
- **文本分支**：token embedding（`input_ids -> embeddings`）
- **噪声分支**：
  - bucket 模式：各维 `noise_ids` 对应 embedding 的归因
  - continuous 模式：`noise_values`（7 维）投影分支的归因

### 2.3 归因目标（Targets）
- **Task1/3 KV-NER**：
  - 目标 A：某一 token 的预测标签 logit（KEY / VALUE）
  - 目标 B：句级目标（所有 KEY/VALUE token logit 求和）
  - 说明：CRF 解码前后目标不同，建议以 **CRF 前 emission logits** 为主目标，避免不可导路径。
- **Task2 EBQA**：
  - 目标 A：start logit at predicted start
  - 目标 B：end logit at predicted end
  - 目标 C：`start+end` 联合分数

### 2.4 Baseline 设计（IG 必需）
- 文本 baseline：`[PAD]` 序列或全 `[MASK]` 序列（两者择一并固定）
- 噪声 baseline：
  - bucket：全 anchor bin（“完美文本”bin）
  - continuous：7 维全 0 或“完美噪声向量” `[1,1,0,0,0,0,0]`
- 建议同时记录两种 baseline 的敏感性，正文固定一种，附录报告一致性。

### 2.5 超参数建议
- `n_steps`: 32 / 64（主报告使用 64）
- `internal_batch_size`: 8~32（按显存）
- 使用 `LayerIntegratedGradients` 优先归因到 embedding 层，减少离散 token 处理复杂度

---

## 3. 样本集构建（解释性评估集）

构建统一 `analysis_set.jsonl`（建议 300~500 条），按难度与噪声分层抽样：

1. **噪声分层**（基于 `conf_avg` 或综合噪声分）
   - High-quality / Mid / Low-quality（按分位数切三组）
2. **任务分层**
   - Task1：Key 边界清晰 vs 边界模糊
   - Task2：短值（日期/数值）vs 长值（诊断描述）
   - Task3：一对一键值 vs 一键多值
3. **预测结果分层**
   - TP（正确）/ FP（误检）/ FN（漏检）各取一定比例

---

## 4. 量化指标（不仅看热力图）

### 4.1 归因可置信性（Faithfulness）
1. **Deletion AOPC**
   - 按归因分数从高到低移除 top-k token/noise，观察目标分数下降曲线
2. **Comprehensiveness**
   - 去掉 top-k 重要片段后，预测分数下降幅度
3. **Sufficiency**
   - 仅保留 top-k 重要片段时，预测分数保留程度

### 4.2 归因合理性（Plausibility）
1. **Span Overlap@k**
   - top-k 归因 token 与标注 KEY/VALUE span 重叠率
2. **Boundary Focus Score**
   - Key-Value 边界窗口内归因占比（验证 KV-MLM 假设）
3. **Noise Utilization Score**
   - 噪声分支归因占比，按高/中/低噪声分组比较

### 4.3 稳定性（Stability）
- 同一样本不同 seed 的归因 rank 相关（Spearman）
- 同一语义模板样本归因分布方差

---

## 5. 实验矩阵（可直接执行）

## 5.1 P0：核心结论矩阵

| ExpID | 模型 | 任务 | 关注问题 | 输出 |
|---|---|---|---|---|
| IG-1 | Full | Task1/3 | Key/Value 与边界归因是否集中 | 热力图 + Boundary Focus |
| IG-2 | Full | Task2 | 问题词与答案片段是否主导 start/end | 热力图 + Span Overlap |
| IG-3 | NoNoise vs Full | Task1/2/3 | 低质量 OCR 场景下噪声归因与性能关联 | Noise Utilization + AOPC |
| IG-4 | NoNSP vs Full | Task2/3 | KV 逻辑相关 token 归因是否下降 | Comprehensiveness |
| IG-5 | NoMLM vs Full | Task1/3 | Key 边界与医学实体归因是否变弱 | Boundary Focus + Overlap |

## 5.2 P1：增强结论（可选）

| ExpID | 变量 | 任务 | 目标 |
|---|---|---|---|
| IG-6 | noise_mode: bucket/linear/mlp | Task2/3 | 比较噪声分支可解释性形态 |
| IG-7 | NSP ratio: 1:1 / 3:1 / 1:3 | Task2 | 比较逻辑判别 token 的归因差异 |

---

## 6. 工程落地步骤（建议新增脚本）

> 下列脚本是“建议新增”，用于把现有训练/推理流程和 IG 分析串起来。

1. `experiments/interpretability/build_analysis_set.py`
   - 输入：预测结果 + GT + 噪声特征
   - 输出：`analysis_set.jsonl`
2. `experiments/interpretability/run_ig_kvner.py`
   - 对 KV-NER 样本计算 token/noise IG
3. `experiments/interpretability/run_ig_ebqa.py`
   - 对 EBQA 样本计算 start/end IG
4. `experiments/interpretability/eval_faithfulness.py`
   - 计算 deletion/comprehensiveness/sufficiency
5. `experiments/interpretability/plot_attribution.py`
   - 生成可视化图与汇总表

---

## 7. 命令模板（与你现有 pipeline 对齐）

### 7.1 先获得各模型预测结果
- Task1/3：沿用 `../pipelines/pipeline_xiaorong.md` 中 `compare_models.py` + 对齐 + scorer 流程
- Task2：沿用 `../pipelines/pipeline_task2_xiaorong.md` 中 `predict_ebqa.py` + 聚合 + 对齐 + scorer 流程

### 7.2 归因分析命令模板

```bash
cd /data/ocean/DAPT
git pull
conda activate medical_bert
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 1) 构建统一分析集
python experiments/interpretability/build_analysis_set.py \
  --task task3 \
   --pred_file runs/macbert_eval_aligned_preds.jsonl \
   --gt_file runs/macbert_eval_aligned_gt.jsonl \
   --raw_file biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \
   --output runs/ig/analysis_set_task3.jsonl

# 2) 跑 KV-NER IG
python experiments/interpretability/run_ig_kvner.py \
   --config dapt_eval_package/pre_struct/kv_ner/kv_ner_config_macbert.json \
   --model_dir runs/kv_ner_finetuned_macbert/best \
   --analysis_set runs/ig/analysis_set_task3.jsonl \
  --ig_steps 64 \
  --baseline pad \
   --output runs/ig/kvner_macbert_task3_ig.jsonl

# 3) 跑 EBQA IG
python experiments/interpretability/run_ig_ebqa.py \
   --model_dir runs/ebqa_macbert/best \
   --tokenizer macbert_staged_output/final_staged_model \
   --data_path data/kv_ner_prepared_comparison/ebqa_eval_real.jsonl \
  --ig_steps 64 \
  --baseline pad \
   --output runs/ig/ebqa_macbert_task2_ig.jsonl

# 4) Faithfulness 量化
python experiments/interpretability/eval_faithfulness.py \
   --ig_file runs/ig/kvner_macbert_task3_ig.jsonl \
  --metric deletion_aopc comprehensiveness sufficiency \
   --output runs/ig/kvner_macbert_task3_faithfulness.json
```

---

## 8. 结果产出模板（论文可直接用）

### 8.1 表格
1. **主表：归因量化**
   - 行：Full / NoNoise / NoNSP / NoMLM
   - 列：AOPC、Comprehensiveness、Span Overlap、Boundary Focus、Noise Utilization
2. **分层表：按噪声等级**
   - High/Mid/Low 三组分别统计

### 8.2 图片
1. Task1/3 token 归因热力图（正确样本 + 错误样本各 3~5 例）
2. Task2 QA start/end 归因热力图（短值/长值各 3 例）
3. Deletion 曲线（Full vs Ablation）
4. 噪声分支归因占比柱状图（按噪声等级分组）

---

## 9. 风险点与规避

1. **CRF 不可导路径**
   - 归因目标使用 CRF 前 emission logits，避免对 Viterbi 解码做梯度。
2. **离散输入导致归因不稳定**
   - 用 `LayerIntegratedGradients` 对 embedding 层归因，减少离散 token id 的问题。
3. **长文本显存开销高**
   - 固定最大长度（如 512），并先在分析集小批量跑通再扩展。
4. **baseline 选择影响绝对值**
   - 报告相对比较（模型间差异）并附 baseline 敏感性检查。

---

## 10. 两周执行排期（建议）

- **Day 1-2**：实现归因脚本（KV-NER/EBQA）+ 小样本 sanity check
- **Day 3-4**：生成 Full 模型的 Task1/2/3 IG 与初版图
- **Day 5-7**：跑 NoNoise / NoNSP / NoMLM 对比
- **Day 8-10**：Faithfulness 与分层统计
- **Day 11-12**：整理论文图表与 case study
- **Day 13-14**：补充 P1（noise_mode 或 nsp ratio）

---

## 11. 完成判据（DoD）

- [ ] Task1/2/3 各至少 1 个模型完成 IG 归因
- [ ] Full vs 三个关键消融（NoNoise/NoNSP/NoMLM）完成量化对比
- [ ] 至少 1 张“噪声分层”对比图 + 1 张“deletion 曲线”
- [ ] 归因结论可对应回论文三项创新点
- [ ] 关键结论可复现（命令、输入、输出路径明确）
