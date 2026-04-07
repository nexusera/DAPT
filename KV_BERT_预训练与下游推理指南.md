# KV-BERT：预训练与下游推理说明

本文档汇总当前实验仓库中 **KV-BERT**（噪声鲁棒、面向半结构化键值抽取的预训练框架）的**数据准备、预训练、推理与评测**流程，并与 `pipeline_new.md`、`pipeline_xiaorong.md`、`pipeline_task2_xiaorong.md` 及 `paper.md` 中的设定对齐。路径以服务器示例 `/data/ocean/DAPT` 为准，本地开发时可按需替换。

---

## 1. 方法概览（与论文一致）

- **基座**：MacBERT（12 层、hidden 768、12 头），见 `train_dapt_macbert_staged.py`。
- **输入**：标准 Token / Position / Segment 嵌入，外加 **Noise Embedding**：对 OCR 输出的 **7 维物理特征**（置信度统计、标点异常率、断字率、版面对齐等）做建模，与文本表示相加后送入 Encoder（论文公式与 `paper.md` 中 Methodology 一致）。
- **预训练任务**：
  - **KV-MLM**：在 `word_ids`（Jieba + 医学/键名词表引导的全词 ID）上做 **KV-aware 全词掩码**，可选与普通 token 级 MLM 对照（`--mlm_masking kv_wwm` vs `token`）。
  - **KV-NSP**：将经典「句对连续性」改为 **键–值是否匹配** 的二分类；负样本含 **倒序（reverse）** 与 **随机 value（random）**，比例可由 `--nsp_reverse_negative_ratio` / `--nsp_random_negative_ratio` 控制。
- **噪声融合**：默认 **分桶查表（bucket）**；亦支持 `linear` / `mlp` / `concat_linear` 等变体（与消融实验对应）。

---

## 2. 预训练前：数据与词表

### 2.1 去重与可选重采样

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 提取与 MD5 去重 | `extract_and_dedup_json_v2.py` | 多源 JSON/TXT → 干净文本，如 `train.txt` |
| 分源重采样（可选） | `resample_mix.py` | 调整临床/书籍/论文/通用等占比 |
| 长行滑窗（可选） | `chunk_long_lines.py` | 如 window=1000, stride=500，减轻 512 截断 |

### 2.2 词表与 Tokenizer

1. **OCR 词表挖掘**：`train_ocr_clean.py`（语料默认指向 chunked 文本）。
2. **（可选）LLM 过滤**：`filter_vocab_with_llm.py`，需本地 API。
3. **合并 Tokenizer**：`final_merge_v9_regex_split_slim.py` → 如 `my-medical-tokenizer/`。
4. **Jieba 词典**：`generate_jieba_vocab.py` → `vocab_for_jieba.txt`，供数据集构建与 `word_ids` 一致。

### 2.3 噪声分桶边界（一次性）

用全量 OCR JSON 拟合 `NoiseFeatureProcessor`，写出 `noise_bins.json`，供数据集与训练共用：

- 脚本逻辑见 `pipeline_new.md` 第 2.1 节内联 Python（`noise_feature_processor`）。

### 2.4 数据集构建（分路，避免噪声错配）

- **OCR 路**（有真实噪声对齐）  
  - 导出文本：`export_ocr_texts.py`  
  - `build_dataset_final_slim.py`：**`--no_shuffle_split`** 保持与 OCR JSON 顺序一致  
  - `add_noise_features.py`：写入 `noise_values`（7 维连续值，按 token/word 对齐）  
  - 校验：`verify_noise_alignment.py`
- **非 OCR 路**（书籍/指南/百科等）  
  - 同一脚本构建，**可 `--shuffle_split`**；无 OCR 元信息时，训练时由 collator 填「完美」噪声向量 `[1,1,0,0,0,0,0]`（与 `pipeline_new.md` 一致）。
- **合并（可选）**：`merge_datasets.py`，用 `ocr_repeat` / `non_ocr_repeat` 只调 train 配比；**禁止**把 OCR 噪声按错误索引塞给非 OCR 样本。

合并后常用软链统一入口：

```bash
ln -sfn /data/ocean/DAPT/workspace/processed_dataset_merged \
        /data/ocean/DAPT/workspace/processed_dataset
```

---

## 3. 预训练：`train_dapt_macbert_staged.py`

### 3.1 核心逻辑

- **分阶段交替**：`num_rounds` 轮，每轮先 **MLM**（`mlm_epochs_per_round`）再 **KV-NSP**（`nsp_epochs_per_round`），默认如 3 轮 ×（1 epoch MLM + 3 epoch NSP）。
- **KV-NSP 数据**：`--nsp_data_dir` 指向伪标签 JSON（如 `pseudo_kv_labels_filtered.json`），内部使用 `kv_nsp/dataset.py` 的 `KVDataset`。
- **噪声**：`--noise_bins_json` 与数据集内 `noise_values` 一致；非 OCR 样本在 collator 中映射为完美桶或连续值。

### 3.2 默认与常用超参（摘录）

| 参数 | 典型值 | 含义 |
|------|--------|------|
| `--learning_rate` | 5e-5 | 预训练学习率 |
| `--num_rounds` | 3 | MLM/NSP 交替轮数 |
| `--mlm_epochs_per_round` | 1 | 每轮 MLM epoch 数 |
| `--nsp_epochs_per_round` | 3 | 每轮 KV-NSP epoch 数 |
| `--mlm_probability` | 0.15 | MLM 掩码比例 |
| `--max_length` | 512 | 最大序列长度 |
| `--mlm_masking` | `kv_wwm` / `token` | KV 全词掩码 vs 普通 MLM 消融 |
| `--noise_mode` | `bucket` / `linear` / `mlp` / `concat_linear` | 噪声嵌入形式 |
| `--nsp_negative_prob` | 0.5 | KV-NSP 中采负样本的总概率 |
| `--nsp_reverse_negative_ratio` | 1 | 负样本中「倒序」相对权重 |
| `--nsp_random_negative_ratio` | 1 | 负样本中「随机 value」相对权重 |

### 3.3 产物目录

- 每轮：`output_dir/round_{k}_mlm/`、`output_dir/round_{k}_nsp/`
- **最终供下游使用**：`output_dir/final_staged_model/`（含权重与 tokenizer；脚本会尽量导出可用的 fast tokenizer 供 `offset_mapping`）。

### 3.4 预训练侧「推理」说明

此处「推理」指 **MLM 验证 loss / KV-NSP 验证** 或导出 checkpoint，无单独 REST 服务。下游任务使用 `final_staged_model` 中的 `AutoModel` 兼容权重。

### 3.5 一键命令模板（主实验）

详见 `pipeline_new.md` 第 3 节。示例：

```bash
cd /data/ocean/DAPT
python /data/ocean/DAPT/train_dapt_macbert_staged.py \
  --output_dir /data/ocean/DAPT/workspace/output_macbert_kvmlm_staged \
  --dataset_path /data/ocean/DAPT/workspace/processed_dataset \
  --nsp_data_dir /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
  --tokenizer_path /data/ocean/DAPT/my-medical-tokenizer \
  --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
  --learning_rate 5e-5 \
  --num_rounds 3 \
  --mlm_epochs_per_round 1 \
  --nsp_epochs_per_round 3 \
  --mlm_probability 0.15 \
  --max_length 512 \
  --mlm_masking kv_wwm
```

消融（MLM 掩码、NSP 负样本比例、noise bucket/linear/mlp/concat_linear）的完整命令同样列于 `pipeline_new.md`。

---

## 4. 下游任务总览

| 任务 | 含义 | 主流程文档 |
|------|------|------------|
| Task 1 / 3 | KV 结构化抽取（span 级等） | `pipeline_xiaorong.md` |
| Task 2 | 基于查询的抽取（EBQA 形式） | `pipeline_task2_xiaorong.md` |

更换预训练 checkpoint 时，**统一修改各 JSON 配置中的 `model_name_or_path` / `tokenizer_name_or_path`** 指向新的 `final_staged_model`。

---

## 5. Task 1 & 3：KV-NER 微调 → 推理 → 评测

### 5.1 环境

```bash
cd /data/ocean/DAPT
conda activate medical_bert   # 以实际环境名为准
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

测试集示例：`biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json`；噪声分桶：`workspace/noise_bins.json`。

### 5.2 训练

```bash
python /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/train_with_noise.py \
  --config /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_macbert.json \
  --noise_bins "/data/ocean/DAPT/workspace/noise_bins.json"
```

不同预训练变体使用不同 `kv_ner_config_*.json`（如 `staged` / `hybrid` / `mtl` / 消融 `no_nsp` / `no_mlm` / `no_noise` / `noise_*`），保证 **`output_dir` 互不覆盖**。

### 5.3 推理（生成预测与 GT 旁路文件）

```bash
python /data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/compare_models.py \
  --ner_config .../kv_ner_config_macbert.json \
  --test_data "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --noise_bins "/data/ocean/DAPT/workspace/noise_bins.json" \
  --output_summary /data/ocean/DAPT/runs/macbert_eval_summary.json
```

`compare_models.py` 会在 `output_summary` 对应路径生成 **`_preds.jsonl`** 与 **`_gt.jsonl`**，供后续评分。

### 5.4 Task 1：Span 对齐 + 打分

Task 1 需先对齐预测与 GT 的 span 表述：

```bash
python /data/ocean/DAPT/scripts/align_for_scorer_span.py \
  --gt_in "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --pred_in "/data/ocean/DAPT/runs/macbert_eval_summary_preds.jsonl" \
  --gt_out "/data/ocean/DAPT/runs/macbert_eval_aligned_gt.jsonl" \
  --pred_out "/data/ocean/DAPT/runs/macbert_eval_aligned_preds.jsonl"
```

```bash
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/scorer.py \
  --pred_file "/data/ocean/DAPT/runs/macbert_eval_aligned_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/runs/macbert_eval_aligned_gt.jsonl" \
  --task_type task1 \
  --overlap_threshold -1 \
  --output_file "/data/ocean/DAPT/runs/macbert_eval_report_task1.json"
```

`--overlap_threshold -1` 用于在评测规则下**忽略 span 位置重叠约束**（与仓库内说明一致）。

### 5.5 Task 3

同一对齐结果上，将 `--task_type` 改为 `task3` 即可（通常不需要 Task 1 那样严格的 span 对齐策略，但流程共用同批文件）。

### 5.6 NSP 比例消融（Task 1/3）

批量跑法：`experiments/nsp_ratio_ablation/run_nsp_ratio_kvner_all.sh`（见 `pipeline_xiaorong.md`）。

---

## 6. Task 2：EBQA 数据准备 → 微调 → 推理 → 评测

### 6.1 环境变量示例

```bash
export PYTHONPATH=$PYTHONPATH:/data/ocean/DAPT
export QUERY_SET="/data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/keys_merged_1027_cleaned.json"
NOISE_BINS="/data/ocean/DAPT/workspace/noise_bins.json"
TOKENIZER_PATH="/data/ocean/DAPT/macbert_staged_output/final_staged_model"   # 与所选预训练一致
```

### 6.2 KV-NER 格式 → EBQA JSONL

对 train / test 各跑一次 `convert_ebqa.py`：

```bash
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/convert_ebqa.py \
  --input_file "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json" \
  --output_file "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_train_real.jsonl" \
  --struct_path $QUERY_SET \
  --tokenizer_name $TOKENIZER_PATH \
  --noise_bins $NOISE_BINS
```

测试集同理，输出如 `ebqa_eval_real.jsonl`。

### 6.3 微调

```bash
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/train_ebqa.py \
  --config /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/ebqa_config_macbert.json
```

### 6.4 推理（QA 级 → 文档级）

```bash
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/predict_ebqa.py \
  --model_dir "/data/ocean/DAPT/runs/ebqa_macbert/best" \
  --tokenizer $TOKENIZER_PATH \
  --data_path "/data/ocean/DAPT/data/kv_ner_prepared_comparison/ebqa_eval_real.jsonl" \
  --output_preds "/data/ocean/DAPT/runs/ebqa_macbert_preds.jsonl"
```

**必须**将 QA 级预测聚合为文档级（否则后续对齐脚本无法工作）：

```bash
python /data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/aggregate_qa_preds_to_doc.py \
  --raw_file "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json" \
  --qa_pred_file "/data/ocean/DAPT/runs/ebqa_macbert_preds.jsonl" \
  --output_file "/data/ocean/DAPT/runs/ebqa_macbert_doc_preds.jsonl" \
  --prefer score
```

### 6.5 对齐与 Task 2 打分

```bash
python /data/ocean/DAPT/dapt_eval_package/MedStruct-S-Benchmark-feature-configurable-metrics/preprocess_ebqa_real_h200.py \
  --gt_file /data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \
  --pred_file /data/ocean/DAPT/runs/ebqa_macbert_doc_preds.jsonl \
  --output_dir aligned_data
```

```bash
cd /data/ocean/DAPT/dapt_eval_package/MedStruct-S-master
python scorer.py \
  --pred_file "/data/ocean/DAPT/aligned_data/aligned_ebqa_macbert_doc_preds.jsonl" \
  --gt_file "/data/ocean/DAPT/aligned_data/gt_ebqa_aligned.jsonl" \
  --query_set $QUERY_SET \
  --task_type task2 \
  --output_file "/data/ocean/DAPT/runs/ebqa_macbert_report_task2.json"
```

注意：新版 `scorer.py` 依赖在 **MedStruct-S-master** 目录下导入 `med_eval`，需先 `cd` 到该目录再执行。

### 6.6 NSP 比例消融（Task 2）

`experiments/nsp_ratio_ablation/run_nsp_ratio_ebqa_all.sh`（见 `pipeline_task2_xiaorong.md`）。

---

## 7. 关键脚本与配置文件索引

| 类别 | 路径 |
|------|------|
| 主预训练 | `train_dapt_macbert_staged.py` |
| 预训练流程说明 | `pipeline_new.md` |
| KV-NER 训练/推理 | `dapt_eval_package/pre_struct/kv_ner/train_with_noise.py`、`compare_models.py` |
| KV-NER 配置 | `dapt_eval_package/pre_struct/kv_ner/kv_ner_config*.json` |
| EBQA 转换/训练/推理 | `pre_struct/ebqa/convert_ebqa.py`、`train_ebqa.py`、`predict_ebqa.py`、`aggregate_qa_preds_to_doc.py` |
| EBQA 配置 | `pre_struct/ebqa/ebqa_config*.json` |
| Task1 span 对齐 | `scripts/align_for_scorer_span.py` |
| 泄漏/对齐检查（预训练数据） | `scripts/check_pretrain_test_leakage.py`（若需保证划分无泄漏可配合使用） |
| 论文方法与符号 | `paper.md` |

---

## 8. 常见问题（摘自 `pipeline_new.md`）

- **噪声错配**：先跑 `verify_noise_alignment.py`；匹配率低说明 OCR 与 dataset 顺序不一致，应重建 OCR 路再合并。
- **Token 越界 / NaN**：用当前 tokenizer 重建数据；检查 `noise_values` 无 NaN；可暂时关闭 bf16 做短跑验证。
- **端口冲突**：分布式训练时显式指定 `MASTER_PORT` 等。
- **Fast tokenizer 异常**（中文变 UNK、大颗粒 token）：检查 `vocab.txt` 与 `tokenizer.json` 一致性，按训练脚本说明修复 fast backend，而非简单「去空格」规避。

---

## 9. 文档修订说明

- 本文档由仓库内 `pipeline_new.md`、`pipeline_xiaorong.md`、`pipeline_task2_xiaorong.md` 与 `paper.md` 整理而成；**具体绝对路径、GPU 编号、conda 环境名**请以实际部署为准。
- 新增预训练消融或下游实验时，优先复制一份 JSON 配置并修改 `output_dir` 与 `model_name_or_path`，避免相互覆盖。
