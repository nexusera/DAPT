# Tokenizer 消融实验计划

目标：评估不同词表扩展策略对 KV-BERT 预训练与下游任务（Task1/Task2/Task3）的影响，验证“盲目扩词伤性能、LLM 过滤与 keys 加入的价值”。

---

## 实验概览

Variants:
- T1 (BASE): 原始 tokenizer（不加 keys、不加 OCR vocab）。
- T2 (+keys): 在 base 上加入 `biaozhu_keys_only.txt` 中的 key 词。
- T3 (+ocr_raw): 在 base 上直接合并 OCR 挖掘出的 raw vocab（未过滤）。
- T4 (+ocr_llm): 在 base 上合并 OCR vocab，但先用 LLM 过滤（`kept_vocab.txt`），并加入 keys —— 论文最终版。

优先级：先做 T1/T4/T3 快速比较（1-seed），确定趋势后对 T2 与关键对比做 3-seed。

---

## 统一约定（与 `DAPT/ABLATION_PLAN.md` 保持一致）
- Backbone：MacBERT-base
- 数据切分：统一 train/valid/test（如 FULL 实验所用）
- Seed：优先 1-seed 快跑；核心对比补 3-seed（例如 42/43/44）
- 评估指标：Task1/Task2/Task3 的 F1/EM/Acc（与论文一致）
- 训练超参：与 FULL 保持一致（learning rate、batch size、max_length 等）

---

## 文件与脚本参考（来自 pipeline）
- OCR vocab 挖掘：`/data/ocean/DAPT/train_ocr_clean.py`
- LLM 过滤：`/data/ocean/DAPT/filter_vocab_with_llm.py`
- Tokenizer 合并（历史脚本，含硬编码路径，不建议做消融入口）：`/data/ocean/DAPT/final_merge_v9_regex_split_slim.py`
- Jieba 词典生成（历史脚本，含硬编码路径，不建议做消融入口）：`/data/ocean/DAPT/generate_jieba_vocab.py`
- 数据构建（word_ids 对齐）：`/data/ocean/DAPT/build_dataset_final_slim.py`
- 训练入口（MacBERT staged, KV-MLM+KV-NSP+noise）：`/data/ocean/DAPT/train_dapt_macbert_staged.py`

**本消融推荐统一入口（为可复现性、避免硬编码路径）**：
- `/data/ocean/DAPT/experiments/tokenizer_ablation/`（本次新增的一套可配置脚本）

---

## 具体流程（远端可复制执行，统一绝对路径）

> 重要路径约定：你的远端仓库根目录是 `/data/ocean/DAPT`，不存在 `/data/ocean/DAPT/DAPT`。

### 0) 固化可复现信息（强烈建议每次实验先跑）

```bash
cd /data/ocean/DAPT

# 记录代码版本
git rev-parse HEAD | tee -a /data/ocean/DAPT/ablation_tokenizer_gitsha.log
git status --porcelain | tee -a /data/ocean/DAPT/ablation_tokenizer_gitsha.log

# 记录环境（可选）
python -V | tee -a /data/ocean/DAPT/ablation_tokenizer_env.log
python -m pip freeze | tee -a /data/ocean/DAPT/ablation_tokenizer_env.log
```

### 1) 配置实验参数（只改一次）

```bash
cd /data/ocean/DAPT/experiments/tokenizer_ablation

# 复制配置模板
cp -n config.env.example config.env

# 编辑 config.env：把 KEYS/OCR_VOCAB/TRAIN_FILE/OUT_ROOT 改成你实际路径
vim config.env

# 校验必需文件是否存在，并创建输出目录
bash 00_check_env.sh
```

> 说明：本目录脚本会把所有输出写到 `OUT_ROOT`（你在 config.env 里配置），建议按日期分目录，例如：
> `OUT_ROOT="/data/ocean/DAPT/ablation/tokenizer/2026-03-04"`

### 2)（可选）生成 OCR vocab 与 LLM 过滤产物

如果你已经有：
- `/data/ocean/DAPT/workspace/medical_vocab_ocr_only/vocab.txt`
- `/data/ocean/DAPT/workspace/kept_vocab.txt`

可以跳过本节。

```bash
# 2.1 OCR vocab 挖掘
python /data/ocean/DAPT/train_ocr_clean.py

# 2.2 LLM 过滤（需要本地 LLM 服务）
cd /data/ocean/DAPT
export LLF_API_BASE="http://127.0.0.1:8008/v1"
export OPENAI_API_KEY="EMPTY"
python /data/ocean/DAPT/filter_vocab_with_llm.py \
  --vocab /data/ocean/DAPT/workspace/medical_vocab_ocr_only/vocab.txt \
  --kept /data/ocean/DAPT/workspace/kept_vocab.txt \
  --dropped /data/ocean/DAPT/workspace/dropped_vocab.txt \
  --batch_size 64 \
  --topn 50000
```

### 3) 生成四个 tokenizer 变体（T1~T4）

```bash
cd /data/ocean/DAPT/experiments/tokenizer_ablation

bash 10_make_tokenizers.sh 2>&1 | tee -a "$(pwd)/tokenizer_build.log"

# 查看各 tokenizer 的词表大小（如果保存结构包含 vocab.txt）
find "$(grep '^OUT_ROOT=' config.env | cut -d'=' -f2 | tr -d '"')/tokenizers" -maxdepth 2 -name vocab.txt -print -exec wc -l {} \;
```

输出目录（默认）：
- `${OUT_ROOT}/tokenizers/t1_base`
- `${OUT_ROOT}/tokenizers/t2_keys`
- `${OUT_ROOT}/tokenizers/t3_ocr_raw`（若 OCR_VOCAB_RAW 存在）
- `${OUT_ROOT}/tokenizers/t4_ocr_llm_keys`（必跑）

### 4) 生成每个变体对应的 Jieba 词典（避免消融 confound）

```bash
cd /data/ocean/DAPT/experiments/tokenizer_ablation
bash 20_make_jieba_dicts.sh 2>&1 | tee -a "$(pwd)/jieba_build.log"
```

输出目录：`${OUT_ROOT}/jieba/*.txt`

### 5) 重建数据集（word_ids 对齐）

```bash
cd /data/ocean/DAPT/experiments/tokenizer_ablation
bash 30_build_datasets.sh 2>&1 | tee -a "$(pwd)/dataset_build.log"
```

输出目录：`${OUT_ROOT}/datasets/processed_dataset_t{1,2,3,4}`

### 6) Quick-run（强烈建议先做，用小语料验证全链路可跑）

> 说明：`train_dapt_macbert_staged.py` 不支持 `--max_steps` 这类短跑参数；
> quick-run 的推荐方式是用小语料（减少每个 epoch 的耗时）。

```bash
cd /data/ocean/DAPT/experiments/tokenizer_ablation

# 6.1 生成小语料
bash 31_make_quick_corpus.sh

# 6.2 用小语料重建 quick datasets
bash 32_build_datasets_quick.sh
```

### 7) 预训练 quick（每个 tokenizer 变体跑 1-round/1-epoch 做 sanity）

以下命令按 T1/T3/T4 先跑（T2 视时间再补）。把 `${OUT_ROOT}` 换成你的真实输出根目录。

```bash
# 例：T4 quick
python /data/ocean/DAPT/train_dapt_macbert_staged.py \
  --output_dir ${OUT_ROOT}/runs/t4_quick \
  --dataset_path ${OUT_ROOT}/datasets_quick/processed_dataset_t4 \
  --tokenizer_path ${OUT_ROOT}/tokenizers/t4_ocr_llm_keys \
  --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
  --nsp_data_dir /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
  --num_rounds 1 \
  --mlm_epochs_per_round 1 \
  --nsp_epochs_per_round 1 \
  --learning_rate 5e-5 \
  2>&1 | tee ${OUT_ROOT}/runs/t4_quick/train.log

# 例：T1 quick
python /data/ocean/DAPT/train_dapt_macbert_staged.py \
  --output_dir ${OUT_ROOT}/runs/t1_quick \
  --dataset_path ${OUT_ROOT}/datasets_quick/processed_dataset_t1 \
  --tokenizer_path ${OUT_ROOT}/tokenizers/t1_base \
  --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
  --nsp_data_dir /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
  --num_rounds 1 \
  --mlm_epochs_per_round 1 \
  --nsp_epochs_per_round 1 \
  --learning_rate 5e-5 \
  2>&1 | tee ${OUT_ROOT}/runs/t1_quick/train.log
```

### 8) 正式跑（建议）

在 quick-run 确认链路无误后，再用 full dataset 跑 1-seed：

```bash
# 例：T4 full（1-seed）
python /data/ocean/DAPT/train_dapt_macbert_staged.py \
  --output_dir ${OUT_ROOT}/runs/t4_full_seed42 \
  --dataset_path ${OUT_ROOT}/datasets/processed_dataset_t4 \
  --tokenizer_path ${OUT_ROOT}/tokenizers/t4_ocr_llm_keys \
  --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
  --nsp_data_dir /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
  --num_rounds 3 \
  --mlm_epochs_per_round 1 \
  --nsp_epochs_per_round 3 \
  --learning_rate 5e-5 \
  2>&1 | tee ${OUT_ROOT}/runs/t4_full_seed42/train.log
```

> 备注：脚本内部分别会创建 `round_*_mlm/round_*_nsp/` 子目录。你也可以把 `--output_dir` 设成包含 expid 的路径便于管理。

> 说明：上面 0~8 节已经给出**推荐的统一可复现流程**（基于 `/data/ocean/DAPT/experiments/tokenizer_ablation/`）。
> 本文档不再建议用旧的 `final_merge_v9_regex_split_slim.py`/`generate_jieba_vocab.py` 作为消融入口（它们有硬编码路径，容易造成不同机器不可复现）。

8) 下游微调评估（用同一训练脚本/配置对比）

- 若你使用统一 finetune 脚本，请把 `model_name_or_path` 指向 quick-run 输出或最终预训练输出，然后评估 Task1/2/3。

9) 记录模板（CSV/TSV 推荐）

| ExpID | Tokenizer | VocabSize | DatasetPath | Seed | Task1_F1 | Task2_F1 | Task3_F1 | TrainTime | Notes |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| T1 | base | 21128 | /.../processed_dataset_t1 | 42 |  |  |  |  |  |

---

## 成功判定（建议）
- 若 T4 显著优于 T1/T3（在 Task3 上提升 ≥0.5% absolute 且在 3-seed 上稳定），说明 LLM 过滤与 keys 合并带来增益。
- 若 T3 低于 T1，说明盲目扩词在实际场景中会伤性能，需在论文中强调“过滤”必要性。

---

## 预算与时间建议
- 生成 tokenizer + vocab 处理：几小时以内（磁盘+CPU）
- 重建数据集（单次）：数小时，视数据大小与 num_proc
- 快跑训练：使用 `datasets_quick`（小语料），数小时内（单卡）
- 完整 3-seed 对比（若做预训练级别）：需要相当多资源，建议只对最终候选做 3-seed。

---

## 输出与论文写法建议
- 主文表放 T1/T4/T3 的对比（3-seed），T2 放在补充或附录（若空间不足）。
- 给出 vocab size、OOV 率（或未知 token 比例）、以及典型错误案例（盲目扩词引入的稀疏子词示例）。

---

如果你同意，我下一步会：
- 把 `manage_todo_list` 中第 2 项状态标为 in-progress（准备 tokenizer 变体）；
- 或者我可以直接把每个变体的具体命令脚本写成一组 runnable shell 脚本放在 `DAPT/experiments/tokenizer_ablation/` 下。

请选择下一步（直接写脚本 / 先更新 todo / 先跑哪个变体）。
