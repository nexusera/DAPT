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
- OCR vocab 挖掘：`DAPT/train_ocr_clean.py`
- LLM 过滤：`DAPT/filter_vocab_with_llm.py`
- Tokenizer 合并：`DAPT/final_merge_v9_regex_split_slim.py`
- Jieba 词典生成：`DAPT/generate_jieba_vocab.py`
- 数据构建：`DAPT/build_dataset_final_slim.py`（或 `retokenize_processed_dataset_with_wordids.py`）
- 训练入口：`DAPT/train_dapt_macbert_staged.py`

---

## 具体流程（可复制的步骤）

1) 准备 base tokenizer（T1）

```bash
# 假设已有基础 tokenizer 目录 my-medical-tokenizer/base
# 若没有，先复制官方/公司 base tokenizer 到工作目录
cp -r /path/to/base-tokenizer /data/ocean/DAPT/my-medical-tokenizer/base
```

2) 生成 OCR vocab（用于 T3/T4）

```bash
# 在 pipeline 中运行 OCR 词表挖掘（以 train_chunked.txt 为语料）
python DAPT/train_ocr_clean.py \
  --corpus /data/ocean/DAPT/workspace/train_chunked.txt \
  --output_dir /data/ocean/DAPT/workspace/medical_vocab_ocr_only
# 结果在 /data/ocean/DAPT/workspace/medical_vocab_ocr_only/vocab.txt
```

3) 对 OCR vocab 做 LLM 过滤（生成 kept_vocab.txt，用于 T4）

```bash
# 需要本地 LLM 服务（如 pipeline 文档所述）
cd /data/ocean/DAPT
export LLF_API_BASE="http://127.0.0.1:8008/v1"
export OPENAI_API_KEY="EMPTY"
python filter_vocab_with_llm.py \
  --vocab /data/ocean/DAPT/workspace/medical_vocab_ocr_only/vocab.txt \
  --kept /data/ocean/DAPT/workspace/kept_vocab.txt \
  --dropped /data/ocean/DAPT/workspace/dropped_vocab.txt \
  --batch_size 64 \
  --topn 50000
```

4) 生成各变体的 tokenizer（使用 `final_merge_v9_regex_split_slim.py`）

```bash
# T2: base + keys
python DAPT/final_merge_v9_regex_split_slim.py \
  --base_tokenizer /data/ocean/DAPT/my-medical-tokenizer/base \
  --keys_file DAPT/biaozhu_keys_only.txt \
  --output_dir /data/ocean/DAPT/my-medical-tokenizer/t2_keys

# T3: base + ocr_raw
python DAPT/final_merge_v9_regex_split_slim.py \
  --base_tokenizer /data/ocean/DAPT/my-medical-tokenizer/base \
  --ocr_vocab /data/ocean/DAPT/workspace/medical_vocab_ocr_only/vocab.txt \
  --output_dir /data/ocean/DAPT/my-medical-tokenizer/t3_ocr_raw

# T4: base + ocr_llm_filtered + keys
python DAPT/final_merge_v9_regex_split_slim.py \
  --base_tokenizer /data/ocean/DAPT/my-medical-tokenizer/base \
  --ocr_vocab /data/ocean/DAPT/workspace/kept_vocab.txt \
  --keys_file DAPT/biaozhu_keys_only.txt \
  --output_dir /data/ocean/DAPT/my-medical-tokenizer/t4_ocr_llm_keys
```

- 运行后请记录每个 tokenizer 的 `vocab.txt` 大小（`wc -l`）并保存为日志。

```bash
wc -l /data/ocean/DAPT/my-medical-tokenizer/t4_ocr_llm_keys/vocab.txt
```

5) 生成对应的 Jieba 词典（若 pipeline 要求）

```bash
# 若 final_merge 脚本已生成 vocab_for_jieba，可跳过；否则：
python DAPT/generate_jieba_vocab.py \
  --kept_vocab /data/ocean/DAPT/workspace/kept_vocab.txt \
  --keys_file DAPT/biaozhu_keys_only_min5.txt \
  --output /data/ocean/DAPT/vocab_for_jieba.txt
```

6) 为每个 tokenizer 重建/retokenize 数据集

```bash
# 示例：为 T4 生成 processed dataset（OCR 路或 non-ocr 路视需求）
python DAPT/build_dataset_final_slim.py \
  --train_file /data/ocean/DAPT/workspace/train_chunked.txt \
  --output_path /data/ocean/DAPT/workspace/processed_dataset_t4 \
  --tokenizer_path /data/ocean/DAPT/my-medical-tokenizer/t4_ocr_llm_keys \
  --shuffle_split

# 对 OCR-only 数据集请使用 --no_shuffle_split 并对应 OCR 路的 train_file
```

7) 快跑验证训练（短训练或少量 steps）

```bash
# 建议先做 short sanity run（节省算力），例如 1 epoch 或指定 steps
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 CUDA_VISIBLE_DEVICES=0 \
python DAPT/train_dapt_macbert_staged.py \
  --dataset_dir /data/ocean/DAPT/workspace/processed_dataset_t4 \
  --output_dir /data/ocean/DAPT/workspace/output_tokenizer_t4_quick \
  --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
  --max_steps 500 \
  --per_device_train_batch_size 32 \
  --logging_steps 50
```

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
- 快跑训练（max_steps=500）：数小时（单卡）
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
