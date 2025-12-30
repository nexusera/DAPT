KV 预训练合集（KV-MLM & KV-NSP）
================================

目的
----
将 KV 相关的预训练脚本集中到 `kv_pretrain/`，方便在本地或远端（如 `/data/ocean/BERT_DAPT`）统一管理与同步。包含：
- KV-MLM / 通用 MLM / 位置扩展 MLM 训练脚本
- KV-NSP 任务（判定键值是否匹配）
- 数据检查与重分词辅助脚本
- 原 pipeline 核心数据处理脚本已整体迁移自 `stage_1/`。

目录结构
--------
- `train_dapt_kvmlm.py`：KV-aware MLM 训练（保留 `word_ids`，不扩词表）。
- `train_dapt_mlm.py`：在扩展词表后继续通用 MLM 训练。
- `train_dapt_base_mlm_resize.py`：仅扩展 position embeddings，再做通用 MLM。
- `apply_added_vocab.py`：向基座模型注入新增词并 `resize_token_embeddings`。
- `check_processed_dataset.py`：检查 `processed_dataset` 中是否含 `input_ids` / `word_ids`。
- `retokenize_processed_dataset_with_wordids.py`：用当前 tokenizer 重新分词，生成带 `word_ids` 的数据。
- `extract_and_dedup_json_v2.py`：源数据清洗与去重，生成 `train.txt` 和按源拆分文件。
- `train_ocr_clean.py`：OCR 词表挖掘（WordPiece）与噪声清洗，生成 `medical_vocab_ocr_only/vocab.txt`。
- `final_merge_v9_regex_split_slim.py`：精简版 tokenizer 合并（去掉外部词典），产出 `my-medical-tokenizer/`。
- `generate_jieba_vocab.py`：从 OCR 词表生成 `vocab_for_jieba.txt` 供 Jieba 使用。
- `build_dataset_final_slim.py`：基于 Jieba + tokenizer 构建带 `word_ids` 的 `processed_dataset/`。
- `train_dapt_distributed.py`：RoBERTa slim 版 8 卡 KV-aware MLM（WWM）训练脚本。
- `kv_nsp/`
  - `dataset.py`：动态负采样（hard reverse + random value）生成 KV-NSP 样本。
  - `run_train.py`：KV-NSP 二分类训练脚本（Trainer + Bert/RoBERTa）。

使用约定
--------
- 请在仓库根目录运行脚本，保持相对路径有效：`cd /data/ocean/BERT_DAPT`（远端）或本地同级路径。
- 默认基座为 `hfl/chinese-roberta-wwm-ext`（slim 流程），如需替换请修改命令中的 `model_name_or_path`。
- 输出目录务必分开，避免覆盖不同实验。

数据管线（核心步骤，已从 `stage_1/` 迁入）
--------------------------------------
0) 源数据清洗与去重（生成 `train.txt` + 按源拆分）
```bash
cd /data/ocean/BERT_DAPT
python kv_pretrain/extract_and_dedup_json_v2.py
```

1) OCR 词表挖掘与清洗（`medical_vocab_ocr_only/vocab.txt`）
```bash
python kv_pretrain/train_ocr_clean.py
```

2) 精简版 tokenizer 合并（去外部词典，产出 `my-medical-tokenizer/`）
```bash
python kv_pretrain/final_merge_v9_regex_split_slim.py
```

3) 生成 Jieba 外挂词典（`vocab_for_jieba.txt`）
```bash
python kv_pretrain/generate_jieba_vocab.py
```

4) 构建对齐数据集（含 `word_ids`），保存到 `processed_dataset/`
```bash
python kv_pretrain/build_dataset_final_slim.py
```

5) RoBERTa slim 版分布式 KV-aware MLM 训练（WWM）
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
  kv_pretrain/train_dapt_distributed.py \
  --output_dir /data/ocean/bpe_workspace/output_medical_bert_v2_8gpu
```

KV-NSP 训练（键值匹配二分类）
--------------------------
数据：Label Studio 导出的 5 个 JSON（例：`/data/ocean/FT_workspace/ner-finetune/data`）。  
命令示例（RoBERTa 基座，90/10 划分，动态 50% 负采样）：
```bash
cd /data/ocean/BERT_DAPT
python kv_pretrain/kv_nsp/run_train.py \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --data_dir /data/ocean/FT_workspace/ner-finetune/data \
  --output_dir /data/ocean/bpe_workspace/output_kv_nsp_roberta \
  --max_length 256 \
  --negative_prob 0.5 \
  --hard_negative_prob 0.5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32
```
要点：
- 输入格式 `[CLS] Key [SEP] Value [SEP]`，`token_type_ids` 自动区分段 A/B。
- 负样本包含“倒序”(Value→Key) 和 “随机 Value” 两种，比例由 `hard_negative_prob` 控制。
- `compute_metrics` 返回 Accuracy/Precision/Recall/F1，日志与模型保存到 `--output_dir`。

KV-MLM 流程（slim 版本，RoBERTa 基座）
-----------------------------------
1) 数据检查  
```bash
cd /data/ocean/BERT_DAPT
python kv_pretrain/check_processed_dataset.py /data/ocean/bpe_workspace/processed_dataset
```
确认 `input_ids` 与 `word_ids` 均存在。若 `word_ids` 不匹配当前 tokenizer：
```bash
python kv_pretrain/retokenize_processed_dataset_with_wordids.py
```

2) （可选）扩展词表并保存 checkpoint  
```bash
python kv_pretrain/apply_added_vocab.py
# 产出：output_medical_bert_add_vocab/final_model
```

3) KV-aware MLM 训练（不扩表，保留 word_ids）  
```bash
CUDA_VISIBLE_DEVICES=3,4,5 torchrun --nproc_per_node=3 \
  kv_pretrain/train_dapt_kvmlm.py 2>&1 | tee /data/ocean/bpe_workspace/train_kvmlm.log
# 输出目录：脚本内默认 output_medical_bert_kvmlm（可在脚本中修改）
```
注意：保持 `remove_unused_columns=False`，以便 collator 读取 `word_ids`。

4) 扩表后通用 MLM（继续学新词语义）  
```bash
CUDA_VISIBLE_DEVICES=3 python kv_pretrain/train_dapt_mlm.py 2>&1 | tee /data/ocean/bpe_workspace/train_dapt_mlm.log
# 输出：output_medical_bert_add_vocab_mlm/final_model
```

5) 仅扩 position embeddings 的通用 MLM（基座词表不变）  
```bash
CUDA_VISIBLE_DEVICES=3,4,5 torchrun --nproc_per_node=3 \
  kv_pretrain/train_dapt_base_mlm_resize.py 2>&1 | tee /data/ocean/bpe_workspace/train_base_mlm_resize.log
# 输出：output_medical_bert_base_mlm_resize/final_model
```

同步与路径提醒
--------------
- 远端：在 `/data/ocean/` 下 `git pull` 后，脚本路径更新为 `kv_pretrain/...`。请同步更新启动命令。
- 若旧流程引用 `stage_1/train_dapt_kvmlm.py` 等路径，请改为 `kv_pretrain/train_dapt_kvmlm.py`。
- 输出目录建议使用绝对路径区分不同实验（如 `/data/ocean/bpe_workspace/output_kv_nsp_roberta`）。

FAQ 快记
--------
- 训练爆显存：降低 `per_device_train_batch_size` 或开启梯度累积；KV-MLM 可适当减小 `MAX_SEQ_LEN`（脚本内常量）。
- bfloat16 不可用：在对应脚本中改用 `fp16` 或关闭混合精度。
- 评估下游：KV-NSP 直接用 Trainer evaluate；KV-MLM 完成后请按各自下游微调配置（见 `pre_struct/kv_ner/*.json`）进行评估。

