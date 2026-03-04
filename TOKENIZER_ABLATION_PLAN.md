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

## 一句话入口（只看一个文件）

从“准备数据 → 构建 4 套 tokenizer/dataset → 四个变体训练（quick/full）”的所有可复制命令，已合并到：

- [DAPT/experiments/tokenizer_ablation/README.md](DAPT/experiments/tokenizer_ablation/README.md)

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


