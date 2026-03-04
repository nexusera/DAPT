# Tokenizer Ablation Runner

这套脚本用于复现/跑通 Tokenizer 相关消融（T1~T4），并尽量复用你现有 pipeline：
- Tokenizer 生成：使用本目录的 `build_tokenizer_variant.py`（可配置，避免改动你现有脚本里硬编码的 `/data/ocean` 路径）
- Jieba 词典生成：使用本目录的 `build_jieba_dict.py`（每个 tokenizer 变体一份，避免 confound）
- 数据集构建：直接调用 `DAPT/build_dataset_final_slim.py`（支持 `--tokenizer_path/--vocab_for_jieba/--keys_file`）

## 快速开始

1. 复制配置并修改路径：

```bash
cd DAPT/experiments/tokenizer_ablation
cp config.env.example config.env
# 然后编辑 config.env
```

2. 生成 tokenizer 变体：

```bash
bash 10_make_tokenizers.sh
```

3. 生成每个变体对应的 Jieba 词典：

```bash
bash 20_make_jieba_dicts.sh
```

4. 为每个 tokenizer 重建数据集：

```bash
bash 30_build_datasets.sh
```

（可选）先用小语料 quick-run：

```bash
bash 31_make_quick_corpus.sh
bash 32_build_datasets_quick.sh
```

## 输出结构（默认）

- `${OUT_ROOT}/tokenizers/t1_base/` ...
- `${OUT_ROOT}/tokenizers/t2_keys/` ...
- `${OUT_ROOT}/tokenizers/t3_ocr_raw/` ...
- `${OUT_ROOT}/tokenizers/t4_ocr_llm_keys/` ...

- `${OUT_ROOT}/jieba/t1_base.txt` ...
- `${OUT_ROOT}/datasets/processed_dataset_t1/` ...

## 说明

- T1 会把 base tokenizer “快照”保存到输出目录，用于保证版本可复现。
- 若某个输入词表不存在（例如没有 OCR raw/kept），脚本会报错并退出，避免你跑到一半才发现数据缺失。
