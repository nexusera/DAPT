# Tokenizer Ablation Runner

这套脚本用于复现/跑通 Tokenizer 相关消融（T1~T4），并尽量复用你现有 pipeline。

本文档是 **唯一入口**：从准备数据 → 构建 4 套 tokenizer/dataset → 四个变体训练（quick/full）都在这里。

> 远端路径假设：仓库根目录为 `/data/ocean/DAPT`（不存在 `/data/ocean/DAPT/DAPT`）。

## 你要做什么（四个全跑）

- **准备数据（一次）**：`00_check_env.sh` → `10_make_tokenizers.sh` → `20_make_jieba_dicts.sh` → `30_build_datasets.sh`
- **关键修复（一次）**：`15_repair_fast_tokenizers.sh`（为每个 tokenizer 变体重建 `tokenizer.json`，保证下游 Fast Tokenizer 的 `return_offsets_mapping` 可用）
- **训练（四个全跑）**：用 `train_dapt_macbert_staged.py` 分别跑 T1/T2/T3/T4（建议先 quick，确认无误后再 full）
- Tokenizer 生成：使用本目录的 `build_tokenizer_variant.py`（可配置，避免改动你现有脚本里硬编码的 `/data/ocean` 路径）
- Jieba 词典生成：使用本目录的 `build_jieba_dict.py` 生成**一份共享词典**（VIP 基础词表 + keys_min5 + OCR kept vocab），用于所有变体
- 数据构建是“两阶段”（消融更干净）：
	- Stage A：用共享 Jieba 词典对语料分词，产出 `words` 数据集（一次性）
	- Stage B：对同一份 `words`，分别用不同 tokenizer 生成各自的 `input_ids/word_ids`
- 数据集构建：按 `docs/pipelines/pipeline_new.md` 的规则分两路构建再合并
	- 非 OCR 路：可 `shuffle_split`
	- OCR 路：必须 `--no_shuffle_split`，并在构建后调用 `add_noise_features.py` 写入 `noise_values`，再用 `verify_noise_alignment.py` 抽检
	- 最终用 `merge_datasets.py` 合并为训练用的 merged dataset

## 快速开始

### 0)（可选但强烈建议）记录可复现信息

```bash
cd /data/ocean/DAPT
git rev-parse HEAD | tee -a /data/ocean/DAPT/ablation_tokenizer_gitsha.log
git status --porcelain | tee -a /data/ocean/DAPT/ablation_tokenizer_gitsha.log

python -V | tee -a /data/ocean/DAPT/ablation_tokenizer_env.log
python -m pip freeze | tee -a /data/ocean/DAPT/ablation_tokenizer_env.log
```

### 1) 准备配置（只改一次）

```bash
cd /data/ocean/DAPT/experiments/tokenizer_ablation

# 若仓库已包含 config.env，可直接用；需要自定义时再修改。


# 如需修改 OUT_ROOT / 输入文件路径，请编辑 config.env
# vim config.env

# 校验必需文件是否存在，并创建输出目录
bash 00_check_env.sh
```

### 2) 生成 4 个 tokenizer 变体（T1~T4）

```bash
cd /data/ocean/DAPT/experiments/tokenizer_ablation
bash 10_make_tokenizers.sh 2>&1 | tee -a "$(pwd)/tokenizer_build.log"
```

### 2.5)（强烈建议）修复/生成 Fast Tokenizer 配置（tokenizer.json）

背景：我们在消融中会合并/扩充 `vocab.txt`。如果目录里残留了旧的 `tokenizer.json`（Fast backend 配置），
`use_fast=True` 可能加载到不匹配的后端，导致中文短语被切成单个 `[UNK]`。

下游 KV-NER/EBQA 依赖 `return_offsets_mapping=True`（通常需要 Fast tokenizer），所以必须保证 fast/slow 行为一致。

补充说明（关于“现在用 vocab.txt 了，是否还需要 repair/slow？”）：

- 现在我们确实以 `vocab.txt` 为准（新增词直接写入 vocab），这是正确方向。
- 但 **fast tokenizer 是否可靠** 不只取决于 `vocab.txt`，还取决于目录里是否存在/残留不匹配的 `tokenizer.json`。
	- 一旦 `tokenizer.json` 与 `vocab.txt` 不一致，`use_fast=True` 可能加载到旧后端，出现 all-UNK 或 offsets 异常。
- 因此：
	- **下游（KV-NER/EBQA）仍建议跑 `15_repair_fast_tokenizers.sh`**：它会按 `vocab.txt` 确定性重建 fast 后端并做 offsets/self-test。
	- **预训练阶段不强依赖 fast**（不需要 offsets），你可以继续默认用 slow（更稳定、变量更少），也可以在 fast 已通过 repair+self-test 后改用 fast。

```bash
cd /data/ocean/DAPT/experiments/tokenizer_ablation
bash 15_repair_fast_tokenizers.sh 2>&1 | tee -a "$(pwd)/tokenizer_repair.log"
```

这一步会对每个变体：
- 备份并重建 `tokenizer.json`
- 运行快速自检（同一目录分别用 fast/slow tokenize 一组中文 probe）

### 3) 生成每个变体对应的 Jieba 词典（避免消融 confound）

> 更新：为保证“只消融 tokenizer、不消融 jieba 分词边界”，现在改为**只生成一份共享 Jieba 词典**。

```bash
cd /data/ocean/DAPT/experiments/tokenizer_ablation
bash 20_make_jieba_dicts.sh 2>&1 | tee -a "$(pwd)/jieba_build.log"
```

### 4) 为每个 tokenizer 重建数据集（non-OCR + OCR(with noise) -> merged）

该步骤会严格遵守 `docs/pipelines/pipeline_new.md` 的数据规则：

- non-OCR 路：允许 shuffle
- OCR 路：强制不 shuffle，构建后写入 `noise_values`，并运行 `verify_noise_alignment.py` 抽检
- 最终使用 `merge_datasets.py` 合并成训练用的 merged dataset

tmux new -s run30
```bash
cd /data/ocean/DAPT/experiments/tokenizer_ablation
bash 30_build_datasets.sh 2>&1 | tee -a "$(pwd)/dataset_build.log"
```

实现细节（便于你核对“变量控制”是否符合预期）：

- `30_build_datasets.sh` 会先用共享 Jieba 词典构建两份 `words` 数据集（non-OCR/OCR 各一份）
- 再对这两份 `words` 分别用 t1~t4 tokenizer 生成各自的 `processed_dataset_t*`

### 5)（强烈建议）Quick 数据集（小语料）

`train_dapt_macbert_staged.py` 不支持 `--max_steps`，因此 quick-run 推荐用“小语料”来缩短每个 epoch。

```bash
cd /data/ocean/DAPT/experiments/tokenizer_ablation
bash 31_make_quick_corpus.sh
bash 32_build_datasets_quick.sh
```

## 训练（四个全跑）

### 6) Quick 训练（四个变体都跑一遍 sanity）

把下面的 `${OUT_ROOT}` 换成你 `config.env` 里的 `OUT_ROOT`（默认是 `/data/ocean/DAPT/ablation/tokenizer`）。

```bash
OUT_ROOT=/data/ocean/DAPT/ablation/tokenizer

# T1 quick
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

# T2 quick
python /data/ocean/DAPT/train_dapt_macbert_staged.py \
	--output_dir ${OUT_ROOT}/runs/t2_quick \
	--dataset_path ${OUT_ROOT}/datasets_quick/processed_dataset_t2 \
	--tokenizer_path ${OUT_ROOT}/tokenizers/t2_keys \
	--noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
	--nsp_data_dir /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
	--num_rounds 1 \
	--mlm_epochs_per_round 1 \
	--nsp_epochs_per_round 1 \
	--learning_rate 5e-5 \
	2>&1 | tee ${OUT_ROOT}/runs/t2_quick/train.log

# T3 quick（仅当你确实有 OCR raw vocab 时才会生成这个 tokenizer 目录）
python /data/ocean/DAPT/train_dapt_macbert_staged.py \
	--output_dir ${OUT_ROOT}/runs/t3_quick \
	--dataset_path ${OUT_ROOT}/datasets_quick/processed_dataset_t3 \
	--tokenizer_path ${OUT_ROOT}/tokenizers/t3_ocr_raw \
	--noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
	--nsp_data_dir /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
	--num_rounds 1 \
	--mlm_epochs_per_round 1 \
	--nsp_epochs_per_round 1 \
	--learning_rate 5e-5 \
	2>&1 | tee ${OUT_ROOT}/runs/t3_quick/train.log

# T4 quick
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
```

### 7) Full 训练（四个变体全量训练，1-seed）

```bash
OUT_ROOT=/data/ocean/DAPT/ablation/tokenizer

# 注意：
# - 如果你想“先写一行变量、下一行再运行 python”，必须用 export；否则 python 子进程看不到变量。
# - 更保险的一行写法是：CUDA_VISIBLE_DEVICES=2,3 OUT_ROOT=... python ...（只对这一条命令生效）。
# - 请避开被其他任务占满的 GPU（例如你日志里 vLLM 占用的 GPU6/7）。

# T1 full
tmux new -s t1_full
cd /data/ocean/DAPT
conda activate medical_bert

export CUDA_VISIBLE_DEVICES=2,3
export OUT_ROOT=/data/ocean/DAPT/ablation/tokenizer
python /data/ocean/DAPT/train_dapt_macbert_staged.py \
	--output_dir ${OUT_ROOT}/runs/t1_full_seed42 \
	--dataset_path ${OUT_ROOT}/datasets/processed_dataset_t1 \
	--tokenizer_path ${OUT_ROOT}/tokenizers/t1_base \
	--noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
	--nsp_data_dir /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
	--num_rounds 3 \
	--mlm_epochs_per_round 1 \
	--nsp_epochs_per_round 3 \
	--learning_rate 5e-5 \
	2>&1 | tee ${OUT_ROOT}/runs/t1_full_seed42/train.log

# T2 full
tmux new -s t2_full

export CUDA_VISIBLE_DEVICES=2,3
export OUT_ROOT=/data/ocean/DAPT/ablation/tokenizer
python /data/ocean/DAPT/train_dapt_macbert_staged.py \
	--output_dir ${OUT_ROOT}/runs/t2_full_seed42 \
	--dataset_path ${OUT_ROOT}/datasets/processed_dataset_t2 \
	--tokenizer_path ${OUT_ROOT}/tokenizers/t2_keys \
	--noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
	--nsp_data_dir /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
	--num_rounds 3 \
	--mlm_epochs_per_round 1 \
	--nsp_epochs_per_round 3 \
	--learning_rate 5e-5 \
	2>&1 | tee ${OUT_ROOT}/runs/t2_full_seed42/train.log

# T3 full（同 quick：仅当你确实生成了 t3_ocr_raw tokenizer）
tmux new -s t3_full

export CUDA_VISIBLE_DEVICES=4,5
export OUT_ROOT=/data/ocean/DAPT/ablation/tokenizer
python /data/ocean/DAPT/train_dapt_macbert_staged.py \
	--output_dir ${OUT_ROOT}/runs/t3_full_seed42 \
	--dataset_path ${OUT_ROOT}/datasets/processed_dataset_t3 \
	--tokenizer_path ${OUT_ROOT}/tokenizers/t3_ocr_raw \
	--noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
	--nsp_data_dir /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
	--num_rounds 3 \
	--mlm_epochs_per_round 1 \
	--nsp_epochs_per_round 3 \
	--learning_rate 5e-5 \
	2>&1 | tee ${OUT_ROOT}/runs/t3_full_seed42/train.log

# T4 full
tmux new -s t4_full
export CUDA_VISIBLE_DEVICES=0,1,4,5
export OUT_ROOT=/data/ocean/DAPT/ablation/tokenizer
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

## 输出结构（默认）

- `${OUT_ROOT}/tokenizers/t1_base/` ...
- `${OUT_ROOT}/tokenizers/t2_keys/` ...
- `${OUT_ROOT}/tokenizers/t3_ocr_raw/` ...
- `${OUT_ROOT}/tokenizers/t4_ocr_llm_keys/` ...

- `${OUT_ROOT}/jieba/shared_kept_keys_min5.txt`（共享 Jieba userdict）

- `${OUT_ROOT}/datasets_words/nonocr_words/`（non-OCR 共享 words 数据集）
- `${OUT_ROOT}/datasets_words/ocr_words/`（OCR 共享 words 数据集）

- `${OUT_ROOT}/datasets/processed_dataset_t1/` ...（最终 merged，用于训练）
- `${OUT_ROOT}/datasets/nonocr/processed_dataset_t1/` ...（中间产物）
- `${OUT_ROOT}/datasets/ocr/processed_dataset_t1_with_noise/` ...（中间产物）

## 说明

- T1 会把 base tokenizer “快照”保存到输出目录，用于保证版本可复现。
- 若某个输入词表不存在（例如没有 OCR raw/kept），脚本会报错并退出，避免你跑到一半才发现数据缺失。

## 我到底应该看哪个文件？

- 只看这一份即可：`/data/ocean/DAPT/experiments/tokenizer_ablation/README.md`
