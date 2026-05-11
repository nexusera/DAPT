# 下游 KV-NER 微调与评估
tmux new -s ner
tmux attach -t ner
## 方案 A：精简版（slim DAPT，去 medical_dict）

训练（在 ner-finetune 根目录运行，使用 slim 配置）：
```bash
cd /data/ocean/FT_workspace/ner-finetune
CUDA_VISIBLE_DEVICES=2 python pre_struct/kv_ner/train.py --config pre_struct/kv_ner/kv_ner_config_1224.json 2>&1 | tee train_ner_1224.log
```
如要多卡（需脚本支持分布式）：
```bash
CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 pre_struct/kv_ner/train.py --config pre_struct/kv_ner/kv_ner_config_slim.json 2>&1 | tee train_ner_slim.log
```

评估/预测（跑之前的 evaluate 流程）：
```bash
cd /data/ocean/FT_workspace/ner-finetune

CUDA_VISIBLE_DEVICES=4 python pre_struct/kv_ner/evaluate.py \
  --config pre_struct/kv_ner/kv_ner_config_1224.json \
  --model_dir runs/kv_ner4_bioe_slim_exp1/best \
  --test_data data/kv_ner_prepared/val_eval.jsonl \
  --output_dir data/kv_ner_eval_bioe_slim_exp1 \
  2>&1 | tee eval_ner_slim_exp1.log

CUDA_VISIBLE_DEVICES=4 python pre_struct/kv_ner/evaluate.py \
	--config pre_struct/kv_ner/kv_ner_config_1224.json \  # 换成本次配置（已指向 exp1 模型/分词器）
	--model_dir runs/kv_ner4_bioe_slim_exp1/best \        # 对应本次微调 best
	--test_data data/kv_ner_prepared/val_eval.jsonl \
	--output_dir data/kv_ner_eval_bioe_slim_exp1 \        # 避免覆盖旧评估
	2>&1 | tee eval_ner_slim_exp1.log                     # 日志改名防覆盖
```

---

## 方案 B：全量版（含 medical_dict，原始配置）

训练（使用全量 tokenizer 与配置）：
```bash
cd /data/ocean/FT_workspace/ner-finetune
CUDA_VISIBLE_DEVICES=3 python pre_struct/kv_ner/train.py --config pre_struct/kv_ner/kv_ner_config.json 2>&1 | tee train_ner_full.log
```
如要多卡（需脚本支持分布式）：
```bash
CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 pre_struct/kv_ner/train.py --config pre_struct/kv_ner/kv_ner_config.json 2>&1 | tee train_ner_full.log
```
 CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 pre_struct/kv_ner/train.py --config pre_struct/kv_ner/kv_ner_config.json 2>&1 | tee train_ner_full.log

评估/预测：
```bash
cd /data/ocean/FT_workspace/ner-finetune
CUDA_VISIBLE_DEVICES=3 python pre_struct/kv_ner/evaluate.py \
	--config pre_struct/kv_ner/kv_ner_config.json \
	--model_dir runs/kv_ner4_bioe/best \
	--test_data data/kv_ner_prepared/val_eval.jsonl \
	--output_dir data/kv_ner_eval_bioe \
	2>&1 | tee eval_ner_full.log
```

---

## 通用说明（适用于两种方案）

若更换模型或数据：
- `CUDA_VISIBLE_DEVICES=3` 指定空闲的 GPU（根据 `nvidia-smi` 调整）
- `--model_dir` 指向你要评估的 checkpoint（如新的 best 目录）
- `--test_data` 改为要评估的 JSONL 文件路径（默认用 val_eval.jsonl）
- `--output_dir` 改为新的结果输出目录以免覆盖

如需替换上游 DAPT/Tokenizer，请修改对应配置文件（如 `pre_struct/kv_ner/kv_ner_config_slim.json`）：
- `model_name_or_path`：指向新的 DAPT 输出目录下的 `final_model`（如 `/data/ocean/bpe_workspace/output_medical_bert_exp1/final_model`）
- `tokenizer_name_or_path`：同上，指向同一个 `final_model` 目录（或新的分词器目录）
- `predict.model_dir` / `evaluate.model_dir`：指向 NER 训练产出的 best 目录（如 `runs/kv_ner4_bioe_slim_exp1/best`，可在命令行通过 `--output_dir` 设置不同 runs 目录）
示例（配置片段）：
```
"model_name_or_path": "/data/ocean/bpe_workspace/output_medical_bert_exp1/final_model",
"tokenizer_name_or_path": "/data/ocean/bpe_workspace/output_medical_bert_exp1/final_model",
"predict": {
  "model_dir": "runs/kv_ner4_bioe_slim_exp1/best"
},
"evaluate": {
  "model_dir": "runs/kv_ner4_bioe_slim_exp1/best"
}
```

如需更换数据：将配置里的 `data_path` / `val_data_path` / `eval_data_path` / `input_path` 替换为新数据路径；确保路径相对于运行目录或改为绝对路径。

对比实验建议：
- 同时保留 slim 和 full 两个配置文件，分别指向不同的 tokenizer / DAPT 输出目录
- 使用不同的 `--output_dir` 保存结果（如 `data/kv_ner_eval_bioe_slim` vs `data/kv_ner_eval_bioe_full`）
- 收集 `eval_summary.json` 对比 F1、key-level mismatch 等指标


## 方案 C：纯基座 RoBERTa-wwm-ext 微调与评估（新）
tmux new -s ner
tmux attach -t ner

如果你想直接查看基座模型 `hfl/chinese-roberta-wwm-ext` 做下游 KV-NER 的效果，可以使用下面的配置和命令。你本机的本地缓存路径为 `/data/ocean/cache/huggingface/hub/models--hfl--chinese-roberta-wwm-ext`，我们已生成对应的配置文件：

- 配置文件：`my-bert-finetune/pre_struct/kv_ner/kv_ner_config_roberta_base.json`

- 训练（单卡示例）：
```bash
cd /data/ocean/FT_workspace/ner-finetune
CUDA_VISIBLE_DEVICES=4 python pre_struct/kv_ner/train.py --config pre_struct/kv_ner/kv_ner_config_roberta_base.json 2>&1 | tee train_ner_roberta_base.log
```

- 训练（多卡示例，脚本需支持分布式）：
```bash
cd /data/ocean/FT_workspace/ner-finetune
CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 pre_struct/kv_ner/train.py --config pre_struct/kv_ner/kv_ner_config_roberta_base.json 2>&1 | tee train_ner_roberta_base.log
```

- 评估/预测：
```bash
cd /data/ocean/FT_workspace/ner-finetune
CUDA_VISIBLE_DEVICES=3 python pre_struct/kv_ner/evaluate.py \
	--config pre_struct/kv_ner/kv_ner_config_roberta_base.json \
	--model_dir runs/kv_ner4_bioe_roberta_base/best \
	--test_data data/kv_ner_prepared/val_eval.jsonl \
	--output_dir data/kv_ner_eval_bioe_roberta_base \
	2>&1 | tee eval_ner_roberta_base.log
```

说明：在配置中 `model_name_or_path` 与 `tokenizer_name_or_path` 指向本地缓存目录（上面路径），Transformers 会直接加载；如果你希望改为使用在线 HF ID，可以把两项改为 `hfl/chinese-roberta-wwm-ext`。

---
**消融实验（三组）**

目标：和纯基座模型对比，做三组消融：

A) 只增加词表（其他不变）
- 思路：把 `my-medical-tokenizer` 的新增词注入基座模型 embedding（不做额外 DAPT 微调），保存为新的 checkpoint，再用该 checkpoint 做下游 KV-NER 微调。
- 新增文件（已创建）：
	- `apply_added_vocab.py`：加载基座模型与 `my-medical-tokenizer`，执行 `model.resize_token_embeddings(len(tokenizer))` 并保存到 `/data/ocean/bpe_workspace/output_medical_bert_add_vocab/final_model`。
	- 配置：`my-bert-finetune/pre_struct/kv_ner/kv_ner_config_add_vocab.json`（已添加），用于下游微调，`model_name_or_path` 指向上面保存的位置，`tokenizer_name_or_path` 指向 `my-medical-tokenizer`。

运行步骤示例：
```bash
# 1) 在 DAPT workspace 生成带新词的 checkpoint
cd /data/ocean/bpe_workspace
conda activate medical_bert
python apply_added_vocab.py


**只用官方MLM训练（加入词表后继续训练）**

- **目的**：在把 [my-medical-tokenizer](my-medical-tokenizer) 的新词注入基座模型 embedding 后，继续做标准的原始 MLM 训练（非 KV-MLM），让模型真正学会新词的语义表示。

- **脚本**：已添加 [train_dapt_mlm.py](train_dapt_mlm.py) 到仓库根目录。

- **运行示例（单卡）**：
```bash
cd /data/ocean/bpe_workspace
conda activate medical_bert
# 单卡训练
CUDA_VISIBLE_DEVICES=3 python train_dapt_mlm.py 2>&1 | tee train_dapt_mlm.log
```

- **运行示例（多卡分布式，torchrun）**：
```bash
cd /data/ocean/bpe_workspace
conda activate medical_bert
CUDA_VISIBLE_DEVICES=3,4,5 torchrun --nproc_per_node=3 train_dapt_mlm.py 2>&1 | tee train_dapt_mlm.log
```

- **日志与输出**：默认输出目录为 `output_medical_bert_add_vocab_mlm`（可在 `train_dapt_mlm.py` 中修改）。训练结束会把最终模型与 tokenizer 保存到 `output_medical_bert_add_vocab_mlm/final_model`。

- **依赖（建议安装）**：如果你的环境还没装，请使用匹配 CUDA 的 `torch`，然后安装下面的包：
```bash
# 安装示例（按需调整 torch 版本以匹配 CUDA）
pip install -U transformers datasets accelerate sentencepiece tokenizers
```

- **重要注意事项**：
	- `train_dapt_mlm.py` 使用官方 `DataCollatorForLanguageModeling` 做动态遮盖，和 KV-MLM 不同：遮盖是按 token 随机的，不会参考 `word_ids`。
	- 脚本默认 `bf16=True`（参照原脚本设置）。如果你的 GPU/驱动或 PyTorch 不支持 bfloat16，请把 `bf16=False` 或改用 `fp16`（需要在 `TrainingArguments` 中调整或设置 `--fp16`）。
	- 确保 `processed_dataset` 存在且可被 `datasets.load_from_disk` 加载（脚本默认路径为 `/data/ocean/bpe_workspace/processed_dataset`）。数据应包含 `train` 和 `test` split，且已用 `tokenizer` 进行编码（即包含 `input_ids` 字段），否则需要先做 tokenization。
	- 在注入新词后（使用 `apply_added_vocab.py`）会执行 `model.resize_token_embeddings(len(tokenizer))`，务必先把 tokenizer（即 `my-medical-tokenizer`）放在 `TOKENIZER_PATH` 指定目录并能被 `AutoTokenizer.from_pretrained` 加载。
	- 若显存或吞吐受限，可减小 `per_device_train_batch_size` 或 `gradient_accumulation_steps`，或把 `MAX_SEQ_LEN` 降低。

- **如果需要我可以**：
	- 把脚本的 `bf16` 开关改为自动检测并回退到 `fp16/float32` 的更健壮实现；
	- 生成一个把原始文本转成 HuggingFace `datasets` tokenized 格式的辅助脚本（如果你当前的 `processed_dataset` 不是 tokenized）。


# 2) 用生成的 checkpoint 做下游 KV-NER 微调
cd /data/ocean/FT_workspace/ner-finetune
CUDA_VISIBLE_DEVICES=3 python pre_struct/kv_ner/train.py --config pre_struct/kv_ner/kv_ner_config_add_vocab.json 2>&1 | tee train_ner_add_vocab.log
```

C) 只权重克隆 + 普通MLM（基座模型）
- **目的**：在基座模型上只扩展 position embeddings 到1024（权重克隆），然后做标准MLM DAPT，不扩展词表，用于对比权重克隆对长上下文的影响。
- **新增脚本**：`train_dapt_base_mlm_resize.py`（基于 `train_dapt_mlm.py`，但使用基座tokenizer，不resize_token_embeddings，只resize position embeddings）。
- **运行示例（多卡）**：
```bash
cd /data/ocean/bpe_workspace
conda activate medical_bert
CUDA_VISIBLE_DEVICES=3,4,5 torchrun --nproc_per_node=3 train_dapt_base_mlm_resize.py 2>&1 | tee train_base_mlm_resize.log
```
- **输出**：`output_medical_bert_base_mlm_resize/final_model`。
- **下游微调**：
```bash
cd /data/ocean/FT_workspace/ner-finetune
CUDA_VISIBLE_DEVICES=3 python pre_struct/kv_ner/train.py --config pre_struct/kv_ner/kv_ner_config_base_mlm_resize.json 2>&1 | tee train_ner_base_mlm_resize.log
```
- **评估**：
```bash
cd /data/ocean/FT_workspace/ner-finetune
CUDA_VISIBLE_DEVICES=3 python pre_struct/kv_ner/evaluate.py \
	--config pre_struct/kv_ner/kv_ner_config_base_mlm_resize.json \
	--model_dir runs/kv_ner4_bioe_base_mlm_resize/best \
	--test_data data/kv_ner_prepared/val_eval.jsonl \
	--output_dir data/kv_ner_eval_bioe_base_mlm_resize \
	2>&1 | tee eval_ner_base_mlm_resize.log
```



---

**只做 kv-MLM（不扩展词表）—— 新增脚本与数据检查**

- **目标**：在不改变模型词表的前提下，使用 kv-aware 掩码（基于 `word_ids`）训练模型，使模型更关注 Key/Value 位置的信息。

- **新增脚本**：
	- `train_dapt_kvmlm.py`：独立 kv-MLM 训练脚本（不会调用 `resize_token_embeddings`，保持词表不变）。
	- `check_processed_dataset.py`：用于在训练机上检查 `processed_dataset` 是否包含 `input_ids` 与 `word_ids`，并打印样例与字段类型。

- **数据检查（必须先运行）**：在训练机上运行：
```bash
cd /data/ocean/bpe_workspace
conda activate medical_bert
python /data/ocean/bpe_workspace/check_processed_dataset.py /data/ocean/bpe_workspace/processed_dataset
```

 期望输出关键项（示例）：
	- "Dataset splits: ['train', 'test']"
	- 每个 split 列出 Columns，且包含 `input_ids` 与 `word_ids`。
	- 首条样本展示 `input_ids`（list[int]）与 `word_ids`（list[int|None]），并打印：
		`input_ids_ok=True, word_ids_ok=True`


cd /data/ocean/bpe_workspace
		conda activate medical_bert
		python retokenize_processed_dataset_with_wordids.py
- **运行 kv-MLM（示例）**：
```bash
cd /data/ocean/bpe_workspace
conda activate medical_bert
# 单机多卡示例
CUDA_VISIBLE_DEVICES=3,4,5 torchrun --nproc_per_node=3 train_dapt_kvmlm.py 2>&1 | tee train_kvmlm.log
```

- **注意事项**：
	- `train_dapt_kvmlm.py` 保持 `remove_unused_columns=False`，Trainer 会把 `word_ids` 传给自定义 collator；不要把 `remove_unused_columns` 改为 `True` 否则 `word_ids` 会被移除。
	- 模型与 tokenizer 的词表必须一致：如果不扩表，请使用基座模型与对应分词器（例如 `hfl/chinese-roberta-wwm-ext`）。
	- 若 `check_processed_dataset.py` 返回 `word_ids_ok=False`，说明需要先运行 `build_dataset_final_slim.py` 或额外的 tokenization 脚本来生成对齐的 `word_ids`。
	- 若训练脚本报错 embedding 越界（如 device-side assert），说明 processed_dataset 是用扩展过的 tokenizer 生成，需用当前基座 tokenizer 重新分词：
		- 若只做官方MLM，可用 `retokenize_processed_dataset.py`（不保留 word_ids）。
		- 若要做 KV-MLM（需 word_ids），请用 `retokenize_processed_dataset_with_wordids.py`：
		```bash
		cd /data/ocean/bpe_workspace
		conda activate medical_bert
		python retokenize_processed_dataset_with_wordids.py
		```
		重新生成后再运行 kv-MLM 训练脚本。

**并行运行注意（你当前场景）**

- 你提到正在用 GPU 3 卡做“只扩展词表”的训练，可以同时在 GPU 4 与 5 上运行“只做 kv-MLM”的 DAPT，前提条件：
	- 两个训练过程使用不同的 GPU（通过 `CUDA_VISIBLE_DEVICES` 明确分配），并且各自 `OUTPUT_DIR` 不相同，避免写同一目录或覆盖同一 checkpoint。
	- 避免同时对同一 tokenizer/checkpoint 进行写操作（例如一边保存 tokenizer 或 final_model 到同一路径会产生冲突）。建议：`train_dapt_mlm.py` 输出到 `output_medical_bert_add_vocab_mlm`，`train_dapt_kvmlm.py` 输出到 `output_medical_bert_kvmlm`，`apply_added_vocab.py` 输出到 `output_medical_bert_add_vocab`，互不冲突。
	- 并行运行时注意 I/O 压力（同时写大量检查点会增加磁盘压力）；如必要，把 `save_steps` 调大或在一个实验完成后再做最终合并。

**kv-MLM DAPT 完成后下游微调（必须做）**

- 在 kv-MLM 训练结束并保存 checkpoint（`output_medical_bert_kvmlm/final_model`）后，请用该 checkpoint 做下游 KV-NER 微调以衡量改动效果。已为此创建/更新下游配置：
	- `my-bert-finetune/pre_struct/kv_ner/kv_ner_config_kvmlm.json`，其 `model_name_or_path` 已指向 `/data/ocean/bpe_workspace/output_medical_bert_kvmlm/final_model`。

- 下游训练示例（在 ner-finetune 根目录运行）：
```bash
cd /data/ocean/FT_workspace/ner-finetune
conda activate medical_bert
CUDA_VISIBLE_DEVICES=3 python pre_struct/kv_ner/train.py --config pre_struct/kv_ner/kv_ner_config_kvmlm.json 2>&1 | tee train_ner_kvmlm.log
```

- 推荐：在对比实验中保持下游训练超参数（seed、batch size、epoch）与其他实验一致，仅替换 `model_name_or_path` 与 `tokenizer_name_or_path`，便于公平比较。

## 微调后评估（Evaluate）

- 说明：下游微调完成后应立即运行评估脚本获取最终指标（F1/Precision/Recall 等）并保存预测文件与 eval summary，便于不同实验对比。
- 使用已有脚本 `pre_struct/kv_ner/evaluate.py`，无需创建新文件。

## 示例（`add_vocab` 实验）：
```bash
cd /data/ocean/FT_workspace/ner-finetune
conda activate medical_bert
CUDA_VISIBLE_DEVICES=3 python pre_struct/kv_ner/evaluate.py \
	--config pre_struct/kv_ner/kv_ner_config_add_vocab.json \
	--model_dir runs/kv_ner4_bioe_add_vocab/best \
	--test_data data/kv_ner_prepared/val_eval.jsonl \
	--output_dir data/kv_ner_eval_bioe_add_vocab \
	2>&1 | tee eval_ner_add_vocab.log
```

## 示例（`kv-MLM` 实验）：
```bash
cd /data/ocean/FT_workspace/ner-finetune
conda activate medical_bert
CUDA_VISIBLE_DEVICES=3 python pre_struct/kv_ner/evaluate.py \
	--config pre_struct/kv_ner/kv_ner_config_kvmlm.json \
	--model_dir runs/kv_ner4_bioe_kvmlm/best \
	--test_data data/kv_ner_prepared/val_eval.jsonl \
	--output_dir data/kv_ner_eval_bioe_kvmlm \
	2>&1 | tee eval_ner_kvmlm.log