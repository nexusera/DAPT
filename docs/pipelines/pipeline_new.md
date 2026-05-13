# DAPT 全流程（含噪声对齐、去重、KV-NSP）
写在最前：重要！！！当前的基座模型已经从RoBERTa换为了MacBert
## 0. 数据去重与清洗
- 脚本：`scripts/data/extract_and_dedup_json_v2.py`
- 作用：扫描多源 JSON/TXT，提取文本，MD5 去重。
- 产出：`/data/ocean/DAPT/workspace/train.txt`（干净文本）；可按源拆分辅助重采样。


```

（可选）分源重采样，调整占比
python scripts/data/resample_mix.py \
  --clinical /data/ocean/DAPT/workspace/train_clinical.txt \
  --book_core /data/ocean/DAPT/workspace/train_book_core.txt \
  --book_old /data/ocean/DAPT/workspace/train_book_old.txt \
  --paper /data/ocean/DAPT/workspace/train_paper.txt \
  --general /data/ocean/DAPT/workspace/train_general.txt \
  --supplement /data/ocean/DAPT/workspace/train_supplement.txt \
  --wiki_med /data/ocean/DAPT/workspace/wiki_data/wiki_med_zh_local.txt \
  --wiki_general /data/ocean/DAPT/workspace/wiki_data/wiki_general_med_zh_local.txt \
  --med_book /data/ocean/DAPT/workspace/train_med_book.txt \
  --general2 /data/ocean/DAPT/general_data/fineweb_edu_sample.jsonl.gz \
  --weights  \
    0.35 \
    0.15 \
    0.00 \
    0.08 \
    0.05 \
    0.02 \
    0.07 \
    0.02 \
    0.13 \
    0.13 \
  --output /data/ocean/DAPT/workspace/train_resampled.txt

# 当前示例权重合计为 1.0，可按需求微调；若精确要求“医疗≈50%，病例≈35%，通用≈15%”，可适当提高 med_book/wiki_med/（或 paper）并下调 general/general2。
```

（可选，降低 512 截断）字符级滑窗切分长行
```bash
python chunk_long_lines.py \
  --input /data/ocean/DAPT/workspace/train_resampled.txt \
  --output /data/ocean/DAPT/workspace/train_chunked.txt \
  --window 1000 \
  --stride 500
```
# 若未重采样input则用 train.txt
后续构建数据集时，将 `TRAIN_FILE` 指向 `train_chunked.txt`，减少长文本被 512 截断。


## 1. 词表与 Tokenizer
1) OCR 词表挖掘（脚本已预设：`CORPUS_FILE=/data/ocean/DAPT/workspace/train_chunked.txt`，`OUTPUT_DIR=/data/ocean/DAPT/workspace/medical_vocab_ocr_only`）
```bash
python train_ocr_clean.py
```
2) （可选）LLM 过滤（需本地 LLM 服务，保留医学相关词）
```bash
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
3) Tokenizer 合并（精简版，使用 LLM 精修 OCR 词表 + 纯键名 `biaozhu_keys_only.txt`，输出到 `my-medical-tokenizer/`）
```bash
python final_merge_v9_regex_split_slim.py
```
4) Jieba 词典（与合并策略保持一致，使用 kept_vocab.txt + `biaozhu_keys_only_min5.txt`，生成 `/data/ocean/DAPT/vocab_for_jieba.txt`，供 `build_dataset_final_slim.py` / `scripts/data/retokenize_processed_dataset_with_wordids.py` 使用）
```bash
python generate_jieba_vocab.py
```

## 2. 数据集构建（分路，避免噪声错配）
### 2.1 预计算分桶边界（一次）
```bash
python - <<'PY'
import json
from noise_feature_processor import NoiseFeatureProcessor
import os

OCR_JSON = "/home/ocean/semi_label/ocr_rerun/char_ocr_9297.json"
with open(OCR_JSON, "r", encoding="utf-8") as f:
    obj = json.load(f)
if isinstance(obj, dict):
    for k in ["data","ocr_list","items"]:
        if k in obj and isinstance(obj[k], list):
            obj = obj[k]; break
    else:
        obj = [obj]
proc = NoiseFeatureProcessor()
proc.fit_bins(obj)
out = "/data/ocean/DAPT/workspace/noise_bins.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
proc.save(out)
print("saved", out)
PY
```

### 2.2 OCR 专用路（有噪声特征）
1) 导出 OCR 文本（保持顺序）
```bash
python scripts/data/export_ocr_texts.py \
  --ocr_json /home/ocean/semi_label/ocr_rerun/char_ocr_9297.json \
  --output /data/ocean/DAPT/workspace/train_ocr_9297.txt
```
2) 构建带 word_ids 的 dataset（命令行指定 train_file/output_path）
```bash
python build_dataset_final_slim.py \
  --train_file /data/ocean/DAPT/workspace/train_ocr_9297.txt \
  --output_path /data/ocean/DAPT/workspace/processed_dataset_ocr9297 \
  --tokenizer_path /data/ocean/DAPT/my-medical-tokenizer \
  --no_shuffle_split   # OCR 路保持顺序，避免对齐错位
# 如先做滑窗，则将 train_file 改为 /data/ocean/DAPT/workspace/train_chunked.txt
```
   - 产出：`processed_dataset_ocr9297`
3) 加噪声特征（连续值 -> 分桶 ID 在训练时处理，此处只写连续值）
```bash
python add_noise_features.py \
  --dataset /data/ocean/DAPT/workspace/processed_dataset_ocr9297 \
  --output /data/ocean/DAPT/workspace/processed_dataset_ocr9297_with_noise \
  --ocr_json /home/ocean/semi_label/ocr_rerun/char_ocr_9297.json \
  --bins_json /data/ocean/DAPT/workspace/noise_bins.json \
  --num_proc 16
```
   - OCR 样本：存储真实 7 维连续值到 `noise_values`（对齐 word_ids→token）。
   - 非 OCR 样本：存储完美物理值 `[1.0,1.0,0,0,0,0,0]`，分桶映射在 collator。
4) 对齐校验
```bash
python verify_noise_alignment.py \
  --dataset /data/ocean/DAPT/workspace/processed_dataset_ocr9297_with_noise \
  --ocr_json /home/ocean/semi_label/ocr_rerun/char_ocr_9297.json \
  --check_samples 50 \
  --tokenizer /data/ocean/DAPT/my-medical-tokenizer
```
   目标：高匹配率、噪声覆盖率 ~100%。

### 2.3 非 OCR 路（无噪声特征）
- 书籍/指南/百科/20w 病历等：命令行指定 train_file/output_path（无 noise_values，训练时自动填完美噪声）
```bash
python build_dataset_final_slim.py \
  --train_file /data/ocean/DAPT/workspace/train_chunked.txt \
  --output_path /data/ocean/DAPT/workspace/processed_dataset_nonocr \
  --tokenizer_path /data/ocean/DAPT/my-medical-tokenizer \
  --shuffle_split      # 非 OCR 路可打乱
```

### 2.4 合并（可选）
已分别按 2.2/2.3 得到：
- OCR: `/data/ocean/DAPT/workspace/processed_dataset_ocr9297_with_noise`
- 非 OCR: `/data/ocean/DAPT/workspace/processed_dataset_nonocr`

合并示例（保持分割一致，防止按索引错配噪声）：
```bash
python merge_datasets.py \
  --ocr_dataset /data/ocean/DAPT/workspace/processed_dataset_ocr9297_with_noise \
  --non_ocr_dataset /data/ocean/DAPT/workspace/processed_dataset_nonocr \
  --output_path /data/ocean/DAPT/workspace/processed_dataset_merged \
  --ocr_repeat 1 \
  --non_ocr_repeat 1 \
```
  - `ocr_repeat/non_ocr_repeat` 仅作用于 train 分割，用于配比；其余分割保持 1:1。  
- 禁止按索引把 OCR 特征硬塞给无 OCR 元信息的样本。
```
#（可选）校验对齐
python verify_noise_alignment.py \
  --dataset /data/ocean/DAPT/workspace/processed_dataset_merged \
  --ocr_json /home/ocean/semi_label/ocr_rerun/char_ocr_9297.json \
  --check_samples 50 \
  --split all
  
```
## 3. 最新的训练（KV-aware MLM + 噪声+KV-NSP）
训练的脚本为train_dapt_macbert_staged.py

### 3.1 KV-MLM 消融：普通 MLM（不使用 KV 全词掩码）作为对照

说明：当前的 KV-aware MLM 通过 `build_dataset_final_slim.py` 产出的 `word_ids`（由 jieba + keys 词典引导）实现“全词掩码”。
为了做消融对照，我们新增了一个开关：
- `--mlm_masking kv_wwm`：使用 `word_ids` 的全词掩码（现有默认行为，KV-MLM）
- `--mlm_masking token`：普通 MLM（按 token 随机掩码，忽略 `word_ids`）

两组实验除 `--mlm_masking` 外保持完全一致（同一数据、同一 tokenizer、同一超参、同一轮次/epoch 配置），以保证可比性。

#### A) KV-MLM（现有默认，KV 全词掩码）
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
  --mlm_masking kv_wwm \
  2>&1 | tee /data/ocean/DAPT/workspace/output_macbert_kvmlm_staged/train.log
```

#### B) 普通 MLM 对照（不使用 KV 全词掩码）
```bash
cd /data/ocean/DAPT
CUDA_VISIBLE_DEVICES=4,5
tmux attach -t mlm

CUDA_VISIBLE_DEVICES=4,5 python /data/ocean/DAPT/train_dapt_macbert_staged.py \
  --output_dir /data/ocean/DAPT/workspace/output_macbert_plainmlm_staged \
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
  --mlm_masking token \
  2>&1 | tee /data/ocean/DAPT/workspace/output_macbert_plainmlm_staged/train.log
```

产物目录结构与原 staged 训练一致：
- `.../round_{k}_mlm/`、`.../round_{k}_nsp/`
- 最终模型：`.../final_staged_model/`

### 3.2 KV-NSP 负样本比例消融：reverse / random

当前 KV-NSP 已支持把负样本中的两种策略拆开配置：
- `--nsp_reverse_negative_ratio`：倒序负样本（reverse）的权重
- `--nsp_random_negative_ratio`：随机 value 负样本（random）的权重
- `--nsp_negative_prob`：总负样本概率，默认仍为 `0.5`

说明：
- 若设为 `--nsp_reverse_negative_ratio 1 --nsp_random_negative_ratio 1`，即原始 `1:1`。
- 若设为 `3:1`，则负样本内部约 75% 为 reverse、25% 为 random。
- 若设为 `1:3`，则负样本内部约 25% 为 reverse、75% 为 random。

建议除比例外，其余超参完全保持一致，以保证消融公平。

#### A) 基线：reverse/random = 1:1
```bash
python /data/ocean/DAPT/train_dapt_macbert_staged.py \
  --output_dir /data/ocean/DAPT/workspace/output_macbert_nsp_ratio_1_1 \
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
  --mlm_masking kv_wwm \
  --nsp_negative_prob 0.5 \
  --nsp_reverse_negative_ratio 1 \
  --nsp_random_negative_ratio 1
```

#### B) reverse 偏置：reverse/random = 3:1
```bash
python /data/ocean/DAPT/train_dapt_macbert_staged.py \
  --output_dir /data/ocean/DAPT/workspace/output_macbert_nsp_ratio_3_1 \
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
  --mlm_masking kv_wwm \
  --nsp_negative_prob 0.5 \
  --nsp_reverse_negative_ratio 3 \
  --nsp_random_negative_ratio 1
```

#### C) random 偏置：reverse/random = 1:3
```bash
python /data/ocean/DAPT/train_dapt_macbert_staged.py \
  --output_dir /data/ocean/DAPT/workspace/output_macbert_nsp_ratio_1_3 \
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
  --mlm_masking kv_wwm \
  --nsp_negative_prob 0.5 \
  --nsp_reverse_negative_ratio 1 \
  --nsp_random_negative_ratio 3
```

完成预训练后，下游 Task1/3 与 Task2 的微调、推理、评测流程不变，只需把对应配置中的 `model_name_or_path` / `tokenizer_name_or_path` 改到各自 ratio 实验生成的最终模型目录。

### 3.3 Noise Embedding 消融：Bucket vs Linear vs MLP

当前 `train_dapt_macbert_staged.py` 已支持：
- `--noise_mode bucket`：现有分桶查表基线
- `--noise_mode linear`：对 7 维连续噪声直接做 `Linear(7→hidden)`
- `--noise_mode mlp`：对 7 维连续噪声做 2-layer MLP 投影

建议除 `--noise_mode`（以及 `mlp` 时的 `--noise_mlp_hidden_dim`）外，其余设置完全一致。

#### A) Bucket 基线
```bash
python /data/ocean/DAPT/train_dapt_macbert_staged.py \
  --output_dir /data/ocean/DAPT/workspace/output_ablation_noise_bucket \
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
  --mlm_masking kv_wwm \
  --noise_mode bucket
```

#### B) Linear 连续噪声
```bash
python /data/ocean/DAPT/train_dapt_macbert_staged.py \
  --output_dir /data/ocean/DAPT/workspace/output_ablation_noise_linear \
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
  --mlm_masking kv_wwm \
  --noise_mode linear
```

#### C) MLP 连续噪声
```bash
python /data/ocean/DAPT/train_dapt_macbert_staged.py \
  --output_dir /data/ocean/DAPT/workspace/output_ablation_noise_mlp \
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
  --mlm_masking kv_wwm \
  --noise_mode mlp \
  --noise_mlp_hidden_dim 128
```

对应下游可直接使用以下配置：
- KV-NER：`kv_ner_config_noise_bucket.json` / `kv_ner_config_noise_linear.json` / `kv_ner_config_noise_mlp.json`
- EBQA：`ebqa_config_noise_bucket.json` / `ebqa_config_noise_linear.json` / `ebqa_config_noise_mlp.json`

### 3.4 Noise Embedding 融合方式消融：Bucket-Add vs Concat-Linear

**动机**：现有 `bucket` 模式直接把 7 路分桶嵌入求和（Add），不同维度的嵌入信号可能互相干扰或相互抵消。
新提出的 `concat_linear` 模式改为先 **Concat（拼接）** 再接一层 **Linear 映射**，保留各维嵌入的独立性，由线性层学习最优融合权重。

实现细节：
- 每路特征独立 Embedding 表：`Embedding(n_bins+1, embed_dim)`，默认 `embed_dim=64`
- 7 路拼接：`[batch, seq, 7×64=448]`
- 线性映射：`Linear(448 → 768)` + Dropout
- 可学习系数 `alpha` 控制残差强度（与 bucket 模式一致）

#### A) Concat-Linear 预训练（新增实验）
```bash
cd /data/ocean/DAPT

python /data/ocean/DAPT/train_dapt_macbert_staged.py \
  --output_dir /data/ocean/DAPT/workspace/output_ablation_noise_concat_linear \
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
  --mlm_masking kv_wwm \
  --noise_mode concat_linear \
  --noise_concat_embed_dim 64 \
  2>&1 | tee /data/ocean/DAPT/workspace/output_ablation_noise_concat_linear/train.log
```

> 说明：`--noise_concat_embed_dim 64` 意味着 7×64=448 维拼接后映射至 768。如需更大容量可改为 `128`（7×128=896 → Linear(896,768)）。

#### B) 一键消融实验（仅跑新增 Concat-Linear 部分）

以下脚本与第 3.3 节 Bucket/Linear/MLP 实验共享同一基线，仅增量补跑 `concat_linear`：

```bash
#!/bin/bash
set -e

BASE_CMD="python /data/ocean/DAPT/train_dapt_macbert_staged.py \
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
  --mlm_masking kv_wwm"

# ---- D) Concat-Linear（新融合方式，embed_dim=64）----
echo "=== [D] concat_linear (embed_dim=64) ==="
OUT=/data/ocean/DAPT/workspace/output_ablation_noise_concat_linear
mkdir -p "$OUT"
$BASE_CMD \
  --output_dir "$OUT" \
  --noise_mode concat_linear \
  --noise_concat_embed_dim 64 \
  2>&1 | tee "$OUT/train.log"

echo "=== All new ablation runs completed ==="
```

保存为 `run_ablation_concat.sh` 后执行：
```bash
chmod +x run_ablation_concat.sh && bash run_ablation_concat.sh
```

完成预训练后，下游 KV-NER 微调时将 `model_name_or_path` / `tokenizer_name_or_path` 指向
`/data/ocean/DAPT/workspace/output_ablation_noise_concat_linear/final_staged_model`，
其余微调/推理/评测步骤与第 3.3 节保持一致。
<!-- ## 3. 训练（KV-aware MLM + 噪声）
1) 指向对齐后的数据集（单独 OCR 或合并后）：
```bash
ln -sfn /data/ocean/DAPT/workspace/processed_dataset_merged \
       /data/ocean/DAPT/workspace/processed_dataset
```
2) 启动训练（避免端口冲突）：
```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 CUDA_VISIBLE_DEVICES=3,4 \
torchrun --nproc_per_node=2 --master_port=29505 \
  train_dapt_distributed.py \
  --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
  --output_dir /data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu_noise_v2 \
  2>&1 | tee training_noise_v2.log
```
   - 首次可关闭 bf16 做短程 sanity check，确认 loss 非 NaN。

## 4. KV-NSP 流程（键值匹配二分类）
1) 数据：Label Studio 导出的 5 个 JSON（示例路径 `/data/ocean/FT_workspace/ner-finetune/data`）
2) 训练示例：
```bash
python kv_nsp/run_train.py \
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
- 输入格式 `[CLS] Key [SEP] Value [SEP]`，自动使用 token_type_ids 区分段。
- 负样本：倒序 + 随机 value，比例由 `hard_negative_prob` 控制。 -->

## 5. 下游微调 / 评估
- 使用最新预训练模型（含噪声）做 NER/其它任务，保持同超参对比，修改配置中的 `model_name_or_path`、`tokenizer_name_or_path` 指向新的 `final_model`。

## 常见问题
- 噪声错配：先跑 `verify_noise_alignment.py`，匹配率低说明 OCR 与 dataset 顺序不一致，必须按 OCR-only 路重建并再合并。
- token 越界 / NaN：用当前 tokenizer 重建数据；确保无样本 token_id >= vocab size；噪声特征无 NaN/Inf；必要时关闭 bf16 做短程验证。
- 端口冲突：显式 `--master_port`（如 29505），不要用默认 29500。

- fast tokenizer 异常（中文变成“带空格的大 token”且 token id 映射为 UNK）：优先检查 `vocab.txt` 与 `tokenizer.json` 是否一致，并用修复脚本重建 fast backend；不要首先通过“去空格”来规避。
