# DAPT 全流程（含噪声对齐、去重、KV-NSP）

## 0. 数据去重与清洗
- 脚本：`extract_and_dedup_json_v2.py`
- 作用：扫描多源 JSON/TXT，提取文本，MD5 去重。
- 产出：`/data/ocean/DAPT/workspace/train.txt`（干净文本）；可按源拆分辅助重采样。


```

（可选）分源重采样，调整占比
python resample_mix.py \
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
4) Jieba 词典（与合并策略保持一致，使用 kept_vocab.txt + `biaozhu_keys_only_min5.txt`，生成 `/data/ocean/DAPT/vocab_for_jieba.txt`，供 `build_dataset_final_slim.py` / `retokenize_processed_dataset_with_wordids.py` 使用）
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
python export_ocr_texts.py \
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
  
```
## 3. 训练（KV-aware MLM + 噪声）
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
- 负样本：倒序 + 随机 value，比例由 `hard_negative_prob` 控制。

## 5. 下游微调 / 评估
- 使用最新预训练模型（含噪声）做 NER/其它任务，保持同超参对比，修改配置中的 `model_name_or_path`、`tokenizer_name_or_path` 指向新的 `final_model`。

## 常见问题
- 噪声错配：先跑 `verify_noise_alignment.py`，匹配率低说明 OCR 与 dataset 顺序不一致，必须按 OCR-only 路重建并再合并。
- token 越界 / NaN：用当前 tokenizer 重建数据；确保无样本 token_id >= vocab size；噪声特征无 NaN/Inf；必要时关闭 bf16 做短程验证。
- 端口冲突：显式 `--master_port`（如 29505），不要用默认 29500。

