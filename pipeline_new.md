# DAPT 全流程（含噪声对齐、去重、KV-NSP）

## 0. 数据去重与清洗
- 脚本：`extract_and_dedup_json_v2.py`
- 作用：扫描多源 JSON/TXT，提取文本，MD5 去重。
- 产出：`/data/ocean/bpe_workspace/train.txt`（干净文本）；可按源拆分辅助重采样。

（可选）分源重采样，调整占比
```bash
python resample_mix.py \
  --clinical /data/ocean/bpe_workspace/train_clinical.txt \
  --book_core /data/ocean/bpe_workspace/train_book_core.txt \
  --book_old /data/ocean/bpe_workspace/train_book_old.txt \
  --paper /data/ocean/bpe_workspace/train_paper.txt \
  --general /data/ocean/bpe_workspace/train_general.txt \
  --supplement /data/ocean/bpe_workspace/train_supplement.txt \
  --weights 0.25 0.35 0.05 0.10 0.20 0.05 \
  --output /data/ocean/bpe_workspace/train_resampled.txt
```
（可选，降低 512 截断）字符级滑窗切分长行
```bash
python chunk_long_lines.py \
  --input /data/ocean/bpe_workspace/train_resampled.txt \  # 若未重采样则用 train.txt
  --output /data/ocean/bpe_workspace/train_chunked.txt \
  --window 1000 --stride 500
```
后续构建数据集时，将 `TRAIN_FILE` 指向 `train_chunked.txt`，减少长文本被 512 截断。
（可选，降低 512 截断）字符级滑窗切分长行
```bash
python chunk_long_lines.py \
  --input /data/ocean/bpe_workspace/train.txt \
  --output /data/ocean/bpe_workspace/train_chunked.txt \
  --window 1000 --stride 500
```
后续构建数据集时，将 `TRAIN_FILE` 指向 `train_chunked.txt`，减少长文本被 512 截断。

## 1. 词表与 Tokenizer
1) OCR 词表挖掘：`train_ocr_clean.py` → `medical_vocab_ocr_only/vocab.txt`
2) （可选）LLM 过滤：`filter_vocab_with_llm.py` → `kept_vocab.txt`
3) Tokenizer 合并（精简版）：`final_merge_v9_regex_split_slim.py` → `my-medical-tokenizer/`
4) Jieba 词典：`generate_jieba_vocab.py` → `vocab_for_jieba.txt`

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
out = "/data/ocean/bpe_workspace/noise_bins.json"
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
  --output /data/ocean/bpe_workspace/train_ocr_9297.txt
```
2) 构建带 word_ids 的 dataset  
   - 设置 `TRAIN_FILE=train_ocr_9297.txt` 运行 `build_dataset_final_slim.py`（如开启滑窗流程，则 TRAIN_FILE 指向 `train_chunked.txt`）  
   - 产出：`processed_dataset_ocr9297`
3) 加噪声特征（连续值 -> 分桶 ID 在训练时处理，此处只写连续值）
```bash
python add_noise_features.py \
  --dataset /data/ocean/bpe_workspace/processed_dataset_ocr9297 \
  --output /data/ocean/bpe_workspace/processed_dataset_ocr9297_with_noise \
  --ocr_json /home/ocean/semi_label/ocr_rerun/char_ocr_9297.json \
  --bins_json /data/ocean/bpe_workspace/noise_bins.json \
  --num_proc 32
```
   - OCR 样本：存储真实 7 维连续值到 `noise_values`（对齐 word_ids→token）。
   - 非 OCR 样本：存储完美物理值 `[1.0,1.0,0,0,0,0,0]`，分桶映射在 collator。
4) 对齐校验
```bash
python verify_noise_alignment.py \
  --dataset /data/ocean/bpe_workspace/processed_dataset_ocr9297_with_noise \
  --ocr_json /home/ocean/semi_label/ocr_rerun/char_ocr_9297.json \
  --check_samples 50 \
  --tokenizer /data/ocean/bpe_workspace/my-medical-tokenizer
```
   目标：高匹配率、噪声覆盖率 ~100%。

### 2.3 非 OCR 路（无噪声特征）
- 书籍/指南/百科/20w 病历等：运行 `build_dataset_final_slim.py`，如果使用滑窗，确保 `TRAIN_FILE` 指向 `train_chunked.txt`；不含 `noise_values` 字段即可（训练时 collator 会自动填完美物理值并分桶）。

### 2.4 合并（可选）
- 使用 `datasets.concatenate_datasets` 将 OCR 数据集与非 OCR 数据集合并，可按权重重采样。  
- 禁止按索引把 OCR 特征硬塞给无 OCR 元信息的样本。

## 3. 训练（KV-aware MLM + 噪声）
1) 指向对齐后的数据集（单独 OCR 或合并后）：
```bash
ln -sfn /data/ocean/bpe_workspace/processed_dataset_ocr9297_with_noise \
       /data/ocean/bpe_workspace/processed_dataset
```
2) 启动训练（避免端口冲突）：
```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 CUDA_VISIBLE_DEVICES=3,4 \
torchrun --nproc_per_node=2 --master_port=29505 \
  train_dapt_distributed.py \
  --noise_bins_json /data/ocean/bpe_workspace/noise_bins.json \
  --output_dir /data/ocean/bpe_workspace/output_medical_bert_v2_8gpu_noise_v2 \
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

