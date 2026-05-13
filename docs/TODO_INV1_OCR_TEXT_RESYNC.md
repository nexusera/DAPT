# TODO：修复训练数据 INV-1 对齐违例

> 创建时间：2026-04-28  
> 优先级：中（不影响已训练模型，但会导致下一轮训练的噪声通道错位）

---

## 问题描述

在 `biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json` 和
`real_test_with_ocr.json` 中，约一半样本存在：

```
ocr_text  ≠  ''.join(w['words'] for w in words_result)
```

典型表现：`ocr_text` 中英文单词无空格（`SecondAffiliatedHospital`），
而 `words_result` 中对应条目保留了原始空格（`Second Affiliated Hospital`）。

**后果**：训练时文本输入（`ocr_text`）与按 `words_result` 展开的噪声向量
（`noise_values_per_word`）从第一个不一致字符起产生逐字符错位，
导致模型学到的噪声特征对应的是错误的字符。

**根因**：历史数据通过旧版流程生成，`fetch_and_merge_baidu_ocr.py` 在
添加 `--no_sync_ocr_text` 豁免前已落库。详见 `docs/OCR_TEXT_AND_NOISE_ALIGNMENT.md`。

---

## CI 检测命令

```bash
cd /data/ocean/DAPT
python3 scripts/ci_check_ocr_alignment.py \
    --files biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json \
            biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \
    --inv1_only   # 只检查 INV-1，忽略 noise 格式差异
```

当前预期输出：约 **1700+ 条** INV-1 违例（`real_train` + `real_test` 合计）。

---

## 修复步骤（远端 H200）

```bash
cd /data/ocean/DAPT
git pull

# 步骤 1：重新同步 ocr_text（将旧值备份到 ocr_text_before_ocr_sync）
# 如果 OCR JSON 已在 biaozhu_with_ocr/ 或 biaozhu_with_ocr_noise/ 目录
python3 fetch_and_merge_baidu_ocr.py \
    --input_glob "biaozhu_with_ocr_noise/*.json" \
    --output_dir biaozhu_with_ocr_noise_synced/
# 注意：不要加 --no_sync_ocr_text

# 步骤 2：重新运行噪声特征计算（如果 ocr_raw.words_result 已包含概率字段）
python3 compute_noise_from_ocr.py \
    --inputs biaozhu_with_ocr_noise_synced/*.json \
    --output_dir biaozhu_with_ocr_noise_synced_with_noise/

# 步骤 3：重新拆分 train/test 并准备 HuggingFace 数据集
python3 add_noise_features.py \
    --train_json biaozhu_with_ocr_noise_synced_with_noise/real_train.json \
    --test_json  biaozhu_with_ocr_noise_synced_with_noise/real_test.json \
    --noise_bins_json workspace/noise_bins.json \
    --output_dir biaozhu_with_ocr_noise_prepared_v2/

# 步骤 4：验证修复
python3 scripts/ci_check_ocr_alignment.py \
    --files biaozhu_with_ocr_noise_prepared_v2/real_train_with_ocr.json \
            biaozhu_with_ocr_noise_prepared_v2/real_test_with_ocr.json \
    --strict
# 预期：[PASS] 所有对齐不变量检查通过
```

---

## 注意事项

- 修复后须重新训练 KV-NER 模型（噪声通道对齐改变会影响 embedding 输入）。
- 训练前确认 `noise_bins.json` 与新数据集的特征分布一致，必要时重新 fit：
  ```bash
  python3 dapt_eval_package/pre_struct/kv_ner/generate_noise_bins.py \
      --input biaozhu_with_ocr_noise_prepared_v2/real_train_with_ocr.json \
      --output workspace/noise_bins_v2.json
  ```
- 旧 checkpoint 路径：`runs/kv_ner_finetuned_noise_bucket/best`  
  新训练建议输出到：`runs/kv_ner_finetuned_noise_bucket_v2/`

---

## 参考文档

- `docs/OCR_TEXT_AND_NOISE_ALIGNMENT.md`（根因与已有修复说明）
- `scripts/ci_check_ocr_alignment.py`（5 条对齐不变量检查，支持 `--inv1_only`）
- `docs/guides/fetch_and_merge_baidu_ocr.md`（ocr_text 同步流程详细说明）
