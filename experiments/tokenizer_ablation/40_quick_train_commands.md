# Quick Train（Tokenizer 消融用）

> 训练脚本 `DAPT/train_dapt_macbert_staged.py` 本身不支持 `--max_steps`，所以 quick-run 建议用 `datasets_quick/`（小语料）来缩短每个 epoch。

示例（T4 quick）：

```bash
python DAPT/train_dapt_macbert_staged.py \
  --output_dir /data/ocean/DAPT/ablation/tokenizer/runs/t4_quick \
  --dataset_path /data/ocean/DAPT/ablation/tokenizer/datasets_quick/processed_dataset_t4 \
  --tokenizer_path /data/ocean/DAPT/ablation/tokenizer/tokenizers/t4_ocr_llm_keys \
  --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
  --nsp_data_dir /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
  --num_rounds 1 \
  --mlm_epochs_per_round 1 \
  --nsp_epochs_per_round 1 \
  --learning_rate 5e-5
```

T1/T2/T3 对应替换 `dataset_path` 与 `tokenizer_path` 即可。
