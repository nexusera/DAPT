# 微调与评估流水（含 DAPT 噪声特征）

## 数据来源
- 输入：`/data/ocean/DAPT/biaozhu_with_ocr_noise/merged_*.json`
- 可用（已计算 noise）：`merged_huizhenbingli_with_ocr.json`、`merged_huojianbingli_with_ocr.json`、`merged_menzhenbingli_with_ocr.json`、`merged_shuhoubingli_with_ocr.json`
- 暂排除：`merged_ruyuanjilu_with_ocr.json`（图片缺失未算出 noise）

## 划分 train/dev/test
```
python - <<'PY'
import json, random
from pathlib import Path

files = [
    "biaozhu_with_ocr_noise/merged_huizhenbingli_with_ocr.json",
    "biaozhu_with_ocr_noise/merged_huojianbingli_with_ocr.json",
    "biaozhu_with_ocr_noise/merged_menzhenbingli_with_ocr.json",
    "biaozhu_with_ocr_noise/merged_shuhoubingli_with_ocr.json",
]
items = []
for f in files:
    items.extend(json.load(Path(f).open()))

random.seed(42)
random.shuffle(items)

n = len(items)
n_train = int(n * 0.8)
n_dev   = int(n * 0.1)
splits = {
    "train": items[:n_train],
    "dev":   items[n_train:n_train+n_dev],
    "test":  items[n_train+n_dev:],
}
for name, data in splits.items():
    p = Path(f"biaozhu_with_ocr_noise/{name}.jsonl")
    with p.open("w", encoding="utf-8") as w:
        for obj in data:
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(name, len(data), "->", p)
PY
```

- 按需合并/切分，保留 `noise_values` 字段
- 若训练脚本要求 JSONL，请转换为一行一个对象
- 保持标签/键值对的跨度信息不变

## 带噪声的微调示例
```
cd /data/ocean/DAPT
python pre_struct/kv_ner/train.py \
  --model_name_or_path /data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu_noise_v2/final_model \
  --train_file biaozhu_with_ocr_noise/train.jsonl \
  --eval_file biaozhu_with_ocr_noise/dev.jsonl \
  --output_dir runs/dapt_kv_ner_finetuned_v1 \
  --noise_bins_json /path/to/noise_bins.json \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.1 \
  --max_seq_length 512
```
说明：
- `noise_bins_json` 必须与 DAPT 预训练时的分桶文件一致
- 代码中的 `NoiseCollator` 会消费 `noise_values`；缺失时自动回退为“完美值”
- 监控 loss/metric，按需调学习率、epoch、batch size

## 评估示例
```
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
  --model_path runs/dapt_kv_ner_finetuned \
  --test_data /path/to/test.jsonl \
  --noise_bins_json /path/to/noise_bins.json \
  --output_summary runs/eval_dapt_finetuned.json
```

## 数据更新时的步骤
1) 新图片先跑 OCR 合并：`fetch_and_merge_baidu_ocr.py`
2) 计算 7 维噪声：`compute_noise_from_ocr.py --inputs ... --output_dir ...`
3) 重新生成 train/dev/test JSONL
4) 重新训练与评估
