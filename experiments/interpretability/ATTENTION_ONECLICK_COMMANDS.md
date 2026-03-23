# 注意力可解释性一键命令总表（KV-NSP / Noise-Embedding / KV-MLM）

本文件汇总三类实验的一键执行命令，便于后续直接复用。  
默认在远端 `/data/ocean/DAPT` 下执行。

---

## 0) 首次准备（只需一次）

```bash
cd /data/ocean/DAPT
chmod +x experiments/interpretability/run_attention_kv_nsp_oneclick.sh
chmod +x experiments/interpretability/run_attention_noise_compare_oneclick.sh
chmod +x experiments/interpretability/run_attention_kv_mlm_oneclick.sh
```

---

## 1) KV-NSP 注意力可视化（最终主命令）

### 1.1 最简一条命令
```bash
bash /data/ocean/DAPT/experiments/interpretability/run_attention_kv_nsp_oneclick.sh
```

### 1.2 常用可配置命令（示例）
```bash
MODEL_DIR=/data/ocean/DAPT/workspace/output_ablation_noise_bucket/final_staged_model \
GPU_ID=0 \
MAX_SAMPLES_PER_GROUP=200 \
PROGRESS_EVERY=10 \
bash /data/ocean/DAPT/experiments/interpretability/run_attention_kv_nsp_oneclick.sh
```

产出目录：`/data/ocean/DAPT/runs/attention_kv_nsp_时间戳`

---

## 2) Noise-Embedding 注意力可解释性（with vs without 对照）

该脚本会自动跑两次 KV-NSP 注意力分析并生成对照报告：  
- with noise model  
- without noise model

### 2.1 最简一条命令
```bash
bash /data/ocean/DAPT/experiments/interpretability/run_attention_noise_compare_oneclick.sh
```

### 2.2 常用可配置命令（示例）
```bash
WITH_NOISE_MODEL_DIR=/data/ocean/DAPT/workspace/output_ablation_noise_bucket/final_staged_model \
WITHOUT_NOISE_MODEL_DIR=/data/ocean/DAPT/workspace/output_ablation_no_noise/final_no_noise_model \
GPU_ID=0 \
MAX_SAMPLES_PER_GROUP=200 \
PROGRESS_EVERY=10 \
bash /data/ocean/DAPT/experiments/interpretability/run_attention_noise_compare_oneclick.sh
```

> 说明：若不显式传 `WITHOUT_NOISE_MODEL_DIR`，脚本会自动按以下顺序探测：  
> 1) `/data/ocean/DAPT/workspace/output_ablation_no_noise/final_no_noise_model`  
> 2) `/data/ocean/DAPT/workspace/output_no_noise_baseline/final_no_noise_model`  
> 3) `/data/ocean/DAPT/workspace/output_ablation_no_noise/final_staged_model`

关键产出：
- `.../with_noise/summary.json`
- `.../without_noise/summary.json`
- `.../compare_summary.json`
- `.../compare_report.md`

---

## 3) KV-MLM 注意力可视化（增强可解释性）

该脚本做两类 mask 策略分析：
- `entity`：偏“实体重建”视角
- `boundary`：偏“边界依赖”视角

### 3.1 最简一条命令
```bash
bash /data/ocean/DAPT/experiments/interpretability/run_attention_kv_mlm_oneclick.sh
```

### 3.2 常用可配置命令（示例）
```bash
MODEL_DIR=/data/ocean/DAPT/workspace/output_ablation_noise_bucket/final_staged_model \
GPU_ID=0 \
MASK_STRATEGY=both \
MASK_SPAN_LEN=1 \
MAX_SAMPLES_PER_GROUP=120 \
PROGRESS_EVERY=10 \
bash /data/ocean/DAPT/experiments/interpretability/run_attention_kv_mlm_oneclick.sh
```

关键产出：
- `.../summary.json`
- `.../report.md`
- `.../cases/*.png`
- `.../figures/*.png`

---

## 4) 快速查看最新结果目录

```bash
cd /data/ocean/DAPT/runs
ls -dt attention_kv_nsp_* | head -n 1
ls -dt attention_noise_compare_* | head -n 1
ls -dt attention_kv_mlm_* | head -n 1
```

---

## 5) 建议执行顺序（论文写作）

1. 先跑 KV-NSP（已有主结果）  
2. 再跑 Noise-Embedding 对照（补“鲁棒性机制解释”）  
3. 最后跑 KV-MLM（补“实体/边界重建机制解释”）

