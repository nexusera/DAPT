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
WITHOUT_NOISE_MODEL_DIR=/data/ocean/DAPT/workspace/output_no_noise_baseline/final_no_noise_model \
# 推荐传入带 noise_level/conf_avg 的元数据文件（用于 high/medium/low 分桶）
# 例如可用真实数据：/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json
NOISE_META_FILE=/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \
GPU_ID=0 \
MAX_SAMPLES_PER_GROUP=200 \
PROGRESS_EVERY=10 \
bash /data/ocean/DAPT/experiments/interpretability/run_attention_noise_compare_oneclick.sh
```

> 默认路径已固定为你确认的真实路径：  
> `WITHOUT_NOISE_MODEL_DIR=/data/ocean/DAPT/workspace/output_no_noise_baseline/final_no_noise_model`
>
> 若不传 `NOISE_META_FILE`，噪声分桶可能只有 `unknown`，无法做 high/medium/low 对比。

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
MODEL_TAG=main \
GPU_ID=0 \
MASK_STRATEGY=both \
MASK_SPAN_LEN=1 \
MAX_SAMPLES_PER_GROUP=120 \
PROGRESS_EVERY=10 \
bash /data/ocean/DAPT/experiments/interpretability/run_attention_kv_mlm_oneclick.sh
```

### 3.3 `w/o KV-MLM` 对照命令（建议做）
```bash
MODEL_DIR=/data/ocean/DAPT/workspace/output_ablation_no_mlm/final_no_mlm_model \
TOKENIZER_PATH=/data/ocean/DAPT/workspace/output_ablation_no_mlm/final_no_mlm_model \
MODEL_TAG=no_kvmlm \
GPU_ID=0 \
MASK_STRATEGY=both \
MASK_SPAN_LEN=1 \
MAX_SAMPLES_PER_GROUP=120 \
PROGRESS_EVERY=10 \
bash /data/ocean/DAPT/experiments/interpretability/run_attention_kv_mlm_oneclick.sh
```

> 默认不会覆盖：脚本输出目录格式为  
> `attention_kv_mlm_${MODEL_TAG}_${RUN_TAG}`，主模型与对照模型会分开保存。
>
> 另外，KV-MLM 一键脚本默认会优先使用 `MODEL_DIR` 下的 tokenizer，避免 `vocab size mismatch`。

### 3.4 Main vs w/o KV-MLM 显著性检验（新增）
```bash
cd /data/ocean/DAPT
MAIN_DIR=$(ls -dt runs/attention_kv_mlm_main_* | head -n 1)
ABL_DIR=$(ls -dt runs/attention_kv_mlm_no_kvmlm_* | head -n 1)
OUT_CMP="runs/attention_kv_mlm_compare_$(date +%Y%m%d_%H%M%S)"

python /data/ocean/DAPT/experiments/interpretability/compare_kv_mlm_runs.py \
  --main_metrics "${MAIN_DIR}/per_sample_metrics.jsonl" \
  --abl_metrics "${ABL_DIR}/per_sample_metrics.jsonl" \
  --output_dir "${OUT_CMP}"

cat "${OUT_CMP}/kv_mlm_main_vs_no_kvmlm_report.md"
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

---

## 6) 远端结果同步到本地（GitHub）

> 建议每次实验单独分支，避免互相覆盖。

### 6.1 远端 push
```bash
cd /data/ocean/DAPT
BRANCH=exp/attn-results-$(date +%Y%m%d_%H%M)
git checkout -b "$BRANCH"

# 按需添加你这次实验目录（示例）
git add runs/attention_kv_nsp_* runs/attention_noise_compare_* runs/attention_kv_mlm_*
git commit -m "Add attention interpretability outputs"
git push -u origin "$BRANCH"
echo "$BRANCH"
```

### 6.2 本地 pull
```bash
cd /Users/shanqi/Documents/BERT_DAPT
git fetch origin
# 把分支名替换为远端打印出来的 BRANCH
git checkout exp/attn-results-YYYYMMDD_HHMM
git pull
```

---

## 7) 贴给助手评估结果（固定命令）

```bash
cd /data/ocean/DAPT

# 1) Noise-Embedding compare 最新结果
OUT_NOISE=$(ls -dt runs/attention_noise_compare_* | head -n 1)
echo "OUT_NOISE=${OUT_NOISE}"
python - <<'PY'
import json, os, glob
cand = glob.glob("runs/attention_noise_compare_*")
if not cand:
    raise SystemExit("No runs/attention_noise_compare_* found")
out = max(cand, key=os.path.getmtime)
print("## noise compare:", out)
print(open(f"{out}/compare_report.md","r",encoding="utf-8").read())
print("## compare_summary.json")
print(open(f"{out}/compare_summary.json","r",encoding="utf-8").read())
PY

# 2) KV-MLM 主模型最新结果
OUT_MLM_MAIN=$(ls -dt runs/attention_kv_mlm_main_* | head -n 1)
echo "OUT_MLM_MAIN=${OUT_MLM_MAIN}"
python - <<'PY'
import json, os, glob
cand = glob.glob("runs/attention_kv_mlm_main_*")
if cand:
    out = max(cand, key=os.path.getmtime)
else:
    fallback = glob.glob("runs/attention_kv_mlm_*")
    if not fallback:
        raise SystemExit("No runs/attention_kv_mlm_* found")
    out = max(fallback, key=os.path.getmtime)
print("## kv-mlm(main):", out)
print(open(f"{out}/report.md","r",encoding="utf-8").read())
print("## summary.json")
print(open(f"{out}/summary.json","r",encoding="utf-8").read())
PY

# 3) KV-MLM 对照模型（w/o KV-MLM，若已跑）
OUT_MLM_ABL=$(ls -dt runs/attention_kv_mlm_no_kvmlm_* 2>/dev/null | head -n 1 || true)
echo "OUT_MLM_ABL=${OUT_MLM_ABL}"
python - <<'PY'
import os, glob
cand = glob.glob("runs/attention_kv_mlm_no_kvmlm_*")
if not cand:
    print("## kv-mlm(no_kvmlm): not found, skip")
    raise SystemExit(0)
out = max(cand, key=os.path.getmtime)
print("## kv-mlm(no_kvmlm):", out)
print(open(f"{out}/report.md","r",encoding="utf-8").read())
print("## summary.json")
print(open(f"{out}/summary.json","r",encoding="utf-8").read())
PY

# 4) Main vs w/o KV-MLM 显著性检验结果（若已跑 compare）
OUT_MLM_CMP=$(ls -dt runs/attention_kv_mlm_compare_* 2>/dev/null | head -n 1 || true)
echo "OUT_MLM_CMP=${OUT_MLM_CMP}"
if [ -n "$OUT_MLM_CMP" ]; then
  cat "${OUT_MLM_CMP}/kv_mlm_main_vs_no_kvmlm_report.md"
fi
```

