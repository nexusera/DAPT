# MLM Ablation 远端使用说明

本目录用于比较两种 MLM 训练方式：

- `kvmlm`：`mlm_masking=kv_wwm`（当前主流程）
- `plainmlm`：`mlm_masking=token`（普通 token-level MLM）

其余训练流程保持一致（同一 staged 预训练、同一 KV-NSP、同一 noise 配置）。

## 1. 环境

```bash
cd /data/ocean/DAPT
git pull
conda activate medical_bert
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 2. 预训练（推荐：6卡并行，2卡/variant）

```bash
cd /data/ocean/DAPT

GPU_LIST=0,1,2,3 \
VARIANTS=kvmlm,plainmlm \
PARALLEL=1 \
RESUME=0 CLEAN_BEFORE_RUN=1 \
PRETRAIN_LAUNCHER=torchrun \
NPROC_PER_NODE=2 \
MASTER_PORT_BASE=29621 \
PER_DEVICE_TRAIN_BATCH_SIZE=16 \
GRADIENT_ACCUMULATION_STEPS=4 \
DATALOADER_NUM_WORKERS=4 \
NOISE_MODE=bucket \
NSP_LOSS_GUARD_ENABLE=1 \
NSP_LOSS_GUARD_ROUND=2 \
NSP_LOSS_GUARD_TARGET=0.6931 \
NSP_LOSS_GUARD_TOL=0.005 \
bash experiments/mlm_ablation/run_mlm_pretrain_all.sh \
| tee runs/mlm_ablation_logs/_pretrain.$(date +%F_%H%M%S).log
```

说明：
- 该命令仅用 4 张卡（2×2）即可并行跑完两个变体。
- 若希望串行：`PARALLEL=0 GPU_LIST=0,1`。

## 3. KV-NER（微调 + 推理 + Task1/3）

```bash
cd /data/ocean/DAPT

GPU_LIST=0,1 \
VARIANTS=kvmlm,plainmlm \
PARALLEL=1 \
RESUME=0 \
NOISE_MODE=bucket \
bash experiments/mlm_ablation/run_mlm_kvner_all.sh \
| tee runs/mlm_ablation_logs/_kvner.$(date +%F_%H%M%S).log
```

## 4. EBQA（微调 + 推理 + Task2）

```bash
cd /data/ocean/DAPT

GPU_LIST=0,1 \
VARIANTS=kvmlm,plainmlm \
PARALLEL=1 \
RESUME=0 \
NOISE_MODE=bucket \
bash experiments/mlm_ablation/run_mlm_ebqa_all.sh \
| tee runs/mlm_ablation_logs/_ebqa.$(date +%F_%H%M%S).log
```

## 5. 结果路径

- 预训练模型：
  - `/data/ocean/DAPT/workspace/output_ablation_mlm_kvmlm/final_staged_model`
  - `/data/ocean/DAPT/workspace/output_ablation_mlm_plainmlm/final_staged_model`
- KV-NER 评测：
  - `/data/ocean/DAPT/runs/mlm_kvmlm_report_task1.json`
  - `/data/ocean/DAPT/runs/mlm_kvmlm_report_task3.json`
  - `/data/ocean/DAPT/runs/mlm_plainmlm_report_task1.json`
  - `/data/ocean/DAPT/runs/mlm_plainmlm_report_task3.json`
- EBQA 评测：
  - `/data/ocean/DAPT/runs/mlm_kvmlm_report_task2.json`
  - `/data/ocean/DAPT/runs/mlm_plainmlm_report_task2.json`
