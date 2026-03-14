# KV-NSP 比例消融（reverse/random）远端使用说明

本文档说明如何在远端机器 /data/ocean/DAPT 上，分阶段跑 3 组 KV-NSP 负样本比例消融：

- 1:1
- 3:1
- 1:3

默认每组都包含：

1) 预训练
2) Task1/3（KV-NER）微调 + 推理 + 评测
3) Task2（EBQA）微调 + 推理 + 评测

---

## 1. 代码与环境准备

```bash
cd /data/ocean/DAPT
conda activate medical_bert
# 或 source /data/ocean/DAPT/.venv/bin/activate
git pull
```

建议先检查关键参数是否已生效：

```bash
grep -n 'nsp_reverse_negative_ratio\|nsp_random_negative_ratio' /data/ocean/DAPT/train_dapt_macbert_staged.py
```

并确认脚本实际使用的 Python（避免误用 base 环境）：

```bash
cd /data/ocean/DAPT
PYTHON_BIN=/home/ocean/.conda/envs/medical_bert/bin/python \
bash experiments/nsp_ratio_ablation/run_nsp_ratio_pretrain_all.sh
```

若你已经 `conda activate medical_bert`，也可不显式传 `PYTHON_BIN`。

---

## 2. 分阶段脚本（推荐）

脚本目录：

- experiments/nsp_ratio_ablation/run_nsp_ratio_pretrain_all.sh
- experiments/nsp_ratio_ablation/run_nsp_ratio_kvner_all.sh
- experiments/nsp_ratio_ablation/run_nsp_ratio_ebqa_all.sh
- experiments/nsp_ratio_ablation/run_nsp_ratio_ablation_all.sh

默认变体为：

- r11 -> reverse/random = 1:1
- r31 -> reverse/random = 3:1
- r13 -> reverse/random = 1:3

---

## 3. 预训练阶段

串行（单卡，推荐先跑通）：

```bash
tmux attach -t nsp_ab

cd /data/ocean/DAPT
GPU_LIST=5 PARALLEL=0 RESUME=1 \
bash experiments/nsp_ratio_ablation/run_nsp_ratio_pretrain_all.sh
```

并行（三卡）：

```bash
cd /data/ocean/DAPT
GPU_LIST=0,1,2 PARALLEL=1 RESUME=1 \
bash experiments/nsp_ratio_ablation/run_nsp_ratio_pretrain_all.sh
```

预训练输出模型：

- /data/ocean/DAPT/workspace/output_ablation_nsp_ratio_1_1/final_staged_model
- /data/ocean/DAPT/workspace/output_ablation_nsp_ratio_3_1/final_staged_model
- /data/ocean/DAPT/workspace/output_ablation_nsp_ratio_1_3/final_staged_model

---

## 4. Task1/3（KV-NER）阶段

```bash
cd /data/ocean/DAPT
GPU_LIST=0 PARALLEL=0 RESUME=1 \
bash experiments/nsp_ratio_ablation/run_nsp_ratio_kvner_all.sh
```

输出报告：

- /data/ocean/DAPT/runs/nsp_ratio_1_1_report_task1.json
- /data/ocean/DAPT/runs/nsp_ratio_1_1_report_task3.json
- /data/ocean/DAPT/runs/nsp_ratio_3_1_report_task1.json
- /data/ocean/DAPT/runs/nsp_ratio_3_1_report_task3.json
- /data/ocean/DAPT/runs/nsp_ratio_1_3_report_task1.json
- /data/ocean/DAPT/runs/nsp_ratio_1_3_report_task3.json

---

## 5. Task2（EBQA）阶段

```bash
cd /data/ocean/DAPT
GPU_LIST=0 PARALLEL=0 RESUME=1 \
bash experiments/nsp_ratio_ablation/run_nsp_ratio_ebqa_all.sh
```

输出报告：

- /data/ocean/DAPT/runs/nsp_ratio_1_1_report_task2.json
- /data/ocean/DAPT/runs/nsp_ratio_3_1_report_task2.json
- /data/ocean/DAPT/runs/nsp_ratio_1_3_report_task2.json

---

## 6. 一键总入口

```bash
cd /data/ocean/DAPT
GPU_LIST=0 PARALLEL=0 RESUME=1 \
bash experiments/nsp_ratio_ablation/run_nsp_ratio_ablation_all.sh
```

按需只跑部分阶段：

```bash
cd /data/ocean/DAPT
RUN_PRETRAIN=1 RUN_KVNER=1 RUN_EBQA=0 \
GPU_LIST=0 PARALLEL=0 RESUME=1 \
bash experiments/nsp_ratio_ablation/run_nsp_ratio_ablation_all.sh
```

---

## 7. 常用环境变量

- VARIANTS：默认 r11,r31,r13
- GPU_LIST：默认 0,1,2
- PARALLEL：0 串行；1 并行
- RESUME：1 断点续跑；0 强制重跑
- NSP_NEGATIVE_PROB：总负样本概率，默认 0.5
- NUM_ROUNDS / MLM_EPOCHS_PER_ROUND / NSP_EPOCHS_PER_ROUND：预训练轮次

示例（只跑 1:1 和 3:1）：

```bash
cd /data/ocean/DAPT
VARIANTS=r11,r31 GPU_LIST=0,1 PARALLEL=1 RESUME=1 \
bash experiments/nsp_ratio_ablation/run_nsp_ratio_pretrain_all.sh
```

---

## 8. 日志与排错

日志目录默认：

- /data/ocean/DAPT/runs/nsp_ratio_ablation_logs

失败时脚本会自动输出：

- 失败阶段
- 日志路径
- 日志尾部（最近 80 行）

你也可以手动查看：

```bash
tail -n 100 /data/ocean/DAPT/runs/nsp_ratio_ablation_logs/r11_pretrain.gpu0.log
```
