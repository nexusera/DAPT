# Noise Ablation 远端使用说明

本文说明如何在远端机器 `/data/ocean/DAPT` 上分阶段运行 `bucket / linear / mlp` 三组噪声消融实验。

## 1. 代码同步

你会通过 GitHub 同步代码，因此默认假设远端仓库路径为：

- `/data/ocean/DAPT`

进入目录：

```bash
cd /data/ocean/DAPT
conda activate medical_bert   
```

如果需要先更新代码：

```bash
git pull
```

---

## 2. 推荐执行方式

三个阶段已经解耦，推荐按下面顺序运行：

1. 预训练
2. KV-NER 微调 + 推理 + Task1/3 评测
3. EBQA 微调 + 推理 + Task2 评测

对应脚本：

- `experiments/noise_ablation/run_noise_pretrain_all.sh`
- `experiments/noise_ablation/run_noise_kvner_all.sh`
- `experiments/noise_ablation/run_noise_ebqa_all.sh`

总入口脚本仍然保留：

- `experiments/noise_ablation/run_noise_ablation_all.sh`

但建议优先使用上面 3 个分阶段脚本。

---

## 3. 环境准备

如果你在远端使用 conda 或 venv，请先激活环境，再执行脚本。

示例：

```bash
cd /data/ocean/DAPT
source /data/ocean/DAPT/.venv/bin/activate
```

如果远端不是这个环境路径，请替换成你自己的 Python 环境。

可先快速检查 Python：

```bash
python -V
```

---

## 4. 预训练阶段

### 4.1 串行执行（推荐先用这个）

```bash
cd /data/ocean/DAPT
bash experiments/noise_ablation/run_noise_pretrain_all.sh
```

默认会顺序跑：

- `bucket`
- `linear`
- `mlp`

默认使用：

- `GPU_LIST=0,1,2`
- `PARALLEL=0`
- `RESUME=1`

其中 `PARALLEL=0` 表示串行执行，只会使用 `GPU_LIST` 的第一张卡。

### 4.2 指定单卡串行

```bash
cd /data/ocean/DAPT
GPU_LIST=3 PARALLEL=0 RESUME=1 \
bash experiments/noise_ablation/run_noise_pretrain_all.sh
```

### 4.3 三卡并行
tmux attach -t noise_ab

```bash
cd /data/ocean/DAPT
GPU_LIST=2,3,4 PARALLEL=1 RESUME=1 \
bash experiments/noise_ablation/run_noise_pretrain_all.sh
```

### 4.4 常用可选参数

例如：

```bash
cd /data/ocean/DAPT
GPU_LIST=0 PARALLEL=0 RESUME=1 \
NUM_ROUNDS=3 MLM_EPOCHS_PER_ROUND=1 NSP_EPOCHS_PER_ROUND=3 \
MLP_HIDDEN_DIM=128 MLM_MASKING=kv_wwm \
bash experiments/noise_ablation/run_noise_pretrain_all.sh
```

### 4.5 预训练产物位置

三组模型默认输出到：

- `/data/ocean/DAPT/workspace/output_ablation_noise_bucket/final_staged_model`
- `/data/ocean/DAPT/workspace/output_ablation_noise_linear/final_staged_model`
- `/data/ocean/DAPT/workspace/output_ablation_noise_mlp/final_staged_model`

---

## 5. KV-NER 阶段（Task1 / Task3）

该阶段会自动完成：

1. 生成运行期配置
2. 微调 KV-NER
3. 推理
4. 对齐预测
5. 评测 Task1
6. 评测 Task3

### 5.1 运行命令

```bash
cd /data/ocean/DAPT
GPU_LIST=0 PARALLEL=0 RESUME=1 \
bash experiments/noise_ablation/run_noise_kvner_all.sh
```

### 5.2 三卡并行

```bash
cd /data/ocean/DAPT
GPU_LIST=0,1,2 PARALLEL=1 RESUME=1 \
bash experiments/noise_ablation/run_noise_kvner_all.sh
```

### 5.3 输出结果

每组会生成：

- Task1 报告
  - `/data/ocean/DAPT/runs/noise_bucket_report_task1.json`
  - `/data/ocean/DAPT/runs/noise_linear_report_task1.json`
  - `/data/ocean/DAPT/runs/noise_mlp_report_task1.json`

- Task3 报告
  - `/data/ocean/DAPT/runs/noise_bucket_report_task3.json`
  - `/data/ocean/DAPT/runs/noise_linear_report_task3.json`
  - `/data/ocean/DAPT/runs/noise_mlp_report_task3.json`

- 微调后的最佳模型目录
  - `/data/ocean/DAPT/runs/kv_ner_finetuned_noise_bucket/best`
  - `/data/ocean/DAPT/runs/kv_ner_finetuned_noise_linear/best`
  - `/data/ocean/DAPT/runs/kv_ner_finetuned_noise_mlp/best`

---

## 6. EBQA 阶段（Task2）

该阶段会自动完成：

1. 转换训练集为 EBQA JSONL
2. 转换测试集为 EBQA JSONL
3. 微调 EBQA
4. 推理
5. 聚合到文档级
6. 对齐
7. 评测 Task2

### 6.1 运行命令

```bash
cd /data/ocean/DAPT
GPU_LIST=0 PARALLEL=0 RESUME=1 \
bash experiments/noise_ablation/run_noise_ebqa_all.sh
```

### 6.2 三卡并行

```bash
cd /data/ocean/DAPT
GPU_LIST=0,1,2 PARALLEL=1 RESUME=1 \
bash experiments/noise_ablation/run_noise_ebqa_all.sh
```

### 6.3 输出结果

每组会生成：

- Task2 报告
  - `/data/ocean/DAPT/runs/noise_bucket_report_task2.json`
  - `/data/ocean/DAPT/runs/noise_linear_report_task2.json`
  - `/data/ocean/DAPT/runs/noise_mlp_report_task2.json`

- 微调后的最佳模型目录
  - `/data/ocean/DAPT/runs/ebqa_noise_bucket/best`
  - `/data/ocean/DAPT/runs/ebqa_noise_linear/best`
  - `/data/ocean/DAPT/runs/ebqa_noise_mlp/best`

---

## 7. 一次性串联运行

如果你仍然想一次跑多个阶段，可以使用总入口脚本。

### 7.1 三阶段全跑

```bash
cd /data/ocean/DAPT
GPU_LIST=0 PARALLEL=0 RESUME=1 \
bash experiments/noise_ablation/run_noise_ablation_all.sh
```

### 7.2 只跑部分阶段

只跑预训练 + KV-NER：

```bash
cd /data/ocean/DAPT
RUN_PRETRAIN=1 RUN_KVNER=1 RUN_EBQA=0 \
GPU_LIST=0 PARALLEL=0 RESUME=1 \
bash experiments/noise_ablation/run_noise_ablation_all.sh
```

只跑 KV-NER + EBQA（跳过预训练，要求预训练模型已存在）：

```bash
cd /data/ocean/DAPT
RUN_PRETRAIN=0 RUN_KVNER=1 RUN_EBQA=1 \
GPU_LIST=0 PARALLEL=0 RESUME=1 \
bash experiments/noise_ablation/run_noise_ablation_all.sh
```

---

## 8. 日志与报错定位

所有日志默认放在：

- `/data/ocean/DAPT/runs/noise_ablation_logs`

例如：

- `bucket_pretrain.gpu0.log`
- `linear_kvner_train.gpu1.log`
- `mlp_ebqa_train.gpu2.log`

脚本已经内置失败处理：

- 任一步命令报错会立刻停止
- 会打印失败阶段
- 会打印日志路径
- 会回显日志最后 80 行，方便快速定位

如果要手动查看日志：

```bash
cd /data/ocean/DAPT
ls runs/noise_ablation_logs
```

查看某个日志尾部：

```bash
tail -n 100 /data/ocean/DAPT/runs/noise_ablation_logs/bucket_pretrain.gpu0.log
```

持续跟踪日志：

```bash
tail -f /data/ocean/DAPT/runs/noise_ablation_logs/bucket_pretrain.gpu0.log
```

---

## 9. 断点续跑

默认：

- `RESUME=1`

含义：

- 如果预期输出已经存在，则自动跳过该步骤
- 适合长时间实验中断后继续跑

如果你想强制重跑某个阶段：

```bash
cd /data/ocean/DAPT
RESUME=0 GPU_LIST=0 PARALLEL=0 \
bash experiments/noise_ablation/run_noise_kvner_all.sh
```

---

## 10. 最推荐的实际执行顺序

如果你准备正式跑远端实验，建议直接按下面顺序：

### 第一步：预训练

```bash
cd /data/ocean/DAPT
source /data/ocean/DAPT/.venv/bin/activate
GPU_LIST=0 PARALLEL=0 RESUME=1 \
bash experiments/noise_ablation/run_noise_pretrain_all.sh
```

### 第二步：KV-NER

```bash
cd /data/ocean/DAPT
source /data/ocean/DAPT/.venv/bin/activate
GPU_LIST=0 PARALLEL=0 RESUME=1 \
bash experiments/noise_ablation/run_noise_kvner_all.sh
```

### 第三步：EBQA

```bash
cd /data/ocean/DAPT
source /data/ocean/DAPT/.venv/bin/activate
GPU_LIST=0 PARALLEL=0 RESUME=1 \
bash experiments/noise_ablation/run_noise_ebqa_all.sh
```

---

## 11. 补充说明

1. 这些脚本默认优先按远端路径 `/data/ocean/DAPT` 运行。
2. 若该路径不存在，脚本才会回退到当前仓库所在路径。
3. KV-NER 和 EBQA 阶段要求对应的预训练模型已经存在；如果你跳过预训练，请确认以下目录已生成：
   - `workspace/output_ablation_noise_bucket/final_staged_model`
   - `workspace/output_ablation_noise_linear/final_staged_model`
   - `workspace/output_ablation_noise_mlp/final_staged_model`
4. `bucket / linear / mlp` 三组结果文件名都已经固定，便于后续汇总。
