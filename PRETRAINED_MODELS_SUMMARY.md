# 预训练产出模型汇总

本文档用于汇总当前仓库里已经出现过的预训练模型产物、核心特性、默认存放位置、对应训练脚本与命令入口。

主要依据：

- `DAPT/pipeline_new.md`：当前主预训练流程
- `DAPT/experiments/noise_ablation/REMOTE_USAGE.md`
- `DAPT/experiments/mlm_ablation/REMOTE_USAGE.md`
- `DAPT/experiments/nsp_ratio_ablation/REMOTE_USAGE.md`
- `DAPT/train_dapt_macbert_staged.py`
- `DAPT/train_dapt_macbert_no_nsp.py`
- `DAPT/train_dapt_macbert_no_mlm.py`
- `DAPT/train_dapt_macbert_no_noise.py`

## 1. 统一背景配置

当前主流程的共同设定大致如下：

- 基座模型：`hfl/chinese-macbert-base`
- Tokenizer：`/data/ocean/DAPT/my-medical-tokenizer`
- 预训练数据集：`/data/ocean/DAPT/workspace/processed_dataset`
- KV-NSP 数据：`/data/ocean/DAPT/data/pseudo_kv_labels_filtered.json`
- 噪声分桶文件：`/data/ocean/DAPT/workspace/noise_bins.json`
- 主训练脚本：`DAPT/train_dapt_macbert_staged.py`
- 默认 staged 训练产物结构：
  - 中间轮次：`round_{k}_mlm/`、`round_{k}_nsp/`
  - 最终模型：`final_staged_model/`

主流程里常见的默认超参：

- `learning_rate=5e-5`
- `num_rounds=3`
- `mlm_epochs_per_round=1`
- `nsp_epochs_per_round=3`
- `mlm_probability=0.15`
- `max_length=512`

## 2. 总表

| 模型名称 | 主要特性 | 训练脚本/入口 | 默认最终模型目录 |
| --- | --- | --- | --- |
| Main Full（主流程 KV-MLM + KV-NSP + Noise） | KV-aware MLM，全词掩码；带 Noise Embedding；带 KV-NSP staged 交替训练 | `train_dapt_macbert_staged.py` / `pipeline_new.md` | `/data/ocean/DAPT/workspace/output_macbert_kvmlm_staged/final_staged_model` |
| Ablation No NSP | 只保留 MLM，不做 NSP；仍保留噪声建模 | `train_dapt_macbert_no_nsp.py` | `/data/ocean/DAPT/workspace/output_ablation_no_nsp/final_no_nsp_model` |
| Ablation No MLM | 只保留 NSP，不做 MLM；仍保留噪声建模 | `train_dapt_macbert_no_mlm.py` | `/data/ocean/DAPT/workspace/output_ablation_no_mlm/final_no_mlm_model` |
| No Noise Baseline | 做 MLM + NSP，但不使用噪声嵌入 | `train_dapt_macbert_no_noise.py` | `/data/ocean/DAPT/workspace/output_no_noise_baseline/final_no_noise_model`|
| MLM Ablation: `kvmlm` | 和主流程同类的 KV-aware MLM 版本，作为 MLM 消融基线 | `experiments/mlm_ablation/run_mlm_pretrain_all.sh` | `/data/ocean/DAPT/workspace/output_ablation_mlm_kvmlm/final_staged_model` |
| MLM Ablation: `plainmlm` | 普通 token-level MLM，不使用 `word_ids` 做 KV 全词掩码 | `experiments/mlm_ablation/run_mlm_pretrain_all.sh` | `/data/ocean/DAPT/workspace/output_ablation_mlm_plainmlm/final_staged_model` |
| Noise Ablation: `bucket` | 噪声特征采用分桶 Embedding 查表 | `experiments/noise_ablation/run_noise_pretrain_all.sh` | `/data/ocean/DAPT/workspace/output_ablation_noise_bucket/final_staged_model` |
| Noise Ablation: `linear` | 7 维连续噪声直接线性映射到 hidden | `experiments/noise_ablation/run_noise_pretrain_all.sh` | `/data/ocean/DAPT/workspace/output_ablation_noise_linear/final_staged_model` |
| Noise Ablation: `mlp` | 7 维连续噪声经 2-layer MLP 投影 | `experiments/noise_ablation/run_noise_pretrain_all.sh` | `/data/ocean/DAPT/workspace/output_ablation_noise_mlp/final_staged_model` |
| NSP Ratio: `1:1` | KV-NSP 负样本 reverse/random = 1:1 | `experiments/nsp_ratio_ablation/run_nsp_ratio_pretrain_all.sh` | `/data/ocean/DAPT/workspace/output_ablation_nsp_ratio_1_1/final_staged_model` |
| NSP Ratio: `3:1` | KV-NSP 负样本 reverse/random = 3:1 | `experiments/nsp_ratio_ablation/run_nsp_ratio_pretrain_all.sh` | `/data/ocean/DAPT/workspace/output_ablation_nsp_ratio_3_1/final_staged_model` |
| NSP Ratio: `1:3` | KV-NSP 负样本 reverse/random = 1:3 | `experiments/nsp_ratio_ablation/run_nsp_ratio_pretrain_all.sh` | `/data/ocean/DAPT/workspace/output_ablation_nsp_ratio_1_3/final_staged_model` |

### 2.1 Task1/3（KV-NER）微调模型与推理结果路径

这一节汇总上表这些预训练模型在 `Task1/3` 下游流程中的：

- KV-NER 微调后最佳模型目录
- `compare_models.py` 的输出摘要路径
- 对齐后的预测结果路径
- Task1 / Task3 最终评分报告路径

补充说明：

- `compare_models.py --output_summary xxx.json` 会自动生成：
  - `xxx_preds.jsonl`
  - `xxx_gt.jsonl`
- 手工流程（`pipeline_xiaorong.md` 里的 `macbert / no_nsp / no_mlm / no_noise`）对齐文件一般使用：
  - `*_aligned_preds.jsonl`
  - `*_aligned_gt.jsonl`
- 一键脚本流程（`mlm_ablation / noise_ablation / nsp_ratio_ablation`）对齐文件一般使用：
  - `*_task13_aligned_preds.jsonl`
  - `*_task13_aligned_gt.jsonl`

| 预训练模型 | KV-NER 微调后最佳模型目录 | `compare_models` 输出摘要 | 对齐后预测结果 | Task1 报告 | Task3 报告 |
| --- | --- | --- | --- | --- | --- |
| Main Full（主流程 KV-MLM + KV-NSP + Noise） | `/data/ocean/DAPT/runs/kv_ner_finetuned_macbert/best` | `/data/ocean/DAPT/runs/macbert_eval_summary.json` | `/data/ocean/DAPT/runs/macbert_eval_aligned_preds.jsonl` | `/data/ocean/DAPT/runs/macbert_eval_report_task1.json` | `/data/ocean/DAPT/runs/macbert_eval_report_task3.json` |
| Ablation No NSP | `/data/ocean/DAPT/runs/kv_ner_finetuned_no_nsp/best` | `/data/ocean/DAPT/runs/no_nsp_eval_summary.json` | `/data/ocean/DAPT/runs/no_nsp_eval_aligned_preds.jsonl` | `/data/ocean/DAPT/runs/no_nsp_eval_report_task1.json` | `/data/ocean/DAPT/runs/no_nsp_eval_report_task3.json` |
| Ablation No MLM | `/data/ocean/DAPT/runs/kv_ner_finetuned_no_mlm/best` | `/data/ocean/DAPT/runs/no_mlm_eval_summary.json` | `/data/ocean/DAPT/runs/no_mlm_eval_aligned_preds.jsonl` | `/data/ocean/DAPT/runs/no_mlm_eval_report_task1.json` | `/data/ocean/DAPT/runs/no_mlm_eval_report_task3.json` |
| No Noise Baseline | `/data/ocean/DAPT/runs/kv_ner_finetuned_no_noise/best` | `/data/ocean/DAPT/runs/no_noise_eval_summary.json` | `/data/ocean/DAPT/runs/no_noise_eval_aligned_preds.jsonl` | `/data/ocean/DAPT/runs/no_noise_eval_report_task1.json` | `/data/ocean/DAPT/runs/no_noise_eval_report_task3.json` |
| MLM Ablation: `kvmlm` | `/data/ocean/DAPT/runs/kv_ner_finetuned_mlm_kvmlm/best` | `/data/ocean/DAPT/runs/mlm_kvmlm_eval_summary.json` | `/data/ocean/DAPT/runs/mlm_kvmlm_task13_aligned_preds.jsonl` | `/data/ocean/DAPT/runs/mlm_kvmlm_report_task1.json` | `/data/ocean/DAPT/runs/mlm_kvmlm_report_task3.json` |
| MLM Ablation: `plainmlm` | `/data/ocean/DAPT/runs/kv_ner_finetuned_mlm_plainmlm/best` | `/data/ocean/DAPT/runs/mlm_plainmlm_eval_summary.json` | `/data/ocean/DAPT/runs/mlm_plainmlm_task13_aligned_preds.jsonl` | `/data/ocean/DAPT/runs/mlm_plainmlm_report_task1.json` | `/data/ocean/DAPT/runs/mlm_plainmlm_report_task3.json` |
| Noise Ablation: `bucket` | `/data/ocean/DAPT/runs/kv_ner_finetuned_noise_bucket/best` | `/data/ocean/DAPT/runs/noise_bucket_eval_summary.json` | `/data/ocean/DAPT/runs/noise_bucket_task13_aligned_preds.jsonl` | `/data/ocean/DAPT/runs/noise_bucket_report_task1.json` | `/data/ocean/DAPT/runs/noise_bucket_report_task3.json` |
| Noise Ablation: `linear` | `/data/ocean/DAPT/runs/kv_ner_finetuned_noise_linear/best` | `/data/ocean/DAPT/runs/noise_linear_eval_summary.json` | `/data/ocean/DAPT/runs/noise_linear_task13_aligned_preds.jsonl` | `/data/ocean/DAPT/runs/noise_linear_report_task1.json` | `/data/ocean/DAPT/runs/noise_linear_report_task3.json` |
| Noise Ablation: `mlp` | `/data/ocean/DAPT/runs/kv_ner_finetuned_noise_mlp/best` | `/data/ocean/DAPT/runs/noise_mlp_eval_summary.json` | `/data/ocean/DAPT/runs/noise_mlp_task13_aligned_preds.jsonl` | `/data/ocean/DAPT/runs/noise_mlp_report_task1.json` | `/data/ocean/DAPT/runs/noise_mlp_report_task3.json` |
| NSP Ratio: `1:1` | `/data/ocean/DAPT/runs/kv_ner_finetuned_nsp_ratio_1_1/best` | `/data/ocean/DAPT/runs/nsp_ratio_1_1_eval_summary.json` | `/data/ocean/DAPT/runs/nsp_ratio_1_1_task13_aligned_preds.jsonl` | `/data/ocean/DAPT/runs/nsp_ratio_1_1_report_task1.json` | `/data/ocean/DAPT/runs/nsp_ratio_1_1_report_task3.json` |
| NSP Ratio: `3:1` | `/data/ocean/DAPT/runs/kv_ner_finetuned_nsp_ratio_3_1/best` | `/data/ocean/DAPT/runs/nsp_ratio_3_1_eval_summary.json` | `/data/ocean/DAPT/runs/nsp_ratio_3_1_task13_aligned_preds.jsonl` | `/data/ocean/DAPT/runs/nsp_ratio_3_1_report_task1.json` | `/data/ocean/DAPT/runs/nsp_ratio_3_1_report_task3.json` |
| NSP Ratio: `1:3` | `/data/ocean/DAPT/runs/kv_ner_finetuned_nsp_ratio_1_3/best` | `/data/ocean/DAPT/runs/nsp_ratio_1_3_eval_summary.json` | `/data/ocean/DAPT/runs/nsp_ratio_1_3_task13_aligned_preds.jsonl` | `/data/ocean/DAPT/runs/nsp_ratio_1_3_report_task1.json` | `/data/ocean/DAPT/runs/nsp_ratio_1_3_report_task3.json` |

## 3. 各模型详细说明

### 3.1 主模型与核心对照

#### 3.1.1 Main Full

- 训练脚本：`DAPT/train_dapt_macbert_staged.py`
- 主流程命令来源：`DAPT/pipeline_new.md`
- 关键特性：
  - `mlm_masking=kv_wwm`
  - 同时使用 MLM 和 KV-NSP
  - 使用噪声特征
  - staged 交替训练
- 默认输出目录：
  - 根目录：`/data/ocean/DAPT/workspace/output_macbert_kvmlm_staged`
  - 最终模型：`/data/ocean/DAPT/workspace/output_macbert_kvmlm_staged/final_staged_model`

示例命令：

```bash
cd /data/ocean/DAPT
python /data/ocean/DAPT/train_dapt_macbert_staged.py \
  --output_dir /data/ocean/DAPT/workspace/output_macbert_kvmlm_staged \
  --dataset_path /data/ocean/DAPT/workspace/processed_dataset \
  --nsp_data_dir /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
  --tokenizer_path /data/ocean/DAPT/my-medical-tokenizer \
  --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
  --learning_rate 5e-5 \
  --num_rounds 3 \
  --mlm_epochs_per_round 1 \
  --nsp_epochs_per_round 3 \
  --mlm_probability 0.15 \
  --max_length 512 \
  --mlm_masking kv_wwm
```

#### 3.1.2 Ablation No NSP

- 训练脚本：`DAPT/train_dapt_macbert_no_nsp.py`
- 特性：
  - 仅做 MLM
  - 不做 NSP
  - 保留 Noise Embedding
- 最终目录：
  - `/data/ocean/DAPT/workspace/output_ablation_no_nsp/final_no_nsp_model`

建议命令模板：

```bash
cd /data/ocean/DAPT
python /data/ocean/DAPT/train_dapt_macbert_no_nsp.py \
  --output_dir /data/ocean/DAPT/workspace/output_ablation_no_nsp \
  --dataset_path /data/ocean/DAPT/workspace/processed_dataset \
  --tokenizer_path /data/ocean/DAPT/my-medical-tokenizer \
  --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
  --learning_rate 5e-5 \
  --num_epochs 3
```

#### 3.1.3 Ablation No MLM

- 训练脚本：`DAPT/train_dapt_macbert_no_mlm.py`
- 特性：
  - 仅做 NSP
  - 不做 MLM
  - 保留 Noise Embedding
- 最终目录：
  - `/data/ocean/DAPT/workspace/output_ablation_no_mlm/final_no_mlm_model`

建议命令模板：

```bash
cd /data/ocean/DAPT
python /data/ocean/DAPT/train_dapt_macbert_no_mlm.py \
  --output_dir /data/ocean/DAPT/workspace/output_ablation_no_mlm \
  --nsp_data_dir /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
  --tokenizer_path /data/ocean/DAPT/my-medical-tokenizer \
  --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
  --learning_rate 5e-5 \
  --num_epochs 5
```

#### 3.1.4 No Noise Baseline

- 训练脚本：`DAPT/train_dapt_macbert_no_noise.py`
- 特性：
  - 标准 MLM + NSP
  - 不使用 Noise Embedding
  - 使用自定义 tokenizer
- 最终目录名固定为：
  - `final_no_noise_model`

建议命令模板：

```bash
cd /data/ocean/DAPT
python /data/ocean/DAPT/train_dapt_macbert_no_noise.py \
  --output_dir /data/ocean/DAPT/workspace/output_no_noise_baseline \
  --dataset_path /data/ocean/DAPT/workspace/processed_dataset \
  --nsp_data_dir /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json \
  --tokenizer_path /data/ocean/DAPT/my-medical-tokenizer \
  --learning_rate 5e-5 \
  --num_rounds 3 \
  --mlm_epochs_per_round 1 \
  --nsp_epochs_per_round 3
```

### 3.2 MLM 掩码消融

来源文档：`DAPT/pipeline_new.md` 与 `DAPT/experiments/mlm_ablation/REMOTE_USAGE.md`

#### `kvmlm`

- 特性：
  - `mlm_masking=kv_wwm`
  - 使用 `word_ids` 做 KV-aware 全词掩码
  - 默认仍配合 noise bucket + KV-NSP
- 最终模型目录：
  - `/data/ocean/DAPT/workspace/output_ablation_mlm_kvmlm/final_staged_model`

#### `plainmlm`

- 特性：
  - `mlm_masking=token`
  - 普通 token-level MLM
  - 其余设置与 `kvmlm` 保持一致，便于公平对照
- 最终模型目录：
  - `/data/ocean/DAPT/workspace/output_ablation_mlm_plainmlm/final_staged_model`

推荐入口命令：

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
bash experiments/mlm_ablation/run_mlm_pretrain_all.sh
```

### 3.3 Noise Embedding 消融

来源文档：`DAPT/pipeline_new.md` 与 `DAPT/experiments/noise_ablation/REMOTE_USAGE.md`

#### `bucket`

- 特性：分桶 ID -> Embedding，属于当前噪声建模基线
- 最终模型目录：
  - `/data/ocean/DAPT/workspace/output_ablation_noise_bucket/final_staged_model`

#### `linear`

- 特性：7 维连续噪声直接线性映射到 hidden
- 最终模型目录：
  - `/data/ocean/DAPT/workspace/output_ablation_noise_linear/final_staged_model`

#### `mlp`

- 特性：7 维连续噪声先走 2-layer MLP，再映射到 hidden
- 最终模型目录：
  - `/data/ocean/DAPT/workspace/output_ablation_noise_mlp/final_staged_model`

推荐入口命令：

```bash
cd /data/ocean/DAPT
bash experiments/noise_ablation/run_noise_pretrain_all.sh
```

如需显式指定：

```bash
cd /data/ocean/DAPT
GPU_LIST=0 PARALLEL=0 RESUME=1 \
NUM_ROUNDS=3 MLM_EPOCHS_PER_ROUND=1 NSP_EPOCHS_PER_ROUND=3 \
MLP_HIDDEN_DIM=128 MLM_MASKING=kv_wwm \
bash experiments/noise_ablation/run_noise_pretrain_all.sh
```

### 3.4 KV-NSP 负样本比例消融

来源文档：`DAPT/pipeline_new.md` 与 `DAPT/experiments/nsp_ratio_ablation/REMOTE_USAGE.md`

共同说明：

- `nsp_negative_prob=0.5`
- 只调整负样本内部 `reverse/random` 的权重
- 其他训练设置保持不变

#### `1:1`

- 特性：reverse/random 均衡
- 最终模型目录：
  - `/data/ocean/DAPT/workspace/output_ablation_nsp_ratio_1_1/final_staged_model`

#### `3:1`

- 特性：reverse 负样本占优
- 最终模型目录：
  - `/data/ocean/DAPT/workspace/output_ablation_nsp_ratio_3_1/final_staged_model`

#### `1:3`

- 特性：random 负样本占优
- 最终模型目录：
  - `/data/ocean/DAPT/workspace/output_ablation_nsp_ratio_1_3/final_staged_model`

推荐入口命令：

```bash
cd /data/ocean/DAPT
GPU_LIST=0,1,2 PARALLEL=1 RESUME=1 \
bash experiments/nsp_ratio_ablation/run_nsp_ratio_pretrain_all.sh
```

## 4. 已有代码清单

当前与预训练产出模型直接相关的代码/脚本可以按用途分为以下几类：

### 4.1 主训练脚本

- `DAPT/train_dapt_macbert_staged.py`
  - 主流程脚本
  - 支持 `mlm_masking=kv_wwm/token`
  - 支持 `noise_mode=bucket/linear/mlp`
  - 支持 `nsp_reverse_negative_ratio` 和 `nsp_random_negative_ratio`

- `DAPT/train_dapt_macbert_no_nsp.py`
  - No NSP 消融

- `DAPT/train_dapt_macbert_no_mlm.py`
  - No MLM 消融

- `DAPT/train_dapt_macbert_no_noise.py`
  - No Noise 基线

### 4.2 实验批处理脚本

- `DAPT/experiments/noise_ablation/run_noise_pretrain_all.sh`
- `DAPT/experiments/mlm_ablation/run_mlm_pretrain_all.sh`
- `DAPT/experiments/nsp_ratio_ablation/run_nsp_ratio_pretrain_all.sh`

### 4.3 说明文档

- `DAPT/pipeline_new.md`
- `DAPT/experiments/noise_ablation/REMOTE_USAGE.md`
- `DAPT/experiments/mlm_ablation/REMOTE_USAGE.md`
- `DAPT/experiments/nsp_ratio_ablation/REMOTE_USAGE.md`
- `DAPT/pipeline_xiaorong.md`
- `DAPT/pipeline_task2_xiaorong.md`

## 5. 路径别名与冲突说明

### 5.1 `no_noise` 路径存在不一致

仓库中目前至少有两种写法：

- `kv_ner_config_no_noise.json` 指向：
  - `/data/ocean/DAPT/workspace/output_no_noise_baseline/final_no_noise_model`
- `ebqa_config_no_noise.json` 指向：
  - `/data/ocean/DAPT/workspace/output_ablation_no_noise/final_no_noise_model`

而 `train_dapt_macbert_no_noise.py` 本身只规定最终目录名为：

- `final_no_noise_model`

并不强制上层根目录名必须是 `output_no_noise_baseline` 或 `output_ablation_no_noise`。

建议后续统一成一种写法，优先推荐：

- `/data/ocean/DAPT/workspace/output_no_noise_baseline/final_no_noise_model`

因为当前：

- `DAPT/experiments/interpretability/run_attention_noise_compare_oneclick.sh`
- `DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_no_noise.json`

都使用了这个路径。

### 5.2 主模型存在历史路径别名

仓库中除了主流程现在使用的：

- `/data/ocean/DAPT/workspace/output_macbert_kvmlm_staged/final_staged_model`

还存在旧配置写法：

- `/data/ocean/DAPT/macbert_staged_output/final_staged_model`

例如：

- `DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config_macbert.json`
- `DAPT/dapt_eval_package/pre_struct/ebqa/ebqa_config_macbert.json`

仍指向旧路径。若你后续以 `pipeline_new.md` 为主，建议统一改到新的主流程目录。

### 5.3 若按“配置等价”看，以下目录是同类模型但不同实验批次命名

以下几组在配置语义上高度接近，但目录名来自不同实验入口：

- Main Full：
  - `/data/ocean/DAPT/workspace/output_macbert_kvmlm_staged/final_staged_model`
- MLM Ablation 的 `kvmlm`：
  - `/data/ocean/DAPT/workspace/output_ablation_mlm_kvmlm/final_staged_model`
- Noise Ablation 的 `bucket`：
  - `/data/ocean/DAPT/workspace/output_ablation_noise_bucket/final_staged_model`
- NSP Ratio 的 `1:1`：
  - `/data/ocean/DAPT/workspace/output_ablation_nsp_ratio_1_1/final_staged_model`

它们不一定是同一次训练跑出来的权重，但从实验设计角度，都可以视为“Full setting”的不同命名产物或不同批次运行结果。做结果汇总时建议明确标记“主模型/同配置复现/消融基线副本”。

## 6. 建议的后续整理口径

如果后面要继续做实验记录或论文表格，建议统一按下面字段登记每个模型：

- 模型简称
- 训练目的
- 是否使用 Noise
- MLM 类型
- 是否使用 NSP
- NSP 负样本比例
- 训练脚本
- 输出根目录
- 最终模型目录
- 对应下游配置文件

这样后续做 Task1/2/3 评测、注意力可解释性、IG 归因分析时，就不容易混淆不同目录。

一行名利直接查看模型是否都在：
```
python3 -c "import os; m=[('Main Full','/data/ocean/DAPT/workspace/output_macbert_kvmlm_staged/final_staged_model'),('Ablation No NSP','/data/ocean/DAPT/workspace/output_ablation_no_nsp/final_no_nsp_model'),('Ablation No MLM','/data/ocean/DAPT/workspace/output_ablation_no_mlm/final_no_mlm_model'),('No Noise Baseline','/data/ocean/DAPT/workspace/output_no_noise_baseline/final_no_noise_model'),('MLM Ablation: kvmlm','/data/ocean/DAPT/workspace/output_ablation_mlm_kvmlm/final_staged_model'),('MLM Ablation: plainmlm','/data/ocean/DAPT/workspace/output_ablation_mlm_plainmlm/final_staged_model'),('Noise Ablation: bucket','/data/ocean/DAPT/workspace/output_ablation_noise_bucket/final_staged_model'),('Noise Ablation: linear','/data/ocean/DAPT/workspace/output_ablation_noise_linear/final_staged_model'),('Noise Ablation: mlp','/data/ocean/DAPT/workspace/output_ablation_noise_mlp/final_staged_model'),('NSP Ratio: 1:1','/data/ocean/DAPT/workspace/output_ablation_nsp_ratio_1_1/final_staged_model'),('NSP Ratio: 3:1','/data/ocean/DAPT/workspace/output_ablation_nsp_ratio_3_1/final_staged_model'),('NSP Ratio: 1:3','/data/ocean/DAPT/workspace/output_ablation_nsp_ratio_1_3/final_staged_model')]; missing=[n for n,p in m if not os.path.exists(p)]; [print(f'{n:<30} | {os.path.exists(p)}') for n,p in m]; print('-'*50); print('Missing Models:', missing if missing else 'None')"
```