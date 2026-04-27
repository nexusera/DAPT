# KV-BERT / DAPT — 代码评审报告（中文）

**仓库**：`/Users/user/Documents/DAPT`
**分支**：`feature/noise-embedding`（相较 `main` 多 296 个文件 / 约 200 万行，实际源码约 1.5–2 万行）
**HEAD**：`fcf1d61`  "docs: add OCR_TEXT_AND_NOISE_ALIGNMENT guide"
**日期**：2026-04-23
**评审范围**：预训练管线 + 下游 NER/EBQA + FastAPI 服务 + 消融实验

---

## 1. 总体结论

**KV-BERT** 是由 Hao Li 等人（AI Starfish）提出的、面向 OCR 中文临床报告的半结构化键值抽取的噪声鲁棒领域自适应预训练框架。论文贡献三项：**KV-MLM**（按医学实体 / 键值边界的全词掩码）、**KV-NSP**（在键值对上做二分类，带硬负 / 随机负样本）、以及 **Noise-Embedding**（7 维 OCR 质量特征经分桶离散化后求和注入 embedding 层）。基座模型已从 RoBERTa 切换为 **MacBERT**，整条管线从原始 OCR JSON 一直走到 Docker 化的 FastAPI 服务。

**总体评价**：*资深科研工程师水平，目标是发表论文；但工程卫生没有跟上功能扩张。*

| 维度 | 评分（1–5）| 说明 |
|------|:---------:|------|
| 架构 & ML 设计 | **4.5** | 分桶 + anchor bin 设计思路清晰；三项贡献互相独立；消融覆盖充分。 |
| 正确性 | **3** | `main` 分支严重 bug 已修，但在 `add_noise_features.py` 引入新的 5↔7 维度错配；noise shape 校验仍较弱。 |
| 代码卫生 / DRY | **2** | 11 份训练脚本之间 ≥60% 是拷贝代码；没有公共工具模块。 |
| 文档 | **4** | 中英双语 docstring；paper 初稿；管线文档与集成指南齐全。 |
| 可移植性 | **2** | `/data/ocean/DAPT/...` 绝对路径遍布全仓；6+ 脚本内嵌 `sys.path.append`。 |
| 测试覆盖 | **1.5** | 仅 serving 侧 2 个集成测试；噪声、建模、评测无任何单测。 |
| 生产可用度（serving）| **2** | 功能可跑，但无鉴权、无限流、CORS 全放开、Docker base 未 pin。 |

---

## 2. 相较 `main` 的改进

`main` 分支评审提出的 5 个问题，现已修 4 个：

| `main` 上的问题 | `feature/noise-embedding` 状态 |
|----------------|-------------------------------|
| `add_noise_features.py` 按拆分索引对齐错位 | ✅ **已修** — `global_idx = split_offsets[split] + idx`（165–182、268–271 行）。 |
| SymSpell 用 `max_dictionary_edit_distance=32` 建索引 | ✅ **已移除** — 整段 SymSpell / Levenshtein 代码被删，改走分桶路线。 |
| `char_break_norm=24` 导致特征饱和 | ✅ **已移除** — `char_break_ratio` 被硬截断到 0.25（`noise_feature_processor.py:42`）。 |
| `RobertaNoiseEmbeddings.forward` 缺 seq-len 校验 | ✅ **已修** — 直接由 `input_shape[1]` 推出（`noise_bert_model.py:123`）。 |
| `kv_nsp/run_train.py` 用已弃用的 `evaluation_strategy=` | ⚠️ **部分修** — 主预训练脚本已迁移，但 `kv_nsp/run_train.py:192`、`kv_nsp/run_train_with_noise.py:472` 仍用旧关键字。 |

---

## 3. 项目架构

```
 原始 JSON / OCR（多源）
        │
        ▼                                     extract_and_dedup_json_v2.py （MD5 去重）
 train.txt + 按源拆分文件
        │
        ▼                                     resample_mix.py  +  chunk_long_lines.py
 train_resampled.txt → train_chunked.txt
        │
        ▼                                     train_ocr_clean.py  +  filter_vocab_with_llm.py
 medical_vocab_ocr_only/vocab.txt → kept_vocab.txt
        │
        ▼                                     final_merge_v9_regex_split_slim.py
 my-medical-tokenizer/  +  vocab_for_jieba.txt （generate_jieba_vocab.py）
        │
        ▼                                     build_dataset_final_slim.py （Jieba 对齐 WWM）
 processed_dataset/  (input_ids + word_ids)
        │
        ▼                                     NoiseFeatureProcessor.fit_bins → noise_bins.json
        ▼                                     add_noise_features.py
 processed_dataset_with_noise/   (+ noise_values = 每 token 7 维)
        │
        ▼                                     train_dapt_macbert_staged.py （8×H200, 分阶段 MLM↔NSP）
 output_macbert_kvmlm_staged/final_staged_model/
        │
        ├─────►  下游微调：dapt_eval_package/pre_struct/kv_ner/train_with_noise.py
        │                  dapt_eval_package/pre_struct/ebqa/train_ebqa.py
        │
        ├─────►  消融实验：experiments/{mlm,noise,nsp_ratio,tokenizer}_ablation/*
        │
        ├─────►  可解释性：experiments/interpretability/{attention,IG}/*
        │
        └─────►  在线服务：serving/ （FastAPI + BertCRF + 批推理 + 噪声抽取）
```

**数据规模**（源自 paper）：原始 98.2 万行 → 重采样 22.5 万 → 切片 53.7 万；标注 3 582 页 OCR（训练 3 224 / 测试 358）。

---

## 4. 亮点

1. **噪声建模思路扎实** — `NoiseFeatureProcessor` 的分桶设计 + 为 "完美文本" 保留 anchor bin，使 OCR 元数据能以干净-文本相容的方式融入 BERT embedding。
2. **三项贡献切分明确** — KV-MLM / KV-NSP / Noise-Embedding 各自对应 `no_mlm`、`no_nsp`、`no_noise`、`noise_{bucket,linear,mlp,concat_linear}`、`nsp_ratio_{1:1,3:1,1:3}` 等独立消融目录。
3. **关键对齐 bug 已修** — 每个拆分独立的索引 offset 让 OCR 对齐在 train/test 上都正确。
4. **多种训练范式** — staged / MTL / hybrid-masking 皆已落地，覆盖 DAPT 常见套路。
5. **从数据到上线的全生命周期** — 数据清洗、分词器训练、预训练、下游微调（KV-NER / EBQA）、可解释性分析（attention / IG）、产品化 serving 全都有。
6. **论文级文档** — `paper.md`（LLNCS LaTeX）、`pipeline_new.md`、`PRETRAINED_MODELS_SUMMARY.md`、`interview_prep.md`、`KV_BERT_预训练与下游推理指南.md`。
7. **实用训练技巧** — `bf16`、`tf32`、`group_by_length`、梯度检查点、阶段性 curriculum，以及按需开启的 `ddp_find_unused_parameters`。

---

## 5. 问题清单（按严重度）

### 🔴 关键（Critical）

| # | 位置 | 问题 |
|---|------|------|
| **C1** | `add_noise_features.py:96` | `build_zero_feats` 仍返回 `[[0.0]*5 …]` 与 `[[False]*5 …]`，但 `FEATURES` 已有 **7** 项。fallback 路径（第 195 行，OCR 缺失 / 格式异常）会得到 **5 维** `noise_values`，而正常样本是 7 维。HF `save_to_disk` 要么因 schema 漂移报错，要么静默错齐——两种都糟。修：`[[0.0]*len(FEATURES) for _ in range(seq_len)]`。<br>**✅ 已修（f23cbbf）** 将 `[[0.0]*5]`/`[[False]*5]` 改为 `n = len(FEATURES)` 动态计算，fallback 与正常路径维度统一，消除 schema 漂移风险。 |
| **C2** | 所有 `train_dapt_*.py` | **≈60–70% 的代码重复** —— `PrecomputedWWMCollator`、`PerplexityCallback`、`RobertaModelWithNoise`、`MLMStageCollator`、`DynamicNSPDataset` 各自被重新实现 3–6 次。任何一次 bug 修复都要 N 份同步（已经出现漂移，见 M1）。应抽出 `pretraining_common.py`。<br>**✅ 已修（f23cbbf）** 新建 `pretraining_common.py`，提取 `PerplexityCallback`（4 个脚本共用）与 `PrecomputedWWMCollator`（合并 kvmlm 的防御性实现，`max_seq_len` 参数化）；4 个训练脚本改为 import 并删除本地重复定义。`MLMStageCollator`、`DynamicNSPDataset`、`RobertaModelWithNoise` 各版本存在细微差异，留注释待核对后继续合并。 |

### 🟠 高（High）

| # | 位置 | 问题 |
|---|------|------|
| **H1** | `noise_bert_model.py:147–154` | 桶模式 forward 未校验 `noise_ids.shape[-1] == len(FEATURES)`。若出现 5 维 fallback（C1）或缓存被截断，将静默索引错误 embedding 矩阵。<br>**✅ 已修（f23cbbf）** bucket 与 concat_linear 两个分支均加入 `if noise_ids.shape[-1] != len(FEATURES): raise ValueError(...)` 前置校验，C1 修复前若有旧缓存流入会立即报错而非静默污染。 |
| **H2** | `noise_bert_model.py:156–162` | 连续 / linear / mlp 模式同样没有验证 `noise_values` 的 dtype 或 shape。<br>**✅ 已修（f23cbbf）** continuous 分支加入 `noise_values.shape[-1] != len(FEATURES)` 校验，并在 `to(device, dtype=torch.float32)` 前确保 shape 已经正确，不再静默接受错误维度输入。 |
| **H3** | `noise_fusion.py:~126` | `nan_to_num` 在 `clamp` **之后** 才调用，NaN 会先流经 clamp；应当先清 NaN 再 clamp。<br>**✅ 确认已正确（f23cbbf）** 当前版本 `nan_to_num`（第 123 行）已在 `clamp` 等效操作（第 126 行）之前执行，顺序正确。评审报告描述的问题在此版本不存在。在该行添加注释 `# H3: nan_to_num 必须在 clamp 之前` 防止未来改动重新引入。 |
| **H4** | `dapt_eval_package/pre_struct/kv_ner/evaluate.py` vs `evaluate_with_dapt_noise.py` | 约 1 000 行接近相同的预测 / 组合 / 指标代码散在两份文件里，噪声开 / 关两组实验之间极易出现指标漂移。<br>**✅ 已修（f23cbbf）** 新建 `evaluate_core.py`，提取 `set_seed`、`_read_jsonl`、`_normalize_text_for_eval`、`_extract_ground_truth` 四个共享函数；`evaluate.py` 与 `evaluate_with_dapt_noise.py` 均改为从 `evaluate_core` 导入并删除各自的重复定义，消除两组实验之间指标实现漂移的可能。 |
| **H5** | `data_utils.py:226–228`、`train_with_noise.py:325–333`、`compare_models.py:81–98` | `_expand_word_noise_to_chars()` / `_broadcast_global_noise()` 在 **三份** 文件里重复实现。<br>**✅ 已修（f23cbbf）** `data_utils.py` 作为唯一规范定义保持不变；`train_with_noise.py` 两处 import 块补充导入这两个函数并删除本地重复定义；`compare_models.py` import `data_utils` 时补充两函数导入并删除本地重复定义。 |
| **H6** | `train_with_noise.py:602–603` | 取 batch 时类型不安全：`batch.get("noise_ids") if isinstance(batch, dict) else batch.noise_ids`——遇到 tuple / 自定义 collator 会崩。 |
| **H7** | `kv_nsp/run_train.py:192`、`kv_nsp/run_train_with_noise.py:472` | 仍用已弃用的 `evaluation_strategy=`（`transformers ≥ 4.46` 已移除）。 |
| **H8** | `da_core/dataset.py:901, 919` | 调试用 `print()` 尚未清理（每个 `ridx < 5` 的样本都会打印），污染日志且拖慢推理。 |
| **H9** | `serving/app.py:114` | `allow_origins=["*"]` + `allow_headers=["*"]`——任意来源都可以调用 GPU 推理接口。 |
| **H10** | `serving/`（所有路由）| 无鉴权、无限流、无请求 ID 上下游串联。8 卡模型几乎是裸奔。 |
| **H11** | `serving/routers/extract.py:178` | `detail=str(exc)`——把模型路径、CUDA 错误等内部结构直接回给调用方。 |
| **H12** | `serving/core/postprocessor.py:149` | 过滤后未再次检查 `vals` 是否为空，`full_text[v_start:v_end]` 可能抛 `IndexError`。 |
| **H13** | `serving/core/batch_engine.py:132–140` | 批窗口 deadline 锁在"第一条请求"的到达时间，后续缓慢到来的请求会因超时而错过本批。 |
| **H14** | 全仓库 | 除 `serving/test_api.py` 和 `serving/tests/test_extract_from_file.py` 之外 **完全没有单元测试**。噪声分桶、WWM collator、NSP 负采样、评测都没有覆盖。 |

### 🟡 中（Medium）

| # | 位置 | 问题 |
|---|------|------|
| M1 | 多份训练脚本 | `ddp_find_unused_parameters` 在 `train_dapt_distributed.py:395` 是 `False`、在 `train_dapt_mtl.py:566`、`train_dapt_staged.py:477` 是 `True`。staged/MTL 中 head 可能有条件地未参与，错误设为 `False` 会在 step > 0 时抛 DDP 错。 |
| M2 | `train_dapt_mtl.py:568` | 硬编码 `save_safetensors=False` —— 关闭了更安全的序列化方式且无说明。 |
| M3 | 超长文件 | `train_ebqa.py`（1 829）、`da_core/dataset.py`（1 371）、`train_with_noise.py`（1 169）、`evaluate.py`（1 154）、`model_ebqa.py`（994）、`compare_models.py`（969）、`train_dapt_macbert_staged.py`（749）—— 需要按 dataset / model / metrics / training-loop 拆分。 |
| M4 | 全仓库 | `/data/ocean/DAPT/...`、`/home/ocean/...`、`/data/hxzh/...` 绝对路径遍地——应迁入 env 或 `config.yaml`。 |
| M5 | 6+ 脚本 | `sys.path.append(current_dir)` 写法散落，对 cwd 敏感；一旦被当包导入即失效。 |
| M6 | `serving/Dockerfile` | base 镜像 `nvcr.io/nvidia/pytorch:24.03-py3` 未 pin digest。 |
| M7 | `serving/requirements.txt` | 依赖没有 `==X.Y.Z` 版本锁。 |
| M8 | `serving/Dockerfile` | `COPY . /app` 把 notebook / fixture / tools 全部拉进生产镜像，应加 `.dockerignore`。 |
| M9 | `serving/schemas/request.py` | `noise_values` 缺数值范围校验 —— NaN/inf 会污染下游计算。 |
| M10 | `serving/app.py:121–135` | 通用 `Exception` 处理器可能遮蔽 FastAPI 自身的 422 校验响应。 |
| M11 | `serving/app.py:67–69` | lifespan 钩子内的异常被静默吞掉。 |
| M12 | `serving/routers/extract.py:70` | `request_id` 本地生成，没有与上游追踪系统关联。 |
| M13 | `compute_noise_from_ocr.py` vs `noise_feature_processor.py` vs `add_noise_features.py` | 同一套特征抽取代码存在三份。 |
| M14 | `train_dapt_distributed.py:384` vs `train_dapt_macbert_staged.py:591` | 加载 noise processor 的前置检查一份有、一份没有。 |
| M15 | git 历史 | 最近 4 次连续 fix 都是围绕 OCR↔dataset 对齐（`78476fa`、`f01f729`、`fcf1d61`、`54ac0a3`），建议把"对齐不变量"写进 CI 检查。 |

### 🟢 轻微（Minor）

| # | 位置 | 问题 |
|---|------|------|
| N1 | `dapt_eval_package/.../modeling.py:97–98` | 过时注释："noise_embed_dim preserved but no longer used"——参数还在签名里，容易误导后继维护者。 |
| N2 | 全仓库 | 错误处理风格混杂：有的函数抛异常，有的静默返回 `None`（如 `dataset.py:223–224`）。 |
| N3 | `noise_feature_processor.py:157` | `to_id` 对非零值返回 `digitize + 1`，调用方 embedding 表需要开 `NUM_BINS + 1`（0 号为 anchor）；这层契约需文档化。 |
| N4 | 仓库根 | 缺少 `requirements.txt`——只有 `serving/requirements.txt`。 |
| N5 | `paper.md` | 匿名块和作者块同时存在，非匿名提交前需清理。 |
| N6 | `dapt_eval_package/pre_struct/.DS_Store` | 提交了 `.DS_Store`。 |

---

## 6. 分组健康度

| 分组 | 代表文件 | 状态 |
|------|---------|------|
| 数据摄入 | `extract_and_dedup_json_v2.py`、`resample_mix.py`、`chunk_long_lines.py` | 良好 |
| 词表 / Tokenizer | `train_ocr_clean.py`、`filter_vocab_with_llm.py`、`final_merge_v9_regex_split_slim.py`、`generate_jieba_vocab.py` | 良好（LLM 过滤缺断点续跑） |
| 数据集构建 | `build_dataset_final_slim.py`、`add_noise_features.py` | **C1** |
| 噪声核心 | `noise_feature_processor.py`、`noise_embeddings.py`、`noise_bert_model.py`、`noise_fusion.py` | **H1–H3** |
| 预训练脚本 (11) | `train_dapt_distributed.py`、`train_dapt_macbert_staged.py`、`…_no_mlm/no_nsp/no_noise/mtl/hybrid/staged` | **C2**、M1、M2 |
| KV-NSP | `kv_nsp/{dataset_with_noise.py, run_train_with_noise.py, negative_sampling.py}` | H7 |
| 下游微调 | `dapt_eval_package/pre_struct/kv_ner/*`、`ebqa/*` | **H4、H5、H6、H8**、M3 |
| 消融脚本 | `experiments/{mlm,noise,nsp_ratio,tokenizer}_ablation/*.sh` | 以 shell 居多，整体可控 |
| 可解释性 | `experiments/interpretability/*` | 研究级 |
| 在线服务 | `serving/app.py`、`serving/core/*`、`serving/routers/*` | **H9–H13、M6–M12** |
| 校验脚本 | `verify_noise_alignment.py`、`scripts/validate_ner_spans_after_ocr_sync.py`、`scripts/check_pretrain_test_leakage.py` | 有这些脚本是好事 |

---

## 7. 按优先级的推荐

**下一次预训练 / 重跑消融之前必做**

1. **C1** — 修 `add_noise_features.py:96`，改为 `len(FEATURES)`（=7）。
2. **H1** — 在 `noise_bert_model.py` forward 中加 `assert noise_ids.shape[-1] == len(FEATURES)`。
3. **H7** — `kv_nsp/run_train*.py` 从 `evaluation_strategy=` 迁移到 `eval_strategy=`。
4. **H8** — 清掉 `da_core/dataset.py` 的调试 `print()`。
5. 审查 **M1** — 逐个脚本核对 `ddp_find_unused_parameters` 与真实 head 计算图的一致性。

**交付给外部团队之前必做**

6. **C2** — 新建 `pretraining_common.py`（collator、callback、模型包装），删除重复实现。
7. **H4 / H5** — 落地统一的 `noise_utils.py` 与 `evaluate_core.py`。
8. **H14** — 建 `tests/`，最少包括：noise processor 分位数往返、WWM collator 形状/数量、NSP 负采样分布、`add_noise_features` 5 条样本的端到端回归。
9. **M4 / M5** — 引入 `config.yaml` / `.env`，去掉 `sys.path.append` 写法。
10. **M6 / M7 / M8** — pin Docker base 与依赖版本，加 `.dockerignore`。

**在把 `serving/` 暴露到内网之外之前必做**

11. **H9** — 收紧 CORS 到具体域名。
12. **H10** — 加 API key 鉴权与 token bucket 限流。
13. **H11** — 清洗错误响应：内部详写日志、对外只返回不透明错误。
14. **H12 / H13** — 修 postprocessor 的 `IndexError` 路径；批 deadline 每次新请求都要重置。
15. **M9** — `noise_values` 做数值范围校验，拒绝 NaN/inf。

**生活质量改进**

16. 拆分超过 1 000 行的文件（`train_ebqa.py`、`da_core/dataset.py`、`train_with_noise.py`、`evaluate.py`）。
17. 加一个 `CI.yml`，至少跑 `pyflakes` / 新增单测 / 全文件 lint。
18. 下一个 PR 之前把连续四次的 sync/对齐 fix commit 压成一个逻辑变更。

---

## 8. 结论

**科研团队视角**：这是一份可发表、论据齐备的代码库。ML 故事顺、消融矩阵扎实、paper 初稿已经成形，`main` 分支上那几个严重 bug 也都修了。

**生产交接视角**：还没到位。新引入的 5↔7 维错配（C1）可以静默污染一次训练；11 份训练脚本的 ~70% 重复意味着任何一次修改都要手动同步很多份；FastAPI 服务没鉴权；整个管线没有单元测试。把 C1、C2、H1、H4/H5、H9–H11、H14 解决之后，项目就能从"在我的 GPU 集群上能跑"进化到"另一个团队可以安心接手"。

**一句话建议**：今天就修 C1 和 H1；接着预算 2–3 天用于训练脚本去重 + 一套最小单测，然后再启动下一轮全量预训练。
