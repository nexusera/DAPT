# 医疗 BERT DAPT 项目 —— 大厂面试工程细节问答手册

> 适用场景：算法工程师（NLP/大模型方向）面试，涵盖预训练、微调、推理、评测全链路。

---

## 目录

1. [项目全貌速览](#项目全貌速览)
2. [一、Noise-Embedding 机制深挖](#一noise-embedding-机制深挖)
3. [二、KV-MLM 与 KV-NSP 预训练任务拷问](#二kv-mlm-与-kv-nsp-预训练任务拷问)
   - [2.1 如何在无标注语料中识别 KV 边界](#21-如何在无标注语料中识别-kv-边界)
   - [2.2 KV-MLM 的掩码比例与防过拟合策略](#22-kv-mlm-的掩码比例与防过拟合策略)
   - [2.3 KV-NSP 困难负样本的构造](#23-kv-nsp-困难负样本的构造)
   - [2.4 KV-NSP 数据集构建全流程](#24-kv-nsp-数据集构建全流程)
   - [2.5 微调框架与推理框架](#25-微调框架与推理框架)
4. [三、全流程工程 Pipeline 细节验证](#三全流程工程-pipeline-细节验证)
   - [3.1 预训练语料的文档切分与滑窗策略](#31-预训练语料的文档切分与滑窗策略)
   - [3.2 词表扩充与新词 Embedding 初始化](#32-词表扩充与新词-embedding-初始化)
   - [3.3 预训练规模与训练超参数](#33-预训练规模与训练超参数)
   - [3.4 下游任务数据集构建](#34-下游任务数据集构建)
   - [3.5 评测流程](#35-评测流程)
5. [四、常见延伸追问与参考回答](#四常见延伸追问与参考回答)

---

## 项目全貌速览

**项目核心**：在通用预训练模型（MacBERT）基础上，针对医疗 OCR 扫描报告场景，进行领域自适应预训练（DAPT），最终服务于两类下游任务：
- **Task 1/3（KV-NER）**：从医疗报告中抽取键值对实体（命名实体识别范式）。
- **Task 2（EBQA）**：将结构化键值提取转化为基于证据的 QA 匹配范式。

**三大创新点**：
1. **Noise-Embedding**：将 OCR 引擎输出的噪声质量信号嵌入为向量，使模型对扫描质量具备动态感知能力。
2. **KV-MLM（全词掩码 MLM）**：以医疗键值对边界为引导，用 jieba + 专业词典做全词掩码，替代随机掩码。
3. **KV-NSP（键值匹配二分类）**：设计键值对匹配任务，预训练阶段即让模型学习"Key 与 Value 是否配对"。

---

## 一、Noise-Embedding 机制深挖

### 1.1 7 个维度的具体含义与提取方式

| 维度名 | 含义 | 从 OCR 原始输出的提取公式 |
|---|---|---|
| `conf_avg` | 单词识别的平均字符置信度 | `probability.average`（OCR 直接输出） |
| `conf_min` | 单词中最低字符置信度 | `probability.min`（OCR 直接输出） |
| `conf_var_log` | 置信度方差的对数 | `log10(probability.variance + 1e-12)`（加小量防 log(0)） |
| `conf_gap` | 均值与最低值之差，表示置信度离散程度 | `conf_avg - conf_min` |
| `punct_err_ratio` | 单词中非汉字、非数字字符占比，衡量乱码率 | `sum(ch 不是汉字且不是数字) / len(word)` |
| `char_break_ratio` | 字符相对于图像宽度的密度，衡量字符截断程度 | `len(word) / max(1.0, location.width)`，截断上限 0.25 |
| `align_score` | 当前词相对于所属段落平均 top 坐标的偏差，衡量排版对齐程度 | `abs(location.top - 段落平均top)`，截断上限 3500 |

**对齐粒度**：特征按 OCR `words_result`（单词级）提取，通过 `word_ids`（由 jieba 分词产出）将特征从词级对齐到 token 级，保证 tokenizer 子词切分后每个 token 都携带对应的噪声特征。

---

### 1.2 分桶策略：等频分桶 + 锚点桶

**策略选择：等频分桶（Quantile Binning）**。

各维度桶数如下：

| 特征 | 桶数 |
|---|---|
| `conf_avg` | 64 |
| `conf_min` | 64 |
| `conf_var_log` | 32 |
| `conf_gap` | 32 |
| `punct_err_ratio` | 16 |
| `char_break_ratio` | 32 |
| `align_score` | 64 |

**等频分桶的核心代码逻辑**：

```python
qs = np.linspace(0, 1, nb + 1)[1:]       # 生成 nb 个等间距分位点
bounds = np.quantile(non_zero, qs)         # 按分位数计算桶边界
```

**为何选等频分桶而非等宽分桶**：
- OCR 置信度分布高度偏斜（大多数识别置信度 > 0.9），等宽分桶会导致高置信区间桶极度稀疏，模型无法有效区分不同噪声等级。
- 等频分桶保证每个桶内样本数量大致相等，嵌入向量受到均衡训练。

**锚点桶（Anchor Bin）设计**：
- 值为 `0.0` 或非有限值（`NaN`/`Inf`）的 token 统一映射到 ID=0，作为"无特征"锚点。
- 非零值用 `np.digitize` 映射到 `[1, n_bins]`，ID 加 1 以避开锚点位置。
- 非 OCR 来源的语料（书籍、指南等）全部赋予"完美值" `[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]`，经分桶后对应最高质量的桶 ID，与 OCR 样本在同一嵌入空间中形成对比。

**边界值的确定方式**：不人工设定，由实际 OCR 语料（`char_ocr_9297.json`）的数据分布一次性 `fit`，保存为 `noise_bins.json`，训练与推理全程复用同一份边界文件，保证一致性。

---

### 1.3 Noise-Embedding 与文本向量的融合方式

**采用加性融合（直接相加），无维度对齐问题。**

融合公式：

```
embeddings_final = (word_emb + position_emb + token_type_emb)
                 + α × noise_embed
```

其中 `noise_embed` 的计算为：

```
noise_embed = Σ_{i=1}^{7} Embedding_i(bucket_id_i)
```

每个维度独立维护一个 `nn.Embedding(n_bins+1, hidden_size)` 表，7 个维度的查表结果直接逐元素相加，输出维度与 `hidden_size`（768）完全一致，不引入任何维度拼接。

`α` 是一个可学习标量参数，初始化为 `0.1`，训练过程中自动调节噪声信号对主干嵌入的贡献比例。

**参数量增加分析**：

```
额外参数量 = Σ (n_bins_i + 1) × hidden_size
           = (65+65+33+33+17+33+65) × 768
           = 311 × 768 ≈ 238,848 ≈ 23.9 万参数
```

MacBERT-base 主干约 1.02 亿参数，噪声嵌入额外引入约 **0.23%** 的参数量，训练开销可忽略不计。

---

### 1.4 消融实验：去掉 Noise-Embedding 后的性能影响

在消融对照中（实验 7：No-Noise Baseline，`kv_ner_config_no_noise.json`），模型在以下错误类型上性能下降最明显：

| 错误类型 | 原因 | 影响方向 |
|---|---|---|
| **低置信度区域的实体误识别** | 扫描模糊时模型无法区分正常文本与乱码，倾向于将乱码片段也识别为实体 | NER Precision 下降 |
| **排版对齐错位导致的 Value 抽取越界** | 多栏报告中，相邻字段因坐标偏差被误判为同一 KV 对的 Value | QA Recall 下降 |
| **字符截断导致的实体边界偏移** | 字符密度过高（`char_break_ratio` 大）时，实体末尾 token 被错误纳入 | NER F1 下降，尤其在数值型实体（如化验指标数值）上 |

相比之下，去掉 Noise-Embedding 后，在高质量文本（PDF 直接转文字的病历）上性能差距较小，印证了该机制主要针对 OCR 噪声场景发挥作用，消融实验可分层（按 OCR 质量分段）进行以验证此结论。

---

## 二、KV-MLM 与 KV-NSP 预训练任务拷问

### 2.1 如何在无标注语料中识别 KV 边界

**采用"词典引导 + 规则辅助"的弱监督方案，不依赖人工标注。**

**关键词典来源**：
1. `biaozhu_keys_only.txt`：从标注平台（Label Studio）导出的所有标注键名（如"白细胞计数"、"血红蛋白"等），直接用于 jieba 自定义词典，强制分词器将这些专业键名识别为整体 token。
2. `kept_vocab.txt`：经 LLM 过滤后保留的医疗领域词汇，也注入 jieba 词典，扩大全词掩码的覆盖范围。

**`word_ids` 的生成机制**：
- 在 `build_dataset_final_slim.py` 的数据预处理阶段，对每个文本先用注入了专业词典的 jieba 进行分词，再通过 tokenizer 对每个 jieba 词进行子词切分，记录每个 token 对应的 jieba 词的 `word_id`。
- 同一 jieba 词内的所有 token 共享同一 `word_id`，在 KV-MLM 的 collator 中，当某个 token 被选中做 Mask 时，其所属 jieba 词内的所有 token 均同时被 Mask（全词掩码）。

**为何不用 NER 模型预标注**：预训练阶段语料规模达数十万文档，在线推理标注成本过高；词典匹配的召回率对于高频医疗键名已足够，少量词典未覆盖的键名仍以 token 级掩码处理，不影响整体预训练质量。

---

### 2.2 KV-MLM 的掩码比例与防过拟合策略

**掩码比例**：`--mlm_probability 0.15`，与标准 BERT/MacBERT 保持一致。

**掩码实现差异**：标准 MLM 按 token 独立随机采样，KV-MLM 以 `word_id` 为单位做 group 采样——一旦某 word 被选中，其内所有 token 同时掩盖，实际被掩码的 token 比例仍在 15% 附近，但掩码粒度从 subword 提升到 word/phrase 级别。

**防过拟合措施**：
- **分阶段训练（Staged Training）**：`--num_rounds 3 --mlm_epochs_per_round 1 --nsp_epochs_per_round 3`，即 MLM 阶段与 NSP 阶段交替训练（1 epoch MLM → 3 epoch NSP → 循环），防止模型在某一任务上过拟合。
- **语料多样性**：多源混合语料（临床病历 35% + 医学教材 13% + 百科 7% + 通用语料 13% + 论文 8% 等），医疗领域数据约 87%，通用数据约 13%，防止领域过拟合导致通用语言理解能力退化。
- **消融对照（`--mlm_masking token`）**：设置了使用 token 级随机掩码的对照组，可量化 KV 全词掩码的额外收益。

---

### 2.3 KV-NSP 困难负样本的构造

**任务定义**：输入格式为 `[CLS] Key [SEP] Value [SEP]`，使用 `token_type_ids` 区分 Key 段和 Value 段，输出二分类（匹配 / 不匹配）。

**负样本的两种类型**：

**类型 1：倒序负样本（Reverse Negative）**

> 场景举例：同一份检验报告中包含多个键值对：
> - 正样本：`[CLS] 白细胞计数 [SEP] 6.2×10⁹/L [SEP]`（Label=1）
> - 倒序负样本：`[CLS] 血红蛋白 [SEP] 6.2×10⁹/L [SEP]`（将报告中另一个 Key 与当前 Value 配对，Label=0）

这类负样本的语义上是"真实存在的值，配了错误的键"，模型无法通过数值的合理性（"6.2 是合理的血液指标数值"）来判断，必须真正理解键值语义配对关系，是典型的**结构相似型困难负样本**。

**类型 2：随机 Value 负样本（Random Negative）**

> 场景举例：`[CLS] 白细胞计数 [SEP] 阴性 [SEP]`（从其他样本中随机抽取一个 Value，Label=0）

这类负样本较容易区分，用于保持负样本的多样性，防止模型只学倒序策略。

**自动化构造方式**：
- 从 Label Studio 导出的标注 JSON（5 份，含 Key-Value 标注对）中自动读取所有文档的 KV pairs。
- 以 `--nsp_negative_prob 0.5` 控制总负样本概率，以 `--nsp_reverse_negative_ratio` 和 `--nsp_random_negative_ratio` 控制两类负样本的内部比例（消融实验对比了 1:1、3:1、1:3 三种配置）。
- 构造逻辑在 `kv_nsp/negative_sampling.py` 中实现，对每份文档内的 KV pairs 做 intra-document reverse，保证倒序负样本与正样本来自同一份报告，提高样本语义难度。

---

### 2.4 KV-NSP 数据集构建全流程

#### 原始数据来源

来自 Label Studio 标注平台导出的 5 份 JSON 文件，覆盖不同病历类型：

| 文件名 | 来源类型 |
|---|---|
| `ruyuanjilu1119.json` | 入院记录 |
| `menzhenbingli1119.json` | 门诊病历 |
| `shuhoubingli1119.json` | 术后病历 |
| `huojianbingli1119.json` | 活检病历 |
| `huizhenbingli1119.json` | 会诊病历 |

#### 标注格式解析逻辑

Label Studio 导出的每条样本包含 `annotations[].result` 列表，其中：

- **type=`labels`**：实体标注，`value.labels[0]` 为 `"键名"` 或 `"值"`，`value.text` 为实体文本。
- **type=`relation`**：有向关系，`from_id` 指向键名实体 ID，`to_id` 指向值实体 ID。

解析时仅使用 `was_cancelled=False` 的最新标注，按关系连边构建正样本对 `(key_text, value_text)`。

#### 动态负采样机制（运行时生成）

负样本**不预先生成**，在 `Dataset.__getitem__` 每次调用时动态决策，避免负样本集固化导致模型记忆特定模式：

```
取第 i 个正样本 (key_i, value_i)
│
├── P = negative_prob（默认 0.5）→ 保持正样本，label=1
│
└── P = 1 - negative_prob → 构造负样本，label=0
    │
    ├── 抽签：reverse_ratio : random_ratio（默认 1:1）
    │
    ├── 倒序负样本：text_a=value_i, text_b=key_i（键值互换）
    │   └── 若互换后恰好是真实正对（防误标），回退到 random 策略
    │
    └── 随机负样本：text_a=key_i, text_b=random(value_pool)
        └── 重试最多 max_easy_retries=10 次，避免随机到真实正对
            └── 若全部重试失败，回退到 global_random_fallback（跨文档随机抽 key+value）
```

**防假负例（False Negative）机制**：构建 `valid_pairs_set`（所有正样本的集合），每次生成候选负样本时在集合中查找，若命中则重试，确保负样本标签的正确性。

#### 数据集规模与切分

- 按 `train_ratio=0.9` 随机切分训练集/验证集（`sklearn.model_selection.train_test_split`，固定 `random_state=42`）。
- 数据集大小 = 标注正样本对数量（由 5 份 JSON 中的 relation 数决定）；实际训练时因动态负采样，每个 epoch 正/负样本比例约各 50%。

#### 预训练阶段 vs 独立微调阶段的数据路径

| 使用场景 | 数据入口 | 备注 |
|---|---|---|
| 预训练 DAPT（`train_dapt_macbert_staged.py`） | `--nsp_data_dir pseudo_kv_labels_filtered.json` | 兼容单文件或目录，自动扫描 `*.json` |
| 独立 KV-NSP 微调（`kv_nsp/run_train.py`） | `--data_dir` + `--data_files` | 同上 5 份文件，独立作为二分类任务训练 |

---

### 2.5 微调框架与推理框架

#### 预训练阶段：自定义 MTL 框架

| 组件 | 实现 |
|---|---|
| 模型类 | 自定义 `BertForDaptMTL`（继承 `BertPreTrainedModel`） |
| 底座 | `BertModelWithNoise`（替换标准 `BertEmbeddings` 为 `BertNoiseEmbeddings`） |
| 预测头 | `BertPreTrainingHeads`（包含 MLM head + NSP 二分类 head，复用 BERT 原生结构） |
| 损失函数 | `CrossEntropyLoss`（MLM）+ `CrossEntropyLoss`（NSP，mask 掉 label=-100 的 MLM 阶段样本） |
| 训练引擎 | HuggingFace `Trainer`，分阶段实例化：`trainer_mlm` 和 `trainer_nsp` 复用同一 model 对象，权重持续累积 |
| 混合精度 | `fp16=True`（自动检测 CUDA 可用性） |
| 梯度累积 | `gradient_accumulation_steps=4`，有效 batch size = 16 × 4 × 2卡 = 128 |
| DataLoader | `dataloader_num_workers=4`，`pin_memory=True`，`persistent_workers=True` |
| 日志 | TensorBoard（`report_to="tensorboard"`） |
| 检查点策略 | `save_strategy="epoch"`，`save_total_limit=1`（只保留最新 checkpoint，节省磁盘） |

#### 独立 KV-NSP 微调阶段（`kv_nsp/run_train.py`）

| 组件 | 实现 |
|---|---|
| 模型类 | `AutoModelForSequenceClassification(num_labels=2, ignore_mismatched_sizes=True)` |
| 权重来源 | 从 DAPT 预训练输出加载 backbone，自动忽略 MLM head 与 NSP 头的尺寸不匹配 |
| 优化器 | AdamW，`weight_decay=0.01` |
| 学习率 | `3e-5`，`warmup_ratio=0.1` |
| Batch Size | train=16，eval=32 |
| 评估指标 | Accuracy / Precision / Recall / **F1**（以 F1 选最优模型） |
| 最优模型选择 | `load_best_model_at_end=True`，`metric_for_best_model="f1"` |

#### 下游 KV-NER 推理框架

```
train_with_noise.py（微调，config JSON 驱动）
       ↓
compare_models.py（推理，输出 QA 级 _preds.jsonl + _gt.jsonl）
       ↓
align_for_scorer_span.py（Span 对齐，解决 tokenizer 子词边界偏移）
       ↓
scorer.py（Task1/Task3 评分，--overlap_threshold -1 忽略精确位置）
```

#### 下游 EBQA 推理框架

```
convert_ebqa.py（KV-NER JSON → QA JSONL，写入 noise_ids/noise_values）
       ↓
train_ebqa.py（微调，SQuAD Span Extraction 范式）
       ↓
predict_ebqa.py（推理，输出 QA 级 preds.jsonl：report_index + question_key + pred_text + score）
       ↓
aggregate_qa_preds_to_doc.py（QA 级 → 文档级聚合，--prefer score 取最高置信度 span）
       ↓
preprocess_ebqa_real_h200.py（text_hash 对齐，将 doc-level 预测与 GT 的 id 对上）
       ↓
scorer.py --task_type task2（需 cd 到 MedStruct-S-master 目录，依赖本地 med_eval 模块）
```

**关键工程细节**：推理时 `predict_ebqa.py` 输出的是 QA 粒度（数千行，每行对应一个"文档×键名"组合），必须经过 `aggregate_qa_preds_to_doc.py` 聚合回文档粒度（行数 ≈ 测试集文档数，如 355 条），才能与 GT 的文档级格式对齐，否则 scorer 行数不匹配导致评测失败。

---

## 三、全流程工程 Pipeline 细节验证

### 3.1 预训练语料的文档切分与滑窗策略
       📝 预训练切分“三板斧”总结
       1. 设置参数（记住数值）
       窗口 (Window)：1000 字符

       步长 (Stride)：500 字符（重叠率 50%）

       模型上限：512 Token

       2. 为什么窗口设 1000？（溢出策略）
       目的：喂饱模型。

       逻辑：中文“字符”到“Token”有缩水（约 1:0.8）。1000 字符换算后约 800 Token，远超 BERT 的 512 限制。

       结果：即使后续有硬截断，也能保证模型处理的每一块数据都是**满载（满 512 长度）**的，计算效率最高。

       3. 为什么步长设 500？（无死角覆盖）
       目的：防止断章取义。

       逻辑：任何一段文本（如：重要信息A）如果因为太靠后被窗口 1 截断了，由于我们只挪了窗口的一半距离，它必然会出现在窗口 2 的前半段（黄金观察位）。

       结果：整篇文档的所有内容，至少有一次机会以“完整状态”出现在模型的视野内。

       💡 核心一句话口诀
       “大窗口保效率（填满 512），半重叠保质量（接好断头句）。”

       记忆锦囊：
       想象你在给一条长木板刷漆，刷子宽 1 米，但你每次只往前走 0.5 米。虽然多刷了一遍，但保证了漆最厚（数据最足）且没缝隙（信息不断层）

#### 整体切分流程

预训练语料的文本处理分为**两个独立阶段**，各自解决不同粒度的问题：

```
原始多源语料（JSON/TXT）
        ↓ extract_and_dedup_json_v2.py（MD5 去重）
train.txt（每行一篇文档）
        ↓ resample_mix.py（可选，多源配比重采样）
train_resampled.txt
        ↓ chunk_long_lines.py（字符级滑窗切分长行）
train_chunked.txt（每行长度 ≤ window，后续送入 tokenizer）
        ↓ build_dataset_final_slim.py（jieba + tokenizer，生成 input_ids / word_ids）
processed_dataset（HuggingFace Dataset 格式，存盘）
```

---

#### 第一阶段：字符级滑窗切分（`chunk_long_lines.py`）

**核心逻辑**：逐行读取，若行长 ≤ window 则直接输出；若行长 > window，则以 stride 为步长做字符级滑窗，生成若干条重叠子串。

```python
while start < n:
    chunk = line[start : start + window]   # 截取窗口
    chunks.append(chunk)
    if start + window >= n:
        break
    start += stride                         # 步长滑动
```

**超参数设置**：

| 参数 | 值 | 设置理由 |
|---|---|---|
| `--window` | 1000 字符 | 中文 1 字符 ≈ 1~2 个 token；1000 字符经 tokenizer 处理后通常产出 700~900 个 token，搭配后续 512 硬截断，可覆盖绝大多数完整句子，极少在句中间截断 |
| `--stride` | 500 字符（= window / 2） | 50% 重叠率。任何一段连续文本至少被两个窗口覆盖，跨窗口边界的内容（包括可能跨边界的 KV 对）在下一个窗口中仍可被完整看到；同时控制数据膨胀倍数约为 2× |

**重叠的意义**：若某个键值对（如"白细胞计数：6.2×10⁹/L"）恰好被第 k 个窗口从中间切断，50% 重叠保证该内容整体出现在第 k+1 个窗口中，从而在整个 epoch 中至少被模型以完整形式见到一次。

---

#### 第二阶段：分词与 word_ids 对齐（`build_dataset_final_slim.py`）

该阶段对每行文本执行：

```
text → jieba 分词 → BERT tokenizer 子词切分 → word_ids 对齐
```

**关键参数**：

| 参数 | 值 |
|---|---|
| `--max_len` | 512（含 [CLS]/[SEP]） |
| `--batch_size` | 1000（HuggingFace `dataset.map` 的批大小） |
| `--shuffle_split` | OCR 路：`False`（保持顺序，与 OCR JSON 对齐）；非 OCR 路：`True` |
| train/test 划分 | 95% / 5%，`seed=42`，test 集用于计算 PPL |

**超过 512 的硬截断处理**：

```python
if len(tokens) > max_len:
    tokens = tokens[:max_len]       # 前缀截断
    word_ids = word_ids[:max_len]
    tokens[-1] = tokenizer.sep_token_id   # 强制最后一位为 [SEP]
    word_ids[-1] = None
```

---

#### 针对 KV 边界的保护措施

切分层面**没有专门的 KV 感知逻辑**（字符级滑窗本身不感知语义结构），但在分词层面通过以下机制间接保护 Key 不被拆碎：

**机制 1：jieba 注入业务键名词典**

```python
jieba.load_userdict("biaozhu_keys_only_min5.txt")   # 业务 Key（频次 > 5）
jieba.load_userdict("vocab_for_jieba.txt")           # WordPiece 挖掘的医疗高频词
```

jieba 在切词时会将词典中的字符串作为原子单位保留，如"白细胞计数"、"血红蛋白"等字段名不会被拆成"白"+"细胞"+"计数"。这保证了：
- 整个 Key 名称在 `word_ids` 中共享同一 `word_id`。
- KV-MLM 做全词掩码时，Key 内的所有 token 要么全被掩盖，要么全不被掩盖，不存在 Key 名被部分掩盖的情况。

**机制 2：噪声对齐校验（OCR 路）**

OCR 路在数据集构建后强制执行 `verify_noise_alignment.py`，校验文本 token 与 OCR 字符位置的对应关系，目标：噪声覆盖率 ~100%。若对齐率低，说明文本/OCR 顺序在切分阶段发生了错位，需重建数据集。这间接保证了 OCR 路的每份切片与原报告的字符顺序严格一致，不会出现 KV 内容被乱序重组的情况。

**现存局限与可能的改进方向**

字符级滑窗本身无法感知"当前窗口是否将一个 KV 对切断"。一个潜在改进方案是在切分时检测常见的键值分隔符（如"：""="后跟数值的模式），将窗口边界对齐到这些分隔符之后，但该方案对非结构化自由文本（书籍、论文）不适用，且工程复杂度较高。当前用 50% 重叠 + jieba 词典两者结合，已在实践中取得了可接受的效果。

---

### 3.2 词表扩充与新词 Embedding 初始化

**词表扩充流程**：

1. **OCR 语料词频挖掘**（`train_ocr_clean.py`）：在 `train_chunked.txt` 上训练 BPE/字频统计，提取高频医疗 subword。
2. **LLM 过滤**（`filter_vocab_with_llm.py`）：调用本地 LLM API 对候选词进行医学相关性过滤（`--topn 50000`，批量大小 64），保留 `kept_vocab.txt`，丢弃 `dropped_vocab.txt`。
3. **Tokenizer 合并**（`final_merge_v9_regex_split_slim.py`）：将 LLM 精修后的 OCR 词表 + `biaozhu_keys_only.txt`（纯键名词典）合并入 MacBERT 原始词表，输出 `my-medical-tokenizer/`。

**新词 Embedding 初始化方式**：

对于词表中新增的医疗专有词汇，其嵌入向量使用 **正态分布随机初始化**（`mean=0.0, std=0.02`），与 MacBERT 原始词表的初始化方差保持一致，避免新词 embedding 在量级上与原词表产生差异，从而稳定早期训练。

一种更优但未采用的方案是"子词均值初始化"（将新词分解为已有 subword 的平均嵌入），这在少量新词时效果更好，但对于大量领域专属缩写（如"CEA"、"AFP"）该方案效果有限。

---

### 3.2 预训练规模与训练超参数

| 参数 | 值 |
|---|---|
| 基座模型 | MacBERT（`hfl/chinese-macbert-base`，BERT-base 架构，768 hidden） |
| 训练框架 | HuggingFace Transformers `Trainer` |
| 优化器 | AdamW（Trainer 默认） |
| 学习率 | `5e-5` |
| 最大序列长度 | 512 |
| MLM 掩码率 | 0.15 |
| 训练轮次结构 | 3 rounds × (1 epoch MLM + 3 epoch NSP) |
| GPU 配置 | 2 × GPU（`CUDA_VISIBLE_DEVICES=4,5`，型号为数据中心侧 GPU） |
| 并行方式 | 单机双卡 DDP（通过 `torchrun --nproc_per_node=2` 启动，`--master_port=29505`） |
| 语料规模 | 多源混合约数十万文档，OCR 来源 9297 份医疗报告；书籍/病历/维基等合计百万量级语句 |
| 滑窗切分 | window=1000 字符，stride=500 字符（减少 512 截断损失） |
| 混合精度 | `bf16`（短程 sanity check 时可关闭以排查 NaN） |

**语料配比（resample 权重）**：

| 来源 | 权重 |
|---|---|
| 临床病历（clinical） | 0.35 |
| 医学教材核心（book_core） | 0.15 |
| 医学教材（med_book） | 0.13 |
| 通用语料2（fineweb_edu） | 0.13 |
| 医学论文（paper） | 0.08 |
| 维基医学（wiki_med） | 0.07 |
| 通用语料1（general） | 0.05 |
| 维基通用（wiki_general） | 0.02 |
| 补充语料（supplement） | 0.02 |

医疗领域数据合计约 87%，通用数据约 15%，设计意图是维持一定通用语言理解能力，防止领域遗忘。

---

### 3.3 下游任务数据集构建

#### Task 1/3 —— KV-NER（命名实体识别范式）

**数据来源**：Label Studio 导出的 5 份标注 JSON，包含医疗报告原文及对应的键值对实体标注（span 级别）。

**测试集特殊处理**：使用带 OCR 噪声特征的"真实测试集"（`real_test_with_ocr.json`），而非纯净文本，以真实反映 OCR 场景下的模型表现。

**推理后处理**：由于 tokenizer 子词切分可能导致预测 span 与标注 span 边界不完全对齐，引入 `align_for_scorer_span.py` 进行对齐，并使用 `--overlap_threshold -1` 忽略精确位置约束，以 span 文本匹配为主。

---

#### Task 2 —— EBQA（基于证据的 QA 匹配，Evidence-Based QA）

**核心思路**：将 KV 键值提取问题转化为**阅读理解（MRC）范式**：以医疗报告为上下文（Context），以字段名（Key）为问题（Question），让模型从报告中抽取对应的字段值（Value）作为答案。

**数据转化流程（`convert_ebqa.py`）**：

```
输入：KV-NER 格式的 JSON（含 text + kv_pairs）
      + keys_merged_1027_cleaned.json（标准键名集合，即 Query Set）

处理：对每份报告，遍历 Query Set 中的每个键名，
      若报告的 kv_pairs 中包含该键名，则该键名的值为正答案；
      否则答案为空（无该字段）。

输出（JSONL，每行一个 QA 样本）：
{
  "report_index": "doc_001",
  "question_key": "白细胞计数",
  "context": "检查日期：2023-01-01\n白细胞计数：6.2×10⁹/L\n...",
  "answer_start": 18,
  "answer_text": "6.2×10⁹/L",
  "noise_ids": [[...], ...],    # token 级 7 维桶 ID
  "noise_values": [[...], ...]  # token 级 7 维连续值
}
```

**模型输入格式**：

```
[CLS] 白细胞计数 [SEP] 检查日期：2023-01-01 白细胞计数：6.2×10⁹/L... [SEP]
  ↑ token_type_id=0 ↑         ↑ token_type_id=1 ↑
```

模型在上下文部分预测答案的起止位置（start/end logits），即标准 SQuAD 范式的 Span Extraction。

**QA 到文档级结果的聚合（`aggregate_qa_preds_to_doc.py`）**：

```
QA 级预测（数千行，每行一个 question-document 对）
       ↓ 按 report_index 聚合
文档级预测（与测试集文档数相同，每份报告包含所有预测 KV 对）
```

当同一键名存在多个候选 span 时，使用 `--prefer score` 策略选取模型置信度最高的答案。

---

### 3.4 评测流程

**Task 1/3（KV-NER）**：
```
微调 → 推理（compare_models.py，输出 _preds.jsonl）
     → span 对齐（align_for_scorer_span.py）
     → scorer.py（--task_type task1，--overlap_threshold -1）
```

**Task 2（EBQA）**：
```
数据转换（convert_ebqa.py）→ 微调（train_ebqa.py）→ 推理（predict_ebqa.py）
     → QA 聚合（aggregate_qa_preds_to_doc.py）
     → id 对齐（preprocess_ebqa_real_h200.py）
     → scorer.py（--task_type task2，需 cd 到 MedStruct-S-master 目录）
```

---

## 四、常见延伸追问与参考回答

### Q1：为什么选 MacBERT 而不是 RoBERTa 或其他模型？

MacBERT 将 MLM 预训练中的被掩词替换为语义相近词（而非 `[MASK]` token），减轻了 Pretrain-Finetune 的 token 分布差异；其原生使用 wwm（全词掩码）策略，与本项目的 KV-WWM 思路高度契合，在中文医疗文本上的词边界对齐更好。

### Q2：词表扩充后 vocab size 变化多大？对训练有何影响？

通过 `filter_vocab_with_llm.py` 筛选后保留的医疗词汇（`topn=50000`，实际保留量视过滤率而定）加上键名词表，估计扩充后 vocab size 在 **22000~25000** 之间（MacBERT 原始 ~21128）。Vocab size 增大会使 MLM head 的输出层参数量正比增加，因此扩充规模应以实际业务需求为准，过大的词表会明显增加 MLM loss 计算的内存开销。

### Q3：两路数据（OCR vs 非 OCR）如何避免噪声特征错配？

- OCR 路：`build_dataset_final_slim.py` 保持原始文档顺序（`--no_shuffle_split`），`add_noise_features.py` 按文档索引与 `char_ocr_9297.json` 一一对齐，写入真实 7 维连续噪声值。
- 非 OCR 路：无 `noise_values` 字段，Collator 在 batch 组装时自动填充"完美值" `[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]`。
- 合并时：`merge_datasets.py` 按 index 分别从两路 dataset 读取，禁止按索引把 OCR 特征硬塞给无 OCR 元信息的样本。
- 上线前强制执行 `verify_noise_alignment.py`，目标指标为噪声覆盖率 ~100%、文本-噪声高匹配率。

### Q4：staged 训练（交替 MLM + NSP）相比联合多任务有什么优劣？

| | Staged 交替训练 | 联合多任务（MTL，单轮混合） |
|---|---|---|
| **优点** | 各任务 loss scale 独立控制，不存在梯度量级冲突；调参更直观 | 训练步数更少，epoch 利用率更高 |
| **缺点** | 总 epoch 数更多，训练时间较长 | 需要仔细调 loss 权重比例，NSP/MLM 梯度互相干扰时可能两端都不收敛 |

本项目同时保留了 MTL 对照组（`kv_ner_config_mtl.json`），可通过下游 Task1/2 指标量化两种方案的差异。

### Q5：如果模型在推理阶段遇到没有 OCR 元信息的文本怎么处理？

对于推理时没有 OCR 噪声特征的输入，Collator / 推理脚本统一使用"完美值"填充（`noise_ids` 全部对应最高质量桶，等价于告诉模型"本文档 OCR 质量极好"），在这种情况下噪声嵌入退化为一个固定偏置向量，模型行为等价于标准 BERT，不影响纯净文本场景下的推理正确性。

---

*文档最后更新：2026-04-01*
