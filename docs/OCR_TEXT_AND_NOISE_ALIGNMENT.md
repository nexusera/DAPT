# OCR 正文与噪声对齐：问题说明与修复流程

本文说明 **微调数据里 `ocr_text` 与 `ocr_raw.words_result` 不一致** 的根因、已在代码中的修复，以及建议你**从试跑到全量重训**的执行顺序。

---

## 1. 发现了什么问题

### 1.1 现象

对 `biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json` / `real_test_with_ocr.json` 做审计时：

- 约 **一半样本** 满足：`ocr_text` ≠ `"".join(words_result[i]["words"])`（记作 `join(words)`）。

### 1.2 后果

- **训练时** `Sample.text` 来自字段 **`ocr_text`**。
- **字符级噪声** `noise_values` 由 `compute_noise_from_ocr.py` 根据 **`ocr_raw.words_result`** 按词展开，等价于对齐 **`join(words)`** 的字符序列。
- 两者不是同一条字符串时，从第一个不一致字符起，**同一字符下标上的「BERT 输入字」与「噪声向量」错位**。
- NER 标签若按 **`ocr_text`** 标 `start/end`，也会与噪声通道一起处于不一致坐标系。

### 1.3 根因（管线层面）

| 步骤 | 文件 | 旧行为 |
|------|------|--------|
| 合并 OCR | `fetch_and_merge_baidu_ocr.py` | 只写入 `ocr_raw`，**不更新**顶层 `ocr_text`。`ocr_text` 仍来自标注 JSON 原始导出（可能是旧 OCR、平台预览前缀等）。 |
| 算噪声 | `compute_noise_from_ocr.py` | 只根据 `words_result` 写 `noise_values` / `noise_values_per_word`，**不改** `ocr_text`。 |

因此：**正文与 OCR 块长期「不同源」**，不是 `compute_noise_from_ocr.py` 公式算错。

---

## 2. 代码侧已做的修复

### 2.1 `fetch_and_merge_baidu_ocr.py`

- OCR 成功写入 `ocr_raw` 后，**默认**执行：

  `ocr_text = "".join(words_result[].words)`

- 若与旧 `ocr_text` 不同，旧值备份到 **`ocr_text_before_ocr_sync`**。
- 仅当必须兼容历史行为时使用 **`--no_sync_ocr_text`**（**不推荐**）。
- 修复循环内 **`image_field` 未初始化** 可能导致的 `NameError`。

说明见 **`guides/fetch_and_merge_baidu_ocr.md`**。

### 2.2 工程化推理（Serving）

- 约定：`ocr_text` 必须与 **`join(words)`** 一致（与 `compare_models._extract_ocr_text` 一致）。
- 示例请求已修正：`serving/tests/fixtures/sample_ocr_request.json`。

---

## 3. 可执行工具与命令

以下均在 **`/data/ocean/DAPT`**（或你本机 `DAPT` 根目录）下执行，且建议先 **`git pull`**。

### 3.1 审计：`ocr_text` 与 `join(words)` 是否一致

```bash
cd /data/ocean/DAPT

python3 scripts/inspect_kvbert_data_formats.py --audit \
  --train_json biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json

python3 scripts/inspect_kvbert_data_formats.py --audit \
  --test_json biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json
```

**目标**：在**新管线**产出的数据上，`ocr_text≠join(words)` 应为 **0**。

（当前仓库里若仍是旧数据，审计仍会显示大量不一致；需完成下文 **第 4 节** 重新生成。）

### 3.2 校验：`transferred_annotations` 与同步后的 `ocr_text`

同步 `ocr_text` 后，应用 **`scripts/validate_ner_spans_after_ocr_sync.py`** 检查实体下标/文本是否仍落在 `ocr_text` 上：

```bash
cd /data/ocean/DAPT

# 全量检查测试集（退出码 1 表示存在失败样本）
python3 scripts/validate_ner_spans_after_ocr_sync.py \
  --json biaozhu_with_ocr_noise_prepared_v2/real_test_with_ocr.json

# 先试跑 200 条
python3 scripts/validate_ner_spans_after_ocr_sync.py \
  --json biaozhu_with_ocr_noise_prepared_v2/real_test_with_ocr.json \
  --limit 200

# 导出问题 record_id
python3 scripts/validate_ner_spans_after_ocr_sync.py \
  --json biaozhu_with_ocr_noise_prepared_v2/real_train_with_ocr.json \
  --bad_ids_out /tmp/kvner_bad_record_ids.txt
```

逻辑与 `load_labelstudio_export` 中 **Case C** 一致：有 `start/end` 则校验切片；否则用 `ocr_text.find(text)` 模拟锚定。

### 3.3 单条样本结构探查

```bash
cd /data/ocean/DAPT

python3 scripts/inspect_kvbert_data_formats.py --index 0 -v \
  --train_json biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json
```

---

## 4. 建议操作顺序（一步步做）

### 步骤 A：更新代码

```bash
cd /data/ocean/DAPT
git pull
```

### 步骤 B：小规模试跑（强烈建议）

对 **训练或测试 JSON** 只处理前 **50～100 条** OCR，确认无报错、图片路径正确：

```bash
cd /data/ocean/DAPT

python fetch_and_merge_baidu_ocr.py \
  --anno_json /path/to/MedStruct_S_Real_test.json \
  --image_root /data/ocean/semi_struct/all_type_pic_oss_csv \
  --extra_root /data/ocean/medstruct_benchmark/intermediate/annotated_images_for_ocr \
  --extra_root2 /data/ocean/medstruct_benchmark/ocr_results \
  --output biaozhu_with_ocr/real_test_with_ocr_try100.json \
  --use_local_baidu --local_mode accurate \
  --limit 100 --offset 0 --sleep 0.5
```

（`--anno_json` / 根目录请与你们 `docs/guides/fetch_and_merge_baidu_ocr.md` 中一致。）

然后：

```bash
python compute_noise_from_ocr.py \
  --inputs biaozhu_with_ocr/real_test_with_ocr_try100.json \
  --output_dir biaozhu_with_ocr_noise_prepared_try100
```

审计 + 校验：

```bash
python3 scripts/inspect_kvbert_data_formats.py --audit \
  --test_json biaozhu_with_ocr_noise_prepared_try100/real_test_with_ocr_try100.json

python3 scripts/validate_ner_spans_after_ocr_sync.py \
  --json biaozhu_with_ocr_noise_prepared_try100/real_test_with_ocr_try100.json
```

- 若 **`inspect --audit` 仍为大量 mismatch**：检查是否误加 **`--no_sync_ocr_text`**，或 `ocr_raw` 未写入。
- 若 **`validate` 失败比例高**：说明标注 **`start/end` 相对旧串**；需评估 **偏移修正 / 重导出 / 局部重标**（见第 5 节）。

### 步骤 C：全量重建数据（试跑通过後）

1. 按 `docs/guides/fetch_and_merge_baidu_ocr.md` 对 **train / test** 全量跑 **`fetch_and_merge_baidu_ocr.py`**（**不要** `--no_sync_ocr_text`）。
2. 对输出跑 **`compute_noise_from_ocr.py`**，建议输出到新目录，例如：

   `biaozhu_with_ocr_noise_prepared_v2/`

3. 再次运行 **`inspect_kvbert_data_formats.py --audit`** 与 **`validate_ner_spans_after_ocr_sync.py`**。

### 步骤 D：更新训练配置并重训

- 将 `kv_ner_config.json`（或 pipeline 中的 `--input_file` / 数据路径）指向 **`v2`** 数据。
- 与当前 `best` 在**同一测试集**上对比（尤其关注多列表格、钼靶等难例子集）。

### 步骤 E：Serving

- 部署新 checkpoint；请求体仍遵守 **`ocr_text == join(words)`**。

---

## 5. 是否必须「重新补标注」

**不必先假设整库重标。** 应用 **`validate_ner_spans_after_ocr_sync.py`** 看失败比例：

| 结果 | 建议 |
|------|------|
| 失败很少 | 只修少量 `record_id` 或接受自动 `find` 锚定（与 loader 一致）。 |
| 失败集中、且为统一前缀/后缀差异 | 可考虑脚本批量平移 `start/end`（需单独验证）。 |
| 失败比例高、且无规律 | 对问题列表 **分批重标** 或改进 **从 box 重算字符 span** 的导出脚本。 |

---

## 6. 相关文件索引

| 文件 | 作用 |
|------|------|
| `fetch_and_merge_baidu_ocr.py` | 图片 → 百度 OCR → 写入 `ocr_raw`，**默认同步 `ocr_text`** |
| `docs/guides/fetch_and_merge_baidu_ocr.md` | 原始三组命令与说明（已补充本节所述问题） |
| `compute_noise_from_ocr.py` | 从 `ocr_raw` 写 `noise_values` / `noise_values_per_word` |
| `scripts/inspect_kvbert_data_formats.py` | 单条探查 + **`--audit`** |
| `scripts/validate_ner_spans_after_ocr_sync.py` | 同步后 **实体与 `ocr_text` 校验** |
| `dapt_eval_package/pre_struct/kv_ner/data_utils.py` | `load_labelstudio_export`：文本与噪声如何进 `Sample` |

---

## 7. 版本与记录

- 文档与脚本随 DAPT 仓库演进；若论文或对内报告需引用「噪声错位」根因，可指向本节 **§1** 与 **`ocr_text_before_ocr_sync`** 备份字段。
