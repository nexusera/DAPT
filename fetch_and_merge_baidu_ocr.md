请在服务器的 DAPT 目录下依次运行以下三组命令：

1. 处理训练集 (Train)
python fetch_and_merge_baidu_ocr.py \
  --anno_json /data/ocean/medstruct_s/medstruct_s_real/sft_data/MedStruct_S_Real_train.json \
  --image_root /data/ocean/semi_struct/all_type_pic_oss_csv \
  --extra_root /data/ocean/medstruct_benchmark/intermediate/annotated_images_for_ocr \
  --extra_root2 /data/ocean/medstruct_benchmark/ocr_results \
  --output biaozhu_with_ocr/real_train_with_ocr.json \
  --use_local_baidu --local_mode accurate \
  --sleep 0.5

  
2. 处理测试集 (Test)
python fetch_and_merge_baidu_ocr.py \
  --anno_json /data/ocean/medstruct_s/medstruct_s_real/sft_data/MedStruct_S_Real_test.json \
  --image_root /data/ocean/semi_struct/all_type_pic_oss_csv \
  --extra_root /data/ocean/medstruct_benchmark/intermediate/annotated_images_for_ocr \
  --extra_root2 /data/ocean/medstruct_benchmark/ocr_results \
  --output biaozhu_with_ocr/real_test_with_ocr.json \
  --use_local_baidu --local_mode accurate \
  --sleep 0.5
3. 计算噪声特征 (Prepare Noise)
python compute_noise_from_ocr.py \
  --inputs biaozhu_with_ocr/real_train_with_ocr.json biaozhu_with_ocr/real_test_with_ocr.json \
  --output_dir biaozhu_with_ocr_noise_prepared

---

## 常见问题：ocr_text 与 OCR 拼接串不一致

**原因（已修复）**：旧版 `fetch_and_merge_baidu_ocr.py` 只写入 `ocr_raw`，**不更新**顶层 `ocr_text`。  
标注 JSON 里自带的 `ocr_text` 往往来自标注平台/旧 OCR，与**当前图片**调百度得到的 `words_result` 拼出来的字符串不是同一条 → 约一半样本出现「正文与噪声字符错位」。

**当前行为**：合并 OCR 成功后默认执行  
`ocr_text = "".join(words_result[].words)`；若与旧值不同，旧值备份到 `ocr_text_before_ocr_sync`。  
仅当必须兼容旧流程时使用 `--no_sync_ocr_text`（不推荐）。

**⚠️ 标注偏移**：若 `transferred_annotations` 等字段的 **字符级 start/end** 是相对**旧** `ocr_text` 标的，同步后偏移可能错位，需要抽样校验或做 span 映射/重标。若标注主要依赖 **box 像素框** 而非字符下标，影响因导出脚本而异，亦需评测集验证。

4. 合并后建议审计（可选）

```bash
python3 scripts/inspect_kvbert_data_formats.py --audit \
  --test_json biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \
  --train_json biaozhu_with_ocr_noise_prepared/real_train_with_ocr.json
```

目标：`ocr_text≠join(words)` 应为 **0**。