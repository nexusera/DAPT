"""
远端运行的数据字段检查脚本
用法：在远端训练机上运行：
  python check_processed_dataset.py /data/ocean/bpe_workspace/processed_dataset
脚本会打印 dataset 概览、列名、首条样本并验证 `input_ids` 与 `word_ids` 的存在和类型。
期望输出示例要点：
  - columns: 包含 'input_ids' 与 'word_ids'
  - first sample: shows 'input_ids' as list[int], 'word_ids' as list[int|None]
  - summary booleans: input_ids_ok=True, word_ids_ok=True
"""
import sys
import os
from datasets import load_from_disk

def check_dataset(path):
    if not os.path.exists(path):
        print(f"ERROR: dataset path not found: {path}")
        return 2
    ds = load_from_disk(path)
    print("Dataset splits:", list(ds.keys()))
    for split in ds.keys():
        print(f"\n--- Split: {split} ---")
        cols = ds[split].column_names
        print("Columns:", cols)
        if len(ds[split]) == 0:
            print("Empty split")
            continue
        sample = ds[split][0]
        print("First sample keys:", list(sample.keys()))
        # check input_ids
        input_ids_ok = 'input_ids' in sample and isinstance(sample['input_ids'], list) and len(sample['input_ids'])>0
        word_ids_ok = 'word_ids' in sample and isinstance(sample['word_ids'], list)
        print(f"input_ids_ok={input_ids_ok}, word_ids_ok={word_ids_ok}")
        if input_ids_ok:
            print("input_ids (len, types):", len(sample['input_ids']), type(sample['input_ids'][0]))
        if word_ids_ok:
            # show first 30 entries of word_ids
            print("word_ids sample (first 30):", sample['word_ids'][:30])
        # 额外检查：报告样本中 input_ids 的最大值，便于判断是否与模型 vocab 大小冲突
        try:
            max_id = max(sample['input_ids']) if input_ids_ok else None
            print("sample max input_id:", max_id)
        except Exception:
            pass

    return 0

if __name__ == '__main__':
    if len(sys.argv) > 1:
        ds_path = sys.argv[1]
    else:
        ds_path = "/data/ocean/bpe_workspace/processed_dataset"
    rc = check_dataset(ds_path)
    sys.exit(rc)
