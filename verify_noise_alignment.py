#!/usr/bin/env python3
"""
校验 OCR 特征与 dataset 的对齐情况

用法:
    python verify_noise_alignment.py \
      --dataset /data/ocean/bpe_workspace/processed_dataset_with_noise_v2 \
      --ocr_json ~/semi_label/ocr_rerun/char_ocr_9297.json \
      --check_samples 20
"""

import argparse
import json
import os
from datasets import load_from_disk
from transformers import AutoTokenizer

def load_ocr_list(path: str):
    """加载 OCR JSON 列表"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"OCR json not found: {path}")
    
    if path.endswith(".jsonl"):
        ocr_list = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ocr_list.append(json.loads(line))
        return ocr_list
    
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, dict):
        for key in ["data", "ocr_list", "items"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        return [obj]
    raise ValueError(f"Unsupported OCR JSON format: {path}")

def extract_text_from_ocr(ocr_obj):
    """从 OCR JSON 中提取文本（模拟 train.txt 的生成逻辑）"""
    if isinstance(ocr_obj, dict) and "words_result" in ocr_obj:
        texts = []
        for item in ocr_obj.get("words_result", []):
            if isinstance(item, dict) and "words" in item:
                s = item["words"].strip()
                if s:
                    texts.append(s)
        return " ".join(texts)
    return ""

def extract_text_from_dataset_sample(sample, tokenizer):
    """从 dataset 样本中还原文本（通过 tokenizer.decode）"""
    input_ids = sample.get("input_ids", [])
    if not input_ids:
        return ""
    # 解码 tokens，过滤特殊 token
    text = tokenizer.decode(input_ids, skip_special_tokens=True)
    return text.strip()


def has_noise(sample: dict) -> bool:
    """
    检查样本是否存在非零噪声特征。
    兼容字段：
    - noise_values: 连续特征（add_noise_features.py 写入）
    - noise_features: 旧字段，若存在也检查
    """
    for key in ["noise_values", "noise_features"]:
        nv = sample.get(key) or []
        if nv and any(any(v != 0.0 for v in feat) for feat in nv):
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="校验 OCR 特征与 dataset 对齐")
    parser.add_argument("--dataset", type=str, required=True, help="processed_dataset 路径")
    parser.add_argument("--ocr_json", type=str, required=True, help="OCR JSON 路径")
    parser.add_argument("--check_samples", type=int, default=20, help="检查前 N 条样本")
    parser.add_argument("--tokenizer", type=str, default="/data/ocean/bpe_workspace/my-medical-tokenizer", help="tokenizer 路径")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="检查哪个 split：train/test/all（all 会依次检查 train 和 test，并对 test 自动加 offset）",
    )
    args = parser.parse_args()
    
    print(f"加载 dataset: {args.dataset}")
    dataset = load_from_disk(args.dataset)
    
    print(f"加载 OCR JSON: {args.ocr_json}")
    ocr_list = load_ocr_list(args.ocr_json)
    
    print(f"加载 tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    print(f"\n数据集信息:")
    if isinstance(dataset, dict) and "train" in dataset:
        print(f"  Train 样本数: {len(dataset['train'])}")
    if isinstance(dataset, dict) and "test" in dataset:
        print(f"  Test  样本数: {len(dataset['test'])}")
    print(f"  OCR 数据条数: {len(ocr_list)}")
    
    def _split_order(dd):
        order = []
        for s in ["train", "validation", "valid", "dev", "test"]:
            if s in dd:
                order.append(s)
        for s in sorted(dd.keys()):
            if s not in order:
                order.append(s)
        return order

    def _split_offsets(dd):
        offsets = {}
        running = 0
        for s in _split_order(dd):
            offsets[s] = running
            running += len(dd[s])
        return offsets

    if not isinstance(dataset, dict):
        raise ValueError("dataset 必须是 DatasetDict（含 train/test split）")

    split_arg = (args.split or "train").lower()
    if split_arg not in {"train", "test", "all"}:
        raise ValueError("--split 仅支持 train/test/all")

    splits_to_check = []
    if split_arg == "all":
        for s in ["train", "test"]:
            if s in dataset:
                splits_to_check.append(s)
    else:
        if split_arg not in dataset:
            raise ValueError(f"dataset 中不存在 split={split_arg}，现有 splits={list(dataset.keys())}")
        splits_to_check = [split_arg]

    offsets = _split_offsets(dataset)

    for split_name in splits_to_check:
        split_data = dataset[split_name]
        offset = offsets.get(split_name, 0)
        check_count = min(args.check_samples, len(split_data), max(0, len(ocr_list) - offset))

        print(f"\n开始检查 split={split_name} 的前 {check_count} 条样本（ocr_offset={offset}）...")
        print("=" * 80)

        matches = 0
        mismatches = 0
        has_noise_cnt = 0
        no_noise_cnt = 0

        for idx in range(check_count):
            sample = split_data[idx]
            global_idx = offset + idx
            ocr_obj = ocr_list[global_idx] if global_idx < len(ocr_list) else None
        
            # 提取文本
            dataset_text = extract_text_from_dataset_sample(sample, tokenizer)
            ocr_text = extract_text_from_ocr(ocr_obj) if ocr_obj else ""
        
            # 检查噪声特征
            has_noise_feat = has_noise(sample)
            if has_noise_feat:
                has_noise_cnt += 1
            else:
                no_noise_cnt += 1
        
            # 简单匹配检查（取前50字符比较）
            dataset_preview = dataset_text[:50].replace(" ", "")
            ocr_preview = ocr_text[:50].replace(" ", "")
        
            # 如果两者都有文本且相似（简单检查）
            is_match = False
            if dataset_preview and ocr_preview:
                # 简单检查：是否有重叠的字符
                if len(set(dataset_preview) & set(ocr_preview)) > 5:
                    is_match = True
                    matches += 1
                else:
                    mismatches += 1
            elif not dataset_preview and not ocr_preview:
                is_match = True
                matches += 1
            else:
                mismatches += 1
        
            status = "✅ 匹配" if is_match else "❌ 不匹配"
            noise_status = "有噪声" if has_noise_feat else "无噪声"

            print(f"\n样本 {idx} (global={global_idx}): {status} | {noise_status}")
            print(f"  Dataset 文本: {dataset_preview}...")
            print(f"  OCR 文本:     {ocr_preview}...")

        print("\n" + "=" * 80)
        print(f"检查结果统计（split={split_name}，前 {check_count} 条）:")
        print(f"  文本匹配: {matches} / {check_count}")
        print(f"  文本不匹配: {mismatches} / {check_count}")
        print(f"  有噪声特征: {has_noise_cnt} / {check_count}")
        print(f"  无噪声特征: {no_noise_cnt} / {check_count}")

        if check_count == 0:
            print("\n⚠️  未检查任何样本：可能是 OCR JSON 条数不足或 split 为空")
        elif matches / check_count < 0.5:
            print(f"\n⚠️  警告: 匹配率低于 50%，可能存在对齐问题！")
            print(f"   建议: 检查 OCR JSON 和 dataset 的来源是否一致，以及是否发生了 shuffle/过滤")
        else:
            print(f"\n✅ 对齐检查通过（匹配率 {matches/check_count*100:.1f}%）")

        # 额外检查：split 的噪声特征覆盖率（采样）
        print(f"\n检查 split={split_name} 的噪声特征覆盖率（采样）...")
        total_n = len(split_data)
        samples_with_noise = 0
        sample_indices = list(range(0, total_n, max(1, total_n // 1000)))
        for si in sample_indices:
            if has_noise(split_data[si]):
                samples_with_noise += 1
        estimated_coverage = (samples_with_noise / len(sample_indices)) * 100 if sample_indices else 0.0
        print(f"  估计噪声特征覆盖率: {estimated_coverage:.1f}% (基于 {len(sample_indices)} 个采样点)")

if __name__ == "__main__":
    main()

