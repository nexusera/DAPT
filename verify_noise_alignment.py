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

def main():
    parser = argparse.ArgumentParser(description="校验 OCR 特征与 dataset 对齐")
    parser.add_argument("--dataset", type=str, required=True, help="processed_dataset 路径")
    parser.add_argument("--ocr_json", type=str, required=True, help="OCR JSON 路径")
    parser.add_argument("--check_samples", type=int, default=20, help="检查前 N 条样本")
    parser.add_argument("--tokenizer", type=str, default="/data/ocean/bpe_workspace/my-medical-tokenizer", help="tokenizer 路径")
    args = parser.parse_args()
    
    print(f"加载 dataset: {args.dataset}")
    dataset = load_from_disk(args.dataset)
    
    print(f"加载 OCR JSON: {args.ocr_json}")
    ocr_list = load_ocr_list(args.ocr_json)
    
    print(f"加载 tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    print(f"\n数据集信息:")
    print(f"  Train 样本数: {len(dataset['train'])}")
    print(f"  Test  样本数: {len(dataset['test'])}")
    print(f"  OCR 数据条数: {len(ocr_list)}")
    
    print(f"\n开始检查前 {args.check_samples} 条样本的对齐情况...")
    print("=" * 80)
    
    matches = 0
    mismatches = 0
    has_noise = 0
    no_noise = 0
    
    train_data = dataset["train"]
    check_count = min(args.check_samples, len(train_data), len(ocr_list))
    
    for idx in range(check_count):
        sample = train_data[idx]
        ocr_obj = ocr_list[idx] if idx < len(ocr_list) else None
        
        # 提取文本
        dataset_text = extract_text_from_dataset_sample(sample, tokenizer)
        ocr_text = extract_text_from_ocr(ocr_obj) if ocr_obj else ""
        
        # 检查噪声特征
        noise_features = sample.get("noise_features", [])
        has_noise_feat = bool(noise_features and any(
            any(v != 0.0 for v in feat) for feat in noise_features
        ))
        
        if has_noise_feat:
            has_noise += 1
        else:
            no_noise += 1
        
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
        
        print(f"\n样本 {idx}: {status} | {noise_status}")
        print(f"  Dataset 文本: {dataset_preview}...")
        print(f"  OCR 文本:     {ocr_preview}...")
    
    print("\n" + "=" * 80)
    print(f"检查结果统计（前 {check_count} 条）:")
    print(f"  文本匹配: {matches} / {check_count}")
    print(f"  文本不匹配: {mismatches} / {check_count}")
    print(f"  有噪声特征: {has_noise} / {check_count}")
    print(f"  无噪声特征: {no_noise} / {check_count}")
    
    if matches / check_count < 0.5:
        print(f"\n⚠️  警告: 匹配率低于 50%，可能存在对齐问题！")
        print(f"   建议: 检查 OCR JSON 和 dataset 的来源是否一致")
    else:
        print(f"\n✅ 对齐检查通过（匹配率 {matches/check_count*100:.1f}%）")
    
    # 额外检查：dataset 中有多少样本有非零噪声特征
    print(f"\n检查所有 train 样本的噪声特征覆盖率...")
    total_train = len(dataset["train"])
    samples_with_noise = 0
    
    # 采样检查（避免全部遍历）
    sample_indices = list(range(0, total_train, max(1, total_train // 1000)))
    for idx in sample_indices:
        sample = dataset["train"][idx]
        noise_features = sample.get("noise_features", [])
        if noise_features and any(any(v != 0.0 for v in feat) for feat in noise_features):
            samples_with_noise += 1
    
    estimated_coverage = (samples_with_noise / len(sample_indices)) * 100
    print(f"  估计噪声特征覆盖率: {estimated_coverage:.1f}% (基于 {len(sample_indices)} 个采样点)")

if __name__ == "__main__":
    main()

