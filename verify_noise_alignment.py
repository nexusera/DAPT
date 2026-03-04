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
import re
import unicodedata
from datasets import load_from_disk
from transformers import AutoTokenizer


# 与 add_noise_features.py / merge_datasets.py 保持一致：非 OCR 样本默认填充的“完美噪声”
PERFECT_VALUES = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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


_KEEP_CHARS_RE = re.compile(r"[0-9a-z\u4e00-\u9fff]+", re.IGNORECASE)


def normalize_for_match(text: str, *, max_len: int = 200) -> str:
    """用于对齐校验的轻量规范化。

    目的不是还原原文，而是消除 tokenization/decode 与 OCR 文本在:
    - 大小写（THEFIRST vs thefirst）
    - 全角/半角
    - 空格/标点/换行
    上的差异，减少误报。
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # 仅保留: 数字/英文字母/中文（其余符号统一丢弃）
    chunks = _KEEP_CHARS_RE.findall(text)
    if not chunks:
        return ""
    normalized = "".join(chunks)
    return normalized[:max_len]


def is_text_match(dataset_text: str, ocr_text: str) -> bool:
    """启发式匹配：足够鲁棒且便于解释。

    - 优先做包含关系（前缀/子串），解决很多“同一条目但 OCR 更长/更短”的情况。
    - 否则用字符集合交集做一个粗判。
    """
    a = normalize_for_match(dataset_text, max_len=200)
    b = normalize_for_match(ocr_text, max_len=200)
    if not a and not b:
        return True
    if not a or not b:
        return False

    # 包含关系：常见于 dataset 被截断、OCR 多了页眉页脚等
    if len(a) >= 12 and a in b:
        return True
    if len(b) >= 12 and b in a:
        return True

    # 粗略字符重叠：避免重计算，保持脚本轻量
    return len(set(a) & set(b)) > 8


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


def is_perfect_noise_sample(sample: dict) -> bool:
    """判断样本是否为“完美噪声”（通常意味着 non-OCR 路或缺失时的占位）。

    关键技巧：对 merged dataset，non-OCR 的 noise_values 会被 merge_datasets.py 填成 PERFECT_VALUES。
    而 OCR 样本在 add_noise_features.py 中，通常 [CLS]/[SEP] 位置会保持 0 向量，
    因此 noise_values[0] 一般 != PERFECT_VALUES。
    """
    nv = sample.get("noise_values") or sample.get("noise_features") or []
    if not nv:
        return False
    first = nv[0]
    return isinstance(first, list) and len(first) >= 7 and list(first[:7]) == PERFECT_VALUES


def compute_ocr_prefix_len(split_data, *, probe_after_boundary: int = 2000) -> int:
    """在 split 中估计 OCR 前缀块长度。

    约定：未 shuffle 的 merge_datasets.py 会把 OCR split 放在前面，non-OCR 放在后面。
    因此 split 的开头是一段 OCR 样本（non-perfect），随后是 non-OCR（perfect）。

    若检测到 boundary 之后仍然出现 OCR 样本，说明 merge 时 shuffle 了或数据顺序被破坏，
    此时无法在 merged dataset 上做 index-to-ocr_json 对齐校验。
    """
    n = len(split_data)
    if n == 0:
        return 0

    # OCR 样本通常非 perfect；non-OCR 样本通常 perfect
    def is_ocr(i: int) -> bool:
        return not is_perfect_noise_sample(split_data[i])

    # 如果第一条就是 perfect，说明 OCR 前缀长度为 0（可能 shuffle 了，或根本没有 OCR）
    if not is_ocr(0):
        return 0

    # 找到第一个 non-OCR（perfect）的位置作为 boundary
    boundary = n
    for i in range(n):
        if not is_ocr(i):
            boundary = i
            break

    # boundary 后抽查一段：如果还有 OCR，说明被 shuffle，不可验证
    end = min(n, boundary + probe_after_boundary)
    for j in range(boundary, end):
        if is_ocr(j):
            # 返回 -1 作为“不可判定/被 shuffle”信号
            return -1
    return boundary

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

    # NOTE:
    # 对 OCR-only dataset：split 的所有样本都应对应 OCR JSON 的连续片段，可用 split 内 idx + ocr_offset。
    # 对 merged dataset：split 中 OCR 只是一个“前缀块”（若 merge 未 shuffle），其后是 non-OCR。
    # 因此 ocr_offset 不能用 len(split)（也不能用 len(merged train)），而应该用“前一 split 的 OCR 前缀长度”。

    # 先计算每个 split 的 OCR 前缀长度（-1 表示检测到 shuffle，无法校验）
    ocr_prefix = {}
    for s in ["train", "test"]:
        if s in dataset:
            ocr_prefix[s] = compute_ocr_prefix_len(dataset[s])

    # 计算 OCR JSON offset：train 从 0 开始，test 的 offset 等于 train 的 OCR 前缀长度
    ocr_offsets = {}
    ocr_offsets["train"] = 0
    if "test" in dataset:
        train_prefix = ocr_prefix.get("train", 0)
        # 若 train_prefix == -1，说明 merged 的 train 被 shuffle，无法对齐
        ocr_offsets["test"] = train_prefix if isinstance(train_prefix, int) and train_prefix > 0 else 0

    for split_name in splits_to_check:
        split_data = dataset[split_name]

        prefix_len = ocr_prefix.get(split_name, len(split_data))
        if prefix_len == -1:
            print(
                f"\n⚠️  split={split_name} 检测到 OCR/non-OCR 混洗（OCR 不再是前缀块），无法在 merged dataset 上做 index 对齐校验。"
            )
            print("   建议：改为在 OCR-only 数据集（processed_dataset_ocr..._with_noise）上运行 verify。")
            continue

        # 只在 OCR 前缀块上做对齐检查；若这是 OCR-only 数据集，则 prefix_len == len(split)
        offset = ocr_offsets.get(split_name, 0)
        check_count = min(args.check_samples, prefix_len, max(0, len(ocr_list) - offset))

        print(f"\n开始检查 split={split_name} 的前 {check_count} 条 OCR 样本（ocr_offset={offset}，ocr_prefix_len={prefix_len}）...")
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
        
            is_match = is_text_match(dataset_text, ocr_text)
            if is_match:
                matches += 1
            else:
                mismatches += 1

            # 展示用 preview：仍然截断，但用 normalize 结果更便于目检
            dataset_preview = normalize_for_match(dataset_text, max_len=60)
            ocr_preview = normalize_for_match(ocr_text, max_len=60)
        
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

