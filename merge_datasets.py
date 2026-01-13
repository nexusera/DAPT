import argparse
from typing import Dict, List

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

# 非 OCR 语料的“完美”物理噪声占位
PERFECT_VALUES = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 OCR 数据集与非 OCR 数据集合并（保持各自特征，不按索引强行对齐）"
    )
    parser.add_argument(
        "--ocr_dataset",
        type=str,
        required=True,
        help="含噪声字段的 OCR 数据集路径（DatasetDict.save_to_disk 输出）",
    )
    parser.add_argument(
        "--non_ocr_dataset",
        type=str,
        required=True,
        help="非 OCR 数据集路径（可无 noise_values，训练时会填充完美噪声）",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="合并后的输出目录（DatasetDict.save_to_disk）",
    )
    parser.add_argument(
        "--ocr_repeat",
        type=int,
        default=1,
        help="仅对 train 分割：OCR 样本重复/重采样次数（>=1）",
    )
    parser.add_argument(
        "--non_ocr_repeat",
        type=int,
        default=1,
        help="仅对 train 分割：非 OCR 样本重复/重采样次数（>=1）",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="合并前对各分割执行 shuffle（相同 seed 保持可复现）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="shuffle/重采样随机种子",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="填充缺失列时的多进程数量（>=2 启用并行）",
    )
    return parser.parse_args()


def _fill_missing_columns(
    ds: Dataset, target_columns: List[str], desc: str, num_proc: int
) -> Dataset:
    missing = [c for c in target_columns if c not in ds.column_names]
    if not missing:
        return ds

    def filler(example: Dict):
        for col in missing:
            if col == "noise_values":
                # 非 OCR 样本默认填完美值；OCR 样本不会缺失该列
                example[col] = [PERFECT_VALUES for _ in range(len(example["input_ids"]))]
            else:
                example[col] = None
        return example

    map_kwargs = {"desc": desc}
    if num_proc and num_proc > 1:
        map_kwargs["num_proc"] = num_proc
    return ds.map(filler, **map_kwargs)


def merge_split(
    ocr_ds: Dataset,
    non_ocr_ds: Dataset,
    split: str,
    ocr_repeat: int,
    non_ocr_repeat: int,
    shuffle: bool,
    seed: int,
    num_proc: int,
) -> Dataset:
    if ocr_repeat < 1 or non_ocr_repeat < 1:
        raise ValueError("ocr_repeat 与 non_ocr_repeat 必须 >= 1")

    # 对齐列：确保 noise_values 仅填充到缺失侧，避免索引错配
    target_columns = list(set(ocr_ds.column_names) | set(non_ocr_ds.column_names))
    ocr_ds = _fill_missing_columns(ocr_ds, target_columns, f"fill_missing_{split}_ocr", num_proc)
    non_ocr_ds = _fill_missing_columns(non_ocr_ds, target_columns, f"fill_missing_{split}_nonocr", num_proc)

    if shuffle:
        ocr_ds = ocr_ds.shuffle(seed=seed)
        non_ocr_ds = non_ocr_ds.shuffle(seed=seed)

    # 仅对 train 应用重复重采样，其余分割保持 1:1
    ocr_r = ocr_repeat if split == "train" else 1
    non_ocr_r = non_ocr_repeat if split == "train" else 1

    merged = concatenate_datasets([ocr_ds] * ocr_r + [non_ocr_ds] * non_ocr_r)
    return merged


def main():
    args = parse_args()

    print(f"加载 OCR 数据集: {args.ocr_dataset}")
    ocr_dd = load_from_disk(args.ocr_dataset)
    print(f"加载非 OCR 数据集: {args.non_ocr_dataset}")
    non_dd = load_from_disk(args.non_ocr_dataset)

    if not isinstance(ocr_dd, DatasetDict) or not isinstance(non_dd, DatasetDict):
        raise ValueError("两路输入都必须是 DatasetDict（含 train/test 等分割）")

    # 确保分割一致，避免按索引硬对齐
    ocr_splits = set(ocr_dd.keys())
    non_splits = set(non_dd.keys())
    if ocr_splits != non_splits:
        raise ValueError(f"分割不一致: OCR={ocr_splits}, 非OCR={non_splits}")

    merged_splits = {}
    for split in sorted(ocr_splits):
        print(f"合并分割: {split}")
        merged_splits[split] = merge_split(
            ocr_dd[split],
            non_dd[split],
            split=split,
            ocr_repeat=args.ocr_repeat,
            non_ocr_repeat=args.non_ocr_repeat,
            shuffle=args.shuffle,
            seed=args.seed,
            num_proc=args.num_proc,
        )
        print(
            f"  完成: {split} -> OCR {len(ocr_dd[split])} + 非OCR {len(non_dd[split])} => 合并 {len(merged_splits[split])}"
        )

    out = DatasetDict(merged_splits)
    out.save_to_disk(args.output_path)
    print(f"✅ 已保存合并数据集到: {args.output_path}")


if __name__ == "__main__":
    main()

