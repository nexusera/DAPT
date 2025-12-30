import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_from_disk, DatasetDict
from DAPT.noise_embeddings import NoiseFeatureExtractor


def load_medical_dict(path: Optional[str]) -> List[str]:
    if not path:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"medical_dict not found: {path}")
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                words.append(w)
    return words


def load_ocr_list(path: str) -> List[Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"OCR json not found: {path}")
    # 支持 json / jsonl
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # 若上层是 {"data": [...]} 或 {"ocr_list": [...]}
        for key in ["data", "ocr_list", "items"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        # 单个对象也包裹成列表，方便按 idx 对齐
        return [obj]
    raise ValueError("Unsupported OCR JSON format")


def build_zero_feats(seq_len: int):
    return [[0.0] * 5 for _ in range(seq_len)], [[False] * 5 for _ in range(seq_len)]


def main():
    parser = argparse.ArgumentParser(description="Add noise features to processed_dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to processed_dataset (load_from_disk)",
    )
    parser.add_argument(
        "--ocr_json",
        type=str,
        required=True,
        help="path to OCR json (list/json/jsonl) aligned by index with dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="output path to save new dataset with noise fields",
    )
    parser.add_argument(
        "--medical_dict",
        type=str,
        default=None,
        help="optional path to medical dict (one word per line)",
    )
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset)
    if not isinstance(dataset, DatasetDict):
        raise ValueError("Expected a DatasetDict with train/test splits")

    ocr_list = load_ocr_list(args.ocr_json)
    med_dict = load_medical_dict(args.medical_dict)
    extractor = NoiseFeatureExtractor(medical_dict=med_dict)

    def add_noise(example: Dict[str, Any], idx: int):
        word_ids = example.get("word_ids")
        if word_ids is None:
            nf, nm = build_zero_feats(0)
            example["noise_features"] = nf
            example["noise_masks"] = nm
            return example

        if idx < len(ocr_list):
            ocr_obj = ocr_list[idx]
            word_feats, word_masks = extractor.extract_word_features({"ocr": ocr_obj})
            nf, nm = extractor.broadcast_to_subwords(
                word_feats, word_masks, word_ids
            )
            example["noise_features"] = nf.tolist()
            example["noise_masks"] = nm.tolist()
        else:
            nf, nm = build_zero_feats(len(word_ids))
            example["noise_features"] = nf
            example["noise_masks"] = nm
        return example

    new_splits = {}
    for split in dataset:
        ds = dataset[split]
        new_splits[split] = ds.map(
            add_noise,
            with_indices=True,
            desc=f"add_noise_features_{split}",
        )

    out = DatasetDict(new_splits)
    out.save_to_disk(args.output)
    print(f"Saved dataset with noise features to {args.output}")


if __name__ == "__main__":
    main()

