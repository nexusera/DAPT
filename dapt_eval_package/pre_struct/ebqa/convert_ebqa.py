import os
import sys
import json
from typing import Any, Dict, List, Optional

# Ensure project root is in path
sys.path.append(os.getcwd())

try:
    from pre_struct.kv_ner.noise_utils import NoiseFeatureProcessor, PERFECT_VALUES
except Exception:
    from pre_struct.kv_ner.noise_utils import NoiseFeatureProcessor, PERFECT_VALUES  # type: ignore

from pre_struct.ebqa.da_core.dataset import EnhancedQADataset
import argparse


def _get_record_noise(records: List[Dict[str, Any]], ridx: Optional[int]) -> List[List[float]]:
    if ridx is None or not isinstance(ridx, int):
        return []
    if ridx < 0 or ridx >= len(records):
        return []
    nv = records[ridx].get("noise_values")
    return nv if isinstance(nv, list) else []


def _build_noise_ids(sample: Dict[str, Any], record_noise: List[List[float]], processor: NoiseFeatureProcessor) -> Optional[List[List[int]]]:
    offsets = sample.get("offset_mapping") or []
    seq_ids = sample.get("sequence_ids") or []
    chunk_start = int(sample.get("chunk_char_start", 0) or 0)
    if not offsets or not seq_ids:
        return None

    text_len = len(record_noise)
    noise_ids: List[List[int]] = []

    for idx, offset in enumerate(offsets):
        sid = seq_ids[idx] if idx < len(seq_ids) else None
        if sid != 1:
            noise_ids.append(processor.values_to_bin_ids(PERFECT_VALUES))
            continue

        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            noise_ids.append(processor.values_to_bin_ids(PERFECT_VALUES))
            continue

        s, e = offset
        if s is None or e is None or int(e) <= int(s):
            noise_ids.append(processor.values_to_bin_ids(PERFECT_VALUES))
            continue

        abs_s = chunk_start + int(s)
        abs_e = chunk_start + int(e)
        vecs = []
        for ci in range(abs_s, min(abs_e, text_len)):
            v = record_noise[ci]
            if isinstance(v, (list, tuple)) and len(v) == 7:
                vecs.append(v)

        if vecs:
            avg = [sum(col) / len(col) for col in zip(*vecs)]
            noise_ids.append(processor.values_to_bin_ids(avg))
        else:
            noise_ids.append(processor.values_to_bin_ids(PERFECT_VALUES))

    return noise_ids


def run_conversion():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--input_file", type=str, default="data/kv_ner_prepared_comparison/train_flattened.jsonl")
    parser.add_argument("--struct_path", type=str, default="data/kv_ner_prepared_comparison/keys_merged_1027_cleaned.json")
    parser.add_argument("--noise_bins", type=str, required=True)
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file or "data/kv_ner_prepared_comparison/ebqa_train.jsonl"
    struct_path = args.struct_path
    tokenizer_name = args.tokenizer_name or os.environ.get("HF_TOKENIZER_NAME") or "/mnt/windows/Users/Admin/LLM/models/bert_multi_language/"

    print(f"Converting {input_file} -> {output_file} using schema {struct_path}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Noise bins: {args.noise_bins}")

    ds = EnhancedQADataset(
        data_path=input_file,
        report_struct_path=struct_path,
        only_title_keys=True,
        tokenizer_name=tokenizer_name,
        max_seq_len=512,
        autobuild=True,
        show_progress=True,
        keep_debug_fields=True,
    )

    processor = NoiseFeatureProcessor.load(args.noise_bins)

    samples = ds.samples if hasattr(ds, "samples") else None
    if samples is None:
        raise RuntimeError("Dataset did not populate samples!")

    records = ds.records if hasattr(ds, "records") else []

    print(f"Saving {len(samples)} samples to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for s in samples:
            rec_noise = _get_record_noise(records, s.get("report_index"))
            noise_ids = _build_noise_ids(s, rec_noise, processor)

            out = dict(s)
            if noise_ids is not None:
                out["noise_ids"] = noise_ids
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[OK] Saved {len(samples)} training samples to {output_file}")


if __name__ == "__main__":
    run_conversion()
