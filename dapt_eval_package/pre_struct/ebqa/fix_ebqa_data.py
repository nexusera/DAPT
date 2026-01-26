import json
import os
from typing import Any, Dict, List, Optional

try:
    from pre_struct.kv_ner.noise_utils import PERFECT_VALUES
except Exception:
    PERFECT_VALUES = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _expand_word_noise_to_chars(ocr_raw: Dict[str, Any], noise_values_per_word: Optional[List[List[float]]]):
    if not (isinstance(ocr_raw, dict) and isinstance(noise_values_per_word, list)):
        return None
    words_result = ocr_raw.get("words_result") or []
    if not isinstance(words_result, list):
        return None
    char_noise = []
    for wr, nv in zip(words_result, noise_values_per_word):
        if not isinstance(nv, (list, tuple)) or len(nv) != 7:
            continue
        word = (wr or {}).get("words", "") if isinstance(wr, dict) else ""
        repeat = max(1, len(word))
        char_noise.extend([list(nv)] * repeat)
    return char_noise if char_noise else None


def _broadcast_global_noise(noise_values: Any, text_len: int):
    if (
        isinstance(noise_values, list)
        and len(noise_values) == 7
        and all(not isinstance(v, (list, tuple)) for v in noise_values)
    ):
        return [list(noise_values) for _ in range(max(0, text_len))]
    return noise_values


def _normalize_noise(item: Dict[str, Any], report_text: str) -> List[List[float]]:
    data_block = item.get("data", {}) if isinstance(item, dict) else {}
    ocr_raw = data_block.get("ocr_raw") or item.get("ocr_raw")
    per_word_noise = data_block.get("noise_values_per_word") or item.get("noise_values_per_word")
    noise_values = _expand_word_noise_to_chars(ocr_raw, per_word_noise)
    if noise_values is None:
        noise_values = data_block.get("noise_values") or item.get("noise_values")
    noise_values = _broadcast_global_noise(noise_values, len(report_text))

    if not noise_values:
        return [list(PERFECT_VALUES) for _ in range(len(report_text))]

    normed: List[List[float]] = []
    for i in range(len(report_text)):
        if (
            isinstance(noise_values, list)
            and i < len(noise_values)
            and isinstance(noise_values[i], (list, tuple))
            and len(noise_values[i]) == 7
        ):
            normed.append(list(noise_values[i]))
        else:
            normed.append(list(PERFECT_VALUES))
    return normed


def fix_data(input_path: str, output_data_path: str, output_schema_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    os.makedirs(os.path.dirname(output_data_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_schema_path) or ".", exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_keys = set()
    flattened_count = 0

    with open(output_data_path, "w", encoding="utf-8") as fout:
        for item in data:
            report_text = item.get("data", {}).get("text", "")
            if not report_text:
                continue

            report_title = "通用病历"
            spans: Dict[str, str] = {}

            entities: Dict[str, Dict[str, Any]] = {}
            relations: List[Dict[str, Any]] = []
            for ann in item.get("annotations", []):
                for res in ann.get("result", []):
                    if res.get("type") == "labels":
                        entities[res["id"]] = res
                    elif res.get("type") == "relation":
                        relations.append(res)

            for rel in relations:
                key_node = entities.get(rel.get("from_id"))
                val_node = entities.get(rel.get("to_id"))
                if key_node and val_node:
                    key_text = key_node.get("value", {}).get("text", "").strip()
                    val_text = val_node.get("value", {}).get("text", "").strip()
                    if key_text:
                        spans[key_text] = val_text
                        all_keys.add(key_text)

            for node in entities.values():
                labels = node.get("value", {}).get("labels", [])
                if "医院名称" in labels:
                    val_text = node.get("value", {}).get("text", "").strip()
                    spans["医院名称"] = val_text
                    all_keys.add("医院名称")

            record = {
                "report": report_text,
                "report_title": report_title,
                "spans": spans,
                "noise_values": _normalize_noise(item, report_text),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            flattened_count += 1

    print(f"Flattened {flattened_count} records to {output_data_path}")

    schema = {"通用病历": {k: {"别名": [], "类型": "str", "Q": ""} for k in all_keys}}
    with open(output_schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f"Schema saved to {output_schema_path}")
    print(f"Total keys found: {len(all_keys)}")
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="扁平化 Label Studio KV 数据并携带噪声特征")
    parser.add_argument("--input", dest="input_path", required=True, help="Label Studio 导出的 JSON/JSONL 文件")
    parser.add_argument("--output_data", dest="output_data_path", default="data/kv_ner_prepared_comparison/train_flattened.jsonl", help="输出扁平化 jsonl")
    parser.add_argument("--output_schema", dest="output_schema_path", default="data/kv_ner_prepared_comparison/keys_v2.json", help="输出 schema json")
    args = parser.parse_args()

    fix_data(args.input_path, args.output_data_path, args.output_schema_path)
