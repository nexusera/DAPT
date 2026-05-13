import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# 默认频次阈值与长度阈值（主输出）
MIN_THRESHOLD = 10
MAX_LEN = 5

# 额外输出：仅频次阈值 5，不做长度过滤，避免覆盖主输出
ALT_THRESHOLD = 5
ALT_OUTPUT = REPO_ROOT / "biaozhu_keys_freq_min5.txt"

# 默认标注目录（本地运行）：DAPT/biaozhu_data 下的五个文件
DEFAULT_ANNOT_DIR = REPO_ROOT / "biaozhu_data"
DEFAULT_FILES = [
    "ruyuanjilu1119.json",
    "menzhenbingli1119.json",
    "shuhoubingli1119.json",
    "huojianbingli1119.json",
    "huizhenbingli1119.json",
]
DEFAULT_OUTPUT = REPO_ROOT / "biaozhu_keys_freq.txt"


def _extract_text_from_value(value: Any) -> str:
    """安全提取 value 中的文本字段，兼容 list / str。"""
    if not value:
        return ""
    if isinstance(value, dict):
        txt = value.get("text", "")
        if isinstance(txt, list):
            txt = "".join(txt) if txt else ""
        return txt or ""
    if isinstance(value, list):
        return "".join([str(x) for x in value])
    return str(value)


def extract_key_data_from_task(task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """兼容 LabelStudio 导出结构，提取标注为“键名”的文本。"""
    key_data: List[Dict[str, Any]] = []
    task_id = task_data.get("id", "unknown")

    # LabelStudio 导出通常是 task["annotations"] -> 每个 annotation 下的 result 列表
    annotations = task_data.get("label", [])
    if not annotations and "annotations" in task_data:
        annotations = task_data.get("annotations", [])

    for ann in annotations or []:
        if not isinstance(ann, dict):
            continue
        results = ann.get("result")
        # 有些场景 ann 本身就是 result 项
        if results is None:
            results = [ann]
        for res in results:
            if not isinstance(res, dict):
                continue
            # labels 可能在 res["value"]["labels"] 或 res["labels"]
            value = res.get("value", {})
            labels = value.get("labels") or res.get("labels") or res.get("label", [])
            if isinstance(labels, str):
                labels = [labels]
            if "键名" not in labels:
                continue
            text = _extract_text_from_value(value)
            if not text:
                text = res.get("text", "")
            if not text:
                continue
            key_data.append({"key": text, "id": task_id})
    return key_data


def process_annotation_file(file_path: Path) -> List[Dict[str, Any]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取失败 {file_path}: {e}")
        return []

    all_key_data = []
    if isinstance(data, list):
        for task in data:
            all_key_data.extend(extract_key_data_from_task(task))
    elif isinstance(data, dict):
        all_key_data = extract_key_data_from_task(data)
    else:
        print(f"无法识别的JSON结构 -> {file_path}")
    return all_key_data


def find_keys_above_threshold(
    key_data: List[Dict[str, Any]],
    threshold: int,
    max_len: int = None,
) -> Dict[str, Dict[str, Any]]:
    counts = Counter([it["key"] for it in key_data])
    selected = {}
    for k, c in counts.items():
        if c <= threshold:
            continue
        if max_len is not None and len(k) > max_len:
            continue
        selected[k] = {"count": c, "ids": []}
    if not selected:
        return {}
    for it in key_data:
        k = it["key"]
        if k in selected:
            selected[k]["ids"].append(it["id"])
    return selected


def main():
    # 使用默认目录和文件列表
    annot_dir = DEFAULT_ANNOT_DIR
    files = [annot_dir / name for name in DEFAULT_FILES]
    threshold = MIN_THRESHOLD
    out_path = DEFAULT_OUTPUT

    print("标注目录:", annot_dir)
    print("读取文件:")
    for fp in files:
        print(" -", fp)

    key_data = []
    for fp in files:
        key_data.extend(process_annotation_file(fp))

    if not key_data:
        print("未提取到任何键名标注。")
        return

    # 主输出：频次 > MIN_THRESHOLD 且长度限制
    result = find_keys_above_threshold(key_data, threshold, max_len=MAX_LEN)
    kept = len(result)
    total_keys = len(Counter([it["key"] for it in key_data]))
    filtered_out = total_keys - kept

    print(f"总唯一键名数: {total_keys}")
    print(f"保留键名数(频次>{threshold}, 长度<= {MAX_LEN}): {kept}")
    print(f"被筛掉的键名数: {filtered_out}")

    if not result:
        print(f"未找到符合条件的键名（频次 > {threshold} 且长度 <= {MAX_LEN}）。")
        out_path.write_text("", encoding="utf-8")
        print(f"已写出空文件: {out_path}")
        return

    lines = []
    for key in sorted(result.keys()):
        info = result[key]
        lines.append(f"键名: {key}")
        lines.append(f"  出现次数: {info['count']}")
        lines.append(f"  对应ID: {', '.join(map(str, info['ids']))}")
        lines.append("")

    out_text = "\n".join(lines)
    print(out_text)
    try:
        out_path.write_text(out_text, encoding="utf-8")
        print(f"已写入 -> {out_path}")
    except Exception as e:
        print(f"写入失败: {e}")

    # 额外输出：仅频次 > ALT_THRESHOLD（无长度限制），不覆盖主输出
    alt_result = find_keys_above_threshold(key_data, ALT_THRESHOLD, max_len=None)
    alt_lines = []
    for key in sorted(alt_result.keys()):
        info = alt_result[key]
        alt_lines.append(f"键名: {key}")
        alt_lines.append(f"  出现次数: {info['count']}")
        alt_lines.append(f"  对应ID: {', '.join(map(str, info['ids']))}")
        alt_lines.append("")
    alt_text = "\n".join(alt_lines)
    try:
        ALT_OUTPUT.write_text(alt_text, encoding="utf-8")
        print(f"已写入（仅频次>{ALT_THRESHOLD}，无长度过滤）-> {ALT_OUTPUT}")
        print(f"保留键名数(频次>{ALT_THRESHOLD}, 无长度限制): {len(alt_result)}")
    except Exception as e:
        print(f"写入 {ALT_OUTPUT} 失败: {e}")


if __name__ == "__main__":
    main()
