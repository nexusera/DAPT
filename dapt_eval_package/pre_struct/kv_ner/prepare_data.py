#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

if __package__ in (None, ""):
    _PACKAGE_ROOT = Path(__file__).resolve().parents[2]
    import sys

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))
    from pre_struct.kv_ner import config_io
else:
    from . import config_io

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

DEFAULT_LABEL_MAP: Dict[str, str] = {
    "键名": "KEY",
    "键": "KEY",
    "KEY": "KEY",
    "值": "VALUE",
    "Value": "VALUE",
    "VALUE": "VALUE",
    "医院名称": "HOSPITAL",
    "医院": "HOSPITAL",
    "HOSPITAL": "HOSPITAL",
}


def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    items: List[Dict[str, Any]] = []
    for idx, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            items.append(obj)
        except json.JSONDecodeError:
            logger.warning("跳过无法解析的行 %d: %s", idx, path.name)
    return items


def _latest_results(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    annotations = task.get("annotations") or []
    valid = [a for a in annotations if not a.get("was_cancelled")]
    pool = valid if valid else annotations
    if not pool:
        return []

    def _key(anno: Dict[str, Any]) -> Tuple[str, str]:
        return (str(anno.get("updated_at") or ""), str(anno.get("created_at") or ""))

    latest = sorted(pool, key=_key)
    results = latest[-1].get("result") if latest else None
    if isinstance(results, list):
        return results
    return []


def has_valid_annotations(task: Dict[str, Any], label_map: Dict[str, str]) -> bool:
    results = _latest_results(task)
    for res in results:
        if res.get("type") != "labels":
            continue
        labels = res.get("value", {}).get("labels") or []
        for raw in labels:
            if label_map.get(str(raw).strip()):
                return True
    return False


def extract_key_value_pairs(task: Dict[str, Any], label_map: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    text = str(task.get("data", {}).get("ocr_text") or task.get("data", {}).get("text") or "")
    if not text:
        return {}

    results = _latest_results(task)
    if not results:
        return {}

    entities: Dict[str, Dict[str, Any]] = {}
    relations: Dict[str, str] = {}

    for res in results:
        r_type = res.get("type")
        if r_type == "labels":
            value = res.get("value") or {}
            labels = value.get("labels") or []
            if not labels:
                continue
            normalized = label_map.get(str(labels[0]).strip())
            if not normalized:
                continue
            start = value.get("start")
            end = value.get("end")
            if start is None or end is None:
                continue
            start = int(start)
            end = int(end)
            if end <= start or start < 0:
                continue
            entities[str(res.get("id"))] = {
                "label": normalized,
                "start": start,
                "end": end,
                "text": value.get("text") or text[start:end],
            }
        elif r_type == "relation":
            from_id = res.get("from_id")
            to_id = res.get("to_id")
            if isinstance(from_id, str) and isinstance(to_id, str):
                relations[from_id] = to_id

    all_kvs: List[Tuple[str, int, int, str, bool, int]] = []

    for ent_id, ent in entities.items():
        if ent.get("label") != "KEY":
            continue
        key_text = (ent.get("text") or text[ent["start"]:ent["end"]]).strip()
        if not key_text:
            continue
        value_ent = None
        if ent_id in relations:
            cand = entities.get(relations[ent_id])
            if cand and cand.get("label") == "VALUE":
                value_ent = cand

        if value_ent:
            value_text = (value_ent.get("text") or text[value_ent["start"]:value_ent["end"]]).strip()
            all_kvs.append(
                (
                    key_text,
                    value_ent["start"],
                    value_ent["end"],
                    value_text,
                    False,
                    ent["start"],
                )
            )
        else:
            all_kvs.append(
                (
                    key_text,
                    ent["end"],
                    ent["end"],
                    "",
                    False,
                    ent["start"],
                )
            )

    for ent in entities.values():
        if ent.get("label") == "HOSPITAL":
            hospital_text = (ent.get("text") or text[ent["start"]:ent["end"]]).strip()
            if hospital_text:
                all_kvs.append(
                    (
                        "医院名称",
                        ent["start"],
                        ent["end"],
                        hospital_text,
                        True,
                        ent["start"],
                    )
                )

    all_kvs.sort(key=lambda x: (0 if x[4] else 1, x[5]))

    spans: Dict[str, Dict[str, Any]] = {}
    for key_text, start, end, value_text, _, _ in all_kvs:
        spans[key_text] = {"start": start, "end": end, "text": value_text}
    return spans


def split_data(tasks: List[Dict[str, Any]], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(tasks)
    rng.shuffle(shuffled)

    total = len(shuffled)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    train_tasks = shuffled[:n_train]
    val_tasks = shuffled[n_train : n_train + n_val]
    test_tasks = shuffled[n_train + n_val :]
    return train_tasks, val_tasks, test_tasks


def save_labelstudio_format(tasks: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("保存 Label Studio 格式: %s (%d 条)", output_path, len(tasks))


def save_evaluation_format(tasks: List[Dict[str, Any]], output_path: Path, label_map: Dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for task in tqdm(tasks, desc=f"导出 {output_path.name}"):
            text = str(task.get("data", {}).get("ocr_text") or task.get("data", {}).get("text") or "")
            title = str(task.get("data", {}).get("category") or task.get("data", {}).get("report_title") or "")
            if not text.strip():
                continue
            spans = extract_key_value_pairs(task, label_map)
            if not spans:
                continue
            f.write(
                json.dumps(
                    {
                        "report_index": str(task.get("id", "")),
                        "report_title": title,
                        "report": text,
                        "spans": spans,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    logger.info("保存评估格式: %s (%d 条)", output_path, count)


def prepare_data(
    input_paths: List[str],
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    config_path: Optional[str] = None,
) -> None:
    label_map = dict(DEFAULT_LABEL_MAP)
    if config_path:
        try:
            cfg = config_io.load_config(config_path)
            # 合并：配置优先，但保留默认里的“键名/值/医院名称”等常见标签
            cfg_map = config_io.label_map_from(cfg)
            label_map.update(cfg_map)
        except Exception as e:
            logger.warning("读取配置失败，将使用默认标签映射: %s", e)

    paths = [Path(p) for p in input_paths]
    all_tasks: List[Dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            logger.warning("文件不存在，跳过: %s", p)
            continue
        loaded = _read_json_or_jsonl(p)
        logger.info("读取 %s: %d 条", p.name, len(loaded))
        all_tasks.extend(loaded)

    valid_tasks = [t for t in all_tasks if has_valid_annotations(t, label_map)]
    logger.info("有效任务: %d / %d", len(valid_tasks), len(all_tasks))

    train_tasks, val_tasks, test_tasks = split_data(valid_tasks, train_ratio, val_ratio, seed)
    logger.info("划分: train=%d, val=%d, test=%d", len(train_tasks), len(val_tasks), len(test_tasks))

    out_dir = Path(output_dir)
    save_labelstudio_format(train_tasks, out_dir / "train.json")
    save_labelstudio_format(val_tasks, out_dir / "dev.json")
    save_labelstudio_format(test_tasks, out_dir / "test.json")

    save_evaluation_format(train_tasks, out_dir / "train_eval.jsonl", label_map)
    save_evaluation_format(val_tasks, out_dir / "dev_eval.jsonl", label_map)
    save_evaluation_format(test_tasks, out_dir / "test_eval.jsonl", label_map)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 KV-NER 训练与评估数据")
    parser.add_argument("--input", nargs="+", required=True, help="输入 Label Studio 导出的 JSON 或 JSONL")
    parser.add_argument("--output_dir", type=str, default="prepared_data", help="输出目录")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--config", type=str, default=None, help="可选配置文件，读取 label_map")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_data(
        input_paths=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        config_path=args.config,
    )
