# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import random


def _tighten_span(text: str, s: int, e: int) -> tuple[int, int]:
    """将 [s, e) 的子串做首尾空白收紧（右开区间）。"""
    if not text:
        return s, e
    n = len(text)
    s = max(0, min(s, n))
    e = max(0, min(e, n))
    if e <= s:
        return s, e
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def _load_jsonl_or_json(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"data_path not found: {path}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _save_jsonl(items: List[Dict[str, Any]], out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _preview(txt: str, n_left: int = 200) -> str:
    txt = (txt or "").replace("\n", "\\n")
    return txt if len(txt) <= n_left else (txt[:n_left] + " ...")


def _dedup_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


class BalancedKVSeparator:
    """在 '：' 与 ' ' 之间均衡选择分隔符。"""

    def __init__(self) -> None:
        self.usage = {"colon": 0, "space": 0}

    def choose(self) -> str:
        c, s = self.usage["colon"], self.usage["space"]
        if c <= s:
            self.usage["colon"] = c + 1
            return "："
        else:
            self.usage["space"] = s + 1
            return " "


# === 最简均衡切分：按 report_title 抽 10% 测试集 ===
def split_train_test_balanced_by_title(
    data_path: str,
    out_train_json: str,
    out_test_json: str,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Any]:
    """按 report_title 分组，抽取约 test_ratio 的测试集（每组至少 1 条），写出 train/test JSON。"""
    records = _load_jsonl_or_json(data_path)
    rng = random.Random(seed)

    # 分组
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        title = str(rec.get("report_title")).strip()
        rec.pop("report_composed",None)
        groups.setdefault(title, []).append(rec)

    train_set: List[Dict[str, Any]] = []
    test_set: List[Dict[str, Any]] = []
    per_title = {}

    for title, items in groups.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        # 每组至少 1 条测试样本
        k = max(1, int(round(n * test_ratio)))
        if n >= 2:
            k = min(k, n - 1)
        # 拆分
        test_slice = items[:k]
        train_slice = items[k:]
        test_set.extend(test_slice)
        train_set.extend(train_slice)
        per_title[title or ""] = {"n": n, "k": k}

    # 写出 JSON（列表）
    for out_path, data in ((out_train_json, train_set), (out_test_json, test_set)):
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "n_total": len(records),
        "n_train": len(train_set),
        "n_test": len(test_set),
        "test_ratio_req": test_ratio,
        "per_title": per_title,
        "out_train": str(out_train_json),
        "out_test": str(out_test_json),
    }
    return summary


# === Label Studio 项目 -> clean_ocr 结构 转换 ===
def convert_labelstudio_project_to_clean_records(
    in_path: str,
    out_path: Optional[str] = None,
    max_report_tokens: int = 512,
    tokenizer_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """将 Label Studio 导出的项目 JSON 转为 clean_ocr 数据结构的列表。

    输入为 Label Studio 项目导出的 JSON（列表，每个元素为一个任务）。转换输出的每条记录包含：
    - report_title: 优先从 data["category"|"report_title"|"title"|"reportType"] 提取
    - report:       优先从 data["ocr_text"|"text"|"report"|"ocr"] 提取
    - 其它字段:     从标注结果 result/annotations/predictions 中的 span.label -> span.text 映射

    对于同一 label 多个值，默认保留"文本最长"的一个；无法找到文本时忽略该项。
    
    当 report 超过 max_report_tokens 时，会在合适的字段值后插入 \n\n 分段符。

    Args:
        in_path:  Label Studio 项目导出的 JSON 文件路径
        out_path: 可选。若提供则将转换后的列表以 JSON（list）格式写出
        max_report_tokens: report 的最大 token 数阈值，超过则智能分段
        tokenizer_name: 可选的 tokenizer 路径，用于准确计算 token 数

    Returns:
        List[Dict[str, Any]]: clean_ocr 风格的记录列表
    """
    p = Path(in_path)
    if not p.exists():
        raise FileNotFoundError(f"Label Studio project json not found: {in_path}")

    raw_text = p.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []
    try:
        tasks = json.loads(raw_text)
    except Exception as e:
        raise ValueError(f"Invalid Label Studio JSON: {e}")

    if not isinstance(tasks, list):
        raise ValueError("Expect a list of tasks in Label Studio export JSON")

    # 初始化 tokenizer（用于准确计算 token 数）
    tokenizer = None
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            pass
    
    def _count_tokens(text: str) -> int:
        """计算文本的 token 数"""
        if tokenizer:
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        # fallback: 按字符数估算（中文约 1.5 字符/token）
        return len(text) // 2 + len(text) % 2
    
    def _smart_segment_report(
        report_text: str, 
        fields: Dict[str, str], 
        max_tokens: int
    ) -> str:
        """当 report 超长时，在字段值后智能插入 \\n\\n 分段符
        
        策略：
        1. 先找到所有字段值在 report 中的位置（按出现顺序）
        2. 累积 token，当超过阈值时在上一个字段值后插入 \\n\\n
        3. 保持字段值的完整性，不在值中间切分
        """
        total_tokens = _count_tokens(report_text)
        if total_tokens <= max_tokens:
            return report_text  # 不超长，原样返回
        
        # 查找所有字段值在 report 中的位置（按出现位置排序）
        value_positions: List[Tuple[int, int, str]] = []  # (start, end, value)
        for key, value in fields.items():
            value_s = str(value).strip()
            if not value_s or len(value_s) < 3:  # 跳过过短的值
                continue
            # 查找所有出现位置
            start = 0
            while True:
                pos = report_text.find(value_s, start)
                if pos == -1:
                    break
                value_positions.append((pos, pos + len(value_s), value_s))
                start = pos + 1
        
        if not value_positions:
            return report_text  # 没找到任何字段值，返回原文
        
        # 按位置排序并去重
        value_positions = sorted(set(value_positions), key=lambda x: x[0])
        
        # 智能分段：每隔 max_tokens 在字段值后插入 \\n\\n
        segments = []
        last_end = 0
        current_tokens = 0
        
        for i, (start, end, value) in enumerate(value_positions):
            # 当前片段：从上次结束到当前值结束
            chunk = report_text[last_end:end]
            chunk_tokens = _count_tokens(chunk)
            
            # 如果加上这段会超长，且已有内容，则在上个位置分段
            if current_tokens + chunk_tokens > max_tokens and segments:
                # 插入分段符
                segments.append("\n\n")
                current_tokens = 0
            
            # 添加当前片段
            segments.append(chunk)
            current_tokens += chunk_tokens
            last_end = end
        
        # 添加剩余部分
        if last_end < len(report_text):
            segments.append(report_text[last_end:])
        
        return "".join(segments)

    def _pick_annotation_result(task: Dict[str, Any]) -> List[Dict[str, Any]]:
        # 1) annotations 中最近一次未取消的 result
        anns = task.get("annotations")
        if isinstance(anns, list) and anns:
            # 先按 was_cancelled 过滤，再按 updated_at/created_at 排序
            valid = [a for a in anns if not a.get("was_cancelled")]
            pool = valid if valid else anns
            # 取最后一个（通常是最新）
            pool_sorted = sorted(
                pool,
                key=lambda x: (
                    str(x.get("updated_at") or ""),
                    str(x.get("created_at") or ""),
                ),
            )
            if pool_sorted:
                r = pool_sorted[-1].get("result")
                if isinstance(r, list):
                    return r
        # 2) 顶层 result（某些导出格式会直接给）
        r = task.get("result")
        if isinstance(r, list):
            return r
        # 3) predictions 中的 result（若有）
        preds = task.get("predictions")
        if isinstance(preds, list) and preds:
            # 取最后一个 prediction
            pr = preds[-1].get("result")
            if isinstance(pr, list):
                return pr
        return []

    out: List[Dict[str, Any]] = []

    for task in tasks:
        data = task.get("data") or {}
        if not isinstance(data, dict):
            data = {}

        # 标题与原文
        report_title = (
            str(
                data.get("category")
                or data.get("report_title")
                or data.get("title")
                or data.get("reportType")
                or ""
            )
        ).strip()

        report_text = (
            str(
                data.get("ocr_text")
                or data.get("text")
                or data.get("report")
                or data.get("ocr")
                or ""
            )
        )

        fields: Dict[str, str] = {}

        # 从标注结果提取 label -> text
        results = _pick_annotation_result(task)
        for r in results:
            if not isinstance(r, dict):
                continue
            val = r.get("value") or {}
            if not isinstance(val, dict):
                continue
            # span 文本
            txt = val.get("text")
            if not isinstance(txt, str):
                continue
            txt_s = txt.strip()
            if not txt_s:
                continue
            # label 名（取首个）
            labels = val.get("labels")
            label_name = None
            if isinstance(labels, list) and labels:
                label_name = str(labels[0]).strip()
            elif isinstance(labels, str) and labels.strip():
                label_name = labels.strip()
            if not label_name:
                continue
            # 多值冲突：保留更长的一个
            old = fields.get(label_name, "")
            if len(txt_s) >= len(old):
                fields[label_name] = txt_s

        # 跳过空标注样本
        if not fields:
            continue

        if not report_text and fields:
            # 无原文时，尝试从 r["text"] 合并（极端 fallback）
            report_text = " ".join(sorted(set(fields.values()), key=len, reverse=True))

        # 智能分段：当 report 超过 max_report_tokens 时，在字段值后插入 \n\n
        if report_text and max_report_tokens > 0:
            report_text = _smart_segment_report(report_text, fields, max_report_tokens)

        rec = {"report_title": report_title, "report": report_text}
        # 合并字段
        for k, v in fields.items():
            rec[k] = v
        # 可选：记录本条新增的键名（仅用于观测，训练不会依赖）
        if fields:
            rec["added_keys"] = list(fields.keys())

        out.append(rec)

    if out_path:
        Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    return out
