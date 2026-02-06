from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

# 可选进度条
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

API_KEY = "haixin_csco1435tG8y98hTa717"
BASE_URL = "https://qwen3.yoo.la/v1"

logger = logging.getLogger("extract_keys")


def call_model_once(
    prompt: str,
    model: str = "qwen3-32b",
    base_url: str = BASE_URL,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> Any:
    api_key = api_key or API_KEY
    logger.debug(f"LLM调用 model={model} timeout={timeout}")
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        timeout=timeout,
    )
    data = response.choices[0].message.content
    if isinstance(data, dict):
        if "llm_ret" in data:
            return data["llm_ret"]
        if isinstance(data.get("data"), dict) and "llm_ret" in data["data"]:
            return data["data"]["llm_ret"]
    return data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _strip_json_fence(x: str) -> str:
    return x.replace("```json", "").replace("```", "").strip()


def _query_alias_by_model(
    unknown_key: str,
    candidates: List[str],
    model_checker: Callable[[str], Any],
) -> str:
    cand_str = ", ".join(candidates)
    prompt = (
        "你是信息抽取助手。现在有一个字段名候选集合，以及一个待判断字段，"
        "请判断该字段是否是候选集合中某个字段的同义词或别名，并严格以 JSON 返回。\n"
        f"候选字段集合: [{cand_str}]\n"
        f"待判断字段: {unknown_key}\n"
        '输出格式严格为: {"alias_of": "<候选字段或 NA>"}。\n'
        "判断规则如下：\n"
        "- 若该字段与某候选字段语义一致或为其别名，则返回该字段名；\n"
        "- 若无明显同义或别名关系，alias_of 必须为 NA；\n"
        "- 仅输出 JSON，不得包含任何额外说明。"
    )
    try:
        resp = model_checker(prompt)  # 调模型
    except Exception as e:
        print(f"模型调用失败{e}")
        return "NA"

    # 解析 JSON
    try:
        obj = json.loads(resp) if isinstance(resp, str) else resp
        if isinstance(obj, dict):
            alias_of = obj.get("alias_of")
            if isinstance(alias_of, str) and alias_of.strip():
                alias_of = alias_of.strip()
                if alias_of.upper() == "NA":
                    return "NA"
                # 直接命中或无空白命中
                if alias_of in candidates:
                    return alias_of
                alias_of2 = alias_of.replace(" ", "")
                for c in candidates:
                    if alias_of2 == c.replace(" ", ""):
                        return c
            return "NA"
    except Exception:
        pass

    # 兜底：文本包含
    if isinstance(resp, str):
        text = resp.strip()
        for c in candidates:
            if c and c in text:
                return c
    return "NA"


def _atomic_write_json(obj: Any, out_path: str) -> None:
    """原子写文件，避免部分写入造成损坏。"""
    import os, tempfile

    d = json.dumps(obj, ensure_ascii=False, indent=2)
    dirn = os.path.dirname(out_path) or "."
    os.makedirs(dirn, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=dirn, delete=False
    ) as tf:
        tmp = tf.name
        tf.write(d)
        tf.flush()
    os.replace(tmp, out_path)


# -------- 规范化工具（关键：医院名规整） --------
def _norm_str(x: Any) -> str:
    s = str(x or "").strip()
    # 消除常见全角/空白
    s = s.replace("\u3000", " ")
    s = " ".join(s.split())
    return s


def _norm_hospital(h: Any) -> str:
    s = _norm_str(h).upper()
    if s in {"", "NA", "N/A", "NULL", "NONE", "—", "-", "无", "未知", "不详"}:
        return "NA"
    return _norm_str(h)  # 保留原大小写/格式（非NA）


def build_key_alias_map(
    report_info_jsonl: str = "report_info.jsonl",
    mapping_json: str = "base_report_keys_mapping.json",
    *,
    out_path: Optional[str] = None,  # 输出路径（可选）
    model_checker: Optional[Callable[[str], Any]] = None,  # 模型函数（可注入）
    save_every: int = 100,  # 每多少行落盘一次
) -> Dict[str, Dict[str, List[Tuple[str, str, str]]]]:
    """
    输出结构:
      {category: {canonical: [(alias_key, hospital, path), ...]}}
    规则：
      - 以 mapping_json 为初始 canonical 列表；
      - 遍历 jsonl：若 key 命中已有 canonical => 跳过；否则询问 LLM 是否为别名；
        * 若为别名：记录 (alias, hospital, path)
        * 若 NA：将该 key 动态晋升为新的 canonical
      - 每处理 1 行，立即用最新的 {category: keys} 更新内存 category_map；
      - 每处理 save_every 行，写一次 out_path；
      - 结束后再写一次 out_path。
      - 证据去重维度：(category, canonical, alias, hospital_norm) —— 不含 path，
        避免同院同别名多次插入（尤其 hospital==NA 时）。
    """
    # 选模型
    if model_checker is None:
        try:
            model_checker = call_model_once  # type: ignore[name-defined]
        except NameError as e:
            raise RuntimeError(
                "未提供 model_checker，且未找到 call_model_once。请传入 model_checker 或实现 call_model_once。"
            ) from e

    # 1) 读初始映射（category -> canonical 列表）
    with open(mapping_json, "r", encoding="utf-8") as f:
        category_map: Dict[str, List[str]] = json.load(f)

    # 2) 读所有记录（为了进度条总数）
    records = load_jsonl(report_info_jsonl)
    total = len(records)

    # 3) 输出容器：{category: {canonical: [(alias,hospital,path), ...]}}
    by_category: Dict[str, Dict[str, List[Tuple[str, str, str]]]] = {}
    for cat, canons in category_map.items():
        by_category[cat] = {c: [] for c in canons}

    # 4) 证据去重：不含 path
    seen_alias_host: set[Tuple[str, str, str, str]] = set()
    #    结构：(category, canonical, alias, hospital_norm)

    # 5) 缓存 (key, hospital_norm) -> canonical 或 "NA"
    cache: Dict[Tuple[str, str], str] = {}

    # 6) 进度条
    pbar = tqdm(total=total, desc="build-key-alias", ncols=100) if tqdm else None

    # 7) 逐行处理
    for idx, rec in enumerate(records, start=1):
        category = _norm_str(rec.get("category", ""))
        hospital_raw = rec.get("hospital", "")
        hospital = _norm_hospital(hospital_raw)
        path = _norm_str(rec.get("path", ""))
        keys = list(rec.get("keys", []) or [])

        # 确保该类目存在于两个映射中
        if category not in by_category:
            by_category[category] = {}
        if category not in category_map:
            category_map[category] = []

        # 当前类目的候选（动态增长）
        cat_map = by_category[category]
        if not cat_map:
            for cano in category_map.get(category, []):
                cat_map.setdefault(cano, [])

        # 处理本行 keys
        for key in keys:
            key = _norm_str(key)
            if not key:
                continue

            candidate_list = list(cat_map.keys())
            candidate_set = set(candidate_list)

            # 本身就是 canonical（完全匹配）
            if key in candidate_set:
                continue

            # 询问缓存/模型 —— 缓存维度 (key, hospital_norm)
            cache_key = (key, hospital)
            canonical = cache.get(cache_key)
            if canonical is None:
                canonical = _query_alias_by_model(key, candidate_list, model_checker)
                # 只缓存命中的 canonical；NA 不缓存以便未来候选扩张后可重试
                if canonical != "NA":
                    cache[cache_key] = canonical

            if canonical != "NA" and canonical in candidate_set:
                # —— 去重维度：不含 path —— #
                guard = (category, canonical, key, hospital)
                if guard not in seen_alias_host:
                    cat_map.setdefault(canonical, []).append((key, hospital, path))
                    seen_alias_host.add(guard)
            else:
                # 动态晋升为新的 canonical
                if key not in cat_map:
                    cat_map[key] = []

            # 用最新 keys 更新内存中的 category_map（供下一行使用）
            category_map[category] = list(cat_map.keys())

        # 二次确保
        category_map[category] = list(by_category[category].keys())

        # 进度 & 批量落盘
        if pbar:
            pbar.update(1)
        else:
            if idx % 100 == 0 or idx == total:
                print(f"[progress] {idx}/{total}")

        if out_path and (idx % save_every == 0):
            _atomic_write_json(by_category, out_path)

    # 结束后最终落盘
    if out_path:
        _atomic_write_json(by_category, out_path)

    if pbar:
        pbar.close()

    return by_category


if __name__ == "__main__":
    result = build_key_alias_map(
        report_info_jsonl="report_info.jsonl",
        mapping_json="base_report_keys_mapping.json",
        out_path="full_report_keys_mapping.json",
        save_every=10000,
    )
    # 统计
    num_categories = len(result)
    num_canonical = sum(len(inner) for inner in result.values())
    num_pairs = sum(sum(len(v) for v in inner.values()) for inner in result.values())
    print(
        f"[DONE] categories={num_categories} canonical_keys={num_canonical} "
        f"alias_pairs={num_pairs} -> key_alias_index.json"
    )
