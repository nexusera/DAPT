# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Callable
from .utils import _tighten_span


def extract_spans(
    report: str,
    keys: List[str],
    title: str,
    alias_map: Dict[str, Dict[str, List[str]]],
    choose_sep: Callable[[], str],
    validate_spans: bool = False,
    colon_only: bool = True,
    generic_heading_break: bool = True,
    expected_map: Optional[Dict[str, str]] = None,
    title_only_alias: bool = True,
) -> Dict[str, Tuple[int, int]]:
    """
    基于 value 的原样抽取：
      - 仅使用 expected_map[key]（即 value）在 report 中做精确子串查找，返回 (start, end)（char 索引，右开区间）。
      - 不使用 key/同义词/分隔符/标题等信息；不做任何兜底或模糊匹配。
      - 仅规范化换行：\\r\\n/\\r -> \\n。

    参数说明：
      - 为兼容旧调用，保留 alias_map/choose_sep/colon_only/... 参数，但不参与逻辑。
      - 当未提供 expected_map 或 value 为空/未命中时，返回 (-1, -1)。
      - 若 validate_spans=True，会在未命中时打印轻量告警。
    """
    spans: Dict[str, Tuple[int, int]] = {}
    if not keys:
        return spans

    # 仅统一换行，保持原索引语义稳定
    text = (report or "").replace("\r\n", "\n").replace("\r", "\n")

    for k in keys:
        val = ""
        if expected_map is not None:
            try:
                val = str(expected_map.get(k, "") or "").strip()
            except Exception:
                val = ""

        if not val:
            spans[k] = (-1, -1)
            if validate_spans:
                print(f"[WARN] empty expected value for key '{k}' -> s=e=-1")
            continue

        pos = text.find(val)
        if pos == -1:
            spans[k] = (-1, -1)
            if validate_spans:
                preview = (val[:60] + "…") if len(val) > 60 else val
                print(
                    f"[WARN] value not found for key '{k}': {repr(preview)} -> s=e=-1"
                )
            continue

        s, e = pos, pos + len(val)
        # 一致性处理：保持右开区间并去掉包裹空白（通常 val 已经 strip，此步多为稳健性）
        s, e = _tighten_span(text, s, e)
        spans[k] = (s, e)

    return spans
