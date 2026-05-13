#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查预训练语料是否包含下游评测测试集文本（数据泄露粗检）。

依据 docs/pipelines/pipeline_new.md（预训练语料如 train.txt / train_chunked.txt）与
docs/pipelines/pipeline_xiaorong.md / docs/pipelines/pipeline_task2_xiaorong.md（测试集 real_test_with_ocr.json）。

方法（默认）：
  从测试集每条样本抽取若干长度 >= min_match_len 的文本窗口作为“指纹”，
  以 Rabin-Karp 滚动哈希在语料上流式扫描；命中后再做字符串校验，减少误报。

注意：
  - 这是启发式检测：过短的指纹会产生大量误报；Normalization 差异（空白/换行）
    可能导致漏报。请结合 min_match_len 与 manual review。
  - 若已安装 pyahocorasick，可加 --use_ahocorasick 使用 AC 自动机（多模式匹配）。

示例：
  python DAPT/scripts/check_pretrain_test_leakage.py \\
    --corpus /data/ocean/DAPT/workspace/train_chunked.txt \\
    --corpus /data/ocean/DAPT/workspace/train_ocr_9297.txt \\
    --test_json /data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json \\
    --min_match_len 96 \\
    --max_windows_per_doc 64 \\
    --report_json leakage_report.json
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# 可选：AC 自动机
# ---------------------------------------------------------------------------
try:
    import ahocorasick  # type: ignore
except Exception:  # pragma: no cover
    ahocorasick = None


# ---------------------------------------------------------------------------
# 测试集加载与文本抽取（与 compare_models.load_data / OCR 逻辑对齐）
# ---------------------------------------------------------------------------


def load_json_or_jsonl(path: str) -> List[dict]:
    """读取 JSON 数组或 JSONL。"""
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    with opener(path, mode, encoding="utf-8") as f:  # type: ignore[arg-type]
        head = ""
        while True:
            ch = f.read(1)
            if not ch:
                return []
            if not ch.isspace():
                head = ch
                break
        f.seek(0)
        if head in "[{":
            obj = json.load(f)
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
            if isinstance(obj, dict):
                return [obj]
            return []
    out: List[dict] = []
    with opener(path, mode, encoding="utf-8") as f:  # type: ignore[arg-type]
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                if isinstance(o, dict):
                    out.append(o)
            except json.JSONDecodeError:
                continue
    return out


def extract_ocr_text(ocr_raw: Any) -> str:
    if not ocr_raw:
        return ""
    if isinstance(ocr_raw, str):
        return ocr_raw
    if isinstance(ocr_raw, dict):
        words_result = ocr_raw.get("words_result")
        if isinstance(words_result, list):
            words: List[str] = []
            for w in words_result:
                if isinstance(w, dict) and "words" in w:
                    words.append(str(w["words"]))
            if words:
                return "".join(words)
    return str(ocr_raw)


def extract_text_segments_from_item(item: dict) -> List[str]:
    """从单条测试样本中收集可能出现在预训练中的纯文本片段。"""
    parts: List[str] = []

    for key in ("full_text", "text", "ocr_text", "content", "report_title"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())

    spans = item.get("spans")
    if isinstance(spans, dict):
        for _k, v in spans.items():
            if isinstance(v, dict):
                t = v.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())

    kvp = item.get("key_value_pairs")
    if isinstance(kvp, list):
        for p in kvp:
            if not isinstance(p, dict):
                continue
            key_obj = p.get("key")
            if isinstance(key_obj, dict):
                kt = key_obj.get("text")
                if isinstance(kt, str) and kt.strip():
                    parts.append(kt.strip())
            vt = p.get("value_text")
            if isinstance(vt, str) and vt.strip():
                parts.append(vt.strip())

    ocr = extract_ocr_text(item.get("ocr_raw"))
    if ocr.strip():
        parts.append(ocr.strip())

    ta = item.get("transferred_annotations")
    if isinstance(ta, list):
        for anno in ta:
            if not isinstance(anno, dict):
                continue
            t = anno.get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())

    # 去重保序
    seen: Set[str] = set()
    uniq: List[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def normalize_ws(s: str) -> str:
    return " ".join(s.split())


def sample_windows(
    text: str,
    window_len: int,
    max_windows: int,
    rng: random.Random,
    stride: Optional[int],
) -> List[str]:
    """从一段文本中抽取固定长度子串窗口（用于指纹）。"""
    t = text
    if len(t) < window_len:
        return [t] if len(t) >= max(1, window_len // 2) else []

    step = stride if stride and stride > 0 else window_len
    indices = list(range(0, len(t) - window_len + 1, step))
    if len(indices) > max_windows:
        indices = rng.sample(indices, max_windows)
    else:
        rng.shuffle(indices)
        indices = indices[:max_windows]

    out: List[str] = []
    for i in sorted(set(indices)):
        out.append(t[i : i + window_len])
    return out


def item_id(item: dict, idx: int) -> str:
    for k in ("id", "record_id", "sample_id", "task_id"):
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v)
    return f"sample_{idx}"


# ---------------------------------------------------------------------------
# Rabin-Karp 多模式（定长）索引
# ---------------------------------------------------------------------------

MOD = (1 << 61) - 1
BASE = 1315423911


def _pow_base(n: int) -> int:
    return pow(BASE, n, MOD)


def build_fixed_len_index(
    patterns: Sequence[str], length: int
) -> Dict[int, List[str]]:
    """所有模式串必须等长 = length。"""
    buckets: Dict[int, List[str]] = defaultdict(list)
    for p in patterns:
        if len(p) != length:
            continue
        h = 0
        for ch in p:
            h = (h * BASE + ord(ch)) % MOD
        buckets[h].append(p)
    return dict(buckets)


@dataclass
class StreamMatch:
    corpus_file: str
    char_offset: int
    line: int
    col: int
    needle: str
    test_id: str
    window_index: int


def scan_corpus_stream_rk(
    corpus_paths: Sequence[str],
    needles_meta: Sequence[Tuple[str, str, int]],
    window_len: int,
    normalize_corpus: bool,
) -> List[StreamMatch]:
    """
    needles_meta: (needle, test_id, window_index)
    按行扫描：窗口不跨行（与常见 train.txt 一行一段/滑窗格式一致）。
    """
    patterns = [n for n, _, _ in needles_meta]
    idx_map: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for needle, tid, wi in needles_meta:
        idx_map[needle].append((tid, wi))

    hash_buckets = build_fixed_len_index(patterns, window_len)
    pow_base_L = _pow_base(window_len)

    matches: List[StreamMatch] = []

    def open_text(path: str) -> io.TextIOBase:
        if path.endswith(".gz"):
            return gzip.open(path, "rt", encoding="utf-8", errors="replace")
        return open(path, "r", encoding="utf-8", errors="replace")

    for cpath in corpus_paths:
        with open_text(cpath) as f:
            line_no = 0
            for raw_line in f:
                line_no += 1
                line_text = normalize_ws(raw_line) if normalize_corpus else raw_line.rstrip("\n\r")
                buf: deque[str] = deque()
                rolling = 0
                col = 0
                for ch in line_text:
                    col += 1
                    if len(buf) == window_len:
                        left = buf.popleft()
                        rolling = (rolling - ord(left) * pow_base_L) % MOD
                        rolling = (rolling * BASE + ord(ch)) % MOD
                        buf.append(ch)
                    else:
                        buf.append(ch)
                        rolling = (rolling * BASE + ord(ch)) % MOD

                    if len(buf) < window_len:
                        continue

                    cand_list = hash_buckets.get(rolling)
                    if not cand_list:
                        continue
                    window = "".join(buf)
                    for cand in cand_list:
                        if window != cand:
                            continue
                        for tid, wi in idx_map.get(cand, []):
                            matches.append(
                                StreamMatch(
                                    corpus_file=cpath,
                                    char_offset=0,
                                    line=line_no,
                                    col=col - window_len + 1,
                                    needle=cand,
                                    test_id=tid,
                                    window_index=wi,
                                )
                            )

    return matches


def scan_corpus_naive_short(
    corpus_paths: Sequence[str],
    short_meta: Sequence[Tuple[str, str, int]],
    normalize_corpus: bool,
) -> List[StreamMatch]:
    """对长度 != L 的少量指纹做子串扫描（短样本全文）。"""
    matches: List[StreamMatch] = []

    def open_text(path: str) -> io.TextIOBase:
        if path.endswith(".gz"):
            return gzip.open(path, "rt", encoding="utf-8", errors="replace")
        return open(path, "r", encoding="utf-8", errors="replace")

    for cpath in corpus_paths:
        with open_text(cpath) as f:
            line_no = 0
            for raw_line in f:
                line_no += 1
                line_text = normalize_ws(raw_line) if normalize_corpus else raw_line.rstrip("\n\r")
                for needle, tid, wi in short_meta:
                    if needle and needle in line_text:
                        col = line_text.find(needle) + 1
                        matches.append(
                            StreamMatch(
                                corpus_file=cpath,
                                char_offset=0,
                                line=line_no,
                                col=col,
                                needle=needle,
                                test_id=tid,
                                window_index=wi,
                            )
                        )
    return matches


def scan_corpus_aho(
    corpus_paths: Sequence[str],
    needles: Sequence[str],
    needle_to_meta: Dict[str, List[Tuple[str, int]]],
    normalize_corpus: bool,
) -> List[StreamMatch]:
    if ahocorasick is None:
        raise RuntimeError("pyahocorasick 未安装")
    A = ahocorasick.Automaton()
    for i, nd in enumerate(needles):
        if not nd:
            continue
        A.add_word(nd.encode("utf-8"), (i, nd))
    A.make_automaton()

    matches: List[StreamMatch] = []

    def open_text(path: str) -> io.TextIOBase:
        if path.endswith(".gz"):
            return gzip.open(path, "rt", encoding="utf-8", errors="replace")
        return open(path, "r", encoding="utf-8", errors="replace")

    for cpath in corpus_paths:
        with open_text(cpath) as f:
            line_no = 0
            for raw_line in f:
                line_no += 1
                line_text = normalize_ws(raw_line) if normalize_corpus else raw_line.rstrip("\n\r")
                line_bytes = line_text.encode("utf-8")
                for end_idx, (_i, nd_str) in A.iter(line_bytes):
                    wbytes = nd_str.encode("utf-8")
                    start_b = end_idx - len(wbytes) + 1
                    if start_b < 0:
                        continue
                    prefix = line_bytes[:start_b].decode("utf-8", errors="replace")
                    col_here = len(prefix) + 1
                    for tid, wi in needle_to_meta.get(nd_str, []):
                        matches.append(
                            StreamMatch(
                                corpus_file=cpath,
                                char_offset=0,
                                line=line_no,
                                col=col_here,
                                needle=nd_str,
                                test_id=tid,
                                window_index=wi,
                            )
                        )

    return matches


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="预训练语料 vs 下游测试集泄露检查")
    p.add_argument(
        "--corpus",
        action="append",
        required=True,
        help="预训练文本文件，可重复传入多个（如 train_chunked.txt + train_ocr_9297.txt）",
    )
    p.add_argument(
        "--test_json",
        action="append",
        default=[],
        help="下游测试集 JSON/JSONL（默认：pipeline 中的 real_test_with_ocr.json），可多次指定",
    )
    p.add_argument(
        "--min_match_len",
        type=int,
        default=96,
        help="指纹最小长度（字符数）。越长误报越少，可能漏报略增。",
    )
    p.add_argument(
        "--max_windows_per_doc",
        type=int,
        default=48,
        help="每条测试样本从拼接文本中最多采样的定长窗口数",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=0,
        help="窗口步长；0 表示等于 min_match_len（不重叠）",
    )
    p.add_argument(
        "--normalize_test",
        action="store_true",
        help="对测试集文本做空白归一化后再抽窗口",
    )
    p.add_argument(
        "--normalize_corpus",
        action="store_true",
        help="对语料每行做空白归一化后再匹配（与测试 normalize 一起用更一致）",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="采样窗口的随机种子",
    )
    p.add_argument(
        "--use_ahocorasick",
        action="store_true",
        help="使用 pyahocorasick 扫描（变长 needle 需长度一致时仍用定长窗口）",
    )
    p.add_argument(
        "--report_json",
        type=str,
        default="",
        help="将结果写入 JSON 文件",
    )
    p.add_argument(
        "--max_report_matches",
        type=int,
        default=500,
        help="最多写入报告的命中条数（防止爆炸）",
    )

    args = p.parse_args(list(argv) if argv is not None else None)

    default_tests = [
        "/data/ocean/DAPT/biaozhu_with_ocr_noise_prepared/real_test_with_ocr.json",
    ]
    test_paths = list(args.test_json) if args.test_json else default_tests

    all_items: List[dict] = []
    for tp in test_paths:
        if not os.path.isfile(tp):
            print(f"[WARN] 测试集文件不存在，跳过: {tp}", file=sys.stderr)
            continue
        all_items.extend(load_json_or_jsonl(tp))

    if not all_items:
        print("[ERROR] 未加载到任何测试样本；请检查 --test_json 路径。", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    L = args.min_match_len
    stride = args.stride if args.stride > 0 else L

    needles_meta: List[Tuple[str, str, int]] = []
    wi_global = 0

    for idx, item in enumerate(all_items):
        tid = item_id(item, idx)
        segments = extract_text_segments_from_item(item)
        big = "\n".join(segments)
        if args.normalize_test:
            big = normalize_ws(big)
        if len(big) < L:
            # 整条过短：仍尝试作为唯一指纹
            if len(big) >= max(32, L // 3):
                needles_meta.append((big, tid, wi_global))
                wi_global += 1
            continue
        wins = sample_windows(big, L, args.max_windows_per_doc, rng, stride)
        for w in wins:
            needles_meta.append((w, tid, wi_global))
            wi_global += 1

    # 去重 needle，保留多 test_id 映射
    needle_to_meta: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for needle, tid, wi in needles_meta:
        needle_to_meta[needle].append((tid, wi))

    uniq_needles = list(needle_to_meta.keys())
    print(
        f"[INFO] 测试样本数={len(all_items)}，唯一指纹数={len(uniq_needles)}，"
        f"min_match_len={L}，语料文件数={len(args.corpus)}",
        file=sys.stderr,
    )

    missing_corpus = [c for c in args.corpus if not os.path.isfile(c)]
    if missing_corpus:
        for c in missing_corpus:
            print(f"[ERROR] 语料文件不存在: {c}", file=sys.stderr)
        return 2

    if args.use_ahocorasick:
        if ahocorasick is None:
            print("[ERROR] 请求 --use_ahocorasick 但未安装 pyahocorasick。", file=sys.stderr)
            return 2
        matches = scan_corpus_aho(
            args.corpus,
            uniq_needles,
            dict(needle_to_meta),
            args.normalize_corpus,
        )
    else:
        # 定长窗口：Rabin–Karp 流式扫描；其余（极短样本）走朴素子串匹配
        meta_f = [(n, tid, wi) for n, tid, wi in needles_meta if len(n) == L]
        meta_short = [(n, tid, wi) for n, tid, wi in needles_meta if len(n) != L]
        matches = scan_corpus_stream_rk(args.corpus, meta_f, L, args.normalize_corpus)
        if meta_short:
            if len(meta_short) > 5000:
                print(
                    f"[WARN] 非定长指纹条数={len(meta_short)} 较多，朴素扫描可能较慢。可考虑 --use_ahocorasick。",
                    file=sys.stderr,
                )
            matches.extend(
                scan_corpus_naive_short(args.corpus, meta_short, args.normalize_corpus)
            )

    # 按 test_id 聚合
    by_test: Dict[str, List[StreamMatch]] = defaultdict(list)
    for m in matches:
        by_test[m.test_id].append(m)

    print("\n========== 扫描结果 ==========", file=sys.stderr)
    print(f"命中条数（总）: {len(matches)}", file=sys.stderr)
    print(f"涉及测试样本数: {len(by_test)}", file=sys.stderr)

    if matches:
        print("\n[ALERT] 发现可能的测试集文本片段出现在预训练语料中，请人工复核。", file=sys.stderr)
        show = matches[: min(20, len(matches))]
        for m in show:
            preview = m.needle[:60] + ("..." if len(m.needle) > 60 else "")
            print(
                f"  - test_id={m.test_id}  file={m.corpus_file}  L{m.line}:C{m.col}  {preview!r}",
                file=sys.stderr,
            )
        if len(matches) > 20:
            print(f"  ... 另有 {len(matches) - 20} 条命中未显示", file=sys.stderr)
    else:
        print("\n[OK] 未发现定长指纹命中（不代表绝对无泄露，见脚本头部说明）。", file=sys.stderr)

    if args.report_json:
        rep = {
            "config": {
                "corpus": args.corpus,
                "test_json": test_paths,
                "min_match_len": L,
                "max_windows_per_doc": args.max_windows_per_doc,
                "stride": stride,
                "normalize_test": args.normalize_test,
                "normalize_corpus": args.normalize_corpus,
                "use_ahocorasick": args.use_ahocorasick,
            },
            "summary": {
                "num_test_items": len(all_items),
                "num_unique_needles": len(uniq_needles),
                "total_matches": len(matches),
                "num_test_ids_with_hits": len(by_test),
            },
            "matches": [],
        }
        lim = args.max_report_matches
        for m in matches[:lim]:
            rep["matches"].append(
                {
                    "test_id": m.test_id,
                    "corpus_file": m.corpus_file,
                    "line": m.line,
                    "col": m.col,
                    "char_offset": m.char_offset,
                    "needle_preview": m.needle[:200],
                    "needle_len": len(m.needle),
                    "window_index": m.window_index,
                }
            )
        if len(matches) > lim:
            rep["truncated"] = True
            rep["truncated_total_matches"] = len(matches)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 已写入报告: {args.report_json}", file=sys.stderr)

    return 1 if matches else 0


if __name__ == "__main__":
    raise SystemExit(main())
