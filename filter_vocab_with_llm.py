#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用本地 Qwen3 32B API 对 WordPiece 候选词表做机器筛选。
默认读取 medical_vocab_ocr_only/vocab.txt，输出：
- kept_vocab.txt      ：保留列表
- dropped_vocab.txt   ：剔除列表（含原因）
- kept_vocab.topN.txt ：按原顺序的前 N 行（可选）

依赖：requests（如无请 pip install requests）

使用示例：
    export LLF_API_BASE="http://127.0.0.1:8008/v1,http://127.0.0.1:8009/v1,http://127.0.0.1:8010/v1,http://127.0.0.1:8011/v1"
    export OPENAI_API_KEY="EMPTY"   # 若服务不校验可用占位
    python filter_vocab_with_llm.py \
        --vocab medical_vocab_ocr_only/vocab.txt \
        --kept kept_vocab.txt \
        --dropped dropped_vocab.txt \
        --batch_size 64 \
        --topn 50000
"""

import argparse
import json
import os
import random
import time
from typing import List, Dict, Tuple

import requests

DEFAULT_MODEL = os.getenv("LLF_MODEL", "qwen3-32b-instruct")
API_BASES = [b.strip() for b in os.getenv("LLF_API_BASE", "").split(",") if b.strip()]
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}


PROMPT_TEMPLATE = """你是医学 NLP 专家，需判定候选分词是否应保留。
判定规则（尽量保守丢弃噪音）：
- 保留：医学实体/术语/检查项目/药物/症状/操作/部位/分期/指标等；常见缩写且医学相关；多词组合且医学相关。
- 删除：人名/机构名（非医学通用词）、纯编号/票据/电话/条码、文件名、路径、日期时间、纯数字或数字+单位噪音、随机编码、版面/表格噪声、明显错别字或无意义碎片、与医学无关的日常或通用短词。
输出 JSON 数组，每个元素 {{"token": "...", "keep": true/false, "reason": "简要原因"}}，只输出数组，不要多余文字。

候选列表：
{tokens}
"""


def pick_base(i: int) -> str:
    if not API_BASES:
        raise RuntimeError("请先设置环境变量 LLF_API_BASE，逗号分隔多个 endpoint")
    return API_BASES[i % len(API_BASES)]


def _parse_json_array(text: str) -> List[Dict]:
    """尽量鲁棒地从模型输出中提取 JSON 数组"""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # 尝试截取首尾中括号
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    raise ValueError(f"无法解析模型输出，原文片段: {text[:400]}")


def call_llm(
    tokens: List[str],
    idx: int,
    model: str = DEFAULT_MODEL,
    timeout: int = 60,
    max_tokens: int = 2048,
    retries: int = 2,
) -> List[Dict]:
    """带重试、多 endpoint 轮询，逐个尝试，直到成功或用尽。"""
    errors = []
    n_bases = len(API_BASES)
    if n_bases == 0:
        raise RuntimeError("请设置环境变量 LLF_API_BASE，例如 http://127.0.0.1:8008/v1")
    for attempt in range(retries + 1):
        base = pick_base(idx + attempt)
        url = f"{base}/chat/completions"
        prompt = PROMPT_TEMPLATE.format(tokens=json.dumps(tokens, ensure_ascii=False, indent=2))
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return _parse_json_array(content)
        except Exception as e:
            errors.append(f"[{base}] {e}")
            time.sleep(0.5)
            continue
    raise RuntimeError("LLM 调用失败，尝试的 endpoint: " + "; ".join(errors))


def load_vocab(path: str, topn: int = None) -> List[str]:
    vocab = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            tok = line.strip()
            if not tok:
                continue
            vocab.append(tok)
            if topn and len(vocab) >= topn:
                break
    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default="medical_vocab_ocr_only/vocab.txt")
    parser.add_argument("--kept", default="kept_vocab.txt")
    parser.add_argument("--dropped", default="dropped_vocab.txt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--topn", type=int, default=None, help="仅取前 N 行做筛选，None 表示全量")
    parser.add_argument("--max_tokens", type=int, default=2048, help="LLM 输出上限，避免截断")
    parser.add_argument("--timeout", type=int, default=60, help="每个请求的超时时间（秒）")
    parser.add_argument("--retries", type=int, default=2, help="每个 batch 的重试次数（轮询其他 BASE）")
    args = parser.parse_args()

    vocab = load_vocab(args.vocab, args.topn)
    print(f"待筛选词数: {len(vocab)}")

    kept = []
    dropped = []
    t0 = time.time()
    for i in range(0, len(vocab), args.batch_size):
        batch = vocab[i:i + args.batch_size]
        try:
            results = call_llm(
                batch,
                idx=i // args.batch_size,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
            )
        except Exception as e:
            print(f"[batch {i//args.batch_size}] 调用失败: {e}")
            continue
        for r in results:
            tok = r.get("token", "").strip()
            keep = bool(r.get("keep", False))
            reason = r.get("reason", "")
            if not tok:
                continue
            if keep:
                kept.append(tok)
            else:
                dropped.append(f"{tok}\t{reason}")
        if (i // args.batch_size) % 10 == 0:
            elapsed = time.time() - t0
            print(f"进度: {i + len(batch)}/{len(vocab)}, elapsed={elapsed:.1f}s, kept={len(kept)}, dropped={len(dropped)}")

    with open(args.kept, "w", encoding="utf-8") as f:
        for k in kept:
            f.write(k + "\n")
    with open(args.dropped, "w", encoding="utf-8") as f:
        for d in dropped:
            f.write(d + "\n")

    print(f"完成。保留 {len(kept)} 个，剔除 {len(dropped)} 个，耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

