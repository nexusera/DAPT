# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Optional
from transformers import BertTokenizerFast as HFTokenizer
import re
from functools import lru_cache


class SemanticChunker:
    """
    语义切块（改进版）：
    - 若提供 budget_tokens，则严格按 token 预算贪心聚合行段，尽量让每个 chunk 的 token 数 ≤ 预算；
    - 否则，当整篇 token 数 ≤ max_tokens_ctx 返回整篇，否则按空行 "\n\n" 粗粒度切块；
    - 对超预算的单段不再二次细切（保持与既有行为一致，编码阶段将会截断）。
    """

    def __init__(
        self,
        tokenizer: "HFTokenizer",
        max_tokens_ctx: int = 350,
        chunk_mode: str = "newline",
    ):
        self.tok = tokenizer
        self.max_tokens_ctx = int(max_tokens_ctx)
        # 保留传入值，split() 方法会根据是否提供 budget_tokens 决定策略
        self.chunk_mode = chunk_mode
    
    @lru_cache(maxsize=10000)
    def _cached_tokenize_len(self, text: str) -> int:
        """缓存tokenize长度计算"""
        try:
            return len(self.tok.tokenize(text))
        except Exception:
            return 0

    def _normalize(self, s: str) -> str:
        # 只做换行归一化（不是兜底策略）
        return (s or "").replace("\r\n", "\n").replace("\r", "\n")

    def line_spans(self, context: str) -> List[Dict]:
        """
        仅按 '\n\n' 切段；返回每段的原文、字符起止（半开区间）与该段 token 数（可选统计，不触发二次切分）。
        """
        context = self._normalize(context)
        spans: List[Dict] = []
        n = len(context)
        i = 0
        while i < n:
            j = context.find("\n\n", i)
            if j == -1:
                j = n
            seg = context[i:j]
            n_tok = self._cached_tokenize_len(seg) if seg else 0
            spans.append({"text": seg, "start": i, "end": j, "n_tok": n_tok})
            i = j + 2 if j < n else j
        return spans

    def split_lines(self, context: str) -> List[Dict]:
        """
        若整篇 token 数 <= max_tokens_ctx：不切块，返回整篇；
        否则：按 '\n\n' 切块返回各段，且不做任何进一步细切。
        """
        context = self._normalize(context)
        try:
            total_tok = len(self.tok.tokenize(context)) if context else 0
        except Exception:
            total_tok = 0

        # 仅当"整篇超预算"时才切块
        if total_tok and total_tok <= self.max_tokens_ctx:
            return [{"text": context, "char_start": 0, "char_end": len(context)}]

        # 超预算：按 '\n\n' 分段；不再进行任何 token 预算相关的合并或二次切分
        lines = self.line_spans(context)
        chunks: List[Dict] = []
        for sp in lines:
            if sp["start"] < sp["end"]:
                chunks.append(
                    {
                        "text": context[sp["start"] : sp["end"]],
                        "char_start": sp["start"],
                        "char_end": sp["end"],
                    }
                )
        return chunks

    def split_with_keys(self, context: str, keys: List[str]) -> List[Dict]:
        """
        当输入超过设定长度时，用本组内的key做分隔，插入尽可能少的\n\n到key前面，
        使得段落能满足设定长度。
        """
        context = self._normalize(context)
        try:
            total_tok = len(self.tok.tokenize(context)) if context else 0
        except Exception:
            total_tok = 0

        # 如果没有超长，直接返回整篇
        if total_tok and total_tok <= self.max_tokens_ctx:
            return [{"text": context, "char_start": 0, "char_end": len(context)}]

        # 查找所有key在文本中的位置
        key_positions = []
        for key in keys:
            # 查找key在文本中的所有出现位置
            for match in re.finditer(re.escape(key), context):
                key_positions.append({
                    'key': key,
                    'start': match.start(),
                    'end': match.end()
                })
        
        # 按位置排序
        key_positions.sort(key=lambda x: x['start'])
        
        # 根据key位置分割文本并尝试满足长度要求
        chunks: List[Dict] = []
        last_pos = 0
        
        for i, key_pos in enumerate(key_positions):
            # 获取从上一个位置到当前key开始位置的文本
            if key_pos['start'] > last_pos:
                segment = context[last_pos:key_pos['start']]
                try:
                    segment_tokens = len(self.tok.tokenize(segment)) if segment else 0
                except Exception:
                    segment_tokens = 0
                    
                # 如果段落太长，需要在key前插入分隔符
                if segment_tokens > self.max_tokens_ctx:
                    # 在当前key前分割
                    chunk_text = context[last_pos:key_pos['start']]
                    chunks.append({
                        "text": chunk_text,
                        "char_start": last_pos,
                        "char_end": key_pos['start']
                    })
                    last_pos = key_pos['start']
        
        # 添加最后一段
        if last_pos < len(context):
            chunk_text = context[last_pos:]
            chunks.append({
                "text": chunk_text,
                "char_start": last_pos,
                "char_end": len(context)
            })
            
        # 如果仍然有超长段落，使用原始的\n\n分割方法作为后备
        final_chunks: List[Dict] = []
        for chunk in chunks:
            try:
                chunk_tokens = len(self.tok.tokenize(chunk["text"])) if chunk["text"] else 0
            except Exception:
                chunk_tokens = 0
                
            if chunk_tokens > self.max_tokens_ctx:
                # 如果单个chunk还是太长，使用原始的line_spans方法分割
                sub_chunks = self.line_spans(chunk["text"])
                for sub_chunk in sub_chunks:
                    if sub_chunk["start"] < sub_chunk["end"]:
                        final_chunks.append({
                            "text": chunk["text"][sub_chunk["start"]:sub_chunk["end"]],
                            "char_start": chunk["char_start"] + sub_chunk["start"],
                            "char_end": chunk["char_start"] + sub_chunk["end"]
                        })
            else:
                final_chunks.append(chunk)
                
        return final_chunks

    def split(self, context: str, budget_tokens: Optional[int] = None) -> List[Dict]:
        """
        对外统一入口：
        - 如提供 budget_tokens（>0），按预算贪心聚合相邻行段；
        - 否则退化为 split_lines。
        """
        context = self._normalize(context)
        if not context:
            return []

        if not budget_tokens or int(budget_tokens) <= 0:
            return self.split_lines(context)

        budget = int(budget_tokens)

        # 先做行段切分并统计每段 token 数
        lines = self.line_spans(context)
        if not lines:
            return [{"text": context, "char_start": 0, "char_end": len(context)}]

        chunks: List[Dict] = []
        cur_start = None
        cur_end = None
        cur_tok_sum = 0

        for sp in lines:
            seg_start, seg_end, seg_tok = int(sp["start"]), int(sp["end"]), int(sp.get("n_tok", 0))

            if cur_start is None:
                # 开启新块
                cur_start = seg_start
                cur_end = seg_end
                cur_tok_sum = seg_tok
                continue

            # 若加入该段仍不超过预算，则继续合并
            if (cur_tok_sum + seg_tok) <= budget:
                cur_end = seg_end
                cur_tok_sum += seg_tok
            else:
                # 先收录当前聚合块
                if cur_end is not None and cur_start is not None and cur_end > cur_start:
                    chunks.append({
                        "text": context[cur_start:cur_end],
                        "char_start": cur_start,
                        "char_end": cur_end,
                    })
                # 以该段开启新块
                cur_start = seg_start
                cur_end = seg_end
                cur_tok_sum = seg_tok

        # 收尾
        if cur_end is not None and cur_start is not None and cur_end > cur_start:
            chunks.append({
                "text": context[cur_start:cur_end],
                "char_start": cur_start,
                "char_end": cur_end,
            })

        return chunks