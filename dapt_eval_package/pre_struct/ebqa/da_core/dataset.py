# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import sys
import re

# ensure repo/package roots are importable (handles execution as script)
_HERE = os.path.abspath(os.path.dirname(__file__))
_PRE_STRUCT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # dapt_eval_package/pre_struct
_PKG_ROOT = os.path.abspath(os.path.join(_PRE_STRUCT_ROOT, ".."))      # dapt_eval_package
for _p in (_HERE, _PRE_STRUCT_ROOT, _PKG_ROOT, os.getcwd()):
    if _p not in sys.path:
        sys.path.append(_p)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
from functools import lru_cache

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from tqdm import tqdm


import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast as HFTokenizer

# 兼容作为模块或脚本直接运行的导入方式
try:
    from .utils import (
        _load_jsonl_or_json,
        _save_jsonl,
        _dedup_keep_order,
        split_train_test_balanced_by_title,
    )
    from .chunking import SemanticChunker
    from . import extraction as EX
except Exception:
    try:
        from pre_struct.ebqa.da_core.utils import (
            _load_jsonl_or_json,
            _save_jsonl,
            _dedup_keep_order,
            split_train_test_balanced_by_title,
        )
        from pre_struct.ebqa.da_core.chunking import SemanticChunker
        from pre_struct.ebqa.da_core import extraction as EX
    except Exception:
        from dapt_eval_package.pre_struct.ebqa.da_core.utils import (  # type: ignore
            _load_jsonl_or_json,
            _save_jsonl,
            _dedup_keep_order,
            split_train_test_balanced_by_title,
        )
        from dapt_eval_package.pre_struct.ebqa.da_core.chunking import SemanticChunker  # type: ignore
        from dapt_eval_package.pre_struct.ebqa.da_core import extraction as EX  # type: ignore

try:
    from pre_struct.map_key2question import convert_key_to_question
except Exception:
    try:
        from dapt_eval_package.pre_struct.map_key2question import convert_key_to_question  # type: ignore
    except Exception:
        from map_key2question import convert_key_to_question  # type: ignore

# --------- Collator（padding + 可选调试字段透传）---------
class QACollator:
    def __init__(
        self,
        pad_id: int = 0,
        pad_token_type_id: int = 0,
        pad_attention_mask: int = 0,
        keep_debug_fields: bool = True,
        vocab_size: int = None,  # 新增：用于兼容性检查
    ) -> None:
        self.torch = torch
        self.pad_id = 0 if pad_id is None else int(pad_id)
        self.pad_tt = 0 if pad_token_type_id is None else int(pad_token_type_id)
        self.pad_am = 0 if pad_attention_mask is None else int(pad_attention_mask)
        self.keep_debug_fields = bool(keep_debug_fields)
        self.vocab_size = vocab_size
        self._warned_vocab = False  # 避免重复警告

    def _to_list(self, x):
        if isinstance(x, self.torch.Tensor):
            return x.tolist()
        return list(x)
    
    def _validate_and_fix_token_ids(self, token_ids_list):
        """验证并修正token ids，确保不超出词表范围
        
        Args:
            token_ids_list: list of lists, 每个子列表是一个样本的token ids
            
        Returns:
            list of lists: 修正后的token ids
        """
        if self.vocab_size is None:
            return token_ids_list  # 没有词表大小信息，跳过验证
        
        fixed_ids = []
        has_invalid = False
        
        for ids in token_ids_list:
            ids_copy = list(ids)
            for i, tid in enumerate(ids_copy):
                if tid >= self.vocab_size:
                    has_invalid = True
                    # 将超出范围的token id替换为unk_token_id (通常是100)
                    # 如果是padding位置(值很大可能是未初始化)，替换为pad_id
                    ids_copy[i] = self.pad_id if tid > 200000 else 100
            fixed_ids.append(ids_copy)
        
        if has_invalid and not self._warned_vocab:
            self._warned_vocab = True
            print(f"\n{'='*80}")
            print(f"[警告] 检测到token id超出模型词表范围！")
            print(f"  - 当前模型词表大小: {self.vocab_size}")
            print(f"  - 数据中存在超出范围的token id")
            print(f"  - 已自动替换为pad_token或unk_token")
            print(f"\n[建议] 您的数据似乎是用另一个tokenizer预处理的。")
            print(f"       建议使用当前tokenizer重新预处理数据以获得最佳效果：")
            print(f"       1. 删除旧的预处理数据")
            print(f"       2. 确保配置文件中的tokenizer_name_or_path正确")
            print(f"       3. 重新运行数据预处理")
            print(f"{'='*80}\n")
        
        return fixed_ids

    def _pad_2d(self, seqs, pad_val: int, validate_vocab: bool = False):
        # 如果需要验证词表（用于input_ids）
        if validate_vocab and self.vocab_size is not None:
            seqs = self._validate_and_fix_token_ids(seqs)
        
        max_len = max(len(s) for s in seqs) if seqs else 0
        out = []
        for s in seqs:
            s = self._to_list(s)
            if len(s) < max_len:
                s = s + [pad_val] * (max_len - len(s))
            out.append(s)
        return self.torch.tensor(out, dtype=self.torch.long)

    def __call__(self, batch):
        input_ids = [self._to_list(b["input_ids"]) for b in batch]
        attention_mask = [self._to_list(b["attention_mask"]) for b in batch]
        token_type_ids = [self._to_list(b.get("token_type_ids", [])) for b in batch]

        out = {
            "input_ids": self._pad_2d(input_ids, self.pad_id, validate_vocab=True),  # 启用词表验证
            "attention_mask": self._pad_2d(attention_mask, self.pad_am),
        }
        if any(len(tt) > 0 for tt in token_type_ids):
            out["token_type_ids"] = self._pad_2d(token_type_ids, self.pad_tt)

        if "start_positions" in batch[0] and "end_positions" in batch[0]:
            starts = [int(b["start_positions"]) for b in batch]
            ends = [int(b["end_positions"]) for b in batch]
            out["start_positions"] = self.torch.tensor(starts, dtype=self.torch.long)
            out["end_positions"] = self.torch.tensor(ends, dtype=self.torch.long)

        #  始终处理训练必需的字段（不管debug模式）
        for train_field in ["length_reasonableness", "is_short_field"]:
            if train_field in batch[0]:
                vals = [b.get(train_field) for b in batch]
                if train_field == "length_reasonableness":
                    vfix = [(1.0 if v is None else float(v)) for v in vals]
                    out[train_field] = self.torch.tensor(vfix, dtype=self.torch.float32)
                elif train_field == "is_short_field":
                    vfix = [(False if v is None else bool(v)) for v in vals]
                    out[train_field] = self.torch.tensor(vfix, dtype=self.torch.bool)

        # noise_ids padding（形状：B x L x F，默认 F=7，缺省填0）
        if any("noise_ids" in b for b in batch):
            max_len = out["input_ids"].size(1)
            noise_raw = []
            max_feat = 0
            for b in batch:
                seq = b.get("noise_ids")
                if isinstance(seq, self.torch.Tensor):
                    seq = seq.tolist()
                seq = seq or []
                noise_raw.append(seq)
                if seq and isinstance(seq[0], (list, tuple)):
                    max_feat = max(max_feat, len(seq[0]))
            max_feat = max(1, max_feat or 7)
            noise_pad = self.torch.zeros(
                (len(batch), max_len, max_feat), dtype=self.torch.long
            )
            for bi, seq in enumerate(noise_raw):
                seq_len = len(input_ids[bi]) if bi < len(input_ids) else 0
                for ti in range(min(seq_len, max_len)):
                    vals = seq[ti] if ti < len(seq) else None
                    if isinstance(vals, self.torch.Tensor):
                        vals = vals.tolist()
                    if isinstance(vals, (list, tuple)):
                        padded = list(vals) + [0] * (max_feat - len(vals))
                    elif vals is None:
                        padded = [0] * max_feat
                    else:
                        padded = [int(vals)] + [0] * (max_feat - 1)
                    noise_pad[bi, ti, :] = self.torch.tensor(
                        padded[:max_feat], dtype=self.torch.long
                    )
            out["noise_ids"] = noise_pad

        if self.keep_debug_fields:
            dbg_keys = (
                "question_key",
                "chunk_index",
                "report_index",
                "offset_mapping",
                "sequence_ids",
                "chunk_char_start",
                "chunk_char_end",
                "chunk_text",
            )
            for k in dbg_keys:
                vals = [b.get(k) for b in batch]
                if any(v is not None for v in vals):
                    out[k] = vals
        return out


# -----------------------------
# 数据集
# -----------------------------
class EnhancedQADataset(Dataset):
    """从报告构建抽取式 QA 样本：值精确匹配取 char span，映射到 token，必要时分块；键集合可按标题限定。"""
    
    @staticmethod 
    def _is_short_field(field_value: str, tokenizer=None) -> bool:
        """基于实际value长度动态识别短字段
        
        Args:
            field_value: 字段的实际值
            tokenizer: 用于计算token长度（可选）
            
        Returns:
            bool: 是否为短字段
        """
        field_value = str(field_value).strip()
        if not field_value:
            return True  # 空值认为是短字段
            
        # 1. 基于字符长度（中文）
        char_len = len(field_value)
        if char_len <= 8:  # 8个字符以内（包括中、英）认为是短字段
            return True
            
        # 2. 基于token长度（如果有tokenizer）
        if tokenizer is not None:
            try:
                if hasattr(tokenizer, 'encode'):
                    token_len = len(tokenizer.encode(field_value, add_special_tokens=False))
                elif hasattr(tokenizer, '__call__'):
                    token_len = len(tokenizer(field_value, add_special_tokens=False)['input_ids'])
                else:
                    token_len = char_len  # fallback
                    
                if token_len <= 6:  # 6个token以内认为是短字段
                    return True
            except Exception:
                pass  # tokenizer失败时忽略，仅用字符长度判断
                
        # 3. 基于内容模式（简单启发式）
        # 纯数字、日期格式、简单词语等
        import re
        if re.match(r'^[\d\-/\s]{1,15}$', field_value):  # 日期、数字格式
            return True
        if re.match(r'^[a-zA-Z\s]{1,10}$', field_value):  # 简短英文
            return True
        if re.match(r'^[\u4e00-\u9fff]{1,5}$', field_value):  # 1-5个中文字符
            return True
            
        return False

    def __init__(
        self,
        data_path: str,
        data_paths: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        tokenizer_name: Optional[str] = None,
        max_seq_len: int = 512,
        max_tokens_ctx: int = 480,
        max_answer_len: int = 384,
        use_question_templates: bool = True,
        keep_debug_fields: bool = False,
        negative_downsample: float = 1.0,
        seed: int = 42,
        autobuild: bool = True,
        show_progress: bool = True,
        chunk_mode: str = "budget",
        # 新增：结构映射
        report_struct_path: Optional[str] = None,
        # True: 只用结构映射里本标题的键；False: 结构键 ∪ 记录键
        only_title_keys: bool = False,
        # 新增：并发选项
        use_concurrent_build: bool = False,  # 禁用多线程，避免tokenizer争用
        max_workers: Optional[int] = None,
        # 推理模式：不使用记录中的值作为 gold，让模型自主预测
        inference_mode: bool = False,
        # 动态答案长度：根据训练数据中的实际答案长度动态调整
        dynamic_answer_length: bool = True,
    ) -> None:
        self._user_tok = tokenizer
        self._tokenizer_name = tokenizer_name
        self.max_seq_len = int(max_seq_len)
        self.max_tokens_ctx = int(max_tokens_ctx)
        self.max_answer_len = int(max_answer_len)
        self.use_question_templates = bool(use_question_templates)
        self.keep_debug_fields = bool(keep_debug_fields)
        self.negative_downsample = float(negative_downsample)
        self._base_seed = int(seed)
        self._rng = random.Random(self._base_seed)
        self.show_progress = bool(show_progress)
        self.chunk_mode = str(chunk_mode)
        
        # 并发选项
        self.use_concurrent_build = bool(use_concurrent_build)
        self.max_workers = max_workers
        # 推理模式标记
        self.inference_mode = bool(inference_mode)
        # 动态答案长度标记
        self.dynamic_answer_length = bool(dynamic_answer_length)

        # 结构映射加载
        self.only_title_keys = bool(only_title_keys)
        self.struct_map: Dict[str, List[str]] = {}
        if report_struct_path:
            try:
                self.struct_map = json.loads(
                    Path(report_struct_path).read_text(encoding="utf-8")
                )
            except Exception:
                self.struct_map = {}

        # 不再使用标题边界检测
        
        # 长度先验容器（用于动态长度合理性评分）
        self.length_priors: Dict[str, Dict[str, float]] = {}

        # 支持多文件聚合
        agg_records: List[Dict] = []
        if data_paths:
            for p in list(data_paths):
                try:
                    r = _load_jsonl_or_json(p)
                    if isinstance(r, list):
                        agg_records.extend(r)
                except Exception as e:
                    if self.show_progress:
                        print(f"[WARN] failed to load {p}: {e}")
        else:
            raw = _load_jsonl_or_json(data_path)
            agg_records = raw if isinstance(raw, list) else []

        self.records: List[Dict] = agg_records
        self.samples: List[Dict] = []

        if autobuild and self.records:
            if self.show_progress:
                print(f"[INFO] Loaded {len(self.records)} reports from {data_path}")
                print("[INFO] Collecting length priors ...")
            # 先收集长度先验（基于训练数据统计）
            self._collect_length_priors()
            if self.show_progress:
                print("[INFO] Building samples ...")
            self.samples = self._build_samples(
                use_concurrent=self.use_concurrent_build,
                max_workers=self.max_workers
            )

    # ---------- Dataset ----------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        out = {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "token_type_ids": torch.tensor(item["token_type_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "start_positions": torch.tensor(item["start_positions"], dtype=torch.long),
            "end_positions": torch.tensor(item["end_positions"], dtype=torch.long),
        }
        if "noise_ids" in item:
            out["noise_ids"] = item.get("noise_ids")
        if self.keep_debug_fields:
            for k in (
                "question_key",
                "chunk_index",
                "offset_mapping",
                "sequence_ids",
                "chunk_char_start",
                "chunk_char_end",
                "report_index",
                "chunk_text",
            ):
                if k in item:
                    out[k] = item[k]
        return out

    # ---------- Tokenizer ----------
    def _get_tok(self):
        if getattr(self, "_user_tok", None) is not None:
            return self._user_tok
        # 优先顺序：用户传入 > 环境变量 > model_path_conf 默认路径
        default_from_conf = None
        try:
            import model_path_conf as _mpc  # type: ignore

            default_from_conf = getattr(
                _mpc, "DEFAULT_TOKENIZER_PATH", getattr(_mpc, "DEFAULT_MODEL_PATH", None)
            )
        except Exception:
            default_from_conf = None

        base_name = (
            self._tokenizer_name
            or os.environ.get("HF_TOKENIZER_NAME")
            or default_from_conf
        )
        if not base_name:
            raise RuntimeError(
                "Tokenizer path not set. Please configure HF_TOKENIZER_NAME or model_path_conf.DEFAULT_TOKENIZER_PATH."
            )
        tok = HFTokenizer.from_pretrained(base_name)
        try:
            tok.model_max_length = int(1e6)
        except Exception:
            pass
        self._user_tok = tok
        return tok
    
    # ---------- 缓存优化 ----------
    @lru_cache(maxsize=50000)
    def _cached_tokenize_len(self, text: str) -> int:
        """缓存tokenize长度计算，避免重复计算"""
        try:
            return len(self._get_tok().tokenize(text))
        except Exception:
            return 0
    
    def _collect_length_priors(self):
        """收集各字段的长度先验统计（p25/p50/p75/sigma），用于稳健的长度合理性评分"""
        from collections import defaultdict
        tok = self._get_tok()
        vals = defaultdict(list)
        
        for rec in self.records:
            keys = self._question_keys_for(rec)
            for k in keys:
                v = str(rec.get(k, "") or "").strip()
                if not v:
                    continue
                try:
                    L = len(tok.encode(v, add_special_tokens=False))
                except Exception:
                    continue
                if L > 0:
                    vals[k].append(L)
        
        priors = {}
        for k, arr in vals.items():
            if len(arr) < 3:  # 样本太少，跳过
                continue
            arr = sorted(arr)
            n = len(arr)
            p25 = arr[int(0.25 * (n - 1))]
            p50 = arr[int(0.50 * (n - 1))]
            p75 = arr[int(0.75 * (n - 1))]
            iqr = max(1.0, float(p75 - p25))
            # 以 IQR 估计稳健方差尺度，近似 σ ≈ IQR / 1.349
            sigma = max(1.0, iqr / 1.349)
            priors[k] = {
                "p25": float(p25),
                "p50": float(p50),
                "p75": float(p75),
                "sigma": float(sigma)
            }
        
        self.length_priors = priors
        if self.show_progress and priors:
            print(f"[INFO] Collected length priors for {len(priors)} keys")
    
    def _length_reasonableness_score(self, key: str, actual_len: int, expected_len: int, is_short: bool) -> float:
        """计算长度合理性分数（0.1-1.0），基于先验统计或平滑函数"""
        import math
        
        # 先尝试 key 先验（优先）
        p = self.length_priors.get(key)
        if p is not None:
            mu = p["p50"]
            sigma = p["sigma"]
            # 短字段可适当收紧（让 σ 更小一些），长字段保持原样
            if is_short:
                sigma = max(1.0, 0.8 * sigma)
            z = abs(float(actual_len) - float(mu)) / (float(sigma) + 1e-6)
            score = math.exp(-(z**2) / 2.0)  # 高斯衰减
        else:
            # 无先验：用"相对误差"的平滑函数（Cauchy-like）
            rel = abs(float(actual_len) - float(expected_len)) / max(1.0, float(expected_len))
            tau = 0.30 if is_short else 0.75  # 短字段更严格
            score = 1.0 / (1.0 + (rel / tau) ** 2)
        
        # 夹紧到 [0.1, 1.0]，避免 0 导致训练时无限加权
        return float(min(1.0, max(0.1, score)))
    
    def _batch_encode_pairs(self, question_chunk_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """批量编码问题-文本对，显著提升性能"""
        if not question_chunk_pairs:
            return []
        
        tok = self._get_tok()
        
        # 分离问题和文本
        questions = [pair[0] for pair in question_chunk_pairs]
        chunks = [pair[1] for pair in question_chunk_pairs]
        
        # 批量编码 - 这比逐个编码快很多
        batch_size = 64  # 增大批次
        results = []
        
        for i in range(0, len(questions), batch_size):
            q_batch = questions[i:i+batch_size]
            c_batch = chunks[i:i+batch_size]
            
            # 批量调用tokenizer
            try:
                encodings = tok(
                    q_batch, c_batch,
                    max_length=self.max_seq_len,
                    truncation="only_second",
                    return_offsets_mapping=True,
                    return_tensors=None,
                    padding=False,  # 不padding，保持原始长度
                )
                
                # 健壮性检查：如果编码数量不一致，整体回退到逐个编码
                if len(encodings.encodings) != len(q_batch):
                    # 某些条目解析失败，整体回退到逐个编码，保证一一对应
                    for question, chunk_text in zip(q_batch, c_batch):
                        results.append(self._encode_pair(question, chunk_text))
                    continue
                
                # 处理批量结果
                for j, (question, chunk_text) in enumerate(zip(q_batch, c_batch)):
                    if j < len(encodings.encodings):
                        enc0 = encodings.encodings[j]
                        
                        # sequence_ids
                        tmp = getattr(enc0, "sequence_ids", None)
                        seq_ids = (
                            tmp() if callable(tmp) else 
                            (tmp if tmp is not None else [None] * len(enc0.ids))
                        )
                        
                        # 防截断检查
                        raw_offsets = enc0.offsets
                        ctx_idx_batch = [i for i, s in enumerate(seq_ids) if s == 1]
                        if ctx_idx_batch and (raw_offsets[ctx_idx_batch[-1]][0] is None or raw_offsets[ctx_idx_batch[-1]][1] is None):
                            # 上下文被截断，回退到单个编码（会抛出异常被上层处理）
                            results.append(self._encode_pair(question, chunk_text))
                            continue
                        
                        # 像 _encode_pair 一样，把非上下文（sequence_ids != 1）的 offset_mapping 置为 (None, None)
                        offset_mapping: List[Tuple[Optional[int], Optional[int]]] = []
                        for k, (s, e) in enumerate(raw_offsets):
                            if seq_ids[k] == 1:
                                offset_mapping.append((int(s), int(e)))
                            else:
                                offset_mapping.append((None, None))
                        
                        feat = {
                            "input_ids": list(enc0.ids),
                            "token_type_ids": list(enc0.type_ids),
                            "attention_mask": [1] * len(enc0.ids),
                            "offset_mapping": offset_mapping,
                            "sequence_ids": list(seq_ids),
                        }
                        
                        results.append(feat)
                    else:
                        # 回退到单个编码
                        results.append(self._encode_pair(question, chunk_text))
                        
            except Exception:
                # 批量失败时回退到逐个编码
                for question, chunk_text in zip(q_batch, c_batch):
                    results.append(self._encode_pair(question, chunk_text))
        
        return results



    # ---------- 文本取回 ----------
    @staticmethod
    def _get_report_text(rec: Dict) -> str:
        rep = rec.get("report") or rec.get("text") or rec.get("ocr_text")
        if isinstance(rep, str) and rep.strip():
            text = rep.replace("\r\n", "\n").replace("\r", "\n")
            # 文本归一化增强：替换特殊空白字符，保持与 tokenizer offset 对齐
            text = text.replace("\u00A0", " ").replace("\u3000", " ")
            # 统一冒号为全角，确保与 extract_spans 的 choose_sep="：" 匹配
            text = text.replace(":", "：")
            return text
        else:
            # 空 report 或非字符串：返回空字符串（上层会跳过）
            return ""

    # ---------- 所有key ----------
    def _question_keys_for(self, rec: Dict) -> List[str]:
        '''确定要提问的键'''
        # Raw KV-NER style records often use `category` as the report title.
        title = str(
            rec.get("report_title")
            or rec.get("report_titles")
            or rec.get("category")
            or rec.get("title")
            or ""
        ).strip()
        title_keys = []
        if title and (title in self.struct_map):
            raw = self.struct_map.get(title) or []
            tmp_keys: List[str] = []
            for it in raw:
                if isinstance(it, str) and it.strip():
                    tmp_keys.append(it.strip())
                elif isinstance(it, dict):
                    for f in ("name", "key", "title"):
                        v = str(it.get(f, "")).strip()
                        if v:
                            tmp_keys.append(v)
            # 去重保序
            seen = set()
            title_keys = [k for k in tmp_keys if not (k in seen or seen.add(k))]

        # 过滤原文及 report_composed / 元字段
        record_keys = []
        meta_keys = {
            "record_id",
            "id",
            "relative_image_path",
            "category",
            "report",
            "text",
            "ocr_text",
            "ocr_raw",
            "transferred_annotations",
            "annotations",
            "relations",
            "noise_values",
            "noise_values_per_word",
            "meta",
            "report_composed",
        }
        for k in rec.keys():
            sk = str(k).strip()
            if not sk:
                continue
            if k in {"report_title", "report_titles"}:
                continue
            if k in meta_keys:
                continue
            record_keys.append(k)

        if self.only_title_keys and title_keys:
            keys = title_keys
        else:
            # 结构映射键 ∪ 记录键（结构优先保证顺序）
            keys = _dedup_keep_order(list(title_keys) + list(record_keys))
        return keys

    # 编码（返回 offset/seq ids）
    def _encode_pair(self, question: str, chunk_text: str) -> Dict:
        tok = self._get_tok()
        enc = tok(
            question,
            chunk_text,
            max_length=self.max_seq_len,
            truncation="only_second",
            return_offsets_mapping=True,
            return_tensors=None,
        )
        enc0 = enc.encodings[0]
        ids_list = list(enc0.ids)
        type_ids = list(enc0.type_ids)
        attn_mask = [1] * len(ids_list)
        tmp = getattr(enc0, "sequence_ids", None)
        seq_ids = (
            tmp()
            if callable(tmp)
            else (tmp if tmp is not None else [None] * len(enc0.ids))
        )
        raw_offsets = enc0.offsets
        
        # 防截断检查：如果上下文最后一个 token 的 offset 为 None，说明被截断了
        ctx_idx = [i for i, s in enumerate(seq_ids) if s == 1]
        if ctx_idx and (raw_offsets[ctx_idx[-1]][0] is None or raw_offsets[ctx_idx[-1]][1] is None):
            # 上下文被截断，抛出异常让上层重新切块
            raise RuntimeError("Context truncated: re-chunk with smaller budget.")
        
        offset_mapping: List[Tuple[Optional[int], Optional[int]]] = []
        for i, (s, e) in enumerate(raw_offsets):
            if seq_ids[i] == 1:
                offset_mapping.append((int(s), int(e)))
            else:
                offset_mapping.append((None, None))
        return {
            "input_ids": ids_list,
            "token_type_ids": type_ids,
            "attention_mask": attn_mask,
            "offset_mapping": offset_mapping,
            "sequence_ids": list(seq_ids),
        }

    # char→token（严格化映射，避免 end 偏大）
    def _char_span_to_token_span(
        self,
        offset_mapping: List[Tuple[Optional[int], Optional[int]]],
        sequence_ids: List[Optional[int]],
        char_start: int,
        char_end: int,
    ) -> Tuple[Optional[int], Optional[int]]:
        ctx = [i for i, sid in enumerate(sequence_ids) if sid == 1]
        if not ctx:
            return (None, None)
        
        # start: 优先选"包含 char_start 的 token"，更精确地复原 gold span
        s_tok: Optional[int] = None
        for i in ctx:
            s, e = offset_mapping[i]
            if s is None or e is None:
                continue
            if s <= char_start < e:
                s_tok = i
                break
        # 备用：第一个 s >= char_start（处理 char_start 恰好在边界的情况）
        if s_tok is None:
            for i in ctx:
                s, e = offset_mapping[i]
                if s is None or e is None:
                    continue
                if s >= char_start:
                    s_tok = i
                    break
        if s_tok is None:
            return (None, None)
        
        # end：更严格的选择，避免吃到下个 token
        e_tok: Optional[int] = None
        # 1) 优先：最后一个 offset_end <= char_end（不越界）
        for i in reversed(ctx):
            s, e = offset_mapping[i]
            if s is None or e is None:
                continue
            if i < s_tok:
                break
            if e <= char_end:
                e_tok = i
                break
        # 2) 若没有，找覆盖 char_end-1 的 token（s < char_end 且 e >= char_end）
        if e_tok is None:
            for i in reversed(ctx):
                s, e = offset_mapping[i]
                if s is None or e is None:
                    continue
                if i < s_tok:
                    break
                if s < char_end and e >= char_end:
                    e_tok = i
                    break
        # 3) 兜底：最后一个 s < char_end
        if e_tok is None:
            for i in reversed(ctx):
                s, e = offset_mapping[i]
                if s is None or e is None:
                    continue
                if i < s_tok:
                    break
                if s < char_end:
                    e_tok = i
                    break
        
        if e_tok is None or e_tok < s_tok:
            return (None, None)
        
        return (s_tok, e_tok)

    # 仅 value→char s/e 抽取
    def _extract_spans_from_report(
        self,
        report: str,
        keys: List[str],
        expected_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Tuple[int, int]]:
        # 过滤空值
        exp = {
            k: v
            for k, v in (expected_map or {}).items()
            if isinstance(v, str) and v.strip()
        }
        return EX.extract_spans(
            report=report,
            keys=keys,
            title="",
            alias_map={},
            choose_sep=lambda: "：",
            validate_spans=False,
            colon_only=True,
            expected_map=exp,
            title_only_alias=True,
        )

    # ---------- 每条报告构建 ----------
    def _build_one_report(self, ridx: int, rec: Dict) -> List[Dict]:
        out: List[Dict] = []
        report = self._get_report_text(rec)
        if not report:
            if ridx < 5: print(f"[DEBUG] ridx={ridx} NO REPORT TEXT found.")
            return out

        tok = self._get_tok()
        chunker = SemanticChunker(
            tok, max_tokens_ctx=self.max_tokens_ctx, chunk_mode=self.chunk_mode
        )

        # 发问键集合：结构映射优先 / 或仅结构
        keys = self._question_keys_for(rec)
        if ridx < 5:
             t = rec.get("report_title", "N/A")
             print(f"[DEBUG] ridx={ridx} title='{t}' keys#={len(keys)} Keys[:3]={keys[:3]}")

        if not keys:  # 早期返回，避免无效处理
            return []

        # 训练时：使用记录中的值作为 gold spans
        # 推理时：让模型自主发现（不依赖预设值），除非记录确实提供了值
        if self.inference_mode:
            # 推理模式：不提供 expected_map，让 extract_spans 返回 (-1,-1)，
            # 这样所有样本都是 no-answer，模型会自主预测
            expected_map = {}
        else:
            # 训练模式：使用记录中的值
            expected_map = {k: str(rec.get(k, "") or "").strip() for k in keys}
        spans = self._extract_spans_from_report(report, keys, expected_map=expected_map)

        rng = random.Random(self._base_seed + int(ridx))

        # 批量生成并将问题 token 长度作为字典缓存
        report_title = rec.get("report_title", "")
        if self.use_question_templates:
            questions_batch = [convert_key_to_question(report_title, key) for key in keys]
        else:
            raise ValueError("use_question_templates must be True")
        
        q_len_map = {key: self._cached_tokenize_len(q) for key, q in zip(keys, questions_batch)}
        q_text_map = {key: q for key, q in zip(keys, questions_batch)}

        # 预计算属性
        val_short_map = {}
        for k in keys:
            v = str(rec.get(k, "") or "").strip()
            val_short_map[k] = EnhancedQADataset._is_short_field(v, tok) if v else False

        # 提前获取全文分块（用于无答案时的滑动窗口，避免在循环内重复调用 line_spans）
        report_lines = chunker.line_spans(report)
        
        # 预计算“无答案候选集”以供复用
        cached_neg_candidates = None
        if report_lines and len(report_lines) > 3:
            cached_neg_candidates = []
            window_size = min(5, max(3, len(report_lines) // 4))
            step = max(1, window_size // 2)
            for i in range(0, len(report_lines), step):
                end_idx = min(i + window_size, len(report_lines))
                s0, e0 = report_lines[i]["start"], report_lines[end_idx-1]["end"]
                cached_neg_candidates.append({"text": report[s0:e0], "char_start": s0, "char_end": e0})
        else:
            cached_neg_candidates = [{"text": report, "char_start": 0, "char_end": len(report)}]

        for key in keys:
            question = q_text_map[key]
            q_tok = q_len_map[key]
            
            # 该 key 的 gold char 范围
            s_abs, e_abs = spans.get(key, (-1, -1))
            has_answer = (s_abs >= 0 and e_abs >= 0 and s_abs < e_abs)
            
            # --- 核心优化 1：提前进行负样本下采样 ---
            # 如果该 key 没答案，它就是一个负样本。我们直接在这里摇号，不中的直接 continue。
            if not has_answer:
                if self.inference_mode:
                    pass # 推理不采样
                else:
                    field_value = str(rec.get(key, "")).strip()
                    field_exists = bool(field_value and field_value != "")
                    
                    if not field_exists:
                        keep_prob = self.negative_downsample * 0.2
                    else:
                        keep_prob = self.negative_downsample
                    
                    if rng.random() > keep_prob:
                        continue # 命中了下采样，直接跳过当前 key 的所有昂贵计算

            # 上下文预算
            ctx_budget = max(1, min(self.max_tokens_ctx, self.max_seq_len - q_tok - 3))

            # 确定候选文本块
            if has_answer:
                if report_lines:
                    L = next((i for i, sp in enumerate(report_lines) if sp["start"] <= s_abs < sp["end"]), 0)
                    R = next((i for i, sp in enumerate(report_lines) if sp["start"] < e_abs <= sp["end"]), len(report_lines) - 1)
                    R2 = min(R + 1, len(report_lines) - 1)
                    s0, e0 = report_lines[L]["start"], report_lines[R2]["end"]
                    candidate = [{"text": report[s0:e0], "char_start": s0, "char_end": e0}]
                else:
                    candidate = [{"text": report, "char_start": 0, "char_end": len(report)}]
            else:
                candidate = cached_neg_candidates

            # 确定最终 chunks
            chunks: List[Dict] = []
            for cand in candidate:
                # 使用缓存的长度计算（如果 candidate 就是全文，可能已经缓存过）
                if self._cached_tokenize_len(cand["text"]) > ctx_budget:
                    if self.inference_mode:
                        # 推理模式：使用最简单的按行切
                        sub_lines = chunker.line_spans(cand["text"])
                        for sl in sub_lines:
                            if sl["start"] < sl["end"]:
                                chunks.append({
                                    "text": cand["text"][sl["start"]:sl["end"]],
                                    "char_start": cand["char_start"] + sl["start"],
                                    "char_end": cand["char_start"] + sl["end"]
                                })
                    else:
                        # 训练模式：使用预算切分
                        sub = chunker.split(cand["text"], budget_tokens=ctx_budget)
                        for ch in sub:
                            ch["char_start"] += cand["char_start"]
                            ch["char_end"] += cand["char_start"]
                        chunks.extend(sub)
                else:
                    chunks.append(cand)

            # 逐 chunk 编码并生成样本
            for ci, ch in enumerate(chunks):
                try:
                    feat = self._encode_pair(question, ch["text"])
                except RuntimeError as e:
                    if "truncated" in str(e).lower():
                        continue # 极个别截断情况跳过即可，不尝试细切以换取速度
                    raise

                start_pos, end_pos = (0, 0)
                length_reasonableness = 1.0
                
                if has_answer:
                    # 复用之前的 s_abs, e_abs
                    ts_abs, te_abs = s_abs, e_abs
                    # 收紧 gold span
                    while ts_abs < te_abs and report[ts_abs].isspace(): ts_abs += 1
                    while te_abs > ts_abs and report[te_abs - 1].isspace(): te_abs -= 1
                    while te_abs > ts_abs and report[te_abs - 1] in "，。,；;、:：)]）】>》": te_abs -= 1
                    
                    if ts_abs >= ch["char_start"] and te_abs <= ch["char_end"] and te_abs > ts_abs:
                        s_rel = ts_abs - ch["char_start"]
                        e_rel = te_abs - ch["char_start"]
                        sp, ep = self._char_span_to_token_span(feat["offset_mapping"], feat["sequence_ids"], s_rel, e_rel)
                        if sp is not None and ep is not None:
                            start_pos, end_pos = sp, ep
                            if self.dynamic_answer_length:
                                field_value = str(rec.get(key, "")).strip()
                                expected_len = self._cached_tokenize_len(field_value) if field_value else (ep - sp + 1)
                                length_reasonableness = self._length_reasonableness_score(
                                    key=key, actual_len=(ep - sp + 1), expected_len=expected_len, is_short=val_short_map[key]
                                )

                # 创建样本
                sample = {
                    "input_ids": feat["input_ids"],
                    "token_type_ids": feat["token_type_ids"],
                    "attention_mask": feat["attention_mask"],
                    "start_positions": start_pos,
                    "end_positions": end_pos,
                    "question_key": key,
                    "chunk_index": ci,
                    "report_index": ridx,
                    "is_short_field": val_short_map[key],
                    "length_reasonableness": length_reasonableness,
                }
                
                if self.keep_debug_fields:
                    sample.update({
                        "offset_mapping": feat["offset_mapping"],
                        "sequence_ids": feat["sequence_ids"],
                        "chunk_char_start": ch["char_start"],
                        "chunk_text": ch["text"],
                    })
                out.append(sample)
        return out
        return out

    # 构建全部样本 - 支持并发优化
    def _build_samples(self, use_concurrent: bool = True, max_workers: Optional[int] = None) -> List[Dict]:
        if not use_concurrent or len(self.records) < 10:
            # 小数据集或禁用并发时使用原始串行方法
            return self._build_samples_serial()
        
        # 并发处理（利用 32 核机器，增加线程数）
        if max_workers is None:
            max_workers = min(16, cpu_count(), len(self.records))  # 增加到 16 线程，更适合 4090 机器
        
        samples: List[Dict] = []
        
        # 简化：直接准备 (ridx, rec) 对
        process_args = [(ridx, rec) for ridx, rec in enumerate(self.records)]
        
        if self.show_progress:
            print(f"[INFO] Building samples with {max_workers} workers...")
        
        # 使用线程池（避免tokenizer的pickle问题）
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if self.show_progress:
                futures = []
                with tqdm(total=len(process_args), desc="Building samples") as pbar:
                    for ridx, rec in process_args:
                        future = executor.submit(self._build_one_report, ridx, rec)
                        futures.append(future)
                    
                    # 使用 as_completed 让进度条不再按顺序阻塞
                    from concurrent.futures import as_completed
                    for future in as_completed(futures):
                        result = future.result()
                        samples.extend(result)
                        pbar.update(1)
            else:
                futures = [executor.submit(self._build_one_report, ridx, rec) for ridx, rec in process_args]
                for future in futures:
                    result = future.result()
                    samples.extend(result)
        
        if self.show_progress:
            from collections import Counter
            n_total = len(samples)
            pos_n = sum(1 for s in samples if int(s.get("start_positions", 0)) != 0 or int(s.get("end_positions", 0)) != 0)
            neg_n = n_total - pos_n
            key_cnt = Counter(str(s.get("question_key", "")) for s in samples)
            top_keys = key_cnt.most_common(10)
            print(f"[STATS] samples={n_total} pos={pos_n} ({(pos_n/max(1,n_total))*100:.1f}%) neg={neg_n}")
            print("[STATS] top question_key:", top_keys)
        
        return samples
    
    # 串行构建方法（原始实现）
    def _build_samples_serial(self) -> List[Dict]:
        samples: List[Dict] = []
        it = (
            tqdm(
                self.records,
                total=len(self.records),
                desc="Building samples",
                dynamic_ncols=True,
            )
            if self.show_progress
            else self.records
        )
        for ridx, rec in enumerate(it):
            samples.extend(self._build_one_report(ridx, rec))
        if self.show_progress:
            from collections import Counter
            n_total = len(samples)
            pos_n = sum(1 for s in samples if int(s.get("start_positions", 0)) != 0 or int(s.get("end_positions", 0)) != 0)
            neg_n = n_total - pos_n
            key_cnt = Counter(str(s.get("question_key", "")) for s in samples)
            top_keys = key_cnt.most_common(10)
            print(f"[STATS] samples={n_total} pos={pos_n} ({(pos_n/max(1,n_total))*100:.1f}%) neg={neg_n}")
            print("[STATS] top question_key:", top_keys)
        return samples
    

    # 导出：按当前抽取逻辑导出测试跨度
    @classmethod
    def export_test_spans(
        cls,
        data_path: str,
        out_path: str,
        show_progress: bool = True,
        report_struct_path: Optional[str] = None,
        only_title_keys: bool = True,
        data_paths: Optional[List[str]] = None,
    ) -> None:
        # 聚合读取
        if data_paths:
            records: List[Dict] = []
            for p in list(data_paths):
                try:
                    r = _load_jsonl_or_json(p)
                    if isinstance(r, list):
                        records.extend(r)
                except Exception:
                    continue
        else:
            records: List[Dict] = _load_jsonl_or_json(data_path)

        # 为导出构建一个临时实例，用于：结构映射、取键集、value→char 抽取
        tok_name = os.environ.get("HF_TOKENIZER_NAME")
        if not tok_name:
            try:
                import model_path_conf as _mpc  # type: ignore

                tok_name = getattr(
                    _mpc, "DEFAULT_TOKENIZER_PATH", getattr(_mpc, "DEFAULT_MODEL_PATH", None)
                )
            except Exception:
                tok_name = None
        if not tok_name:
            raise RuntimeError(
                "Tokenizer path not set. Configure HF_TOKENIZER_NAME or model_path_conf.DEFAULT_TOKENIZER_PATH."
            )
        tmp = cls(
            data_path=(data_path or ""),
            data_paths=data_paths,
            tokenizer_name=tok_name,
            autobuild=False,
            show_progress=False,
            report_struct_path=report_struct_path,
            only_title_keys=only_title_keys,
        )

        results: List[Dict] = []
        iterator = (
            tqdm(
                records,
                total=len(records),
                desc="Export test spans",
                dynamic_ncols=True,
            )
            if show_progress
            else records
        )

        for ridx, rec in enumerate(iterator):
            report = cls._get_report_text(rec)
            keys = tmp._question_keys_for(rec)
            expected_map = {k: str(rec.get(k, "") or "").strip() for k in keys}
            spans = tmp._extract_spans_from_report(
                report, keys, expected_map=expected_map
            )

            per_key: Dict[str, Dict[str, Any]] = {}
            for k in keys:
                s, e = spans.get(k, (-1, -1))
                if s >= 0 and e >= 0 and s < e:
                    per_key[k] = {"start": s, "end": e, "text": report[s:e]}
                else:
                    # 结构里有但记录未出现（或未命中） -> 明确写空
                    per_key[k] = {"start": -1, "end": -1, "text": ""}
            results.append(
                {
                    "report_index": ridx,
                    "report_title": str(rec.get("report_title", "")),
                    "spans": per_key,
                    "report": report,
                }
            )

        _save_jsonl(results, out_path)
        if show_progress:
            print(f"[OK] Exported test spans: {len(results)} reports -> {out_path}")


# 模块级导出包装（保持兼容；可携带结构映射参数）
def export_test_spans(
    data_path: str,
    out_path: str,
    show_progress: bool = True,
    report_struct_path: Optional[str] = None,
    only_title_keys: bool = True,
) -> None:
    return EnhancedQADataset.export_test_spans(
        data_path=data_path,
        out_path=out_path,
        show_progress=show_progress,
        report_struct_path=report_struct_path,
        only_title_keys=only_title_keys,
    )


if __name__ == "__main__":
    import sys
    import json
    import os
    
    # ===== 简洁预计算脚本：支持输入多个 JSON 文件，合并生成到一个 JSONL =====
    
    print("📋 使用方法:")
    print("  python -m pre_struct.ebqa.da_core.dataset <input_json1> [input_json2 ...] <output_jsonl>")
    print("  单文件: python -m pre_struct.ebqa.da_core.dataset data/a.json data/output.jsonl")
    print("  多文件: python -m pre_struct.ebqa.da_core.dataset data/a.json data/b.json data/c.json data/output.jsonl")
    print()
    
    if len(sys.argv) < 3:
        print("❌ 参数不足，请提供至少 1 个输入 JSON 和 1 个输出 JSONL 路径")
        sys.exit(1)
    
    # 最后一个参数是输出路径，前面都是输入文件
    INPUT_JSONS = sys.argv[1:-1]
    OUTPUT_JSONL = sys.argv[-1]
    
    # 验证输入文件存在
    missing_files = [f for f in INPUT_JSONS if not os.path.exists(f)]
    if missing_files:
        print(f"❌ 以下输入文件不存在:")
        for f in missing_files:
            print(f"   - {f}")
        sys.exit(1)
    
    print(f"=== 预计算数据集样本（多文件合并） ===")
    print(f"输入文件数: {len(INPUT_JSONS)}")
    for i, f in enumerate(INPUT_JSONS, 1):
        print(f"  [{i}] {f}")
    print(f"输出: {OUTPUT_JSONL}")
    print()

    # 从环境变量或配置获取 tokenizer 路径
    tokenizer_path = os.environ.get("HF_TOKENIZER_NAME")
    if not tokenizer_path:
        try:
            import model_path_conf as _mpc
            tokenizer_path = getattr(_mpc, "DEFAULT_TOKENIZER_PATH", 
                                    getattr(_mpc, "DEFAULT_MODEL_PATH", None))
        except Exception:
            tokenizer_path = None
    if not tokenizer_path:
        raise RuntimeError(
            "Tokenizer path not set. Configure HF_TOKENIZER_NAME or model_path_conf.DEFAULT_TOKENIZER_PATH."
        )
    
    # 构建数据集（使用 data_paths 参数聚合多个文件）
    ds = EnhancedQADataset(
        data_path="",  # 占位，实际使用 data_paths
        data_paths=INPUT_JSONS,  # 多文件聚合
        tokenizer_name=tokenizer_path,
        max_seq_len=512,
        max_tokens_ctx=500,
        max_answer_len=2000,
        use_question_templates=True,
        keep_debug_fields=True,
        report_struct_path="keys/keys_merged.json",
        only_title_keys=True,
        inference_mode=False,
        dynamic_answer_length=True,
        negative_downsample=0.2,
        chunk_mode="budget",
        seed=42,
        autobuild=True,
        show_progress=True,
    )

    # 保存预计算样本
    _save_jsonl(ds.samples, OUTPUT_JSONL)

    print(f"\n✅ 预计算完成: {OUTPUT_JSONL}")
    print(f"输入记录总数: {len(ds.records)}")
    print(f"输出样本总数: {len(ds.samples)}")

    # 统计正负比例
    pos_count = sum(1 for s in ds.samples if s.get('start_positions', 0) != 0 or s.get('end_positions', 0) != 0)
    neg_count = len(ds.samples) - pos_count
    pos_ratio = pos_count / len(ds.samples) * 100 if ds.samples else 0
    neg_ratio = neg_count / len(ds.samples) * 100 if ds.samples else 0

    print(f"正样本: {pos_count} ({pos_ratio:.1f}%)")
    print(f"负样本: {neg_count} ({neg_ratio:.1f}%)")

    # 检查调试字段
    has_debug_fields = any('chunk_text' in s for s in ds.samples[:100])
    print(f"调试字段就绪: {'✅' if has_debug_fields else '❌'}")
    print(f"\n🎉 数据生成完成！")
