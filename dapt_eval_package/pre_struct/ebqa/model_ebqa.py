# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    BertForQuestionAnswering,
    BertTokenizerFast,
    TrainingArguments,
    Trainer,
)
try:  # pragma: no cover
    from noise_fusion import ContinuousNoiseProjector, build_feature_ranges, uses_bucket_noise, uses_continuous_noise
except Exception:  # pragma: no cover
    import sys
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from noise_fusion import ContinuousNoiseProjector, build_feature_ranges, uses_bucket_noise, uses_continuous_noise

# noise feature meta（7维，与 kv_ner/noise_utils 一致）
try:  # pragma: no cover - optional dependency
    from pre_struct.kv_ner.noise_utils import NUM_BINS as _NOISE_NUM_BINS, FEATURES as _NOISE_FEATURES  # type: ignore
except Exception:  # pragma: no cover
    try:
        from .kv_ner.noise_utils import NUM_BINS as _NOISE_NUM_BINS, FEATURES as _NOISE_FEATURES  # type: ignore
    except Exception:
        _NOISE_FEATURES = [
            "conf_avg",
            "conf_min",
            "conf_var_log",
            "conf_gap",
            "punct_err_ratio",
            "char_break_ratio",
            "align_score",
        ]
        _NOISE_NUM_BINS = {
            "conf_avg": 64,
            "conf_min": 64,
            "conf_var_log": 32,
            "conf_gap": 32,
            "punct_err_ratio": 16,
            "char_break_ratio": 32,
            "align_score": 64,
        }

try:  # optional project-local defaults
    import model_path_conf as _mpc  # type: ignore
    _DEF_MODEL = getattr(_mpc, "DEFAULT_MODEL_PATH", None)
    _DEF_TOK = getattr(_mpc, "DEFAULT_TOKENIZER_PATH", _DEF_MODEL)
except Exception:
    _DEF_MODEL = None
    _DEF_TOK = None

# tighten predicted char spans (trim surrounding whitespaces)
try:
    from .da_core.utils import _tighten_span  
except Exception:  # pragma: no cover
    try:
        from pre_struct.ebqa.da_core.utils import _tighten_span  
    except Exception:
        _tighten_span = None  


# -----------------------------
# tokenizer path helpers
# -----------------------------
def _has_tokenizer_files(path: Optional[str]) -> bool:
    if not path:
        return False
    try:
        if not os.path.isdir(path):
            return False
    except Exception:
        return False
    candidates = [
        "tokenizer.json",
        "vocab.txt",
        "spiece.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    return any(os.path.isfile(os.path.join(path, f)) for f in candidates)


def _resolve_tokenizer_path(
    model_dir_or_id: str, tokenizer_name_or_path: Optional[str]
) -> str:
    """Pick tokenizer: explicit > model dir > saved train_config > fallback."""
    if tokenizer_name_or_path and str(tokenizer_name_or_path).strip():
        return str(tokenizer_name_or_path).strip()
    if _has_tokenizer_files(model_dir_or_id):
        return model_dir_or_id
    try:
        tc = os.path.join(model_dir_or_id, "train_config.json")
        if os.path.isfile(tc):
            meta = json.load(open(tc, "r", encoding="utf-8"))
            cand = (
                meta.get("tokenizer_name")
                or meta.get("tokenizer_name_or_path")
                or meta.get("model_name_or_path")
            )
            if isinstance(cand, str) and cand.strip():
                return cand.strip()
    except Exception:
        pass
    if _DEF_TOK:
        return _DEF_TOK
    raise RuntimeError(
        "Tokenizer path not provided. Please configure model_path_conf.DEFAULT_TOKENIZER_PATH."
    )


def _supports_noise_forward(model: Any) -> bool:
    """Return True if model.forward accepts noise inputs."""
    import inspect

    try:
        params = inspect.signature(model.forward).parameters
        return ("noise_ids" in params) or ("noise_values" in params)
    except Exception:
        return False


class NoiseAwareBertForQuestionAnswering(BertForQuestionAnswering):
    """BERT QA head with optional noise_ids fusion (7-d discrete bins).

    - Each noise dimension has its own embedding; concatenated then projected to hidden size.
    - Added residually to encoder output before QA head.
    """

    def __init__(self, config, use_noise: bool = True, noise_embed_dim: int = 16):
        super().__init__(config)
        self.use_noise = bool(getattr(config, "use_noise", use_noise))
        self.noise_embed_dim = int(getattr(config, "noise_embed_dim", noise_embed_dim))
        self.noise_mode = str(getattr(config, "noise_mode", "bucket") or "bucket").lower()
        self.noise_mlp_hidden_dim = int(getattr(config, "noise_mlp_hidden_dim", 0) or 0) or None
        self.noise_projector = None

        if self.use_noise:
            if uses_bucket_noise(self.noise_mode):
                emb_layers = []
                for feat in _NOISE_FEATURES:
                    nbin = int(_NOISE_NUM_BINS.get(feat, 0))
                    emb_layers.append(nn.Embedding(num_embeddings=nbin + 1, embedding_dim=self.noise_embed_dim))
                self.noise_embeddings = nn.ModuleList(emb_layers)
                self.noise_proj = nn.Linear(len(_NOISE_FEATURES) * self.noise_embed_dim, config.hidden_size)
                self.noise_dropout = nn.Dropout(config.hidden_dropout_prob)
            elif uses_continuous_noise(self.noise_mode):
                self.noise_embeddings = None
                self.noise_proj = None
                self.noise_dropout = None
                self.noise_projector = ContinuousNoiseProjector(
                    config.hidden_size,
                    mode=self.noise_mode,
                    dropout=config.hidden_dropout_prob,
                    mlp_hidden_dim=self.noise_mlp_hidden_dim,
                    feature_ranges=build_feature_ranges(getattr(config, "noise_bin_edges", None)),
                )
            else:
                raise ValueError(f"Unsupported noise_mode: {self.noise_mode}")
        else:
            self.noise_embeddings = None
            self.noise_proj = None
            self.noise_dropout = None

        self.config.use_noise = self.use_noise
        self.config.noise_embed_dim = self.noise_embed_dim
        self.config.noise_mode = self.noise_mode
        self.config.noise_mlp_hidden_dim = self.noise_mlp_hidden_dim

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        noise_ids=None,
        noise_values=None,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        if self.use_noise:
            if uses_bucket_noise(self.noise_mode) and (noise_ids is not None) and self.noise_embeddings is not None:
                if noise_ids.dim() == 2:
                    noise_ids = noise_ids.unsqueeze(-1)
                while noise_ids.dim() < 3:
                    noise_ids = noise_ids.unsqueeze(-1)

                noise_vecs = []
                for idx, emb in enumerate(self.noise_embeddings):
                    if noise_ids.size(-1) <= idx:
                        ids_i = noise_ids[..., -1]
                    else:
                        ids_i = noise_ids[..., idx]
                    ids_i = ids_i.clamp(min=0, max=emb.num_embeddings - 1)
                    noise_vecs.append(emb(ids_i))

                if noise_vecs:
                    noise_cat = torch.cat(noise_vecs, dim=-1)
                    noise_h = self.noise_proj(noise_cat)
                    noise_h = self.noise_dropout(noise_h)
                    sequence_output = sequence_output + noise_h
            elif uses_continuous_noise(self.noise_mode) and (noise_values is not None) and self.noise_projector is not None:
                if noise_values.dim() == 2:
                    noise_values = noise_values.unsqueeze(0)
                sequence_output = sequence_output + self.noise_projector(noise_values.to(sequence_output.device, dtype=torch.float32))

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            start_positions = start_positions.clamp(0, start_logits.size(1))
            end_positions = end_positions.clamp(0, end_logits.size(1))

            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        from transformers.modeling_outputs import QuestionAnsweringModelOutput

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


"""解码器：在单个 chunk 内找最佳 span（字符级位置）。
   使用动态长度 cap，不依赖边界信号。"""
class EBQADecoder:
    def __init__(
        self, 
        tokenizer: BertTokenizerFast, 
        max_answer_len: int = 128, 
        top_k: int = 20,
        # 短字段优化参数
        short_field_boost: float = 0.2,
        # 动态长度 cap 参数
        dyn_alpha: float = 0.90,
        short_cap: int = 6,  # 从10降到6，对中文更严格
        short_caps_by_key: Optional[Dict[str, int]] = None  # 按key自定义cap
    ):
        self.tok = tokenizer
        self.max_answer_len = int(max_answer_len)
        self.top_k = int(top_k)
        # 短字段解码优化
        self.short_field_boost = float(short_field_boost)
        # 动态长度参数
        self.dyn_alpha = float(dyn_alpha)
        self.short_cap = int(short_cap)
        self.short_caps_by_key = short_caps_by_key or {}

    @staticmethod
    
    def _ctx_indices(
        sequence_ids: List[Optional[int]],
        offset_mapping: List[Tuple[Optional[int], Optional[int]]],
    ) -> List[int]:
        """仅保留上下文段的有效 token 索引（seq_id==1 且 offset 有效）。"""
        idx = []
        for i, sid in enumerate(sequence_ids):
            if sid == 1:
                off = offset_mapping[i]
                if off is not None and off[0] is not None and off[1] is not None:
                    idx.append(i)
        return idx

    def _dynamic_cap_for_start(
        self,
        end_logits: np.ndarray,
        ctx_sel: List[int],
        s_abs: int,
        is_short: bool,
        question_key: Optional[str] = None
    ) -> int:
        """基于 end_logits 的后缀分布计算从该 start 起的最长允许长度（动态cap）。
        
        Args:
            end_logits: 所有 token 的 end logits
            ctx_sel: 可选的上下文 token 索引列表
            s_abs: 当前 start token 的绝对索引
            is_short: 是否为短字段
            question_key: 问题的key，用于获取key级的cap
            
        Returns:
            从 s_abs 开始的最大允许 span 长度（token 数）
        """
        # 找到 s_abs 在 ctx_sel 中的位置
        try:
            s_idx = ctx_sel.index(s_abs)
        except ValueError:
            # s_abs 不在 ctx_sel 中，返回默认值
            return self._get_cap_for_key(question_key) if is_short else self.max_answer_len
        
        # 获取从 s_abs 开始的后续 end logits
        tail_indices = ctx_sel[s_idx:]
        if len(tail_indices) <= 1:
            return 1
        
        # 取后续的 end logits 值
        tail_logits = end_logits[tail_indices]
        
        # 计算概率分布（softmax）
        exp_logits = np.exp(tail_logits - np.max(tail_logits))  # 防止溢出
        probs = exp_logits / np.sum(exp_logits)
        
        # 累积概率
        cumsum = np.cumsum(probs)
        
        # 找到累积概率达到 dyn_alpha 的位置
        cap_idx = np.searchsorted(cumsum, self.dyn_alpha)
        cap_idx = min(cap_idx + 1, len(tail_indices))  # +1 因为是长度，最多到全部
        
        # 短字段取更严格的上限（使用key级cap）
        if is_short:
            cap_idx = min(cap_idx, self._get_cap_for_key(question_key))
        
        # 全局兜底
        cap_idx = min(cap_idx, self.max_answer_len)
        
        return max(1, cap_idx)  # 至少长度为 1
    
    def _get_cap_for_key(self, question_key: Optional[str]) -> int:
        """获取指定key的cap值，如果key有自定义cap则使用，否则使用默认short_cap"""
        if question_key and question_key in self.short_caps_by_key:
            return min(self.short_cap, int(self.short_caps_by_key[question_key]))
        return self.short_cap

    def best_span_in_chunk(
        self,
        start_logits: np.ndarray,
        end_logits: np.ndarray,
        offset_mapping: List[Tuple[Optional[int], Optional[int]]],
        sequence_ids: List[Optional[int]],
        chunk_text: str,
        chunk_char_start: int,
        # 短字段标记
        is_short_field: bool = False,
        # key名称，用于获取key级的cap
        question_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """在上下文 token 范围搜索 best span，返回文本/位置与分数。
        使用动态长度 cap，不依赖边界信号。"""
        ctx = self._ctx_indices(sequence_ids, offset_mapping)
        if not ctx:
            return {
                "text": "",
                "score": -1e9,
                "start_char": -1,
                "end_char": -1,
                "token_start": -1,
                "token_end": -1,
                "start_logit": float("-inf"),
                "end_logit": float("-inf"),
                "null_score": (
                    float(start_logits[0] + end_logits[0])
                    if len(start_logits) > 0
                    else -1e9
                ),
            }

        # 不再使用边界限制，使用全部上下文
        ctx_sel = ctx

        s_logits = start_logits[ctx_sel]
        e_logits = end_logits[ctx_sel]

        k = min(self.top_k, len(ctx_sel))
        top_s_rel = np.argpartition(-s_logits, kth=k - 1)[:k]
        top_e_rel = np.argpartition(-e_logits, kth=k - 1)[:k]
        top_s = [ctx_sel[int(i)] for i in top_s_rel]
        top_e = [ctx_sel[int(i)] for i in top_e_rel]

        best = {
            "text": "",
            "score": -1e9,
            "start_char": -1,
            "end_char": -1,
            "token_start": -1,
            "token_end": -1,
            "start_logit": float("-inf"),
            "end_logit": float("-inf"),
            "null_score": (
                float(start_logits[0] + end_logits[0])
                if len(start_logits) > 0
                else -1e9
            ),
        }

        for s_abs in top_s:
            # 计算从当前 start 的动态 cap（使用key级cap）
            dyn_cap = self._dynamic_cap_for_start(end_logits, ctx_sel, s_abs, is_short_field, question_key)
            
            for e_abs in top_e:
                if e_abs < s_abs:
                    continue
                
                # 使用动态 cap 限制 span 长度
                span_len = e_abs - s_abs + 1
                if span_len > dyn_cap:
                    continue
                
                # 基础评分
                score = float(start_logits[s_abs] + end_logits[e_abs])
                
                # 短字段优化评分
                if is_short_field and self.short_field_boost > 0:
                    span_length = e_abs - s_abs + 1
                    # 对短字段，偏好较短的span，给予额外加分
                    if span_length <= 8:  # 8个token以内的span
                        length_bonus = self.short_field_boost * (8 - span_length) / 8
                        score += length_bonus
                
                s_char, e_char = offset_mapping[s_abs][0], offset_mapping[e_abs][1]
                if s_char is None or e_char is None:
                    continue
                if score > best["score"]:
                    # optional tighten on chunk text to trim surrounding spaces
                    s_loc, e_loc = int(s_char), int(e_char)
                    if chunk_text and _tighten_span is not None:
                        try:
                            s_loc, e_loc = _tighten_span(chunk_text, s_loc, e_loc)  
                        except Exception:
                            pass
                    
                    # 收紧后：去尾部标点/空白，再回对齐 token 边界
                    t_s, t_e = s_abs, e_abs
                    if s_loc != s_char or e_loc != e_char:
                        # 去尾部空白/标点
                        _TRIM_TAIL = set(" \t\r\n，。,；;、:：)]）】>》")
                        while e_loc > s_loc and chunk_text and chunk_text[e_loc - 1] in _TRIM_TAIL:
                            e_loc -= 1
                        
                        # 起点 token：第一个 offset_end > s_loc
                        for j in ctx_sel:
                            ss, ee = offset_mapping[j]
                            if ss is None or ee is None:
                                continue
                            if ee > s_loc:
                                t_s = j
                                break
                        
                        # 终点 token：优先覆盖 e_loc-1 的 token
                        t_e_new = None
                        for j in reversed(ctx_sel):
                            ss, ee = offset_mapping[j]
                            if ss is None or ee is None:
                                continue
                            if ss < e_loc and ee >= e_loc:
                                t_e_new = j
                                break
                        # 若没有，退而求其次：最后一个 offset_end <= e_loc
                        if t_e_new is None:
                            for j in reversed(ctx_sel):
                                ss, ee = offset_mapping[j]
                                if ss is None or ee is None:
                                    continue
                                if ee <= e_loc:
                                    t_e_new = j
                                    break
                        # 兜底：至少不小于 t_s
                        t_e = max(t_e_new if t_e_new is not None else t_e, t_s)
                    
                    best.update(
                        {
                            "score": score,
                            "token_start": int(t_s),
                            "token_end": int(t_e),
                            "start_logit": float(start_logits[t_s]),
                            "end_logit": float(end_logits[t_e]),
                            "start_char": int(chunk_char_start + s_loc),
                            "end_char": int(chunk_char_start + e_loc),
                            "text": (chunk_text[s_loc:e_loc] if chunk_text else ""),
                        }
                    )
        # 兜底：若未找到有效组合，用动态 cap 范围内的 argmax 组合一次
        if best["score"] <= -1e8 and len(ctx_sel) > 0:
            s_abs = ctx_sel[int(np.argmax(start_logits[ctx_sel]))]
            # 计算动态 cap
            dyn_cap = self._dynamic_cap_for_start(end_logits, ctx_sel, s_abs, is_short_field)
            
            # 在动态 cap 范围内选择 end
            s_idx = ctx_sel.index(s_abs)
            valid_ends = ctx_sel[s_idx:s_idx+dyn_cap]
            if valid_ends:
                e_abs = valid_ends[int(np.argmax(end_logits[valid_ends]))]
            else:
                e_abs = s_abs
                
            if e_abs < s_abs:
                e_abs = s_abs
            # 基础评分
            s_char, e_char = offset_mapping[s_abs][0], offset_mapping[e_abs][1]
            if s_char is not None and e_char is not None:
                score = float(start_logits[s_abs] + end_logits[e_abs])
                
                # 短字段优化评分
                if is_short_field and self.short_field_boost > 0:
                    span_length = e_abs - s_abs + 1
                    if span_length <= 8:
                        length_bonus = self.short_field_boost * (8 - span_length) / 8
                        score += length_bonus
                
                s_loc, e_loc = int(s_char), int(e_char)
                if chunk_text and _tighten_span is not None:
                    try:
                        s_loc, e_loc = _tighten_span(chunk_text, s_loc, e_loc)  
                    except Exception:
                        pass
                
                # 收紧后：去尾部标点/空白，再回对齐 token 边界
                t_s, t_e = s_abs, e_abs
                if s_loc != s_char or e_loc != e_char:
                    # 去尾部空白/标点（避免 end 落在标点后）
                    _TRIM_TAIL = set(" \t\r\n，。,；;、:：)]）】>》")
                    while e_loc > s_loc and chunk_text and chunk_text[e_loc - 1] in _TRIM_TAIL:
                        e_loc -= 1
                    
                    # 起点 token：第一个 offset_end > s_loc
                    for j in ctx_sel:
                        ss, ee = offset_mapping[j]
                        if ss is None or ee is None:
                            continue
                        if ee > s_loc:
                            t_s = j
                            break
                    
                    # 终点 token：优先覆盖 e_loc-1 的 token（ss < e_loc 且 ee >= e_loc）
                    t_e_new = None
                    for j in reversed(ctx_sel):
                        ss, ee = offset_mapping[j]
                        if ss is None or ee is None:
                            continue
                        if ss < e_loc and ee >= e_loc:
                            t_e_new = j
                            break
                    # 若没有，退而求其次：最后一个 offset_end <= e_loc
                    if t_e_new is None:
                        for j in reversed(ctx_sel):
                            ss, ee = offset_mapping[j]
                            if ss is None or ee is None:
                                continue
                            if ee <= e_loc:
                                t_e_new = j
                                break
                    # 兜底：至少不小于 t_s
                    t_e = max(t_e_new if t_e_new is not None else t_e, t_s)
                
                best.update({
                        "score": score,
                        "token_start": int(t_s),
                        "token_end": int(t_e),
                        "start_logit": float(start_logits[t_s]),
                        "end_logit": float(end_logits[t_e]),
                        "start_char": int(chunk_char_start + s_loc),
                        "end_char": int(chunk_char_start + e_loc),
                        "text": (chunk_text[s_loc:e_loc] if chunk_text else ""),
                    })

        return best


"""模型封装：训练/评估/推理。"""
class EBQAModel:
    """BERT 抽取式问答：训练、评估、推理与解码。"""

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.0,
        num_train_epochs: int = 2,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 16,
        output_dir: str = "./qa_ckpt",
        logging_steps: int = 50,
        save_strategy: str = "epoch",
        fp16: bool = True,
        max_answer_len: int = 128,
        seed: int = 42,
        tokenizer_name_or_path: Optional[str] = None,
        tokenizer: Optional[BertTokenizerFast] = None,
        use_noise: bool = False,
        noise_embed_dim: int = 16,
        noise_mode: str = "bucket",
        noise_mlp_hidden_dim: Optional[int] = None,
    ) -> None:
        if model_name_or_path is None:
            if _DEF_MODEL:
                model_name_or_path = _DEF_MODEL
            else:
                raise RuntimeError(
                    "Model path not provided. Configure model_path_conf.DEFAULT_MODEL_PATH or pass explicitly."
                )
        # tokenizer
        tok_path = _resolve_tokenizer_path(model_name_or_path, tokenizer_name_or_path)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                self.tokenizer = BertTokenizerFast.from_pretrained(tok_path)
            except Exception as exc:
                if _DEF_TOK:
                    self.tokenizer = BertTokenizerFast.from_pretrained(_DEF_TOK)
                else:
                    raise exc

        self.use_noise = bool(use_noise)
        self.noise_embed_dim = int(noise_embed_dim)
        self.noise_mode = str(noise_mode or "bucket").lower()
        self.noise_mlp_hidden_dim = int(noise_mlp_hidden_dim or 0) or None

        config = AutoConfig.from_pretrained(model_name_or_path)
        config.use_noise = self.use_noise
        config.noise_embed_dim = self.noise_embed_dim
        config.noise_mode = self.noise_mode
        config.noise_mlp_hidden_dim = self.noise_mlp_hidden_dim

        # model & args
        model_cls = NoiseAwareBertForQuestionAnswering if self.use_noise else BertForQuestionAnswering
        self.model = model_cls.from_pretrained(model_name_or_path, config=config)
        self.args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            fp16=fp16,
            seed=seed,
        )
        # 初始化解码器，为短字段设置更严格的cap
        self.decoder = EBQADecoder(
            self.tokenizer, 
            max_answer_len=max_answer_len,
            short_cap=6,  # 默认短字段cap从10降到6
        )
        self.max_answer_len = int(max_answer_len)
        self._trainer: Optional[Trainer] = None

    def _should_replace_answer(
        self, new_cand: Dict[str, Any], old_answer: Dict[str, Any], q_key: str
    ) -> bool:
        """
        智能聚合策略：综合评估答案质量，决定是否替换
        
        考虑因素：
        1. Score差距
        2. 答案长度合理性
        3. 答案完整性（是否包含异常标记）
        4. 置信度margin
        """
        new_score = float(new_cand.get("score", -1e9))
        old_score = float(old_answer.get("score", -1e9))
        new_text = str(new_cand.get("text", ""))
        old_text = str(old_answer.get("text", ""))
        
        # 1. Score差距显著（>1.5），直接选高分的
        if new_score > old_score + 1.5:
            return True
        if old_score > new_score + 1.5:
            return False
        
        # 2. Score相近（±1.5内），综合评估答案质量
        
        # 2.1 答案长度（太短或太长都不好）
        new_len = len(new_text)
        old_len = len(old_text)
        
        # 异常短的答案（<2字符）减分
        new_len_penalty = -2.0 if new_len < 2 else 0.0
        old_len_penalty = -2.0 if old_len < 2 else 0.0
        
        # 异常长的答案（>200字符）减分
        new_len_penalty += -1.0 if new_len > 200 else 0.0
        old_len_penalty += -1.0 if old_len > 200 else 0.0
        
        # 2.2 答案完整性（检测异常标记）
        def completeness_score(text):
            score = 0.0
            # 包含字段标题（如"姓名："）→ 可能是错误识别
            if '：' in text or ':' in text:
                # 检查是否包含常见字段名
                common_fields = ['姓名', '性别', '年龄', '科室', '病区', '床号', 
                                '病理号', '住院号', '日期', '时间', '医院']
                if any(field in text for field in common_fields):
                    score -= 1.0  # 包含字段标题，严重减分
            
            # 尾部异常标点 → 可能被截断
            if text.endswith(('、', '，')):
                score -= 0.3
            
            # 开头或结尾有冒号 → 可能是边界问题
            if text.startswith(('：', ':')):
                score -= 0.5
            
            return score
        
        new_complete = completeness_score(new_text)
        old_complete = completeness_score(old_text)
        
        # 2.3 置信度margin（span_score vs null_score的差距）
        new_null = float(new_cand.get("null_score", 0))
        old_null = float(old_answer.get("null_score", 0))
        
        new_margin = new_score - new_null
        old_margin = old_score - old_null
        
        margin_bonus = (new_margin - old_margin) * 0.2  # 置信度差异加权
        
        # 3. 综合评分
        new_total = new_score + new_len_penalty + new_complete + margin_bonus
        old_total = old_score + old_len_penalty + old_complete
        
        return new_total > old_total

    # 训练/评估（保留：你用自定义 loop）
    def build_trainer(self, train_ds, eval_ds=None, data_collator=None) -> Trainer:
        self._trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        return self._trainer

    def train(self) -> Any:
        if self._trainer is None:
            raise RuntimeError("请先调用 build_trainer(train_ds, ...) 构建 Trainer")
        return self._trainer.train()

    def evaluate(self) -> Dict[str, float]:
        if self._trainer is None:
            raise RuntimeError("请先调用 build_trainer(train_ds, ...) 构建 Trainer")
        return self._trainer.evaluate()

    # 推理/解码（ 读取 next_header_* 并传给解码器，禁止跨标题）
    @torch.no_grad()
    def predict(
        self,
        dataset,
        data_collator=None,
        batch_size: int = 32,
        enable_no_answer: bool = True,
        null_threshold: float = 0.0,
        null_agg: str = "max",  # "max" 或 "mean"
        debug_print: bool = False,
    ) -> List[Dict[str, Any]]:
        """对样本前向与解码，聚合到 (report_index, question_key)。
        使用动态长度 cap，不依赖边界信号。"""
        device = self._get_device()
        self.model.to(device).eval()
        
        # 兜底：如未提供 collator，自动创建（避免变长序列拼批错误）
        if data_collator is None:
            try:
                from pre_struct.ebqa.da_core.dataset import QACollator
            except Exception:
                try:
                    from .da_core.dataset import QACollator
                except Exception:
                    QACollator = None
            
            if QACollator is not None:
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                data_collator = QACollator(pad_id=pad_id, keep_debug_fields=True)

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator
        )

        buckets: Dict[Tuple[int, str], Dict[str, Any]] = {}

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            token_type_ids = (
                token_type_ids.to(device) if token_type_ids is not None else None
            )
            noise_ids = batch.get("noise_ids")
            noise_values = batch.get("noise_values")
            kwargs = {}
            if noise_ids is not None and _supports_noise_forward(self.model):
                kwargs["noise_ids"] = noise_ids.to(device)
            if noise_values is not None and _supports_noise_forward(self.model):
                kwargs["noise_values"] = noise_values.to(device, dtype=torch.float32)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
                **kwargs,
            )
            start_logits = outputs.start_logits.detach().cpu().numpy()
            end_logits = outputs.end_logits.detach().cpu().numpy()

            qkeys = batch.get("question_key", ["?"] * input_ids.size(0))
            ridxs = batch.get("report_index", [-1] * input_ids.size(0))

            has_debug = all(
                k in batch
                for k in (
                    "offset_mapping",
                    "sequence_ids",
                    "chunk_char_start",
                    "chunk_text",
                )
            )  # chunk_char_end 不是必需的，不应作为启用条件
            if has_debug:
                offset_mapping = batch["offset_mapping"]
                sequence_ids = batch["sequence_ids"]
                ch_s = batch["chunk_char_start"]
                ch_txt = batch["chunk_text"]

            # 短字段标记
            is_short_fields = batch.get("is_short_field", None)  # list[bool] 或 None

            for i in range(input_ids.size(0)):
                key = (int(ridxs[i]), str(qkeys[i]))
                s_log = start_logits[i]
                e_log = end_logits[i]

                cand = {
                    "text": "",
                    "score": float(np.max(s_log) + np.max(e_log)),
                    "start_char": -1,
                    "end_char": -1,
                    "token_start": int(np.argmax(s_log)),
                    "token_end": int(np.argmax(e_log)),
                    "start_logit": float(np.max(s_log)),
                    "end_logit": float(np.max(e_log)),
                    "null_score": (
                        float(s_log[0] + e_log[0]) if len(s_log) > 0 else -1e9
                    ),
                }

                if has_debug:
                    # 获取短字段标记
                    is_short_field = (
                        bool(is_short_fields[i]) 
                        if is_short_fields is not None and i < len(is_short_fields)
                        else False
                    )

                    cand = self.decoder.best_span_in_chunk(
                        start_logits=s_log,
                        end_logits=e_log,
                        offset_mapping=offset_mapping[i],
                        sequence_ids=list(sequence_ids[i]),
                        chunk_text=str(ch_txt[i]) if ch_txt is not None else "",
                        chunk_char_start=int(ch_s[i]),
                        is_short_field=is_short_field,
                        question_key=str(qkeys[i]),  # 传入question_key
                    )
                    
                    # 调试日志：显示短字段的token长度
                    if debug_print and is_short_field:
                        tok_len = cand["token_end"] - cand["token_start"] + 1
                        print(f"[SHORT] key={qkeys[i]} tok_len={tok_len} text='{cand['text'][:30]}'...")

                slot = buckets.get(key)
                if slot is None:
                    buckets[key] = {
                        "report_index": key[0],
                        "answer": cand,
                        "best_span": float(cand["score"]),
                        "best_null": float(cand.get("null_score", -1e9)),
                        "_null_list": [float(cand.get("null_score", -1e9))],
                        "question_key": str(qkeys[i]),  # 保存question_key用于后续判断
                    }
                else:
                    # 改进的聚合策略：不只看score，综合评估答案质量
                    should_replace = self._should_replace_answer(
                        cand, slot["answer"], str(qkeys[i])
                    )
                    
                    if should_replace:
                        slot["best_span"] = float(cand["score"])
                        slot["answer"] = cand
                    
                    slot["_null_list"].append(float(cand.get("null_score", -1e9)))
                    if null_agg == "max":
                        slot["best_null"] = max(
                            slot["best_null"], float(cand.get("null_score", -1e9))
                        )

        results: List[Dict[str, Any]] = []
        for (r_idx, q_key), slot in buckets.items():
            if null_agg == "mean" and slot.get("_null_list"):
                slot["best_null"] = float(np.mean(slot["_null_list"]))
            best_span = float(slot["best_span"])
            best_null = float(slot["best_null"])
            ans = slot["answer"]

            # no-answer 判定
            if enable_no_answer and (best_null - best_span) > float(null_threshold):
                final = {
                    "text": "",
                    "score": best_null,
                    "start_char": -1,
                    "end_char": -1,
                }
                if debug_print:
                    print(
                        f"[NA] q={q_key} best_null={best_null:.3f} best_span={best_span:.3f} "
                        f"delta={best_null - best_span:.3f} thr={null_threshold:.3f}"
                    )
            else:
                final = {
                    "text": str(ans.get("text", "")),
                    "score": float(ans.get("score", -1e9)),
                    "start_char": int(ans.get("start_char", -1)),
                    "end_char": int(ans.get("end_char", -1)),
                }

            results.append(
                {
                    "report_index": int(r_idx),
                    "question_key": str(q_key),
                    "text": str(final.get("text", "")),
                    "score": float(final.get("score", -1e9)),
                    "start_char": int(final.get("start_char", -1)),
                    "end_char": int(final.get("end_char", -1)),
                    "best_null": best_null,
                    "best_span": best_span,
                }
            )

        return results

    # 工具
    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _get_chunk_text(self, dataset, batch, i: int) -> str:
        if "chunk_text" in batch:
            try:
                return str(batch["chunk_text"][i])
            except Exception:
                pass
        return ""