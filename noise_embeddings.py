import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings


def _clamp01(val: float) -> float:
    """Clamp scalar into [0, 1] range."""
    try:
        return float(max(0.0, min(1.0, val)))
    except Exception:
        return 0.0


def _levenshtein(a: str, b: str, max_dist: int = 64) -> int:
    """
    Minimal implementation of Levenshtein distance.
    A small max_dist lets us early-exit on very dissimilar strings.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    if abs(la - lb) > max_dist:
        return max_dist

    # Ensure a is the shorter sequence for less memory.
    if la > lb:
        a, b = b, a
        la, lb = lb, la

    previous = list(range(lb + 1))
    for i in range(1, la + 1):
        current = [i]
        ai = a[i - 1]
        min_in_row = max_dist
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + cost
            best = insert_cost if insert_cost < delete_cost else delete_cost
            if replace_cost < best:
                best = replace_cost
            current.append(best)
            if best < min_in_row:
                min_in_row = best
        if min_in_row > max_dist:
            return max_dist
        previous = current
    return previous[-1]


class NoiseFeatureExtractor:
    """
    从 OCR JSON 里抽取 5 维噪声特征，并与 tokenizer 的 word_ids 对齐。

    目标特征（均归一化到 [0, 1]）:
      0) conf                : probability.average
      1) dict_edit_dist_norm : 1 / (1 + edit_distance_to_med_dict)
      2) punct_err_ratio     : 非中英文/数字字符占比
      3) align_score         : 词 top 与所属段落平均 top 的归一化偏移
      4) char_break_ratio    : len(words) / location.width
    """

    def __init__(
        self,
        medical_dict: Optional[Iterable[str]] = None,
        default_values: Optional[Dict[str, float]] = None,
        max_edit_distance: int = 32,
        char_break_norm: float = 24.0,
    ) -> None:
        self.medical_dict = {w.lower() for w in (medical_dict or [])}
        self.max_edit_distance = max_edit_distance
        self.char_break_norm = max(1.0, char_break_norm)
        # Defaults represent“高质量 / 无噪声”情形
        self.default_values = {
            "conf": 0.95,
            "dict_edit": 1.0,
            "punct_err": 0.0,
            "align_score": 0.0,
            "char_break_ratio": 0.0,
        }
        if default_values:
            self.default_values.update(default_values)
        self._allowed_re = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")

    # ------------------------
    # 单个维度的特征计算
    # ------------------------
    def _conf(self, item: Dict[str, Any]) -> Tuple[float, bool]:
        prob = item.get("probability", {})
        if isinstance(prob, dict):
            val = prob.get("average")
        else:
            val = prob
        if isinstance(val, (int, float)):
            return _clamp01(float(val)), True
        return self.default_values["conf"], False

    def _dict_edit_score(self, word: str) -> Tuple[float, bool]:
        if not word or not self.medical_dict:
            return self.default_values["dict_edit"], False
        word_l = word.lower()
        best = self.max_edit_distance
        for cand in self.medical_dict:
            if abs(len(cand) - len(word_l)) > self.max_edit_distance:
                continue
            best = min(best, _levenshtein(word_l, cand, self.max_edit_distance))
            if best == 0:
                break
        score = 1.0 / (1.0 + best)
        return _clamp01(score), True

    def _punct_err(self, word: str) -> Tuple[float, bool]:
        if not word:
            return self.default_values["punct_err"], False
        bad = sum(1 for ch in word if not self._allowed_re.match(ch))
        ratio = bad / max(1, len(word))
        return _clamp01(ratio), True

    def _align_score(
        self,
        idx: int,
        locations: List[Dict[str, Any]],
        para_stats: Dict[int, Tuple[float, float]],
        word_to_para: Dict[int, int],
    ) -> Tuple[float, bool]:
        loc = locations[idx] if idx < len(locations) else {}
        top = loc.get("top")
        para_id = word_to_para.get(idx)
        if top is None or para_id is None or para_id not in para_stats:
            return self.default_values["align_score"], False
        mean_top, mean_h = para_stats[para_id]
        offset = abs(float(top) - mean_top)
        denom = max(1.0, mean_h)
        score = _clamp01(offset / denom)
        return score, True

    def _char_break_ratio(self, word: str, loc: Dict[str, Any]) -> Tuple[float, bool]:
        width = loc.get("width")
        if width is None or width <= 0:
            return self.default_values["char_break_ratio"], False
        ratio = len(word) / max(1.0, float(width))
        # 挤压/断裂越严重比值越大，放大到 [0,1] 便于 MLP 学习
        score = min(ratio * self.char_break_norm, 1.0)
        return _clamp01(score), True

    # ------------------------
    # 段落统计
    # ------------------------
    @staticmethod
    def _collect_paragraph_stats(
        words_result: List[Dict[str, Any]],
        paragraphs_result: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[int, int], Dict[int, Tuple[float, float]]]:
        """
        返回: word_idx -> paragraph_id 映射，以及 paragraph_id -> (mean_top, mean_height)
        """
        word_to_para: Dict[int, int] = {}
        para_stats: Dict[int, Tuple[float, float]] = {}
        for pid, para in enumerate(paragraphs_result):
            idxs = para.get("words_result_idx", []) or []
            tops: List[float] = []
            hs: List[float] = []
            for wid in idxs:
                word_to_para[wid] = pid
                if 0 <= wid < len(words_result):
                    loc = words_result[wid].get("location", {}) or {}
                    t = loc.get("top")
                    h = loc.get("height")
                    if isinstance(t, (int, float)):
                        tops.append(float(t))
                    if isinstance(h, (int, float)):
                        hs.append(float(h))
            if tops and hs:
                para_stats[pid] = (sum(tops) / len(tops), sum(hs) / len(hs))
        return word_to_para, para_stats

    # ------------------------
    # 主入口
    # ------------------------
    def extract_word_features(
        self, ocr_json: Dict[str, Any]
    ) -> Tuple[List[List[float]], List[List[bool]]]:
        """
        返回 word-level 特征与有效性掩码。
        - features: List[num_words][5]
        - masks:    List[num_words][5] (True 表示该维度是“真实值”而非默认填充)
        """
        ocr_obj = ocr_json.get("ocr", ocr_json)
        words_result = ocr_obj.get("words_result", []) if isinstance(ocr_obj, dict) else []
        paragraphs_result = ocr_obj.get("paragraphs_result", []) if isinstance(ocr_obj, dict) else []

        word_to_para, para_stats = self._collect_paragraph_stats(words_result, paragraphs_result)
        locations = [w.get("location", {}) or {} for w in words_result]

        features: List[List[float]] = []
        masks: List[List[bool]] = []

        for idx, item in enumerate(words_result):
            word = item.get("words", "") or ""
            loc = locations[idx] if idx < len(locations) else {}

            conf, conf_ok = self._conf(item)
            edit, edit_ok = self._dict_edit_score(word)
            punct, punct_ok = self._punct_err(word)
            align, align_ok = self._align_score(idx, locations, para_stats, word_to_para)
            char_break, char_break_ok = self._char_break_ratio(word, loc)

            feat = [
                _clamp01(conf),
                _clamp01(edit),
                _clamp01(punct),
                _clamp01(align),
                _clamp01(char_break),
            ]
            mask = [conf_ok, edit_ok, punct_ok, align_ok, char_break_ok]
            features.append(feat)
            masks.append(mask)

        return features, masks

    def broadcast_to_subwords(
        self,
        word_features: Sequence[Sequence[float]],
        word_masks: Sequence[Sequence[bool]],
        word_ids: Sequence[Optional[int]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 word-level 特征广播到 subword/token 级别。
        - word_features/masks: 来自 extract_word_features
        - word_ids: tokenizer 编码时得到的 word_ids 列表（长度 = 序列 token 数）
        返回:
          noise_features: [seq_len, 5] float tensor
          noise_masks:    [seq_len, 5] bool tensor，标记该 token 的该维特征是否真实
        """
        seq_len = len(word_ids)
        feats = torch.zeros((seq_len, 5), device=device, dtype=dtype or torch.float32)
        masks = torch.zeros((seq_len, 5), device=device, dtype=torch.bool)
        for tidx, wid in enumerate(word_ids):
            if wid is None or wid < 0:
                continue
            if wid >= len(word_features):
                continue
            feat_vec = torch.tensor(word_features[wid], device=device, dtype=dtype or torch.float32)
            mask_vec = torch.tensor(word_masks[wid], device=device, dtype=torch.bool)
            feats[tidx] = feat_vec
            masks[tidx] = mask_vec
        return feats.clamp_(0.0, 1.0), masks


class RobertaNoiseEmbeddings(RobertaEmbeddings):
    """
    在 word_embeddings 与 position_embeddings 相加前注入噪声特征。
    兼容 DDP：所有参数均注册在模块内，梯度可反向传播。
    """

    def __init__(self, config, noise_dim: int = 5):
        super().__init__(config)
        self.noise_dim = noise_dim or getattr(config, "noise_feature_size", 5)
        self.noise_mlp = nn.Sequential(
            nn.Linear(self.noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.hidden_size),
        )
        # 当某 token 的特征缺失时，用该可学习向量做补充
        self.missing_noise_embedding = nn.Parameter(torch.zeros(config.hidden_size))
        self._reset_noise_parameters()

    def _reset_noise_parameters(self) -> None:
        for module in self.noise_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.missing_noise_embedding)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
        noise_features: Optional[torch.Tensor] = None,
        noise_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 归一化并注入噪声（shape: [batch, seq_len, hidden]）
        if noise_features is not None:
            if noise_features.dim() == 2:
                noise_features = noise_features.unsqueeze(0)
            noise_features = noise_features.to(inputs_embeds.device, inputs_embeds.dtype)
            if noise_features.size(-1) != self.noise_dim:
                raise ValueError(
                    f"Expected noise_features last dim {self.noise_dim}, "
                    f"got {noise_features.size(-1)}"
                )
            noise_features = noise_features.clamp(0.0, 1.0)
            noise_embeds = self.noise_mlp(noise_features)

            if noise_masks is not None:
                if noise_masks.dim() == 2:
                    noise_masks = noise_masks.unsqueeze(0)
                noise_masks = noise_masks.to(inputs_embeds.device)
                missing_ratio = (1.0 - noise_masks.float()).mean(dim=-1, keepdim=True)
                noise_embeds = noise_embeds + missing_ratio * self.missing_noise_embedding.view(
                    1, 1, -1
                )

            inputs_embeds = inputs_embeds + noise_embeds

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

