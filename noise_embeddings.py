# 导入基础Python库
import math  # 数学库（虽然本文件未使用）
import re  # 正则表达式库，用于文本模式匹配
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple  # 类型提示

# 导入深度学习框架
import torch  # PyTorch张量库
from torch import nn  # PyTorch神经网络模块
# 导入HuggingFace的RoBERTa模型中的嵌入层
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings

# 可选：SymSpell 用于高效模糊匹配
try:
    from symspellpy import SymSpell, Verbosity  # type: ignore
except Exception:
    SymSpell = None
    Verbosity = None


def _clamp01(val: float) -> float:
    """
    将数值约束到 [0, 1] 范围内
    
    参数:
        val: 任意数值
        
    返回:
        限制在 [0, 1] 范围内的数值
        
    说明:
        - 如果 val < 0，返回 0.0
        - 如果 val > 1，返回 1.0
        - 如果 0 <= val <= 1，返回 val
        - 如果转换失败，返回 0.0 作为默认值
    """
    try:
        return float(max(0.0, min(1.0, val)))  # min和max结合实现范围约束
    except Exception:
        return 0.0  # 异常时返回最小值0.0


def _levenshtein(a: str, b: str, max_dist: int = 64) -> int:
    """
    计算两个字符串的编辑距离（Levenshtein距离）
    
    参数:
        a: 第一个字符串
        b: 第二个字符串
        max_dist: 最大距离阈值（超过此值返回max_dist）
        
    返回:
        编辑距离（将a转换为b所需的最少编辑操作数）
        
    说明:
        - 编辑操作包括：插入、删除、替换一个字符
        - 如果距离超过max_dist，提前返回max_dist以提高效率
        - 可用于计算医学词与标准医学词典中词汇的相似度
    """
    # 两个字符串相同，距离为0
    if a == b:
        return 0
    la, lb = len(a), len(b)
    # 一个为空的情况
    if la == 0:
        return lb
    if lb == 0:
        return la
    # 长度差异太大，提前返回最大值
    if abs(la - lb) > max_dist:
        return max_dist

    # 为了节省内存，确保a是较短的字符串
    if la > lb:
        a, b = b, a
        la, lb = lb, la

    # 初始化：第一行是 [0, 1, 2, ..., lb]
    previous = list(range(lb + 1))
    # 逐行计算编辑距离矩阵
    for i in range(1, la + 1):
        current = [i]  # 第一列初始化为当前行号
        ai = a[i - 1]  # 当前字符
        min_in_row = max_dist  # 本行最小值
        # 逐列计算
        for j in range(1, lb + 1):
            # 字符匹配则cost=0，不匹配则cost=1
            cost = 0 if ai == b[j - 1] else 1
            # 三种编辑操作的代价
            insert_cost = current[j - 1] + 1   # 插入
            delete_cost = previous[j] + 1      # 删除
            replace_cost = previous[j - 1] + cost  # 替换
            # 选择代价最小的操作
            best = insert_cost if insert_cost < delete_cost else delete_cost
            if replace_cost < best:
                best = replace_cost
            current.append(best)
            # 跟踪该行的最小值
            if best < min_in_row:
                min_in_row = best
        # 若本行最小值超过阈值，提前返回
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
        p0_terms: Optional[Iterable[str]] = None,
    ) -> None:
        # P0: 绝对白名单（Set 查找）
        self.p0_set = {w.lower() for w in (p0_terms or []) if w}

        # P1: 精选医疗实体，SymSpell 近似匹配
        self.max_edit_distance = max_edit_distance
        self._use_symspell = SymSpell is not None
        self.symspell = None
        self.medical_dict = {w.lower() for w in (medical_dict or [])}
        if self._use_symspell and self.medical_dict:
            self.symspell = SymSpell(
                max_dictionary_edit_distance=self.max_edit_distance,
                prefix_length=7,
            )
            for w in self.medical_dict:
                self.symspell.create_dictionary_entry(w, 1)
        else:
            # 回退：按长度分组（低效但保证功能）
            self._dict_by_length: Dict[int, List[str]] = {}
            for w in self.medical_dict:
                length = len(w)
                self._dict_by_length.setdefault(length, []).append(w)

        self.char_break_norm = max(1.0, char_break_norm)
        # Defaults represent"高质量 / 无噪声"情形
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
        """
        计算OCR识别的置信度（confidence）
        
        参数:
            item: OCR一个词的数据对象
            
        返回:
            (conf_score, is_valid) – 置信度值和是否是真实值
            
        说明:
            置信度来自OCR模型输出的probability中的average值
            值为0：低置信度，值为1：高置信度
        """
        prob = item.get("probability", {})  # 获取probability字段，默认为空字典
        # 处理两种类型：字典或浮点数
        if isinstance(prob, dict):
            val = prob.get("average")  # 字典中取average字段
        else:
            val = prob  # 直接使用
        # 检查是否是有效的数值
        if isinstance(val, (int, float)):
            return _clamp01(float(val)), True  # 置信度有效
        return self.default_values["conf"], False  # 使用默认值

    def _dict_edit_score(self, word: str) -> Tuple[float, bool]:
        """
        分层策略：
          P0: 白名单 (set 查找) 命中 -> dist=0, score=1.0
          P1: 医疗实体 SymSpell 近似匹配 (max_edit_distance=2)，取最优 dist
          P2: 兜底 OOD 噪声 -> dist=3, score=0.25
        """
        if not word:
            return self.default_values["dict_edit"], False
        word_l = word.lower()

        # P0: 绝对白名单
        if word_l in self.p0_set:
            return 1.0, True

        # P1: SymSpell 模糊匹配
        if self.symspell is not None:
            suggestions = self.symspell.lookup(
                word_l,
                Verbosity.TOP,
                max_edit_distance=2,
            )
            if suggestions:
                dist = min(suggestion.distance for suggestion in suggestions)
                score = 1.0 / (1.0 + dist)
                return _clamp01(score), True

        # 回退：长度分组（若 SymSpell 不可用）
        if self.medical_dict and self.symspell is None:
            best = self.max_edit_distance
            word_len = len(word_l)
            for cand_len in range(
                max(1, word_len - self.max_edit_distance),
                word_len + self.max_edit_distance + 1,
            ):
                if cand_len not in getattr(self, "_dict_by_length", {}):
                    continue
                for cand in self._dict_by_length[cand_len]:
                    dist = _levenshtein(word_l, cand, self.max_edit_distance)
                    best = min(best, dist)
                    if best == 0:
                        break
                if best == 0:
                    break
            score = 1.0 / (1.0 + best)
            return _clamp01(score), True

        # P2: OOD 噪声（兜底）
        dist = 3
        score = 1.0 / (1.0 + dist)  # 0.25
        return _clamp01(score), True

    def _punct_err(self, word: str) -> Tuple[float, bool]:
        """
        计算词中非法字符的比例
        
        参数:
            word: OCR识别出来的词
            
        返回:
            (err_ratio, is_valid) – 非法字符比例(0-1)和是否是真实值
            
        说明:
            法字符是指：英文字母、中文字符、数字
            例子："一③号"有一个③不是法字符，bad=1, ratio=1/2=0.5
        """
        # 无效情况
        if not word:
            return self.default_values["punct_err"], False
        # 统计不是法字符的个数
        bad = sum(1 for ch in word if not self._allowed_re.match(ch))
        ratio = bad / max(1, len(word))  # 计算比例、防止除以0
        return _clamp01(ratio), True

    def _align_score(
        self,
        idx: int,
        locations: List[Dict[str, Any]],
        para_stats: Dict[int, Tuple[float, float]],
        word_to_para: Dict[int, int],
    ) -> Tuple[float, bool]:
        """
        计算词的位置是否符合段落中的位置分布
        
        参数:
            idx: 词在字段中的索引
            locations: 每个词的位置信息（x, y, width, height）
            para_stats: 段落的较统计信息（平均top、平均高度）
            word_to_para: 词到段落的映射
            
        返回:
            (align_score, is_valid) – 对齐得分(0-1)和是否是真实值
            
        说明:
            - 如果词的位置与段落平均位置一致，score为0
            - 如果词正位置偏离较大，score为1（可能是刀患、搭字）
        """
        # 获取当前词的位置信息
        loc = locations[idx] if idx < len(locations) else {}
        top = loc.get("top")  # y坐标（从上到下的距离）
        # 获取该词所属的段落
        para_id = word_to_para.get(idx)
        # 无效情况：缺少位置信息或段落统计
        if top is None or para_id is None or para_id not in para_stats:
            return self.default_values["align_score"], False
        # 获取段落的平均位置与高度
        mean_top, mean_h = para_stats[para_id]
        # 计算偏离量（年度需要是段落的高度）
        offset = abs(float(top) - mean_top)
        denom = max(1.0, mean_h)  # 防止除以0
        score = _clamp01(offset / denom)  # 比例化为[0,1]
        return score, True

    def _char_break_ratio(self, word: str, loc: Dict[str, Any]) -> Tuple[float, bool]:
        """
        计算词的字符挤压/断裂程度
        
        参数:
            word: OCR识别出来的词
            loc: 词的位置信息（包含width像素）
            
        返回:
            (char_break_score, is_valid) – 挤压程度(0-1)和是否是真实值
            
        说明:
            比例 = 词的字数 / 词的宽度像素
            - 正常准间距时，ratio 不大，score 较低
            - 较挤压时，ratio 若大，score 较高、提示断裂可能性
        """
        # 字幅算那一个。取算渐。
        width = loc.get("width")
        # 无效情况：没有宽度信息或宽度为0
        if width is None or width <= 0:
            return self.default_values["char_break_ratio"], False
        # 有字符数 / 像素宽度
        ratio = len(word) / max(1.0, float(width))
        # 挤压/断裂越严重比值越大，取值*char_break_norm放大了一些
        # 使其更容易被中间层(MLP)学习
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
        收集段落统计信息
        
        参数:
            words_result: OCR识别的所有词的列表
            paragraphs_result: OCR识别的所有段落的列表
            
        返回:
            - word_to_para: 词索引 -> 段落ID的映射
            - para_stats: 段落ID -> (平均top坐标, 平均高度) 的字典
            
        说明:
            这个函数通过段落信息将每个词分配给所属的段落，
            并计算每个段落中的词的平均位置和高度，用于后续的位置对齐评分
        """
        word_to_para: Dict[int, int] = {}  # 词ID -> 段落ID
        para_stats: Dict[int, Tuple[float, float]] = {}  # 段落ID -> (平均top, 平均高)
        # 遍历每个段落
        for pid, para in enumerate(paragraphs_result):
            # 获取该段落包含的词的索引列表
            idxs = para.get("words_result_idx", []) or []
            tops: List[float] = []  # 段落中所有词的top坐标
            hs: List[float] = []    # 段落中所有词的高度
            # 遍历这个段落中的所有词
            for wid in idxs:
                word_to_para[wid] = pid  # 建立映射：这个词属于pid段落
                # 如果词ID在范围内，提取位置信息
                if 0 <= wid < len(words_result):
                    loc = words_result[wid].get("location", {}) or {}
                    t = loc.get("top")      # y坐标
                    h = loc.get("height")   # 高度
                    # 如果这个信息有效，添加到列表
                    if isinstance(t, (int, float)):
                        tops.append(float(t))
                    if isinstance(h, (int, float)):
                        hs.append(float(h))
            # 如果段落中有位置信息，计算平均值
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
        从OCR结果中提取所有词的噪声特征
        
        参数:
            ocr_json: OCR的JSON结果字典，包含words_result和paragraphs_result
            
        返回:
            - features: 词级别的特征列表，形状为 [词数][5维度]
            - masks: 对应的有效性掩码列表，True表示该维度是"真实值"而非默认填充
            
        处理流程:
            1. 从JSON中提取词和段落列表
            2. 计算段落统计信息（平均位置、高度）
            3. 对每个词计算5个维度的特征
            4. 返回特征和掩码
        """
        # 获取OCR数据对象（支持嵌套在"ocr"字段或直接传入）
        ocr_obj = ocr_json.get("ocr", ocr_json)
        # 提取词和段落列表
        words_result = ocr_obj.get("words_result", []) if isinstance(ocr_obj, dict) else []
        paragraphs_result = ocr_obj.get("paragraphs_result", []) if isinstance(ocr_obj, dict) else []

        # 计算段落统计信息
        word_to_para, para_stats = self._collect_paragraph_stats(words_result, paragraphs_result)
        # 预处理：提取所有词的位置信息
        locations = [w.get("location", {}) or {} for w in words_result]

        features: List[List[float]] = []  # 存储特征
        masks: List[List[bool]] = []      # 存储有效性掩码

        # 遍历每个识别出来的词
        for idx, item in enumerate(words_result):
            word = item.get("words", "") or ""  # 获取词文本
            loc = locations[idx] if idx < len(locations) else {}

            # 计算5个维度的特征和有效性标志
            conf, conf_ok = self._conf(item)                                      # 维度0：置信度
            edit, edit_ok = self._dict_edit_score(word)                          # 维度1：词典匹配度
            punct, punct_ok = self._punct_err(word)                              # 维度2：非法字符比例
            align, align_ok = self._align_score(idx, locations, para_stats, word_to_para)  # 维度3：位置对齐
            char_break, char_break_ok = self._char_break_ratio(word, loc)        # 维度4：字符挤压

            # 组织成特征向量（确保每个值都在[0,1]范围内）
            feat = [
                _clamp01(conf),
                _clamp01(edit),
                _clamp01(punct),
                _clamp01(align),
                _clamp01(char_break),
            ]
            # 组织成有效性掩码向量
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
        将word级别的特征"广播"到token级别（subword级别）
        
        参数:
            word_features: 词级别的特征列表[词数][5维度]，来自extract_word_features
            word_masks: 词级别的有效性掩码，来自extract_word_features
            word_ids: BERT tokenizer的word_ids列表
                     表示每个token属于哪个word，长度=序列token数
                     值为None表示特殊token（如[CLS]、[SEP]），值>=0表示对应的word索引
            device: PyTorch设备（CPU或GPU）
            dtype: 数据类型（默认float32）
            
        返回:
            (noise_features, noise_masks)
            - noise_features: [seq_len, 5] 的float张量，seq_len为token数
            - noise_masks: [seq_len, 5] 的bool张量，标记该token的该维特征是否真实
            
        说明:
            因为BERT使用subword分词（一个词可能被分成多个token），
            这个函数将词级别的特征复制给属于该词的所有token
            例如："医生" -> ["医", "生"]，两个token都会继承"医生"的特征
        """
        seq_len = len(word_ids)
        # 初始化张量，默认值为0
        feats = torch.zeros((seq_len, 5), device=device, dtype=dtype or torch.float32)
        masks = torch.zeros((seq_len, 5), device=device, dtype=torch.bool)
        # 遍历每个token
        for tidx, wid in enumerate(word_ids):
            # 如果是特殊token（None）或无效索引，跳过
            if wid is None or wid < 0:
                continue
            # 如果word_id超出范围，跳过
            if wid >= len(word_features):
                continue
            # 获取该word的特征和掩码，并复制到对应的token位置
            feat_vec = torch.tensor(word_features[wid], device=device, dtype=dtype or torch.float32)
            mask_vec = torch.tensor(word_masks[wid], device=device, dtype=torch.bool)
            feats[tidx] = feat_vec
            masks[tidx] = mask_vec
        # 确保所有值都在[0,1]范围内
        return feats.clamp_(0.0, 1.0), masks


class RobertaNoiseEmbeddings(RobertaEmbeddings):
    """
    改进版的RoBERTa嵌入层，在计算最终嵌入前注入噪声特征
    
    功能说明:
        1. 继承RobertaEmbeddings的所有功能（词、位置、token类型嵌入）
        2. 在词嵌入和位置嵌入求和前，添加OCR噪声特征信息
        3. 噪声特征通过MLP网络投影到隐层维度，然后与词嵌入相加
        4. 处理缺失特征：当某些token没有有效的噪声特征时，使用可学习的补充向量
        5. 支持DDP（分布式数据并行）训练：所有参数都注册在模块内，梯度可反向传播
    
    输入输出:
        输入: input_ids、token_type_ids等标准BERT输入，额外还有noise_features和noise_masks
        输出: 注入了噪声特征的嵌入张量，shape=[batch_size, seq_len, hidden_size]
    """

    def __init__(self, config, noise_dim: int = 5):
        """
        初始化RobertaNoiseEmbeddings层
        
        参数:
            config: 模型配置对象（包含hidden_size等参数）
            noise_dim: 噪声特征的维度（默认5维，对应5个噪声维度）
        """
        super().__init__(config)  # 调用父类初始化，包含词嵌入、位置嵌入等
        # 设置噪声特征维度，可以从config中获取
        self.noise_dim = noise_dim or getattr(config, "noise_feature_size", 5)
        # 构建MLP网络：将低维噪声特征投影到高维隐层空间
        # 5维 -> 128维（中间层） -> hidden_size维（与词嵌入维度一致）
        self.noise_mlp = nn.Sequential(
            nn.Linear(self.noise_dim, 128),  # 第一层：扩展维度
            nn.ReLU(),                        # 激活函数
            nn.Linear(128, config.hidden_size),  # 第二层：投影到隐层维度
        )
        # 可学习的补充向量：当token的噪声特征缺失时使用
        # 初始化为零向量，会在训练过程中学习最优值
        self.missing_noise_embedding = nn.Parameter(torch.zeros(config.hidden_size))
        # 初始化网络参数
        self._reset_noise_parameters()

    def _reset_noise_parameters(self) -> None:
        """
        初始化MLP网络和补充向量的参数
        
        说明:
            - MLP的权重使用Xavier均匀初始化（适合深度网络）
            - MLP的偏置初始化为零
            - 补充向量初始化为零
        """
        # 对MLP中的每个线性层进行初始化
        for module in self.noise_mlp:
            if isinstance(module, nn.Linear):
                # Xavier均匀初始化权重（防止梯度消失/爆炸）
                nn.init.xavier_uniform_(module.weight)
                # 偏置初始化为零
                nn.init.zeros_(module.bias)
        # 补充向量初始化为零
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
        """
        前向传播：计算注入了噪声特征的嵌入表示
        
        参数:
            input_ids: token ID列表 [batch_size, seq_len]
            token_type_ids: token类型ID（区分不同句子） [batch_size, seq_len]
            position_ids: 位置ID [batch_size, seq_len]
            inputs_embeds: 如果已有词嵌入，可直接传入，不需要input_ids
            past_key_values_length: 用于缓存推理
            noise_features: OCR噪声特征 [batch_size, seq_len, 5] 或 [seq_len, 5]
            noise_masks: 噪声特征有效性掩码 [batch_size, seq_len, 5] 或 [seq_len, 5]
            
        返回:
            最终嵌入表示 [batch_size, seq_len, hidden_size]
        """
        # 确定输入的形状
        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]  # 去掉最后的嵌入维度
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 如果没有提供token类型ID，使用全零（即都是第一句话）
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        # 如果没有词嵌入，根据input_ids生成
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 注入噪声特征（核心步骤）
        if noise_features is not None:
            # 处理形状：如果是2D张量（没有batch维度），添加batch维度
            if noise_features.dim() == 2:
                noise_features = noise_features.unsqueeze(0)
            # 转移到正确的设备和数据类型
            noise_features = noise_features.to(inputs_embeds.device, inputs_embeds.dtype)
            # 验证噪声特征的维度
            if noise_features.size(-1) != self.noise_dim:
                raise ValueError(
                    f"Expected noise_features last dim {self.noise_dim}, "
                    f"got {noise_features.size(-1)}"
                )
            # 将噪声特征限制在[0,1]范围内
            noise_features = noise_features.clamp(0.0, 1.0)
            # 通过MLP投影到隐层空间
            # 通过MLP投影到隐层空间
            noise_embeds = self.noise_mlp(noise_features)

            # 处理缺失特征：当某个token的噪声特征缺失时，添加补充向量
            if noise_masks is not None:
                # 处理形状：如果是2D张量，添加batch维度
                if noise_masks.dim() == 2:
                    noise_masks = noise_masks.unsqueeze(0)
                # 转移到正确的设备
                noise_masks = noise_masks.to(inputs_embeds.device)
                # 计算缺失特征的比例：有多少维度的特征是缺失的
                # missing_ratio = 1 - masks，然后平均所有维度，得到[batch, seq_len, 1]
                missing_ratio = (1.0 - noise_masks.float()).mean(dim=-1, keepdim=True)
                # 缺失特征越多，添加越多的补充向量
                # 这样可以防止模型对缺失特征过度拟合
                noise_embeds = noise_embeds + missing_ratio * self.missing_noise_embedding.view(
                    1, 1, -1
                )

            # 核心：将噪声嵌入加到词嵌入中
            inputs_embeds = inputs_embeds + noise_embeds

        # 处理位置ID和位置嵌入
        seq_length = input_shape[1]
        # 如果没有提供位置ID，从缓存中获取
        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # 获取token类型嵌入（用于区分不同的句子）
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # 获取位置嵌入（编码位置信息）
        position_embeddings = self.position_embeddings(position_ids)

        # 最终嵌入 = 词嵌入（可能已注入噪声） + 位置嵌入 + token类型嵌入
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # 层归一化（使特征均值为0、方差为1，增强模型稳定性）
        embeddings = self.LayerNorm(embeddings)
        # Dropout正则化（减少过拟合）
        embeddings = self.dropout(embeddings)
        return embeddings

