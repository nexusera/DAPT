# -*- coding: utf-8 -*-
"""
Noise Utils for DAPT Model Evaluation

处理DAPT模型的噪声特征（7维分桶策略）：
- 从预处理数据集中提取 noise_values（连续值）
- 使用 NoiseFeatureProcessor 映射连续值为桶ID
- 生成 noise_ids 张量供模型前向传播

关键设计：
1. OCR样本：从数据集的 noise_values 提取，映射为桶ID
2. 非OCR样本：填充完美物理值 [1.0, 1.0, 0, 0, 0, 0, 0]
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

# 7维噪声特征定义（与DAPT保持一致）
FEATURES = [
    "conf_avg",      # 置信度平均值 (64桶)
    "conf_min",      # 置信度最小值 (64桶)
    "conf_var_log",  # 置信度方差对数 (32桶)
    "conf_gap",      # 置信度差值 (32桶)
    "punct_err_ratio", # 标点错误比例 (16桶)
    "char_break_ratio", # 字符断裂比例 (32桶)
    "align_score",   # 对齐分数 (64桶)
]

NUM_BINS = {
    "conf_avg": 64,
    "conf_min": 64,
    "conf_var_log": 32,
    "conf_gap": 32,
    "punct_err_ratio": 16,
    "char_break_ratio": 32,
    "align_score": 64,
}

# 完美物理值（非OCR样本使用）
PERFECT_VALUES = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 裁剪值（与DAPT保持一致）
CLIP = {
    "char_break_ratio": 0.25,
    "align_score": 3500.0,
}


class NoiseFeatureProcessor:
    """
    将连续噪声值映射为离散桶ID的处理器
    
    功能：
    1. 从JSON加载预计算的分桶边界
    2. 将连续值映射为桶ID（0为anchor）
    3. 生成可用于模型的noise_ids张量
    """
    
    def __init__(self, bins_dict: Optional[Dict[str, List[float]]] = None):
        """
        初始化处理器
        
        Args:
            bins_dict: 分桶边界字典，格式为 {feature_name: [boundary1, boundary2, ...]}
        """
        self.bins = bins_dict or {}
        
    @staticmethod
    def load(filepath: str) -> "NoiseFeatureProcessor":
        """从JSON文件加载分桶边界"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Bins JSON not found: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            bins_dict = json.load(f)
        return NoiseFeatureProcessor(bins_dict)
    
    def value_to_bin_id(self, feature: str, value: float) -> int:
        """
        将单个特征值映射为桶ID
        
        映射规则：
        - 0.0 -> ID=0 (Anchor bin)
        - 其他值根据分位数分桶，ID从1开始
        
        Args:
            feature: 特征名
            value: 连续值
            
        Returns:
            桶ID (0-num_bins)
        """
        if feature not in self.bins:
            return 0
        
        # 完美值（0或接近0）映射为anchor bin 0
        if abs(value) < 1e-6:
            return 0
        
        edges = self.bins[feature]
        # 使用searchsorted找到适当的bin
        # edges已排序，searchsorted返回插入位置，即bin id
        bin_id = np.searchsorted(edges, value, side='left')
        bin_id = min(bin_id, NUM_BINS[feature])  # 确保不超过最大bin数
        return int(bin_id)
    
    def values_to_bin_ids(self, noise_values: List[float]) -> List[int]:
        """
        将7维连续值向量映射为7维桶ID向量
        
        Args:
            noise_values: 长度为7的连续值列表
            
        Returns:
            长度为7的桶ID列表
        """
        bin_ids = []
        for feat, value in zip(FEATURES, noise_values):
            bin_id = self.value_to_bin_id(feat, value)
            bin_ids.append(bin_id)
        return bin_ids


class NoiseCollator:
    """
    DataLoader的collate函数，处理noise特征打包
    
    功能：
    1. 处理变长序列的noise_ids对齐
    2. OCR样本提取noise_values并映射为noise_ids
    3. 非OCR样本填充完美物理值
    4. 生成noise_ids张量 [batch_size, max_seq_len, 7]
    """
    
    def __init__(self, 
                 processor: NoiseFeatureProcessor,
                 tokenizer,
                 max_length: int = 512,
                 pad_token_id: int = 0):
        """
        初始化collator
        
        Args:
            processor: NoiseFeatureProcessor实例
            tokenizer: BERT tokenizer
            max_length: 最大序列长度
            pad_token_id: padding token ID
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        处理批次数据
        
        Args:
            batch: DataLoader返回的批次，每个item是dataset的一条record
            
        Returns:
            包含input_ids、attention_mask、noise_ids等的字典
        """
        batch_size = len(batch)
        
        # 提取input_ids进行padding
        input_ids_list = [item.get('input_ids', []) for item in batch]
        
        # 使用tokenizer进行padding（生成attention_mask）
        encoded = self.tokenizer.pad(
            {'input_ids': input_ids_list},
            padding='longest',
            max_length=self.max_length,
            pad_to_multiple_of=8,
            return_tensors='pt'
        )
        
        # 准备noise_ids [batch_size, padded_length, 7]
        padded_length = encoded['input_ids'].shape[1]
        noise_ids_batch = torch.zeros(
            (batch_size, padded_length, 7),
            dtype=torch.long,
            device=encoded['input_ids'].device
        )
        
        # 处理每个样本
        for i, item in enumerate(batch):
            seq_len = len(item.get('input_ids', []))
            
            # 获取noise_values（可能没有，则填充完美值）
            noise_values_list = item.get('noise_values', None)
            
            if noise_values_list is None or len(noise_values_list) == 0:
                # 非OCR样本：使用完美物理值
                noise_ids_list = [
                    self.processor.values_to_bin_ids(PERFECT_VALUES)
                    for _ in range(seq_len)
                ]
            else:
                # OCR样本：映射连续值为桶ID
                noise_ids_list = []
                for noise_values in noise_values_list[:seq_len]:
                    # noise_values应该是长度为7的列表
                    if isinstance(noise_values, (list, tuple)) and len(noise_values) == 7:
                        bin_ids = self.processor.values_to_bin_ids(noise_values)
                    else:
                        # 格式错误，用完美值
                        bin_ids = self.processor.values_to_bin_ids(PERFECT_VALUES)
                    noise_ids_list.append(bin_ids)
            
            # 填充到padded_length
            for j in range(seq_len):
                noise_ids_batch[i, j, :] = torch.tensor(
                    noise_ids_list[j],
                    dtype=torch.long
                )
            
            # 后续padding部分保持0（anchor bin）
            # 已初始化为0，无需额外操作
        
        # 整合输出
        result = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'noise_ids': noise_ids_batch,
        }
        
        # 如果有token_type_ids，也加入
        if 'token_type_ids' in encoded:
            result['token_type_ids'] = encoded['token_type_ids']
        
        return result


def prepare_noise_ids_for_model(
    batch: Dict[str, torch.Tensor],
    processor: NoiseFeatureProcessor,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    为模型准备noise_ids（在已有batch数据时调用）
    
    如果batch中已有noise_ids（通过NoiseCollator生成），则无需调用此函数。
    此函数用于手动构造或修复noise_ids。
    
    Args:
        batch: 包含input_ids等的batch字典
        processor: NoiseFeatureProcessor实例
        device: 目标设备
        
    Returns:
        补充了noise_ids的batch字典
    """
    if 'noise_ids' in batch:
        if device is not None:
            batch['noise_ids'] = batch['noise_ids'].to(device)
        return batch
    
    # 如果没有noise_ids，构造完美物理值
    batch_size = batch['input_ids'].shape[0]
    seq_len = batch['input_ids'].shape[1]
    
    noise_ids = torch.zeros(
        (batch_size, seq_len, 7),
        dtype=torch.long,
        device=device or batch['input_ids'].device
    )
    
    # 所有位置填充完美物理值对应的桶ID
    perfect_bin_ids = processor.values_to_bin_ids(PERFECT_VALUES)
    for i in range(batch_size):
        for j in range(seq_len):
            noise_ids[i, j, :] = torch.tensor(perfect_bin_ids, dtype=torch.long)
    
    batch['noise_ids'] = noise_ids
    return batch
