#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KV-NER Evaluation with DAPT Noise Support

为支持DAPT预训练模型而扩展的KV-NER评估脚本，特点：
1. 支持加载带有RobertaNoiseEmbeddings的DAPT模型
2. 处理7维分桶噪声特征（noise_ids）
3. 保持与原有KV-NER评估流程的完全兼容性
4. 仅支持Task 1&2 (Keys & Pairs)

使用方式：
    python evaluate_with_dapt_noise.py \
        --model_path /path/to/dapt/model \
        --test_data data/val_eval_titled.jsonl \
        --noise_bins_json /path/to/noise_bins.json \
        --output_summary runs/eval_dapt.json

核心改动点：
- NoiseCollator: 处理noise_ids的对齐与打包
- Model forward: 传入noise_ids张量
- 其他流程保持不变
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import unicodedata
import re
import string

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel

# 添加项目路径
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from pre_struct.kv_ner import config_io
from pre_struct.kv_ner.data_utils import build_bio_label_list
from pre_struct.kv_ner.modeling import BertCrfTokenClassifier
from pre_struct.kv_ner.noise_utils import (
    NoiseFeatureProcessor,
    NoiseCollator,
    PERFECT_VALUES,
    FEATURES,
)
from core.metrics import calculate_task1_stats, calculate_task2_stats, calc_micro_f1

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ============================================================================
# 工具函数与数据加载
# ============================================================================

def set_seed(seed: Optional[int]) -> None:
    """设置随机种子"""
    if seed is None:
        return
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取JSONL文件"""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    results = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _normalize_text_for_eval(s: str) -> str:
    """
    文本归一化：Unicode NFKC、统一连字符、裁剪边界标点
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("—", "-").replace("–", "-")
    s = s.replace("\u3000", " ")
    s = re.sub(r"^\s+|\s+$", "", s)
    edge_punct = "。，、；:;,:()[]{}<>"
    i = 0
    while i < len(s) and s[i] in edge_punct:
        i += 1
    j = len(s)
    while j > i and s[j - 1] in edge_punct:
        j -= 1
    return s[i:j]


def _extract_ground_truth(item: Dict[str, Any]) -> Tuple[set, set]:
    """
    从GT item提取keys和pairs
    
    Returns:
        (gt_keys: set of key texts, gt_pairs: set of (key_text, value_text))
    """
    gt_keys = set()
    gt_pairs = set()
    
    if 'spans' in item:
        for k, v in item['spans'].items():
            v_text = v.get('text', '') if isinstance(v, dict) else ''
            gt_keys.add(k)
            if v_text:
                gt_pairs.add((k, v_text))
    elif 'key_value_pairs' in item:
        for p in item['key_value_pairs']:
            k_text = p['key']['text']
            v_text = p.get('value_text', '')
            gt_keys.add(k_text)
            if v_text:
                gt_pairs.add((k_text, v_text))
    
    return gt_keys, gt_pairs


# ============================================================================
# 简化的KV-NER评估数据集与推理
# ============================================================================

class SimpleEvalDataset(Dataset):
    """
    简化的评估数据集，无需标签
    
    特点：
    - 仅提取input_ids进行推理
    - 支持noise_ids（如果数据集中有）
    - 与标准KV-NER数据格式兼容
    """
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        
        # 使用tokenizer编码
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_token_type_ids=False,
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'text': text,
        }


def predict_with_dapt_model(
    model: BertCrfTokenClassifier,
    dataloader: DataLoader,
    device: torch.device,
    use_noise: bool = True,
) -> List[Dict[str, Any]]:
    """
    使用DAPT模型进行推理
    
    Args:
        model: BertCrfTokenClassifier模型（可能使用DAPT)
        dataloader: 数据加载器
        device: 计算设备
        use_noise: 是否使用noise_ids（DAPT模型）
        
    Returns:
        推理结果列表，每个元素包含label_ids和其他信息
    """
    model.to(device)
    model.eval()
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing"):
            # 准备输入
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # 如果有noise_ids且模型支持，传入
            kwargs = {}
            if use_noise and 'noise_ids' in batch:
                kwargs['noise_ids'] = batch['noise_ids'].to(device)
            
            # 前向传播
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
            
            # 提取CRF解码后的标签序列
            # 假设model返回的predictions是CRF.decode()的结果
            if hasattr(outputs, 'predictions'):
                predictions = outputs.predictions
            else:
                # 如果model返回logits，需要手动decode
                # 这里假设model有crf属性可以用于解码
                logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
                batch_size = logits.shape[0]
                predictions = []
                
                for i in range(batch_size):
                    seq_logits = logits[i].cpu().numpy()
                    if attention_mask is not None:
                        seq_mask = attention_mask[i].cpu().numpy()
                    else:
                        seq_mask = None
                    
                    # 使用CRF解码（如果有）
                    if hasattr(model, 'crf'):
                        pred_ids = model.crf.decode(
                            torch.tensor([seq_logits], device=device),
                            mask=torch.tensor([seq_mask], device=device) if seq_mask is not None else None
                        )[0]
                    else:
                        # 无CRF，直接取argmax
                        pred_ids = torch.argmax(torch.tensor(seq_logits), dim=-1).tolist()
                    
                    predictions.append(pred_ids)
            
            # 整理结果
            for i, (pred_labels, text) in enumerate(zip(predictions, batch.get('text', [''] * len(predictions)))):
                all_predictions.append({
                    'pred_labels': pred_labels,
                    'text': text,
                })
    
    return all_predictions


def assemble_kv_pairs_from_predictions(
    pred_labels: List[int],
    text: str,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
) -> List[Tuple[str, str]]:
    """
    从CRF预测标签组装键值对
    
    BIO标签格式假设：
    - B-KEY / I-KEY
    - B-VALUE / I-VALUE
    - O (其他)
    
    Returns:
        [(key_text, value_text), ...]
    """
    pairs = []
    current_key = None
    current_value = None
    key_start = None
    value_start = None
    
    for idx, label_id in enumerate(pred_labels):
        if label_id < 0 or label_id >= len(id2label):
            continue
        
        label = id2label[label_id]
        
        if label.startswith('B-KEY'):
            # 结束前一个pair
            if current_key and current_value:
                pairs.append((current_key, current_value))
            current_key = text[idx] if idx < len(text) else ''
            current_value = None
            key_start = idx
            
        elif label.startswith('I-KEY'):
            if current_key is not None and idx < len(text):
                current_key += text[idx]
                
        elif label.startswith('B-VALUE'):
            if current_key is None:
                current_key = ''
            current_value = text[idx] if idx < len(text) else ''
            value_start = idx
            
        elif label.startswith('I-VALUE'):
            if current_value is not None and idx < len(text):
                current_value += text[idx]
        
        elif label == 'O':
            # 继续或重置
            pass
    
    # 添加最后的pair
    if current_key and current_value:
        pairs.append((current_key, current_value))
    
    # 归一化
    pairs = [(_normalize_text_for_eval(k), _normalize_text_for_eval(v)) for k, v in pairs]
    pairs = [(k, v) for k, v in pairs if k and v]
    
    return pairs


# ============================================================================
# 主评估逻辑
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KV-NER Model with DAPT Noise Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 评估标准BERT模型
  python evaluate_with_dapt_noise.py \\
    --model_path hfl/chinese-roberta-wwm-ext \\
    --test_data data/val_eval_titled.jsonl \\
    --output_summary runs/eval_roberta.json

  # 评估DAPT预训练模型
  python evaluate_with_dapt_noise.py \\
    --model_path /path/to/dapt/checkpoint \\
    --test_data data/val_eval_titled.jsonl \\
    --noise_bins_json /path/to/noise_bins.json \\
    --output_summary runs/eval_dapt.json
        """
    )
    
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to model checkpoint or HuggingFace model ID"
    )
    parser.add_argument(
        "--test_data",
        required=True,
        help="Path to test JSONL file (must contain 'spans' field)"
    )
    parser.add_argument(
        "--noise_bins_json",
        default=None,
        help="Path to noise bins JSON (for DAPT models). If not provided, assumes standard BERT."
    )
    parser.add_argument(
        "--output_summary",
        default=None,
        help="Output JSON file for evaluation summary"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # ========================================================================
    # 加载测试数据
    # ========================================================================
    logger.info(f"Loading test data: {args.test_data}")
    test_items = _read_jsonl(Path(args.test_data))
    logger.info(f"Loaded {len(test_items)} test samples")
    
    # ========================================================================
    # 加载模型与Tokenizer
    # ========================================================================
    logger.info(f"Loading model: {args.model_path}")
    try:
        # 尝试加载为BertCrfTokenClassifier（通常是微调后的模型）
        config = AutoConfig.from_pretrained(args.model_path)
        
        # 检查是否是DAPT模型（通过检查是否有noise embeddings）
        is_dapt = False
        try:
            model = BertCrfTokenClassifier(model_name_or_path=args.model_path)
            # 检查model.bert.embeddings是否为RobertaNoiseEmbeddings
            if hasattr(model.bert, 'embeddings'):
                emb_class_name = model.bert.embeddings.__class__.__name__
                is_dapt = 'Noise' in emb_class_name
                logger.info(f"Embeddings class: {emb_class_name}")
        except Exception as e:
            logger.warning(f"Could not load as BertCrfTokenClassifier: {e}")
            # 回退到AutoModel
            model = AutoModel.from_pretrained(args.model_path)
        
        logger.info(f"Model is DAPT: {is_dapt}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    
    # ========================================================================
    # 准备Noise处理（如果提供了bins_json）
    # ========================================================================
    processor = None
    use_noise = False
    
    if args.noise_bins_json and os.path.exists(args.noise_bins_json):
        logger.info(f"Loading noise bins: {args.noise_bins_json}")
        processor = NoiseFeatureProcessor.load(args.noise_bins_json)
        use_noise = True
        logger.info("Noise feature processing enabled")
    else:
        logger.info("No noise bins provided, using standard BERT inference")
    
    # ========================================================================
    # 创建数据集与DataLoader
    # ========================================================================
    logger.info("Preparing dataset for evaluation")
    texts = [item['text'] for item in test_items]
    dataset = SimpleEvalDataset(texts, tokenizer, max_length=args.max_length)
    
    # 选择collate_fn
    if use_noise and processor:
        collate_fn = NoiseCollator(
            processor=processor,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
    else:
        collate_fn = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # ========================================================================
    # 推理
    # ========================================================================
    logger.info("Running inference...")
    predictions = predict_with_dapt_model(
        model=model,
        dataloader=dataloader,
        device=device,
        use_noise=use_noise,
    )
    
    # ========================================================================
    # 提取键值对并计算指标
    # ========================================================================
    logger.info("Computing metrics...")
    
    # 获取label映射
    try:
        id2label = model.id2label if hasattr(model, 'id2label') else {}
        label2id = model.label2id if hasattr(model, 'label2id') else {}
    except:
        id2label = {}
        label2id = {}
    
    # 统计指标
    t1_strict_all = {'tp': 0, 'fp': 0, 'fn': 0}
    t1_loose_all = {'ps': 0, 'pc': 0, 'rs': 0, 'rc': 0}
    t2_ss_all = {'tp': 0, 'fp': 0, 'fn': 0}
    t2_sl_all = {'ps': 0, 'pc': 0, 'rs': 0, 'rc': 0}
    t2_ll_all = {'ps': 0, 'pc': 0, 'rs': 0, 'rc': 0}
    
    pred_pairs_all = []  # 保存所有预测的pair供后续使用
    
    for i, (pred, gt) in enumerate(zip(predictions, test_items)):
        pred_labels = pred['pred_labels']
        text = pred['text']
        
        # 从预测标签组装pair
        pred_pairs = assemble_kv_pairs_from_predictions(
            pred_labels, text, id2label, label2id
        )
        pred_pairs_all.append({'pred_pairs': pred_pairs})
        
        # 从GT提取pair
        gt_keys, gt_pairs = _extract_ground_truth(gt)
        
        # Task 1: Keys
        t1_s, t1_l = calculate_task1_stats(
            [p[0] for p in pred_pairs],  # pred_keys
            list(gt_keys),
        )
        for k in t1_strict_all:
            t1_strict_all[k] += t1_s[k]
        for k in t1_loose_all:
            t1_loose_all[k] += t1_l[k]
        
        # Task 2: Pairs
        t2_ss, t2_sl, t2_ll = calculate_task2_stats(pred_pairs, gt_pairs)
        for k in t2_ss_all:
            t2_ss_all[k] += t2_ss[k]
        for k in t2_sl_all:
            t2_sl_all[k] += t2_sl[k]
        for k in t2_ll_all:
            t2_ll_all[k] += t2_ll[k]
    
    # 计算最终指标
    t1_strict_metrics = calc_micro_f1(t1_strict_all)
    t1_loose_metrics = calc_micro_f1(t1_loose_all)
    t2_ss_metrics = calc_micro_f1(t2_ss_all)
    t2_sl_metrics = calc_micro_f1(t2_sl_all)
    t2_ll_metrics = calc_micro_f1(t2_ll_all)
    
    # ========================================================================
    # 输出结果
    # ========================================================================
    result = {
        'model': args.model_path,
        'is_dapt': use_noise,
        'num_samples': len(test_items),
        'task1': {
            'strict': t1_strict_metrics,
            'loose': t1_loose_metrics,
        },
        'task2': {
            'strict_strict': t2_ss_metrics,
            'strict_loose': t2_sl_metrics,
            'loose_loose': t2_ll_metrics,
        },
    }
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Is DAPT: {use_noise}")
    logger.info(f"Samples: {len(test_items)}")
    logger.info("\nTask 1 (Keys Discovery):")
    logger.info(f"  Strict: P={t1_strict_metrics['p']:.4f}, R={t1_strict_metrics['r']:.4f}, F1={t1_strict_metrics['f1']:.4f}")
    logger.info(f"  Loose:  P={t1_loose_metrics['p']:.4f}, R={t1_loose_metrics['r']:.4f}, F1={t1_loose_metrics['f1']:.4f}")
    logger.info("\nTask 2 (Key-Value Pairs):")
    logger.info(f"  Strict-Strict: P={t2_ss_metrics['p']:.4f}, R={t2_ss_metrics['r']:.4f}, F1={t2_ss_metrics['f1']:.4f}")
    logger.info(f"  Strict-Loose:  P={t2_sl_metrics['p']:.4f}, R={t2_sl_metrics['r']:.4f}, F1={t2_sl_metrics['f1']:.4f}")
    logger.info(f"  Loose-Loose:   P={t2_ll_metrics['p']:.4f}, R={t2_ll_metrics['r']:.4f}, F1={t2_ll_metrics['f1']:.4f}")
    logger.info("="*80 + "\n")
    
    # 保存结果
    if args.output_summary:
        output_dir = Path(args.output_summary).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(args.output_summary, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {args.output_summary}")
        
        # 同时保存详细的pred_pairs（用于后续处理）
        pred_file = str(args.output_summary).replace('.json', '_preds.jsonl')
        with open(pred_file, 'w', encoding='utf-8') as f:
            for item in pred_pairs_all:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Predictions saved to: {pred_file}")


if __name__ == '__main__':
    main()
