#!/bin/bash
# -*- coding: utf-8 -*-
"""
evaluate_dapt_examples.sh - DAPT模型评估示例脚本集合

包含多个真实场景的评估命令，可直接复制使用。
"""

# ============================================================================
# 示例 1: 基础评估 - 标准BERT (不含DAPT特性)
# ============================================================================
echo "Example 1: 评估标准BERT模型"
echo "Command:"
echo ""
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path hfl/chinese-roberta-wwm-ext \
    --test_data data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
    --output_summary runs/eval_roberta_baseline.json \
    --batch_size 32 \
    --max_length 512


# ============================================================================
# 示例 2: DAPT评估 - 完整配置
# ============================================================================
echo ""
echo "Example 2: 评估DAPT预训练模型（完整配置）"
echo "Command:"
echo ""
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path /data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu/final_model \
    --test_data data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
    --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
    --output_summary runs/eval_dapt_baseline.json \
    --batch_size 32 \
    --max_length 512 \
    --device cuda:0 \
    --seed 42


# ============================================================================
# 示例 3: 小数据集快速测试
# ============================================================================
echo ""
echo "Example 3: 快速测试 (小数据集)"
echo "Command:"
echo ""
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path /data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu/final_model \
    --test_data data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
    --noise_bins_json /data/ocean/DAPT/workspace/noise_bins.json \
    --output_summary runs/eval_dapt_quick.json \
    --batch_size 64 \
    --max_length 512 \
    --device cuda:0


# ============================================================================
# 示例 4: CPU推理 (低资源环境)
# ============================================================================
echo ""
echo "Example 4: CPU推理 (无GPU环境)"
echo "Command:"
echo ""
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path hfl/chinese-roberta-wwm-ext \
    --test_data data/kv_ner_prepared_comparison/val_eval_titled.jsonl \
    --output_summary runs/eval_roberta_cpu.json \
    --batch_size 8 \
    --device cpu


# ============================================================================
# 示例 5: 批量评估脚本
# ============================================================================
echo ""
echo "Example 5: 批量评估多个模型"
echo ""

cat > batch_evaluate_all_models.sh << 'BATCH_SCRIPT'
#!/bin/bash
set -e

# 配置
TEST_DATA="data/kv_ner_prepared_comparison/val_eval_titled.jsonl"
NOISE_BINS="/data/ocean/DAPT/workspace/noise_bins.json"
OUTPUT_DIR="runs"
BATCH_SIZE=32

echo "=========================================="
echo "批量评估所有模型"
echo "=========================================="
echo ""

# 1. 标准BERT基线
echo "[1/3] 评估 RoBERTa (标准BERT基线)..."
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path hfl/chinese-roberta-wwm-ext \
    --test_data "$TEST_DATA" \
    --output_summary "${OUTPUT_DIR}/eval_roberta.json" \
    --batch_size "$BATCH_SIZE"

# 2. MacBERT基线
echo "[2/3] 评估 MacBERT (标准BERT基线)..."
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path hfl/chinese-macbert-base \
    --test_data "$TEST_DATA" \
    --output_summary "${OUTPUT_DIR}/eval_macbert.json" \
    --batch_size "$BATCH_SIZE"

# 3. DAPT模型
echo "[3/3] 评估 DAPT预训练模型..."
python pre_struct/kv_ner/evaluate_with_dapt_noise.py \
    --model_path /data/ocean/DAPT/workspace/output_medical_bert_v2_8gpu/final_model \
    --test_data "$TEST_DATA" \
    --noise_bins_json "$NOISE_BINS" \
    --output_summary "${OUTPUT_DIR}/eval_dapt.json" \
    --batch_size "$BATCH_SIZE"

echo ""
echo "=========================================="
echo "评估完成！结果文件："
echo "- ${OUTPUT_DIR}/eval_roberta.json"
echo "- ${OUTPUT_DIR}/eval_macbert.json"
echo "- ${OUTPUT_DIR}/eval_dapt.json"
echo "=========================================="

BATCH_SCRIPT

chmod +x batch_evaluate_all_models.sh
echo "脚本已创建: batch_evaluate_all_models.sh"


# ============================================================================
# 示例 6: 结果对比脚本
# ============================================================================
echo ""
echo "Example 6: 生成结果对比报告"
echo ""

cat > compare_eval_results.py << 'COMPARE_SCRIPT'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比多个模型的评估结果
"""

import json
import sys
from pathlib import Path

def load_result(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_metrics(metrics):
    return f"P={metrics['p']:.4f} R={metrics['r']:.4f} F1={metrics['f1']:.4f}"

def main():
    results = {}
    
    # 加载结果
    for filepath in sys.argv[1:]:
        name = Path(filepath).stem.replace('eval_', '')
        try:
            results[name] = load_result(filepath)
            print(f"✓ 加载 {filepath}")
        except Exception as e:
            print(f"✗ 加载失败 {filepath}: {e}")
            continue
    
    if not results:
        print("未找到任何结果文件")
        return
    
    # 打印对比表
    print("\n" + "="*100)
    print("模型评估结果对比")
    print("="*100 + "\n")
    
    # Task 1
    print("Task 1: Keys Discovery (属性发现)")
    print("-" * 100)
    for model_name, result in results.items():
        t1_strict = result.get('task1', {}).get('strict', {})
        t1_loose = result.get('task1', {}).get('loose', {})
        print(f"{model_name:20s} | Strict: {format_metrics(t1_strict):40s} | Loose: {format_metrics(t1_loose)}")
    
    # Task 2
    print("\n" + "-" * 100)
    print("Task 2: Key-Value Pairs (键值对提取)")
    print("-" * 100)
    for model_name, result in results.items():
        t2_ss = result.get('task2', {}).get('strict_strict', {})
        print(f"{model_name:20s} | Strict-Strict: {format_metrics(t2_ss)}")
    
    # 性能提升计算
    if len(results) >= 2:
        model_names = list(results.keys())
        baseline = model_names[0]
        
        print("\n" + "="*100)
        print(f"相对于 {baseline} 的性能提升")
        print("="*100 + "\n")
        
        baseline_t2_f1 = results[baseline].get('task2', {}).get('strict_strict', {}).get('f1', 0)
        
        for model_name in model_names[1:]:
            current_t2_f1 = results[model_name].get('task2', {}).get('strict_strict', {}).get('f1', 0)
            improvement = (current_t2_f1 - baseline_t2_f1) / baseline_t2_f1 * 100 if baseline_t2_f1 > 0 else 0
            
            symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
            print(f"{model_name:20s} Task 2 F1: {symbol} {abs(improvement):+.2f}%")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compare_eval_results.py <result1.json> [<result2.json> ...]")
        sys.exit(1)
    main()

COMPARE_SCRIPT

chmod +x compare_eval_results.py
echo "对比脚本已创建: compare_eval_results.py"


# ============================================================================
# 示例 7: 查看详细预测错误分析
# ============================================================================
echo ""
echo "Example 7: 错误分析脚本"
echo ""

cat > analyze_predictions.py << 'ANALYZE_SCRIPT'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析预测结果，找出常见错误类型
"""

import json
import sys
from collections import Counter

def analyze_preds_file(pred_file, gt_file, top_n=20):
    """分析预测文件中的错误"""
    
    # 加载预测和真值
    with open(pred_file, 'r', encoding='utf-8') as f:
        preds = [json.loads(line) for line in f]
    
    with open(gt_file, 'r', encoding='utf-8') as f:
        gts = [json.loads(line) for line in f]
    
    # 统计
    missing_keys = Counter()  # 预测漏掉的key
    extra_keys = Counter()    # 预测多出来的key
    wrong_values = Counter()  # value错误的key
    
    for pred, gt in zip(preds, gts):
        pred_pairs = set((k, v) for k, v in pred.get('pred_pairs', []))
        gt_pairs = set((k, v) for k, v in gt.get('spans', {}).items() if v.get('text'))
        
        # 分析误差
        for k, v in gt_pairs:
            if (k, v) not in pred_pairs:
                missing_keys[k] += 1
        
        for k, v in pred_pairs:
            if (k, v) not in gt_pairs:
                extra_keys[k] += 1
    
    # 打印报告
    print("\n最容易漏掉的属性 (Top 20):")
    for k, count in missing_keys.most_common(top_n):
        print(f"  {k:30s} - 漏掉 {count} 次")
    
    print("\n最容易多预测的属性 (Top 20):")
    for k, count in extra_keys.most_common(top_n):
        print(f"  {k:30s} - 多预 {count} 次")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python analyze_predictions.py <preds.jsonl> <gt.jsonl>")
        sys.exit(1)
    analyze_preds_file(sys.argv[1], sys.argv[2])

ANALYZE_SCRIPT

chmod +x analyze_predictions.py
echo "分析脚本已创建: analyze_predictions.py"


# ============================================================================
# 总结
# ============================================================================
echo ""
echo "=================================================="
echo "示例脚本创建完成！"
echo "=================================================="
echo ""
echo "可用的脚本和用途："
echo "1. evaluate_dapt_examples.sh - 本文件，包含7个示例"
echo "2. batch_evaluate_all_models.sh - 批量评估3个模型"
echo "3. compare_eval_results.py - 对比评估结果"
echo "4. analyze_predictions.py - 分析预测错误"
echo ""
echo "快速开始："
echo "  bash batch_evaluate_all_models.sh"
echo "  python compare_eval_results.py runs/eval_*.json"
echo ""
echo "更多信息请参考: pre_struct/kv_ner/EVALUATE_WITH_DAPT.md"
echo "=================================================="
