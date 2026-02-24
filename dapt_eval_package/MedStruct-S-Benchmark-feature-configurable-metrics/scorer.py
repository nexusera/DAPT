import argparse
import json
import sys
import os
import logging
import datetime
from collections import defaultdict

# 将项目根目录集成到系统路径，确保 med_eval 模块可被正确导入
sys.path.insert(0, os.getcwd())

# 导入评测引擎包中的调度函数
from med_eval.engine import run_evaluation

# 配置全局日志格式
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_jsonl(filepath):
    """
    加载 JSONL 格式的文件。
    逐行解析并处理可能的 JSON 格式错误或空行。
    """
    data = []
    if not filepath or not os.path.exists(filepath):
        logger.error(f"文件未找到: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    logger.warning(f"行解析失败: {line[:50]}... 错误: {e}")
    return data

def load_schema(p):
    """
    加载键名别名映射表 (Schema)。
    用于 Task 2 (值提取) 的别名对齐和标准字段列表确定。
    """
    if not p or not os.path.exists(p):
        logger.warning(f"Schema 文件未找到，Task 2 可能会受限: {p}")
        return {}
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载 Schema 失败: {e}")
        return {}


def main():
    """
    评测主入口：
    1. 解析命令行参数，配置评测引擎行为。
    2. 加载预测文件和真值文件。
    3. 执行标准化转换和任务特定的数据过滤。
    4. 调用引擎计算指标并生成扁平化的标准化 JSON 报告。
    """
    parser = argparse.ArgumentParser(description="医疗结构化数据评测工具 (Unified Modular Scorer)")
    
    # 基础文件 I/O 参数
    parser.add_argument("--pred_file", required=True, help="模型预测结果文件 (.jsonl)")
    parser.add_argument("--gt_file", required=True, help="人工标注真值文件 (.jsonl)")
    parser.add_argument("--schema_file", default="data/kv_ner_prepared_comparison/keys_merged_1027_cleaned.json", help="别名映射及标准字段表")
    parser.add_argument("--output_file", default=None, help="结果 JSON 保存路径")
    parser.add_argument("--task_type", default="all", choices=["task1", "task2", "task3", "all"], help="运行指定任务评测")
    
    # 算法行为控制：通过参数控制归一化、动态阈值和位置校验
    parser.add_argument("--no_normalize", action="store_false", dest="normalize", help="禁用文本归一化（转小写、去空格）")
    parser.add_argument("--similarity_threshold", type=float, default=0.8, help="NED 相似度判定阈值")
    parser.add_argument("--overlap_threshold", type=float, default=0.0, help="Span IoU 重叠度阈值")
    parser.add_argument("--disable_tau", action="store_false", dest="tau_dynamic", help="禁用 Tau 长度自适应动态阈值")
    
    # 元数据信息（仅用于结果汇总报告）
    parser.add_argument("--model_name", default=None, help="模型名称标识")
    parser.add_argument("--dataset_type", default="Original", help="数据集类型标识 (Original/DA)")
    
    parser.set_defaults(normalize=True, tau_dynamic=True)
    args = parser.parse_args()

    # Load IO
    logger.info(f"正在加载预测文件: {args.pred_file}")
    predictions = load_jsonl(args.pred_file)
    logger.info(f"正在加载真值文件: {args.gt_file}")
    ground_truth = load_jsonl(args.gt_file)
    
    # Assert sample counts match
    if len(predictions) != len(ground_truth):
        logger.error(f"Sample count mismatch: Preds={len(predictions)}, GT={len(ground_truth)}")
        sys.exit(1)
        
    num_samples = len(predictions)
    logger.info(f"共处理 {num_samples} 条样本进行对比。")
    
    # Load Schema
    key_alias_map = load_schema(args.schema_file)
    
    # Config object for reporting only
    report_config = {
        "normalize": args.normalize,
        "similarity_threshold": args.similarity_threshold,
        "overlap_threshold": args.overlap_threshold,
        "tau_dynamic": args.tau_dynamic,
        "use_em": True,
        "use_am": True,
        "use_span": True
    }

    # Execute
    logger.info(f"正在启动任务评测: {args.task_type}...")
    
    results = run_evaluation(
        predictions=predictions,
        ground_truth=ground_truth,
        key_alias_map=key_alias_map,
        task_type=args.task_type,
        normalize=args.normalize,
        tau_dynamic=args.tau_dynamic,
        similarity_threshold=args.similarity_threshold,
        overlap_threshold=args.overlap_threshold
    )

    # Assemble final standardized report
    final_report = {
        "summary": {
            "model": args.model_name or os.path.basename(args.pred_file),
            "dataset": args.dataset_type,
            "samples": num_samples,
            "eval_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": report_config
        },
        "tasks": {}
    }

    # 结果字段映射表，用于美化 JSON 键名
    task_map = {
        "Task 1 (Key Discovery)": "task1",
        "Task 2 (Value Extraction)": "task2",
        "Task 3 (E2E Pairing)": "task3"
    }

    # 扁平化数据处理逻辑：将 Task 2 的两个维度提升至顶级任务列表
    for raw_name, data in results.items():
        clean_key = task_map.get(raw_name, raw_name.lower().replace(" ", "_"))
        if clean_key == "task2":
            # 将 Task 2 的 Global/Pos 维度拆分，方便用户解析
            final_report["tasks"]["task2_global"] = data["global"]
            final_report["tasks"]["task2_pos_only"] = data["pos_only"]
        else:
            final_report["tasks"][clean_key] = data

    # 序列化为 JSON 字符串并输出
    report_json = json.dumps(final_report, indent=2, ensure_ascii=False)
    print(report_json)
    
    # 如果指定了输出路径，将结果持久化
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(report_json)
        logger.info(f"报表已保存至: {args.output_file}")

if __name__ == "__main__":
    main()
