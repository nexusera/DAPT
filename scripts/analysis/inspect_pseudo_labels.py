#!/usr/bin/env python3
"""
数据探查脚本：检查伪标签数据的有效性和质量
"""
import json
import argparse
from collections import Counter, defaultdict
import random

def parse_args():
    parser = argparse.ArgumentParser(description="探查伪标签数据质量")
    parser.add_argument("--data_file", type=str, 
                       default="/data/ocean/DAPT/data/pseudo_kv_labels.json",
                       help="伪标签数据文件路径")
    parser.add_argument("--sample_size", type=int, default=10,
                       help="随机抽样查看的样本数量")
    return parser.parse_args()

def load_data(file_path):
    """加载数据文件"""
    print(f"正在加载数据文件: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ 成功加载，共 {len(data)} 条样本\n")
        return data
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None

def check_basic_stats(data):
    """检查基本统计信息"""
    print("=" * 60)
    print("1. 基本统计信息")
    print("=" * 60)
    
    total_samples = len(data)
    print(f"总样本数: {total_samples}")
    
    # 检查数据格式
    valid_samples = 0
    invalid_samples = []
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            invalid_samples.append((i, "不是字典类型"))
            continue
        if "id" not in item:
            invalid_samples.append((i, "缺少 'id' 字段"))
            continue
        if "annotations" not in item:
            invalid_samples.append((i, "缺少 'annotations' 字段"))
            continue
        if not isinstance(item["annotations"], list) or len(item["annotations"]) == 0:
            invalid_samples.append((i, "annotations 为空或不是列表"))
            continue
        
        annotation = item["annotations"][0]
        if "result" not in annotation:
            invalid_samples.append((i, "缺少 'result' 字段"))
            continue
        
        valid_samples += 1
    
    print(f"有效样本数: {valid_samples}")
    print(f"无效样本数: {len(invalid_samples)}")
    
    if invalid_samples:
        print(f"\n⚠️  前 5 个无效样本:")
        for idx, reason in invalid_samples[:5]:
            print(f"  样本 {idx}: {reason}")
    
    return valid_samples, invalid_samples

def analyze_kv_distribution(data):
    """分析键值对分布"""
    print("\n" + "=" * 60)
    print("2. 键值对分布分析")
    print("=" * 60)
    
    key_counter = Counter()
    total_kv_pairs = 0
    kv_count_per_sample = []
    empty_samples = 0
    
    for item in data:
        try:
            results = item.get("annotations", [{}])[0].get("result", [])
            
            # 统计键名
            kv_pairs_in_sample = 0
            for result in results:
                if result.get("type") == "labels":
                    labels = result.get("value", {}).get("labels", [])
                    if "键名" in labels:
                        key_text = result.get("value", {}).get("text", "")
                        if key_text:
                            key_counter[key_text] += 1
                            kv_pairs_in_sample += 1
            
            kv_count_per_sample.append(kv_pairs_in_sample)
            total_kv_pairs += kv_pairs_in_sample
            
            if kv_pairs_in_sample == 0:
                empty_samples += 1
        except Exception as e:
            continue
    
    print(f"总键值对数量: {total_kv_pairs}")
    print(f"平均每个样本的键值对数: {total_kv_pairs / len(data):.2f}")
    print(f"空样本数（无键值对）: {empty_samples}")
    
    if kv_count_per_sample:
        print(f"键值对数量分布:")
        print(f"  最少: {min(kv_count_per_sample)}")
        print(f"  最多: {max(kv_count_per_sample)}")
        print(f"  中位数: {sorted(kv_count_per_sample)[len(kv_count_per_sample)//2]}")
    
    # 显示最常见的键名
    print(f"\n最常见的键名（Top 20）:")
    for key, count in key_counter.most_common(20):
        print(f"  {key}: {count} 次")

def check_data_quality(data):
    """检查数据质量"""
    print("\n" + "=" * 60)
    print("3. 数据质量检查")
    print("=" * 60)
    
    issues = defaultdict(list)
    
    for i, item in enumerate(data):
        try:
            results = item.get("annotations", [{}])[0].get("result", [])
            
            # 检查键值对是否配对
            key_ids = set()
            value_ids = set()
            relations = []
            
            for result in results:
                if result.get("type") == "labels":
                    labels = result.get("value", {}).get("labels", [])
                    result_id = result.get("id", "")
                    text = result.get("value", {}).get("text", "")
                    
                    if "键名" in labels:
                        key_ids.add(result_id)
                        if not text or not text.strip():
                            issues["空键名"].append(i)
                    elif "值" in labels:
                        value_ids.add(result_id)
                        if not text or not text.strip():
                            issues["空值"].append(i)
                
                elif result.get("type") == "relation":
                    from_id = result.get("from_id")
                    to_id = result.get("to_id")
                    if from_id and to_id:
                        relations.append((from_id, to_id))
            
            # 检查关系是否完整
            for from_id, to_id in relations:
                if from_id not in key_ids:
                    issues["关系中的键ID不存在"].append(i)
                if to_id not in value_ids:
                    issues["关系中的值ID不存在"].append(i)
            
            # 检查是否有键没有对应的值
            related_keys = {r[0] for r in relations}
            orphan_keys = key_ids - related_keys
            if orphan_keys:
                issues["孤立键（无对应值）"].append(i)
            
            # 检查是否有值没有对应的键
            related_values = {r[1] for r in relations}
            orphan_values = value_ids - related_values
            if orphan_values:
                issues["孤立值（无对应键）"].append(i)
                
        except Exception as e:
            issues["解析异常"].append((i, str(e)))
    
    print("发现的问题:")
    if issues:
        for issue_type, sample_indices in issues.items():
            unique_count = len(set(sample_indices))
            print(f"  {issue_type}: {unique_count} 个样本")
            if unique_count <= 5:
                print(f"    样本索引: {list(set(sample_indices))}")
    else:
        print("  ✅ 未发现明显问题")

def show_samples(data, sample_size=10):
    """随机抽样显示样例"""
    print("\n" + "=" * 60)
    print(f"4. 随机抽样查看（{sample_size} 个样例）")
    print("=" * 60)
    
    if len(data) == 0:
        print("数据为空，无法抽样")
        return
    
    sample_indices = random.sample(range(len(data)), min(sample_size, len(data)))
    
    for idx, sample_idx in enumerate(sample_indices, 1):
        item = data[sample_idx]
        print(f"\n--- 样例 {idx} (索引 {sample_idx}) ---")
        print(f"ID: {item.get('id', 'N/A')}")
        
        try:
            results = item.get("annotations", [{}])[0].get("result", [])
            
            # 提取键值对
            kv_pairs = {}
            relations = {}
            
            for result in results:
                if result.get("type") == "labels":
                    result_id = result.get("id", "")
                    labels = result.get("value", {}).get("labels", [])
                    text = result.get("value", {}).get("text", "")
                    
                    if "键名" in labels:
                        kv_pairs[result_id] = {"type": "key", "text": text}
                    elif "值" in labels:
                        kv_pairs[result_id] = {"type": "value", "text": text}
                
                elif result.get("type") == "relation":
                    from_id = result.get("from_id")
                    to_id = result.get("to_id")
                    relations[from_id] = to_id
            
            # 显示键值对
            print("键值对:")
            for key_id, key_info in kv_pairs.items():
                if key_info["type"] == "key":
                    value_id = relations.get(key_id)
                    if value_id and value_id in kv_pairs:
                        value_text = kv_pairs[value_id]["text"]
                        print(f"  {key_info['text']}: {value_text}")
                    else:
                        print(f"  {key_info['text']}: (无对应值)")
            
            if not kv_pairs:
                print("  (无键值对)")
                
        except Exception as e:
            print(f"  ❌ 解析失败: {e}")

def check_compatibility(data):
    """检查与 dataset.py 的兼容性"""
    print("\n" + "=" * 60)
    print("5. 与 dataset.py 兼容性检查")
    print("=" * 60)
    
    # 检查必需的字段
    required_fields = ["id", "annotations"]
    required_annotation_fields = ["result"]
    required_result_fields = ["id", "type", "value"]
    
    compatible_count = 0
    incompatible_reasons = []
    
    for i, item in enumerate(data):
        try:
            # 检查顶层字段
            for field in required_fields:
                if field not in item:
                    incompatible_reasons.append((i, f"缺少顶层字段: {field}"))
                    break
            else:
                # 检查 annotations
                annotations = item.get("annotations", [])
                if not annotations:
                    incompatible_reasons.append((i, "annotations 为空"))
                    continue
                
                annotation = annotations[0]
                if "result" not in annotation:
                    incompatible_reasons.append((i, "缺少 result 字段"))
                    continue
                
                results = annotation.get("result", [])
                if not results:
                    incompatible_reasons.append((i, "result 为空"))
                    continue
                
                # 检查 result 中的字段
                for result in results:
                    for field in required_result_fields:
                        if field not in result:
                            incompatible_reasons.append((i, f"result 中缺少字段: {field}"))
                            break
                
                compatible_count += 1
        except Exception as e:
            incompatible_reasons.append((i, f"检查异常: {e}"))
    
    print(f"兼容样本数: {compatible_count} / {len(data)}")
    print(f"兼容率: {compatible_count / len(data) * 100:.2f}%")
    
    if incompatible_reasons:
        print(f"\n⚠️  不兼容样本（前 5 个）:")
        for idx, reason in incompatible_reasons[:5]:
            print(f"  样本 {idx}: {reason}")

def check_useful_samples(data):
    """检查真正有用的样本（过滤掉包含无效键名的样本）"""
    print("\n" + "=" * 60)
    print("6. 有用样本统计（过滤无效键名）")
    print("=" * 60)
    
    # 定义无效键名（这些是 JSON 结构字段或示例字段，不是真实提取的键名）
    invalid_keys = {"id", "value", "name", "text", "key", "type", "label", "labels"}
    
    useful_samples = []
    useless_samples = []
    
    for i, item in enumerate(data):
        try:
            results = item.get("annotations", [{}])[0].get("result", [])
            
            # 提取所有键名
            keys = []
            has_invalid_key = False
            
            for result in results:
                if result.get("type") == "labels":
                    labels = result.get("value", {}).get("labels", [])
                    if "键名" in labels:
                        key_text = result.get("value", {}).get("text", "").strip()
                        if key_text:
                            keys.append(key_text)
                            # 检查是否是无效键名
                            if key_text.lower() in invalid_keys:
                                has_invalid_key = True
            
            if has_invalid_key or not keys:
                useless_samples.append((i, keys))
            else:
                useful_samples.append((i, keys))
        except Exception:
            useless_samples.append((i, []))
    
    print(f"有用样本数: {len(useful_samples)} / {len(data)}")
    print(f"有用样本比例: {len(useful_samples) / len(data) * 100:.2f}%")
    print(f"无用样本数: {len(useless_samples)}")
    
    # 分析无用样本的原因
    invalid_key_count = 0
    empty_key_count = 0
    
    for idx, keys in useless_samples:
        if not keys:
            empty_key_count += 1
        else:
            # 检查是否包含无效键名
            has_invalid = any(k.lower() in invalid_keys for k in keys)
            if has_invalid:
                invalid_key_count += 1
    
    print(f"\n无用样本原因:")
    print(f"  包含无效键名（id/value/name/text/key等）: {invalid_key_count}")
    print(f"  无键名: {empty_key_count}")
    
    # 显示一些有用样本的键名示例
    if useful_samples:
        print(f"\n有用样本的键名示例（前 10 个）:")
        for idx, keys in useful_samples[:10]:
            print(f"  样本 {idx}: {', '.join(keys[:5])}")  # 只显示前5个键名

def main():
    args = parse_args()
    
    # 加载数据
    data = load_data(args.data_file)
    if data is None:
        return
    
    if len(data) == 0:
        print("❌ 数据文件为空")
        return
    
    # 基本统计
    valid_samples, invalid_samples = check_basic_stats(data)
    
    if valid_samples == 0:
        print("\n❌ 没有有效样本，请检查数据格式")
        return
    
    # 键值对分布
    analyze_kv_distribution(data)
    
    # 数据质量
    check_data_quality(data)
    
    # 兼容性检查
    check_compatibility(data)
    
    # 检查真正有用的样本
    check_useful_samples(data)
    
    # 随机抽样
    show_samples(data, args.sample_size)
    
    print("\n" + "=" * 60)
    print("✅ 数据探查完成")
    print("=" * 60)

if __name__ == "__main__":
    main()

