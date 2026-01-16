#!/usr/bin/env python3
"""
统计自动标注数据中每个键名（key）的出现频次，并根据阈值过滤
用于确保BERT模型能够充分学习到键名的embedding

基于BERT学习规律：
- 一个词至少出现5-10次才能学到基本的表示
- 对于下游任务（如结构化抽取），建议10-20次以上更保险
- 如果某个键名出现频率太低，模型可能学不到它的模式
"""
import json
import argparse
from collections import Counter
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="统计键名频次并过滤",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认阈值（10次）
  python analyze_key_frequency.py --input /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json
  
  # 自定义阈值
  python analyze_key_frequency.py --input /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json --min_freq 5
  
  # 输出到指定文件
  python analyze_key_frequency.py --input /data/ocean/DAPT/data/pseudo_kv_labels_filtered.json --output keys_filtered.txt
        """
    )
    parser.add_argument("--input", type=str,
                       default="/data/ocean/DAPT/data/pseudo_kv_labels_filtered.json",
                       help="输入的伪标签数据文件（JSON格式）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出的键名列表文件（每行一个键名，默认不保存）")
    parser.add_argument("--min_freq", type=int, default=10,
                       help="最小频次阈值（建议10-20，默认10）")
    parser.add_argument("--show_stats", action="store_true", default=True,
                       help="显示详细统计信息（默认启用）")
    parser.add_argument("--save_json", type=str, default=None,
                       help="保存频次统计到JSON文件（可选）")
    return parser.parse_args()

def extract_keys_from_data(data_file):
    """从伪标签数据中提取所有键名并统计频次"""
    print(f"正在加载数据文件: {data_file}")
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"✅ 成功加载 {len(data)} 条样本")
    
    key_counter = Counter()
    total_kv_pairs = 0
    samples_with_keys = 0
    
    print("正在统计键名频次...")
    for i, item in enumerate(data):
        try:
            results = item.get("annotations", [{}])[0].get("result", [])
            
            keys_in_sample = []
            for result in results:
                if result.get("type") == "labels":
                    labels = result.get("value", {}).get("labels", [])
                    if "键名" in labels:
                        key_text = result.get("value", {}).get("text", "").strip()
                        if key_text:
                            keys_in_sample.append(key_text)
                            key_counter[key_text] += 1
                            total_kv_pairs += 1
            
            if keys_in_sample:
                samples_with_keys += 1
        
        except Exception as e:
            continue
        
        # 每处理 10000 条打印一次进度
        if (i + 1) % 10000 == 0:
            print(f"  已处理: {i+1}/{len(data)}")
    
    print(f"✅ 统计完成")
    print(f"  总键值对数: {total_kv_pairs}")
    print(f"  包含键名的样本数: {samples_with_keys}")
    print(f"  唯一键名数: {len(key_counter)}\n")
    
    return key_counter, total_kv_pairs, samples_with_keys

def filter_keys_by_frequency(key_counter, min_freq):
    """根据频次阈值过滤键名"""
    filtered_keys = {}
    dropped_keys = {}
    
    for key, count in key_counter.items():
        if count >= min_freq:
            filtered_keys[key] = count
        else:
            dropped_keys[key] = count
    
    return filtered_keys, dropped_keys

def print_statistics(key_counter, filtered_keys, dropped_keys, min_freq, total_kv_pairs):
    """打印详细的统计信息"""
    print("=" * 70)
    print("键名频次统计结果")
    print("=" * 70)
    
    # 总体统计
    total_keys = len(key_counter)
    kept_keys = len(filtered_keys)
    dropped_keys_count = len(dropped_keys)
    
    print(f"\n【总体统计】")
    print(f"  唯一键名总数: {total_keys}")
    print(f"  保留的键名数（频次 >= {min_freq}）: {kept_keys} ({kept_keys/total_keys*100:.2f}%)")
    print(f"  过滤的键名数（频次 < {min_freq}）: {dropped_keys_count} ({dropped_keys_count/total_keys*100:.2f}%)")
    
    # 频次覆盖统计
    total_occurrences = sum(key_counter.values())
    kept_occurrences = sum(filtered_keys.values())
    dropped_occurrences = sum(dropped_keys.values())
    
    print(f"\n【频次覆盖统计】")
    print(f"  总键值对出现次数: {total_occurrences}")
    print(f"  保留键名的覆盖次数: {kept_occurrences} ({kept_occurrences/total_occurrences*100:.2f}%)")
    print(f"  过滤键名的覆盖次数: {dropped_occurrences} ({dropped_occurrences/total_occurrences*100:.2f}%)")
    
    # 频次分布统计
    print(f"\n【频次分布统计】")
    freq_ranges = [
        (1, 1, "仅出现1次"),
        (2, 5, "出现2-5次"),
        (6, 10, "出现6-10次"),
        (11, 20, "出现11-20次"),
        (21, 50, "出现21-50次"),
        (51, 100, "出现51-100次"),
        (101, float('inf'), "出现101次以上"),
    ]
    
    for min_r, max_r, desc in freq_ranges:
        count = sum(1 for c in key_counter.values() if min_r <= c <= max_r)
        if count > 0:
            print(f"  {desc}: {count} 个键名")
    
    # Top 20 最常见的键名
    print(f"\n【最常见的键名（Top 20）】")
    for key, count in key_counter.most_common(20):
        status = "✅" if count >= min_freq else "❌"
        print(f"  {status} {key}: {count} 次")
    
    # 被过滤的低频键名示例
    if dropped_keys:
        print(f"\n【被过滤的低频键名示例（Top 10）】")
        sorted_dropped = sorted(dropped_keys.items(), key=lambda x: x[1], reverse=True)
        for key, count in sorted_dropped[:10]:
            print(f"  {key}: {count} 次")

def save_results(filtered_keys, output_file, save_json_file=None):
    """保存过滤后的键名列表"""
    if output_file:
        print(f"\n正在保存键名列表到: {output_file}")
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        # 按频次降序保存
        sorted_keys = sorted(filtered_keys.items(), key=lambda x: x[1], reverse=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            for key, count in sorted_keys:
                f.write(f"{key}\n")
        
        print(f"✅ 已保存 {len(filtered_keys)} 个键名")
    
    if save_json_file:
        print(f"\n正在保存频次统计到: {save_json_file}")
        os.makedirs(os.path.dirname(save_json_file) if os.path.dirname(save_json_file) else ".", exist_ok=True)
        
        with open(save_json_file, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(filtered_keys.items(), key=lambda x: x[1], reverse=True)), 
                     f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已保存频次统计")

def main():
    args = parse_args()
    
    # 提取键名并统计频次
    key_counter, total_kv_pairs, samples_with_keys = extract_keys_from_data(args.input)
    
    if not key_counter:
        print("❌ 未找到任何键名，请检查数据格式")
        return
    
    # 根据阈值过滤
    filtered_keys, dropped_keys = filter_keys_by_frequency(key_counter, args.min_freq)
    
    # 打印统计信息
    if args.show_stats:
        print_statistics(key_counter, filtered_keys, dropped_keys, args.min_freq, total_kv_pairs)
    
    # 保存结果
    if args.output or args.save_json:
        save_results(filtered_keys, args.output, args.save_json)
    
    # 建议
    print("\n" + "=" * 70)
    print("【BERT学习效果建议】")
    print("=" * 70)
    print(f"当前阈值: {args.min_freq} 次")
    
    if args.min_freq >= 10:
        print("✅ 阈值设置合理")
        print("   对于结构化抽取任务，键名出现 >= 10 次通常能保证BERT学到基本表示")
    elif args.min_freq >= 5:
        print("⚠️  阈值偏低，建议提高到 10 次")
        print("   5次可能刚刚够，但对于关键任务建议使用更高的阈值")
    else:
        print("❌ 阈值过低，强烈建议提高到至少 10 次")
        print("   过低的阈值可能导致BERT学不到键名的有效表示")
    
    print(f"\n💡 建议：")
    print(f"   - 如果训练epochs较少（1-3轮），建议阈值 >= 10")
    print(f"   - 如果训练epochs较多（5轮以上），阈值 5-10 也可以")
    print(f"   - 对于关键医疗术语，建议阈值 >= 20 更保险")
    
    print("\n" + "=" * 70)
    print("✅ 分析完成")
    print("=" * 70)

if __name__ == "__main__":
    main()

