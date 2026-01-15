#!/usr/bin/env python3
"""
过滤伪标签数据，只保留有用的样本（去除包含无效键名的样本）
"""
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="过滤伪标签数据")
    parser.add_argument("--input_file", type=str,
                       default="/data/ocean/DAPT/data/pseudo_kv_labels.json",
                       help="输入的伪标签数据文件")
    parser.add_argument("--output_file", type=str,
                       default="/data/ocean/DAPT/data/pseudo_kv_labels_filtered.json",
                       help="输出的过滤后数据文件")
    parser.add_argument("--invalid_keys", nargs="+",
                       default=["id", "value", "name", "text", "key", "type", "label", "labels"],
                       help="无效键名列表（会被过滤掉）")
    return parser.parse_args()

def is_useful_sample(item, invalid_keys):
    """
    判断样本是否有用
    返回: (is_useful, keys_list)
    """
    try:
        results = item.get("annotations", [{}])[0].get("result", [])
        
        keys = []
        has_invalid_key = False
        
        for result in results:
            if result.get("type") == "labels":
                labels = result.get("value", {}).get("labels", [])
                if "键名" in labels:
                    key_text = result.get("value", {}).get("text", "").strip()
                    if key_text:
                        keys.append(key_text)
                        # 检查是否是无效键名（不区分大小写）
                        if key_text.lower() in [k.lower() for k in invalid_keys]:
                            has_invalid_key = True
        
        # 如果有键名且不包含无效键名，则认为是有用的
        if keys and not has_invalid_key:
            return True, keys
        else:
            return False, keys
    except Exception as e:
        return False, []

def filter_data(input_file, output_file, invalid_keys):
    """过滤数据"""
    print(f"正在加载数据: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"原始样本数: {len(data)}")
    
    useful_samples = []
    useless_samples = []
    
    print("正在过滤数据...")
    for i, item in enumerate(data):
        is_useful, keys = is_useful_sample(item, invalid_keys)
        
        if is_useful:
            useful_samples.append(item)
        else:
            useless_samples.append((i, keys))
        
        # 每处理 10000 条打印一次进度
        if (i + 1) % 10000 == 0:
            print(f"  已处理: {i+1}/{len(data)}, 有用样本: {len(useful_samples)}")
    
    print(f"\n过滤结果:")
    print(f"  有用样本数: {len(useful_samples)} ({len(useful_samples)/len(data)*100:.2f}%)")
    print(f"  无用样本数: {len(useless_samples)} ({len(useless_samples)/len(data)*100:.2f}%)")
    
    # 保存过滤后的数据
    print(f"\n正在保存过滤后的数据到: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(useful_samples, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 过滤完成！已保存 {len(useful_samples)} 条有用样本")
    
    # 统计过滤后的键名分布
    if useful_samples:
        from collections import Counter
        key_counter = Counter()
        for item in useful_samples:
            try:
                results = item.get("annotations", [{}])[0].get("result", [])
                for result in results:
                    if result.get("type") == "labels":
                        labels = result.get("value", {}).get("labels", [])
                        if "键名" in labels:
                            key_text = result.get("value", {}).get("text", "").strip()
                            if key_text:
                                key_counter[key_text] += 1
            except:
                continue
        
        print(f"\n过滤后最常见的键名（Top 10）:")
        for key, count in key_counter.most_common(10):
            print(f"  {key}: {count} 次")

def main():
    args = parse_args()
    
    # 将字符串列表转换为集合（用于快速查找）
    invalid_keys_set = [k.lower() for k in args.invalid_keys]
    
    filter_data(args.input_file, args.output_file, invalid_keys_set)

if __name__ == "__main__":
    main()

