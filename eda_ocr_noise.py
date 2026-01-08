import json
import numpy as np
import pandas as pd
import os
from pathlib import Path

# ================= 配置绝对路径 =================
DATA_PATH = "/home/your_user_name/semi_label/ocr_rerun/char_ocr_9297.json" # 请替换 your_user_name
OUTPUT_DIR = "/data/ocean/DAPT/eda_results"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def calculate_features(item):
    words = item.get("words_result", [])
    if not words: return None
    
    features_list = []
    # 模拟 align_score 的环境（需根据实际JSON结构取值）
    # 这里假设 top 信息在 location 中
    for w in words:
        prob = w.get("probability", {})
        conf_avg = prob.get("average", 0)
        conf_min = prob.get("min", 0)
        conf_var = prob.get("variance", 0)
        
        # 1-4. 置信度相关
        f1 = conf_avg
        f2 = conf_min
        f3 = np.log10(conf_var + 1e-12)
        f4 = conf_avg - conf_min
        
        # 5. punct_err_ratio
        text = w.get("words", "")
        char_cnt = len(text)
        if char_cnt > 0:
            bad_chars = sum(1 for c in text if not (u'\u4e00' <= c <= u'\u9fa5' or c.isalnum()))
            f5 = bad_chars / char_cnt
        else:
            f5 = 0
            
        # 6. char_break_ratio
        width = w.get("location", {}).get("width", 1)
        f6 = char_cnt / max(width, 1)
        
        # 7. align_score (示例逻辑：这里建议之后根据 paragraph 结果细化)
        f7 = w.get("location", {}).get("top", 0) # 暂时取 top 作为观察，后续对齐
        
        features_list.append([f1, f2, f3, f4, f5, f6, f7])
    return features_list

def run_eda():
    print(f"\n[1/3] 正在读取数据: {DATA_PATH} ...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_features = []
    for item in data:
        feats = calculate_features(item)
        if feats: all_features.extend(feats)
    
    df = pd.DataFrame(all_features, columns=[
        'conf_avg', 'conf_min', 'conf_var_log', 'conf_gap', 
        'punct_err_ratio', 'char_break_ratio', 'align_score'
    ])

    # ================= 打印统计核心信息 =================
    print("\n" + "="*50)
    print("🔥 OCR NOISE FEATURE DISTRIBUTION REPORT")
    print("="*50)
    
    # 计算统计量
    stats = df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T
    stats['zero_ratio (%)'] = ((df == 0).sum() / len(df) * 100).round(2)
    stats['unique_vals'] = df.nunique()
    
    # 格式化打印
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(stats[['mean', 'std', 'min', '5%', '50%', '95%', '99%', 'max', 'zero_ratio (%)', 'unique_vals']])
    
    print("\n" + "="*50)
    print("💡 异常值探查摘要")
    print(f"总 Token 数: {len(df)}")
    print(f"1. 完美置信度 (conf_avg == 1.0): {(df['conf_avg'] >= 0.999).sum()} ({((df['conf_avg'] >= 0.999).sum()/len(df)*100):.2f}%)")
    print(f"2. 极低置信度 (conf_avg < 0.5):  {(df['conf_avg'] < 0.5).sum()} ({((df['conf_avg'] < 0.5).sum()/len(df)*100):.2f}%)")
    print(f"3. 疑似乱码块 (punct_ratio > 0.5): {(df['punct_err_ratio'] > 0.5).sum()} ({((df['punct_err_ratio'] > 0.5).sum()/len(df)*100):.2f}%)")
    print("="*50 + "\n")

    # 保存文件备查
    stats.to_csv(f"{OUTPUT_DIR}/feature_stats.csv")

if __name__ == "__main__":
    run_eda()