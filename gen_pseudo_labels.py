import os
import json
import pandas as pd
from vllm import LLM, SamplingParams
import argparse

# ---------------------------------------------------------------------- #
# 配置区
# ---------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser()
    # 指向你的 LoRA 微调后权重目录
    parser.add_argument("--model_path", type=str, default="/data/ocean/medstruct_s_llm/results/lora/Qwen3-0.6B_task2")
    # 输出 JSON 路径，确保 /data/ocean/DAPT/data 目录存在
    parser.add_argument("--output_file", type=str, default="/data/ocean/DAPT/data/pseudo_kv_labels.json")
    # CSV 所在目录列表
    parser.add_argument("--data_dirs", nargs="+", default=[
        "/data/oss/hxzh-mr/all_type_pic_oss_csv/20251204_5w",
        "/data/oss/hxzh-mr/all_type_pic_oss_csv/20251206_5w",
        "/data/oss/hxzh-mr/all_type_pic_oss_csv/20251208_5w",
        "/data/oss/hxzh-mr/all_type_pic_oss_csv/20251210_5w"
    ])
    return parser.parse_args()

def load_all_csv_texts(data_dirs):
    """遍历所有目录，加载所有 CSV 第一列的文本"""
    all_texts = []
    for directory in data_dirs:
        if not os.path.exists(directory):
            print(f"警告：目录不存在 {directory}")
            continue
        for file in os.listdir(directory):
            if file.endswith(".csv"):
                file_path = os.path.join(directory, file)
                try:
                    df = pd.read_csv(file_path)
                    # ！！！注意：这里假设医疗文本在 CSV 的第一列，如果列名是 'text'，请改为 df['text']
                    texts = df.iloc[:, 0].dropna().astype(str).tolist()
                    all_texts.extend(texts)
                except Exception as e:
                    print(f"读取文件 {file_path} 出错: {e}")
    return all_texts

def format_as_label_studio(idx, kv_dict):
    """将 Qwen 输出的 KV 字典转换为你 dataset.py 能识别的 Label Studio 格式"""
    results = []
    count = 0
    for k, v in kv_dict.items():
        k_id, v_id = f"q_{idx}_{count}_k", f"q_{idx}_{count}_v"
        # 键名实体
        results.append({
            "id": k_id, "type": "labels",
            "value": {"labels": ["键名"], "text": str(k)}
        })
        # 值实体
        results.append({
            "id": v_id, "type": "labels",
            "value": {"labels": ["值"], "text": str(v)}
        })
        # 建立关系
        results.append({
            "from_id": k_id, "to_id": v_id, "type": "relation"
        })
        count += 1
    
    return {
        "id": idx,
        "annotations": [{"result": results, "was_cancelled": False}]
    }

# ---------------------------------------------------------------------- #
# 主程序
# ---------------------------------------------------------------------- #
def main():
    args = parse_args()
    
    # 1. 加载数据
    print("正在扫描并读取 20w 条 CSV 数据...")
    raw_texts = load_all_csv_texts(args.data_dirs)
    print(f"成功加载文本总数: {len(raw_texts)}")

    # 2. 初始化 vLLM (Qwen-0.6B 非常小，H200 可以轻松跑起)
    # 如果你的 lora 目录已经是合并后的，正常加载；如果是未合并的，vLLM 需指定 --enable-lora
    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=0.7, # 留一点余量给系统
        trust_remote_code=True,
        max_model_len=1024 # 医疗文本通常不需要太长
    )

    sampling_params = SamplingParams(
        temperature=0.0, # 医疗提取需要极高的确定性
        max_tokens=512,
        stop=["<|im_end|>", "<|endoftext|>"]
    )

    # 3. 构造推理 Prompts (请确保这和你 Task2 训练时的模板一致)
    # 假设模板为: <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
    prompts = [
        f"<|im_start|>system\n你是一个医疗结构化专家，请将文本提取为JSON格式的键值对。<|im_end|>\n<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n"
        for t in raw_texts
    ]

    # 4. 批量推理
    print("开始 vLLM 批量推理标注...")
    outputs = llm.generate(prompts, sampling_params)

    # 5. 解析结果并保存
    final_data = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        try:
            # 尝试解析输出为字典
            # 如果模型输出带 ```json ... ```，需要清理
            clean_text = generated_text.replace("```json", "").replace("```", "").strip()
            kv_dict = json.loads(clean_text)
            
            if isinstance(kv_dict, dict) and kv_dict:
                formatted_item = format_as_label_studio(i, kv_dict)
                final_data.append(formatted_item)
        except:
            # 记录失败或跳过
            continue

    # 6. 写入文件
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"🎉 标注完成！有效样本数: {len(final_data)}")
    print(f"结果已保存至: {args.output_file}")

if __name__ == "__main__":
    main()