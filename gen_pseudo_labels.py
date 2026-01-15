import os
import json
import argparse
import tempfile
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------- #
# 配置区
# ---------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser()
    # 指向你的 LoRA 微调后权重目录（如果已经 merge，可直接填合并后的模型目录）
    parser.add_argument("--model_path", type=str, default="/data/ocean/medstruct_s_llm/results/lora/Qwen3-0.6B_task2")
    # 如果 model_path 是 LoRA 适配器目录，需指定原始底模（HF Repo ID 或本地路径）
    parser.add_argument("--base_model_path", type=str, default=None)
    # 最大上下文长度（tokens），需满足输入+输出总长
    parser.add_argument("--max_model_len", type=int, default=4096)
    # 生成的最大输出 tokens（已测试 1024 可完整闭合 JSON）
    parser.add_argument("--max_output_tokens", type=int, default=2048)
    # 输入截断上限（tokens），None 则按 max_model_len - max_output_tokens - 32 计算
    # 默认 1024 与 quick_sample.py 保持一致，确保输出有足够空间
    parser.add_argument("--max_input_tokens", type=int, default=1536)
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


def resolve_model_and_lora(model_path, base_model_path=None):
    """
    根据传入路径判断是完整模型还是 LoRA 适配器。
    返回 (base_model, lora_path 或 None)。
    """
    # HuggingFace 远程仓库直接返回
    if not os.path.exists(model_path):
        return model_path, None

    config_path = os.path.join(model_path, "config.json")
    adapter_config_path = os.path.join(model_path, "adapter_config.json")

    if os.path.isfile(config_path):
        # 已经是合并后的模型
        return model_path, None

    if os.path.isfile(adapter_config_path):
        base_from_adapter = None
        try:
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
                base_from_adapter = adapter_cfg.get("base_model_name_or_path")
        except Exception:
            base_from_adapter = None

        base_model = base_model_path or base_from_adapter
        if not base_model:
            raise ValueError(
                "检测到 LoRA 目录但未找到 base_model_name_or_path，"
                "请通过 --base_model_path 指定原始底模（HF Repo 或本地路径）。"
            )
        return base_model, model_path

    # 兜底：目录存在但没有 config.json；如果指定了 base_model_path，则按 LoRA 处理
    if base_model_path:
        return base_model_path, model_path

    raise ValueError(
        f"路径 {model_path} 未发现 config.json 或 adapter_config.json，"
        "请传入包含 config.json 的模型目录，或配合 --base_model_path 使用 LoRA。"
    )


def merge_lora_to_disk(base_model, lora_path):
    """
    将 LoRA 权重合并到基座模型，保存到临时目录，返回合并后的路径。
    适配不支持 lora_path/lora_adapters 的 vLLM 版本。
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        raise RuntimeError(
            f"合并 LoRA 需要 transformers/peft/torch，请安装后重试: {e}"
        )

    tmp_dir = tempfile.mkdtemp(prefix="merged_lora_")
    print(f"正在将 LoRA 合并到基座模型，输出到临时目录: {tmp_dir}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, lora_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(tmp_dir)

    # 复制 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(tmp_dir)

    print("LoRA 合并完成，改用合并后的模型进行推理。")
    return tmp_dir


def truncate_by_tokens(text, tokenizer, max_tokens):
    """
    使用 tokenizer 按 token 长度截断输入文本。
    策略：优先保留开头（通常包含关键信息如姓名、性别、年龄、诊断等），
    如果文本过长，从开头截断到 max_tokens。
    """
    if max_tokens is None:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    # 从开头截断，保留最重要的前 max_tokens 个 tokens
    truncated = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
    return truncated

def load_all_texts(data_dirs):
    """递归遍历目录，支持 CSV 第一列、以及 .txt JSON（含 words_result 或 text 字段）"""
    all_texts = []

    def _load_txt_json(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
                if not raw:
                    return None
                data = json.loads(raw)
        except Exception as e:
            print(f"读取 TXT JSON 失败 {txt_path}: {e}")
            return None

        # 兼容 words_result: [{"words": "..."}]
        if isinstance(data, dict):
            if "words_result" in data and isinstance(data["words_result"], list):
                return "".join([item.get("words", "") for item in data["words_result"] if isinstance(item, dict)])
            if "text" in data:
                return str(data["text"])
        return None

    for directory in data_dirs:
        if not os.path.exists(directory):
            print(f"警告：目录不存在 {directory}")
            continue

        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".csv"):
                    try:
                        df = pd.read_csv(file_path)
                        # ！！！注意：这里假设医疗文本在 CSV 的第一列，如果列名是 'text'，请改为 df['text']
                        texts = df.iloc[:, 0].dropna().astype(str).tolist()
                        all_texts.extend(texts)
                    except Exception as e:
                        print(f"读取文件 {file_path} 出错: {e}")
                elif file.endswith(".txt"):
                    text = _load_txt_json(file_path)
                    if text:
                        all_texts.append(text)
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
    raw_texts = load_all_texts(args.data_dirs)
    print(f"成功加载文本总数: {len(raw_texts)}")
    if not raw_texts:
        print("未读取到任何文本，请检查 data_dirs 或 CSV 内容后重试。")
        return

    # 2. 解析模型/LoRA 路径并初始化 vLLM
    base_model, lora_path = resolve_model_and_lora(args.model_path, args.base_model_path)

    # 若是 LoRA，则先合并再交给 vLLM，避免 vLLM 版本兼容问题
    if lora_path:
        merged_path = merge_lora_to_disk(base_model, lora_path)
        base_model = merged_path
        lora_path = None

    # 3. 准备 tokenizer，用于截断输入
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # 4. 采样与长度控制
    max_input_tokens = args.max_input_tokens
    if max_input_tokens is None:
        max_input_tokens = max(args.max_model_len - args.max_output_tokens - 32, 1)

    llm_kwargs = dict(
        model=base_model,
        gpu_memory_utilization=0.7,  # 留一点余量给系统
        trust_remote_code=True,
        max_model_len=args.max_model_len
    )

    llm = LLM(**llm_kwargs)
    print(f"直接加载模型: {base_model}")

    sampling_params = SamplingParams(
        temperature=0.0, # 医疗提取需要极高的确定性
        max_tokens=args.max_output_tokens,
        stop=["<|im_end|>", "<|endoftext|>"]
    )

    # 3. 构造推理 Prompts (请确保这和你 Task2 训练时的模板一致)
    # 假设模板为: <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
    prompts = [
        f"<|im_start|>system\n你是一个医疗结构化专家，请将文本提取为JSON格式的键值对。<|im_end|>\n<|im_start|>user\n{truncate_by_tokens(t, tokenizer, max_input_tokens)}<|im_end|>\n<|im_start|>assistant\n"
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