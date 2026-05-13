import json
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gen_pseudo_labels import (
    resolve_model_and_lora,
    merge_lora_to_disk,
    load_all_texts,
    truncate_by_tokens,
)

# 可按需修改
MODEL_PATH = "/data/ocean/medstruct_s_llm/results/lora/Qwen3-0.6B_task2"
BASE_MODEL_PATH = None  # 若 adapter_config.json 没写底模，则填真实底模路径
DATA_DIRS = [
    "/data/oss/hxzh-mr/all_type_pic_oss_csv/20251204_5w",
]
MAX_MODEL_LEN = 4096
MAX_OUTPUT_TOKENS = 2048
MAX_INPUT_TOKENS = 1024  # 不填则自动用 max_model_len - max_output_tokens - 32
SAMPLE_N = 5

def main():
    # 1) 取少量数据
    texts = load_all_texts(DATA_DIRS)[:SAMPLE_N]
    print(f"采样文本条数: {len(texts)}")

    # 2) 解析/合并 LoRA
    base_model, lora_path = resolve_model_and_lora(MODEL_PATH, BASE_MODEL_PATH)
    if lora_path:
        base_model = merge_lora_to_disk(base_model, lora_path)

    # 3) 截断准备
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    max_input_tokens = MAX_INPUT_TOKENS
    if max_input_tokens is None:
        max_input_tokens = max(MAX_MODEL_LEN - MAX_OUTPUT_TOKENS - 32, 1)

    prompts = [
        f"<|im_start|>system\n你是一个医疗结构化专家，请将文本提取为JSON格式的键值对。<|im_end|>\n"
        f"<|im_start|>user\n{truncate_by_tokens(t, tokenizer, max_input_tokens)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        for t in texts
    ]

    # 4) 推理
    llm = LLM(
        model=base_model,
        trust_remote_code=True,
        gpu_memory_utilization=0.7,
        max_model_len=MAX_MODEL_LEN,
    )
    outs = llm.generate(prompts, SamplingParams(
        temperature=0.0,
        max_tokens=MAX_OUTPUT_TOKENS,
        stop=["<|im_end|>", "<|endoftext|>"],
    ))

    # 5) 打印原始与解析结果
    for i, out in enumerate(outs):
        text = out.outputs[0].text.strip()
        print(f"\n====== 样本 {i} 原始输出 ======\n{text}")
        clean = text.replace("", "").replace("```", "").strip()
        try:
            parsed = json.loads(clean)
            print(f"------ 解析成功: {parsed}")
        except Exception as e:
            print(f"------ 解析失败: {e}")

if __name__ == "__main__":
    main()
# 若需要指定底模：
# CUDA_VISIBLE_DEVICES=1 python scripts/analysis/quick_sample.py
# 并在脚本里填好 BASE_MODEL_PATH，这样可以快速看到模型真实输出和解析情况。
