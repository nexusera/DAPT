"""
用当前 TOKENIZER_PATH 对原始 train.txt 重新分词，生成 processed_dataset，保留 word_ids 字段（jieba 粒度）。
适用于“只用基座 tokenizer，不扩展词表”且要做 KV-MLM 任务的场景。

用法：
  python retokenize_processed_dataset_with_wordids.py

依赖：transformers, datasets, jieba
"""
import os
import re
from pathlib import Path
import jieba
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

WORKSPACE_DIR = "/data/ocean/DAPT/workspace"
TOKENIZER_PATH = "hfl/chinese-roberta-wwm-ext"
TRAIN_FILE = os.path.join(WORKSPACE_DIR, "train.txt")
OUTPUT_PATH = os.path.join(WORKSPACE_DIR, "processed_dataset")
KEYS_FILE = str(Path(__file__).resolve().parent / "biaozhu_keys_only_min5.txt")
# Jieba 词表位置（已移动到 /data/ocean/DAPT/vocab_for_jieba.txt）
VOCAB_FOR_JIEBA = "/data/ocean/DAPT/vocab_for_jieba.txt"
MAX_LEN = 512
RE_LONG_ALNUM = re.compile(r'^[A-Za-z0-9]{6,}$')
RE_LONG_DIGITS = re.compile(r'\d{6,}')

def has_chinese(s: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', s))

def is_noisy_token(tok: str) -> bool:
    if has_chinese(tok):
        return False
    return bool(RE_LONG_ALNUM.match(tok) or RE_LONG_DIGITS.search(tok))

def init_jieba():
    print("正在初始化 Jieba 并加载双路词典...")
    if os.path.exists(VOCAB_FOR_JIEBA):
        jieba.load_userdict(VOCAB_FOR_JIEBA)
        print(f"✅ 已加载 WordPiece 挖掘词表: {VOCAB_FOR_JIEBA}")
    else:
        print(f"⚠️ 警告: 未找到 {VOCAB_FOR_JIEBA}，请检查路径！")
    if os.path.exists(KEYS_FILE):
        jieba.load_userdict(KEYS_FILE)
        print(f"✅ 已加载业务实体: {KEYS_FILE}")

def process_function(examples, tokenizer):
    final_input_ids = []
    final_word_ids = []
    for text in examples['text']:
        if not text.strip():
            continue
        if not has_chinese(text) and RE_LONG_ALNUM.match(text.strip()):
            continue
        words = list(jieba.cut(text))
        tokens = []
        word_ids = []
        tokens.append(tokenizer.cls_token_id)
        word_ids.append(None)
        current_word_index = 0
        for word in words:
            if is_noisy_token(word):
                continue
            word_tokens = tokenizer.tokenize(word)
            word_token_ids = tokenizer.convert_tokens_to_ids(word_tokens)
            if not word_token_ids:
                continue
            tokens.extend(word_token_ids)
            word_ids.extend([current_word_index] * len(word_token_ids))
            current_word_index += 1
        if current_word_index == 0:
            continue
        tokens.append(tokenizer.sep_token_id)
        word_ids.append(None)
        if len(tokens) > MAX_LEN:
            tokens = tokens[:MAX_LEN]
            word_ids = word_ids[:MAX_LEN]
            tokens[-1] = tokenizer.sep_token_id
            word_ids[-1] = None
        final_input_ids.append(tokens)
        final_word_ids.append(word_ids)
    return {"input_ids": final_input_ids, "word_ids": final_word_ids}

def main():
    print(f"加载 Tokenizer: {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    init_jieba()
    print(f"读取原始语料: {TRAIN_FILE}")
    raw_dataset = load_dataset("text", data_files={"train": TRAIN_FILE})
    print("开始分词+对齐...")
    processed_dataset = raw_dataset.map(
        lambda x: process_function(x, tokenizer),
        batched=True,
        batch_size=1000,
        num_proc=1,
        remove_columns=["text"],
    )
    print("划分训练/测试集 (95%/5%)...")
    final_split = processed_dataset["train"].train_test_split(test_size=0.05, seed=42)
    print(f"保存到: {OUTPUT_PATH}")
    DatasetDict(final_split).save_to_disk(OUTPUT_PATH)
    print("✅ 重新分词+对齐完成！")
    # 检查最大 token id
    max_id = max(max(x["input_ids"]) for x in final_split["train"][:1000])
    print(f"采样前1000条最大token id: {max_id}, tokenizer vocab_size: {len(tokenizer)}")
    # 检查 word_ids
    sample = final_split["train"][0]
    print(f"Token总数: {len(sample['input_ids'])}")
    print("前 30 个 Token 与 WordID 的对应关系:")
    tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
    for t, w in zip(tokens[:30], sample['word_ids'][:30]):
        print(f"Token: {t:<10} | WordID: {w}")

if __name__ == "__main__":
    main()
