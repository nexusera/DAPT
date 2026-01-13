import os
import re
from pathlib import Path
import jieba
from datasets import load_dataset
from transformers import AutoTokenizer

# ===========================
# 1. 配置路径（精简版：不加载 medical_dict.txt）
# ===========================
WORKSPACE_DIR = "/data/ocean/DAPT/workspace"
TOKENIZER_PATH = os.path.join(WORKSPACE_DIR, "my-medical-tokenizer")
# 使用重采样+滑窗后的语料
TRAIN_FILE = os.path.join(WORKSPACE_DIR, "train_chunked.txt")
OUTPUT_PATH = os.path.join(WORKSPACE_DIR, "processed_dataset")

# 仅保留 OCR 挖掘与业务 Key 两路词典
KEYS_FILE = str(Path(__file__).resolve().parent / "biaozhu_keys_freq_min5.txt")  # 业务实体 Key（频次>5）
VOCAB_FOR_JIEBA = os.path.join(WORKSPACE_DIR, "vocab_for_jieba.txt") # WordPiece 挖掘的高频词

# 最大长度 (通常预训练设为 512，和 BERT 基座对齐)
MAX_LEN = 512

# 噪音模式：长纯字母数字/长数字串（避免 ID/编码进入词表）
RE_LONG_ALNUM = re.compile(r'^[A-Za-z0-9]{6,}$')
RE_LONG_DIGITS = re.compile(r'\d{6,}')


def has_chinese(s: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', s))


def is_noisy_token(tok: str) -> bool:
    # 仅在“无中文”的情况下拦截长编码/长数字串，避免误杀含中文的日期/数值描述
    if has_chinese(tok):
        return False
    return bool(RE_LONG_ALNUM.match(tok) or RE_LONG_DIGITS.search(tok))

def init_jieba():
    """初始化 Jieba，加载外挂词典（精简版去掉 medical_dict）"""
    print("正在初始化 Jieba 并加载双路词典...")

    # 1. 加载 WordPiece 挖掘的高频词 (解决 brca1基因 等问题)
    if os.path.exists(VOCAB_FOR_JIEBA):
        jieba.load_userdict(VOCAB_FOR_JIEBA)
        print(f"✅ 已加载 WordPiece 挖掘词表: {VOCAB_FOR_JIEBA}")
    else:
        print(f"⚠️ 警告: 未找到 {VOCAB_FOR_JIEBA}，请检查路径！")

    # 2. 加载业务 Key (Source B)
    if os.path.exists(KEYS_FILE):
        jieba.load_userdict(KEYS_FILE)
        print(f"✅ 已加载业务实体: {KEYS_FILE}")

def process_function(examples, tokenizer):
    """
    核心对齐逻辑:
    Text -> Jieba(粗粒度) -> Tokenizer(细粒度) -> Mapping(word_ids)
    """
    final_input_ids = []
    final_word_ids = []

    for text in examples['text']:
        if not text.strip():
            continue

        # 如果整行无中文且形如长编码/长数字，直接跳过
        if not has_chinese(text) and RE_LONG_ALNUM.match(text.strip()):
            continue

        # 1. Jieba 分词 (利用了两路外挂词典)
        words = list(jieba.cut(text))

        tokens = []
        word_ids = []

        # 添加 [CLS]
        tokens.append(tokenizer.cls_token_id)
        word_ids.append(None)  # None 表示不参与 WWM 策略计算

        current_word_index = 0

        for word in words:
            # 跳过无中文的编码/长数字串
            if is_noisy_token(word):
                continue
            # 2. 对每个“词”进行 BERT Tokenize
            word_tokens = tokenizer.tokenize(word)
            word_token_ids = tokenizer.convert_tokens_to_ids(word_tokens)

            if not word_token_ids:
                continue

            tokens.extend(word_token_ids)

            # 3. 让该词的所有 token 共享同一个 index (Whole Word Masking 的基础)
            word_ids.extend([current_word_index] * len(word_token_ids))

            current_word_index += 1

        # 如果全被过滤掉，则跳过该样本
        if current_word_index == 0:
            continue

        # 添加 [SEP]
        tokens.append(tokenizer.sep_token_id)
        word_ids.append(None)

        # 4. 截断处理
        if len(tokens) > MAX_LEN:
            tokens = tokens[:MAX_LEN]
            word_ids = word_ids[:MAX_LEN]
            tokens[-1] = tokenizer.sep_token_id
            word_ids[-1] = None

        final_input_ids.append(tokens)
        final_word_ids.append(word_ids)

    return {"input_ids": final_input_ids, "word_ids": final_word_ids}


def main():
    # 1. 加载 Tokenizer
    print(f"加载 Tokenizer: {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # 2. 初始化 Jieba
    init_jieba()

    # 3. 加载原始文本
    print(f"读取原始语料: {TRAIN_FILE}")
    raw_dataset = load_dataset("text", data_files={"train": TRAIN_FILE})

    # 4. 数据处理
    print("开始构建数据集 (Tokenization + Alignment)...")
    processed_dataset = raw_dataset.map(
        lambda x: process_function(x, tokenizer),
        batched=True,
        batch_size=1000,
        num_proc=1,
        remove_columns=["text"],
    )

    # 5. 划分训练集和测试集 (保留 5% 用于计算 PPL)
    print("正在划分数据集 (Train 95% / Test 5%)...")
    final_split = processed_dataset["train"].train_test_split(test_size=0.05, seed=42)

    # 6. 保存到磁盘
    print(f"保存处理后的数据集到: {OUTPUT_PATH}")
    final_split.save_to_disk(OUTPUT_PATH)

    # 7. 最终检查
    print("\n" + "=" * 30)
    print("✅ 数据构建完成！")
    print(f"Train 集大小: {len(final_split['train'])}")
    print(f"Test  集大小: {len(final_split['test'])}")
    print("=" * 30)

    print("\n=== 数据对齐采样检查 ===")
    sample = final_split['train'][0]
    sample_tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
    sample_wids = sample['word_ids']

    print(f"Token总数: {len(sample_tokens)}")
    print("前 30 个 Token 与 WordID 的对应关系:")
    for t, w in zip(sample_tokens[:30], sample_wids[:30]):
        print(f"Token: {t:<10} | WordID: {w}")


if __name__ == "__main__":
    main()
