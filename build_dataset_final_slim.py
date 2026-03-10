import os
import re
import argparse
from pathlib import Path
import jieba
from datasets import load_dataset
from transformers import AutoTokenizer

DEFAULT_WORKSPACE_DIR = "/data/ocean/DAPT/workspace"
# Tokenizer 现存放在 /data/ocean/DAPT/my-medical-tokenizer
DEFAULT_TOKENIZER_PATH = "/data/ocean/DAPT/my-medical-tokenizer"
# 使用重采样+滑窗后的语料
DEFAULT_TRAIN_FILE = os.path.join(DEFAULT_WORKSPACE_DIR, "train_chunked.txt")
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_WORKSPACE_DIR, "processed_dataset")

# 仅保留 OCR 挖掘与业务 Key 两路词典
DEFAULT_KEYS_FILE = str(Path(__file__).resolve().parent / "biaozhu_keys_only_min5.txt")  # 业务实体 Key（纯键名，频次>5）
# Jieba 词表位置（注意：词表实际存放在 /data/ocean/DAPT/vocab_for_jieba.txt）
DEFAULT_VOCAB_FOR_JIEBA = "/data/ocean/DAPT/vocab_for_jieba.txt"  # WordPiece 挖掘的高频词

# 最大长度 (通常预训练设为 512，和 BERT 基座对齐)
DEFAULT_MAX_LEN = 512
DEFAULT_BATCH_SIZE = 1000
DEFAULT_NUM_PROC = 1
DEFAULT_SHUFFLE_SPLIT = True  # 非 OCR 路可打乱；OCR 路为保持顺序可设为 False

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

def init_jieba(keys_file: str, vocab_for_jieba: str):
    """初始化 Jieba，加载外挂词典（精简版去掉 medical_dict）"""
    print("正在初始化 Jieba 并加载双路词典...")

    # 1. 加载 WordPiece 挖掘的高频词 (解决 brca1基因 等问题)
    if os.path.exists(vocab_for_jieba):
        jieba.load_userdict(vocab_for_jieba)
        print(f"✅ 已加载 WordPiece 挖掘词表: {vocab_for_jieba}")
    else:
        print(f"⚠️ 警告: 未找到 {vocab_for_jieba}，请检查路径！")

    # 2. 加载业务 Key (Source B)
    if os.path.exists(keys_file):
        jieba.load_userdict(keys_file)
        print(f"✅ 已加载业务实体: {keys_file}")

def process_function(examples, tokenizer, max_len: int):
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
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            word_ids = word_ids[:max_len]
            tokens[-1] = tokenizer.sep_token_id
            word_ids[-1] = None

        final_input_ids.append(tokens)
        final_word_ids.append(word_ids)

    return {"input_ids": final_input_ids, "word_ids": final_word_ids}


def parse_args():
    parser = argparse.ArgumentParser(description="构建对齐后的预训练数据（精简版）")
    parser.add_argument(
        "--train_file",
        type=str,
        default=DEFAULT_TRAIN_FILE,
        help="输入语料路径（文本或重采样+滑窗后的文件）",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="输出保存目录（DatasetDict.save_to_disk）",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=DEFAULT_TOKENIZER_PATH,
        help="分词器路径（用于细粒度 tokenization）",
    )
    parser.add_argument(
        "--keys_file",
        type=str,
        default=DEFAULT_KEYS_FILE,
        help="业务 Key 词典路径（Jieba 用户词典）",
    )
    parser.add_argument(
        "--vocab_for_jieba",
        type=str,
        default=DEFAULT_VOCAB_FOR_JIEBA,
        help="WordPiece 高频词词典（Jieba 用户词典）",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help="序列最大长度（包含 [CLS]/[SEP]）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="map 批大小",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=DEFAULT_NUM_PROC,
        help="多进程 map 个数（>=2 启用并行）",
    )
    parser.add_argument(
        "--shuffle_split",
        action="store_true",
        default=DEFAULT_SHUFFLE_SPLIT,
        help="划分 train/test 时是否打乱（OCR 路建议 False 保持与 OCR JSON 对齐顺序）",
    )
    parser.add_argument(
        "--no_shuffle_split",
        action="store_false",
        dest="shuffle_split",
        help="同上，显式关闭打乱",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载 Tokenizer
    print(f"加载 Tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)

    # 2. 初始化 Jieba
    init_jieba(args.keys_file, args.vocab_for_jieba)

    # 3. 加载原始文本
    print(f"读取原始语料: {args.train_file}")
    raw_dataset = load_dataset("text", data_files={"train": args.train_file})

    # 4. 数据处理
    print("开始构建数据集 (Tokenization + Alignment)...")
    map_kwargs = {
        "batched": True,
        "batch_size": args.batch_size,
        "remove_columns": ["text"],
    }
    if args.num_proc and args.num_proc > 1:
        map_kwargs["num_proc"] = args.num_proc

    processed_dataset = raw_dataset.map(
        lambda x: process_function(x, tokenizer, args.max_len),
        **map_kwargs,
    )

    # 5. 划分训练集和测试集 (保留 5% 用于计算 PPL)
    print(f"正在划分数据集 (Train 95% / Test 5%)，shuffle={args.shuffle_split} ...")
    final_split = processed_dataset["train"].train_test_split(
        test_size=0.05, seed=42, shuffle=args.shuffle_split
    )

    # 6. 保存到磁盘
    print(f"保存处理后的数据集到: {args.output_path}")
    final_split.save_to_disk(args.output_path)

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
