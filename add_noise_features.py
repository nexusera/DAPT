# 导入必要的Python库
import argparse  # 用于解析命令行参数
import json  # 用于处理JSON格式的数据
import os  # 用于文件和路径操作
from typing import Any, Dict, Iterable, List, Optional  # 类型提示，让代码更清晰

# 导入HuggingFace的datasets库，用于加载和保存数据集
from datasets import load_from_disk, DatasetDict
# 导入自定义的噪声特征提取器
from noise_embeddings import NoiseFeatureExtractor

# 默认的OCR（光学字符识别）数据源文件路径
# 可以通过命令行参数覆盖这个默认值
# 如果提供多个文件，会按顺序拼接它们的内容
DEFAULT_OCR_JSONS = [
    "/home/ocean/semi_label/ocr_rerun/char_ocr_9297.json",
]
# P0 白名单默认文件（可选）
DEFAULT_TOKENIZER_VOCAB = "/data/ocean/bpe_workspace/my-medical-tokenizer/vocab.txt"
DEFAULT_KEYS_FILE = "/data/ocean/bpe_workspace/keys.txt"
DEFAULT_KEPT_VOCAB = "/data/ocean/bpe_workspace/kept_vocab.txt"


def load_medical_dict(path: Optional[str]) -> List[str]:
    """
    加载医学词典文件
    
    参数:
        path: 医学词典文件的路径（可选）
        
    返回:
        包含所有医学术语的列表
        
    功能说明:
        - 从文本文件中读取医学术语，每行一个词
        - 如果路径为空或None，返回空列表
        - 如果文件不存在，抛出错误提示
    """
    # 如果没有提供路径，返回空列表
    if not path:
        return []
    # 检查文件是否存在，不存在则报错
    if not os.path.exists(path):
        raise FileNotFoundError(f"medical_dict not found: {path}")
    words = []
    # 打开文件，使用UTF-8编码读取
    with open(path, "r", encoding="utf-8") as f:
        # 逐行读取
        for line in f:
            w = line.strip()  # 去除行首行尾的空白字符
            if w:  # 如果这行不是空的
                words.append(w)  # 添加到词列表中
    return words


def load_list_file(path: Optional[str]) -> List[str]:
    """通用读取一行一个条目的列表文件。不存在则返回空列表。"""
    if not path or not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                items.append(w)
    return items


def load_ocr_list(paths: List[str]) -> List[Any]:
    """
    加载OCR识别结果数据
    
    参数:
        paths: OCR数据文件路径列表（可以是多个文件）
        
    返回:
        合并后的OCR数据列表
        
    功能说明:
        - 支持加载多个JSON或JSONL格式的文件
        - 会按照提供的顺序依次拼接所有文件的数据
        - 自动识别不同的JSON文件格式（列表、字典等）
    """
    merged: List[Any] = []  # 用于存储合并后的所有OCR数据
    # 遍历每个文件路径
    for path in paths:
        # 检查文件是否存在
        if not os.path.exists(path):
            raise FileNotFoundError(f"OCR json not found: {path}")
        # 如果是JSONL格式（每行一个JSON对象）
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()  # 去除空白
                    if line:  # 如果不是空行
                        merged.append(json.loads(line))  # 解析JSON并添加
            continue  # 处理下一个文件
        # 如果是普通JSON格式
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)  # 加载整个JSON文件
        # 如果JSON内容是列表，直接拼接
        if isinstance(obj, list):
            merged.extend(obj)
        # 如果JSON内容是字典，尝试找到包含数据的键
        elif isinstance(obj, dict):
            # 尝试常见的数据字段名
            for key in ["data", "ocr_list", "items"]:
                if key in obj and isinstance(obj[key], list):
                    merged.extend(obj[key])  # 拼接找到的列表数据
                    break
            else:
                # 如果没找到这些键，把整个字典当作一条数据
                merged.append(obj)
        else:
            # 不支持的格式
            raise ValueError(f"Unsupported OCR JSON format: {path}")
    return merged


def build_zero_feats(seq_len: int):
    """
    构建全零的噪声特征和掩码
    
    参数:
        seq_len: 序列长度（token的数量）
        
    返回:
        两个列表：
        1. 噪声特征：每个token有5个特征值，全为0.0
        2. 噪声掩码：每个token有5个布尔值，全为False
        
    功能说明:
        当无法提取真实噪声特征时，用这个函数生成占位数据
    """
    return [[0.0] * 5 for _ in range(seq_len)], [[False] * 5 for _ in range(seq_len)]


def main():
    """
    主函数：为数据集添加噪声特征
    
    整体流程:
    1. 解析命令行参数
    2. 加载已处理的数据集
    3. 加载OCR数据和医学词典
    4. 初始化噪声特征提取器
    5. 为每个样本提取并添加噪声特征
    6. 保存新的数据集
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Add noise features to processed_dataset")
    # 添加各种命令行参数
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,  # 必需参数
        help="path to processed_dataset (load_from_disk)",
    )
    parser.add_argument(
        "--ocr_json",
        type=str,
        nargs="+",  # 可以接受多个值
        default=DEFAULT_OCR_JSONS,
        help=f"one or more OCR json/jsonl paths, concatenated in order (default={DEFAULT_OCR_JSONS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,  # 必需参数
        help="output path to save new dataset with noise fields",
    )
    parser.add_argument(
        "--medical_dict",
        type=str,
        default=None,  # 可选参数
        help="optional path to medical dict (one word per line)",
    )
    parser.add_argument(
        "--tokenizer_vocab",
        type=str,
        default=DEFAULT_TOKENIZER_VOCAB,
        help="vocab.txt for tokenizer (P0 白名单之一)",
    )
    parser.add_argument(
        "--keys_file",
        type=str,
        default=DEFAULT_KEYS_FILE,
        help="keys.txt (OCR 业务 Key，P0 白名单之一)",
    )
    parser.add_argument(
        "--kept_vocab",
        type=str,
        default=DEFAULT_KEPT_VOCAB,
        help="可选：LLM 筛选后的 kept_vocab.txt，加入 P0 白名单",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,  # 可选参数
        help="optional number of processes for Dataset.map (CPU并行)",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 从磁盘加载已处理的数据集
    dataset = load_from_disk(args.dataset)
    # 检查数据集格式是否正确（必须包含train/test等分割）
    if not isinstance(dataset, DatasetDict):
        raise ValueError("Expected a DatasetDict with train/test splits")

    # 加载OCR识别结果数据（可能来自多个文件）
    ocr_list = load_ocr_list(args.ocr_json)
    # 加载医学词典
    med_dict = load_medical_dict(args.medical_dict)

    # 准备 P0 白名单集合：tokenizer vocab + keys + kept_vocab
    p0_terms = []
    p0_terms.extend(load_list_file(args.tokenizer_vocab))
    p0_terms.extend(load_list_file(args.keys_file))
    p0_terms.extend(load_list_file(args.kept_vocab))

    # 初始化噪声特征提取器
    extractor = NoiseFeatureExtractor(
        medical_dict=med_dict,
        p0_terms=p0_terms,
    )

    def add_noise(example: Dict[str, Any], idx: int):
        """
        为单个样本添加噪声特征的函数
        
        参数:
            example: 数据集中的一个样本（字典格式）
            idx: 样本在数据集中的索引位置
            
        返回:
            添加了噪声特征后的样本
            
        处理逻辑:
            1. 获取样本的word_ids（词到子词的映射）
            2. 如果word_ids不存在，生成空特征
            3. 如果idx在OCR列表范围内，从OCR数据提取真实噪声特征
            4. 否则生成全零特征作为占位
        """
        # 获取样本中的word_ids字段（记录每个subword属于哪个word）
        word_ids = example.get("word_ids")
        # 如果word_ids不存在，生成空的噪声特征
        if word_ids is None:
            nf, nm = build_zero_feats(0)
            example["noise_features"] = nf
            example["noise_masks"] = nm
            return example

        # 如果当前样本的索引在OCR数据范围内
        if idx < len(ocr_list):
            # 获取对应的OCR数据
            ocr_obj = ocr_list[idx]
            # 从OCR数据中提取词级别的噪声特征
            word_feats, word_masks = extractor.extract_word_features({"ocr": ocr_obj})
            # 将词级别的特征广播到subword级别（因为BERT使用subword分词）
            nf, nm = extractor.broadcast_to_subwords(
                word_feats, word_masks, word_ids
            )
            # 转换为Python列表格式并保存
            example["noise_features"] = nf.tolist()
            example["noise_masks"] = nm.tolist()
        else:
            # 如果没有对应的OCR数据，生成全零特征
            nf, nm = build_zero_feats(len(word_ids))
            example["noise_features"] = nf
            example["noise_masks"] = nm
        return example

    # 用于存储处理后的各个数据分割（如train、test等）
    new_splits = {}
    # 设置Dataset.map()的参数
    map_kwargs = {
        "with_indices": True,  # 传递索引给处理函数
        "desc": None,  # 进度条描述，后面会更新
    }
    # 如果指定了多进程处理
    if args.num_proc and args.num_proc > 1:
        map_kwargs["num_proc"] = args.num_proc

    # 遍历数据集的每个分割（如train、test等）
    for split in dataset:
        ds = dataset[split]  # 获取当前分割的数据
        map_kwargs["desc"] = f"add_noise_features_{split}"  # 设置进度条描述
        # 对当前分割的所有样本应用add_noise函数
        # map会并行处理所有样本（如果设置了num_proc）
        new_splits[split] = ds.map(add_noise, **map_kwargs)

    # 将所有分割重新组合成DatasetDict
    out = DatasetDict(new_splits)
    # 保存到磁盘
    out.save_to_disk(args.output)
    print(f"Saved dataset with noise features to {args.output}")


# 程序入口：当直接运行这个脚本时，执行main函数
if __name__ == "__main__":
    main()

