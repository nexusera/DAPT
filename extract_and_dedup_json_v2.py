import os
import json
import hashlib
from tqdm import tqdm
import re
from pathlib import Path

# ===========================
# 配置路径
# ===========================
# 原始数据根目录（支持多源，新增 20251215 镜像）
RAW_DATA_ROOTS = [
    "/data/oss/hxzh-mr/all_type_pic_oss_csv/20251204_5w",
    "/data/oss/hxzh-mr/all_type_pic_oss_csv/20251206_5w",
    "/data/oss/hxzh-mr/all_type_pic_oss_csv/20251208_5w",
    "/data/oss/hxzh-mr/all_type_pic_oss_csv/20251210_5w",
    #"/hxzh-h200-img/oss/hxzh-mr/all_type_pic_oss_csv/pt=20251215",
    # 新增本地医疗百科/教材资源
    "/data/ocean/DAPT/med_data",
    "/data/ocean/DAPT/workspace/wiki_data",
    # 新增通用中文大语料抽样（FineWeb）
    "/data/ocean/DAPT/general_data",
]

# 旧书籍列表（兼容；若不需要可留空）
BOOK_JSON_FILES = []


# =================================================================
# 1. 核心医学书籍与指南 (DAPT 的基石，建议权重最高，占比约 50-60%)
# =================================================================
MEDICAL_BOOK_FILES = [
    "/data/hxzh/LLaMA-Factory/data/肿瘤学.json",
    "/data/hxzh/LLaMA-Factory/data/肿瘤外科学高级教程.json",
    "/data/hxzh/LLaMA-Factory/data/肿瘤放射治疗学 第5版json.json",
    "/data/hxzh/LLaMA-Factory/data/肿瘤外科病理学.json",
    "/data/hxzh/LLaMA-Factory/data/临床诊断学.json",
    "/data/hxzh/LLaMA-Factory/data/cpt_books.json",
    "/data/hxzh/LLaMA-Factory/data/八年制教材-医学伦理学（第2版）.json",
    "/data/hxzh/LLaMA-Factory/data/45.医学伦理学实践.json",
    "/data/hxzh/LLaMA-Factory/data/zhinan_extend.jsonl",
    "/data/hxzh/LLaMA-Factory/data/zhinan.jsonl",
    "/data/hxzh/LLaMA-Factory/data/2025肿瘤学-考试指导_cptdata.json",
]

# =================================================================
# 2. 医学学术论文 (补充前沿实体与长尾知识，建议中英文混合，占比约 25-30%)
# =================================================================
MEDICAL_PAPER_FILES = [
    "/data/hxzh/LLaMA-Factory/data/cpt_cn.json",  # 中文医学论文
    "/data/hxzh/LLaMA-Factory/data/cpt_en.json",  # 英文医学论文 (非常建议保留，提升底层逻辑)
]

# =================================================================
# 3. 通用领域语料 (防止灾难性遗忘，维持语法稳定性，占比约 10-15%)
# =================================================================
GENERAL_CPT_FILES = [
    "/data/hxzh/LLaMA-Factory/data/social_cpt.jsonl",
]

# =================================================================
# 4. 备选补充语料 
# =================================================================
SUPPLEMENTARY_FILES = [
    "/data/hxzh/LLaMA-Factory/data/zhinan_abstract.jsonl",
    "/data/hxzh/LLaMA-Factory/data/zhinan_refact.jsonl",
]

# 输出文件 (直接覆盖 train.txt，方便后续流程)
OUTPUT_FILE = "/data/ocean/DAPT/workspace/train.txt"
# 分类输出（便于后续重采样配比）
OUTPUT_SPLIT_FILES = {
    "clinical_raw": "/data/ocean/DAPT/workspace/train_clinical.txt",
    "book_core": "/data/ocean/DAPT/workspace/train_book_core.txt",
    "book_old": "/data/ocean/DAPT/workspace/train_book_old.txt",
    "paper": "/data/ocean/DAPT/workspace/train_paper.txt",
    "general": "/data/ocean/DAPT/workspace/train_general.txt",
    "supplement": "/data/ocean/DAPT/workspace/train_supplement.txt",
    "wiki_med": "/data/ocean/DAPT/workspace/train_wiki_med.txt",
    "wiki_general": "/data/ocean/DAPT/workspace/train_wiki_general.txt",
    "med_book": "/data/ocean/DAPT/workspace/train_med_book.txt",
}

# 可选：用于 token 估算的分词器名称；若不可用则退化为“字符≈token”估算
TOKENIZER_NAME = "hfl/chinese-roberta-wwm-ext"

# 最小长度阈值 (低于此长度的文本被视为无效)
MIN_TEXT_LEN = 10 

def get_content_hash(text):
    """计算文本内容的 MD5 (去空格)"""
    clean_text = re.sub(r'\s+', '', text)
    return hashlib.md5(clean_text.encode('utf-8')).hexdigest()

def extract_smart(data):
    """
    智能提取逻辑：区分 Type A (OCR) and Type B (Report)
    """
    extracted_texts = []
    
    # === 策略 1: Type A (百度 OCR 格式) ===
    # 特征: 字典，包含 "words_result" 列表
    if isinstance(data, dict) and "words_result" in data:
        for item in data["words_result"]:
            if isinstance(item, dict) and "words" in item:
                # 提取 words 字段
                s = item["words"].strip()
                if s:
                    extracted_texts.append(s)
        # OCR 结果通常是分行的，我们用空格拼接成一长句，或者保留换行
        # 这里建议用空格拼接，让模型学上下文
        return " ".join(extracted_texts)

    # === 策略 2: Type B (医疗报告 JSON) ===
    # 特征: 列表，内部项包含 data -> report
    if isinstance(data, list):
        found_report = False
        temp_texts = []
        for item in data:
            if isinstance(item, dict) and "data" in item:
                if isinstance(item["data"], dict) and "report" in item["data"]:
                    # 提取 report 字段 (通常是大段文本)
                    r = item["data"]["report"]
                    if r and isinstance(r, str):
                        # 清洗一下可能的 HTML 标签或过多换行
                        clean_r = r.replace("\r", "").replace("\n", " ").strip()
                        temp_texts.append(clean_r)
                        found_report = True
        
        if found_report:
            return " ".join(temp_texts)

    # === 策略 3: 通用 JSON/JSONL 格式，字段包含 text / content / instruction+output / question+answer ===
    # 典型场景：百科/教材/QA（如 med_data/train_encyclopedia.json 每行 {"text": "..."}）
    if isinstance(data, dict):
        # 1) text / content 直接取
        for k in ["text", "content"]:
            if k in data and isinstance(data[k], str) and data[k].strip():
                return data[k].strip()
        # 2) instruction + output / answer
        if "instruction" in data and "output" in data:
            instr = data.get("instruction") or ""
            out = data.get("output") or ""
            combo = f"{instr} {out}".strip()
            if combo:
                return combo
        if "question" in data and "answer" in data:
            q = data.get("question") or ""
            a = data.get("answer") or ""
            combo = f"{q} {a}".strip()
            if combo:
                return combo

    # === 策略 4: 兜底策略 (未知格式) ===
    return ""

def main():
    print(f"正在扫描目录: {RAW_DATA_ROOTS}")
    print(f"书籍 JSON (旧 BOOK_JSON_FILES): {len(BOOK_JSON_FILES)} files")
    print(f"目标输出: {OUTPUT_FILE}")
    
    unique_hashes = set()
    total_files = 0
    duplicate_count = 0
    valid_count = 0
    parse_error_count = 0

    # 统计来源与长度（更细分）
    source_stats = {
        "clinical_raw": {"lines": 0, "chars": 0, "files": 0, "parse_errors": 0},
        "book_core": {"lines": 0, "chars": 0, "files": 0, "parse_errors": 0},
        "book_old": {"lines": 0, "chars": 0, "files": 0, "parse_errors": 0},
        "paper": {"lines": 0, "chars": 0, "files": 0, "parse_errors": 0},
        "general": {"lines": 0, "chars": 0, "files": 0, "parse_errors": 0},
        "supplement": {"lines": 0, "chars": 0, "files": 0, "parse_errors": 0},
        "wiki_med": {"lines": 0, "chars": 0, "files": 0, "parse_errors": 0},
        "wiki_general": {"lines": 0, "chars": 0, "files": 0, "parse_errors": 0},
        "med_book": {"lines": 0, "chars": 0, "files": 0, "parse_errors": 0},
        "general2": {"lines": 0, "chars": 0, "files": 0, "parse_errors": 0},
    }
    source_chars = {"ocr": 0, "book": 0}  # 保留旧字段用于 token 估算
    
    # 用于采样的缓存
    sample_outputs = []
    
    # 确保输出目录存在
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    for p in OUTPUT_SPLIT_FILES.values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    def write_record(clean_line: str, category: str, f_out_main, split_handlers):
        """写入主文件与分类文件"""
        f_out_main.write(clean_line + "\n")
        if category in split_handlers:
            split_handlers[category].write(clean_line + "\n")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        split_handlers = {k: open(path, "w", encoding="utf-8") for k, path in OUTPUT_SPLIT_FILES.items()}
        # 遍历目录（临床原始数据 + 本地新增 wiki/med_data 等）
        for RAW_DATA_ROOT in RAW_DATA_ROOTS:
            for root, dirs, files in os.walk(RAW_DATA_ROOT):
                # 扫描 .txt/.json/.csv/.jsonl/.jsonl.gz 文件
                target_files = [
                    f for f in files
                    if f.endswith('.txt')
                    or f.endswith('.json')
                    or f.endswith('.csv')
                    or f.endswith('.jsonl')
                    or f.endswith('.jsonl.gz')
                ]
            
                if not target_files:
                    continue
                    
                for file in tqdm(target_files, desc=f"Processing {os.path.basename(root)}"):
                    file_path = os.path.join(root, file)
                    total_files += 1
                    
                    # 根据路径决定分类：med_data -> med_book；wiki_data -> wiki_med/wiki_general；否则默认为 clinical_raw
                    category = "clinical_raw"
                    if "/med_data" in root:
                        category = "med_book"
                    elif "/wiki_data" in root:
                        if "wiki_med" in file:
                            category = "wiki_med"
                        else:
                            category = "wiki_general"

                    source_stats.setdefault(category, {"lines": 0, "chars": 0, "files": 0, "parse_errors": 0})
                    source_stats[category]["files"] += 1
                    try:
                        # 全量读取文件（根据后缀分别处理）
                        if file.endswith('.csv'):
                            with open(file_path, 'r', encoding='utf-8') as f_in:
                                import csv
                                reader = csv.DictReader(f_in)
                                for row in reader:
                                    candidates = []
                                    for key in ["text", "content", "report", "ocr", "paragraph", "instruction", "output"]:
                                        if key in row and row[key]:
                                            candidates.append(str(row[key]).strip())
                                    if not candidates:
                                        for v in row.values():
                                            if isinstance(v, str):
                                                candidates.append(v.strip())
                                    full_text = " ".join([c for c in candidates if c])
                                    if not full_text or len(full_text) < MIN_TEXT_LEN:
                                        continue
                                    file_hash = get_content_hash(full_text)
                                    if file_hash in unique_hashes:
                                        duplicate_count += 1
                                        continue
                                    unique_hashes.add(file_hash)
                                    clean_line = full_text.replace("\n", " ")
                                    write_record(clean_line, category, f_out, split_handlers)
                                    valid_count += 1
                                    source_stats[category]["lines"] += 1
                                    source_stats[category]["chars"] += len(clean_line)
                                    if category == "clinical_raw":
                                        source_chars["ocr"] += len(clean_line)
                                    if len(sample_outputs) < 5:
                                        sample_outputs.append(f"[{file}] {clean_line[:100]}...")
                            continue

                        # JSONL.GZ
                        if file.endswith(".jsonl.gz"):
                            import gzip
                            with gzip.open(file_path, "rt", encoding="utf-8") as f_in:
                                for line in f_in:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        rec = json.loads(line)
                                    except json.JSONDecodeError:
                                        parse_error_count += 1
                                        source_stats["clinical_raw"]["parse_errors"] += 1
                                        continue
                                    text = extract_smart(rec)
                                    if not text or len(text) < MIN_TEXT_LEN:
                                        continue
                                    file_hash = get_content_hash(text)
                                    if file_hash in unique_hashes:
                                        duplicate_count += 1
                                        continue
                                    unique_hashes.add(file_hash)
                                    clean_line = text.replace("\n", " ").strip()
                                    write_record(clean_line, "clinical_raw", f_out, split_handlers)
                                    valid_count += 1
                                    source_stats["clinical_raw"]["lines"] += 1
                                    source_stats["clinical_raw"]["chars"] += len(clean_line)
                                    if len(sample_outputs) < 5:
                                        sample_outputs.append(f"[{file}] {clean_line[:100]}...")
                            continue

                        # JSONL
                        if file.endswith(".jsonl"):
                            with open(file_path, "r", encoding="utf-8") as f_in:
                                for line in f_in:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        rec = json.loads(line)
                                    except json.JSONDecodeError:
                                        parse_error_count += 1
                                        source_stats["clinical_raw"]["parse_errors"] += 1
                                        continue
                                    text = extract_smart(rec)
                                    if not text or len(text) < MIN_TEXT_LEN:
                                        continue
                                    file_hash = get_content_hash(text)
                                    if file_hash in unique_hashes:
                                        duplicate_count += 1
                                        continue
                                    unique_hashes.add(file_hash)
                                    clean_line = text.replace("\n", " ").strip()
                                    write_record(clean_line, category, f_out, split_handlers)
                                    valid_count += 1
                                    source_stats[category]["lines"] += 1
                                    source_stats[category]["chars"] += len(clean_line)
                                    if len(sample_outputs) < 5:
                                        sample_outputs.append(f"[{file}] {clean_line[:100]}...")
                            continue

                        # JSON
                        if file.endswith(".json"):
                            try:
                                with open(file_path, "r", encoding="utf-8") as f_in:
                                    data = json.load(f_in)
                            except json.JSONDecodeError:
                                parse_error_count += 1
                                continue
                            full_text = extract_smart(data)
                            if not full_text or len(full_text) < MIN_TEXT_LEN:
                                continue
                            file_hash = get_content_hash(full_text)
                            if file_hash in unique_hashes:
                                duplicate_count += 1
                                continue
                            unique_hashes.add(file_hash)
                            clean_line = full_text.replace("\n", " ")
                            write_record(clean_line, category, f_out, split_handlers)
                            valid_count += 1
                            source_stats[category]["lines"] += 1
                            source_stats[category]["chars"] += len(clean_line)
                            if category == "clinical_raw":
                                source_chars["ocr"] += len(clean_line)
                            if len(sample_outputs) < 5:
                                sample_outputs.append(f"[{file}] {clean_line[:100]}...")
                            continue

                        # TXT
                        if file.endswith(".txt"):
                            with open(file_path, "r", encoding="utf-8") as f_in:
                                for line in f_in:
                                    clean_line = line.strip()
                                    if len(clean_line) < MIN_TEXT_LEN:
                                        continue
                                    file_hash = get_content_hash(clean_line)
                                    if file_hash in unique_hashes:
                                        duplicate_count += 1
                                        continue
                                    unique_hashes.add(file_hash)
                                    write_record(clean_line, category, f_out, split_handlers)
                                    valid_count += 1
                                    source_stats[category]["lines"] += 1
                                    source_stats[category]["chars"] += len(clean_line)
                                    if len(sample_outputs) < 5:
                                        sample_outputs.append(f"[{file}] {clean_line[:100]}...")
                            continue
                    except Exception as e:
                        # 忽略读取错误，保证主流程不断
                        pass

        # 处理列表类文件（书籍/论文/通用/补充），含 JSON / JSONL
        def process_file_list(file_list, category_key, label):
            nonlocal valid_count, parse_error_count
            for fp in file_list:
                if not os.path.exists(fp):
                    continue
                try:
                    # JSONL 按行读取
                    if fp.endswith(".jsonl"):
                        source_stats[category_key]["files"] += 1
                        with open(fp, "r", encoding="utf-8") as f_jsonl:
                            for line in f_jsonl:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    rec = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                segs = []
                                if isinstance(rec, dict):
                                    for k in ["instruction", "input", "output", "content", "text", "paragraph", "report"]:
                                        if k in rec and isinstance(rec[k], str):
                                            segs.append(rec[k].strip())
                                elif isinstance(rec, str):
                                    segs.append(rec.strip())
                                full_text = " ".join([s for s in segs if s])
                                if len(full_text) < MIN_TEXT_LEN:
                                    continue
                                file_hash = get_content_hash(full_text)
                                if file_hash in unique_hashes:
                                    duplicate_count += 1
                                    continue
                                unique_hashes.add(file_hash)
                                clean_line = full_text.replace("\n", " ")
                                write_record(clean_line, category_key, f_out, split_handlers)
                                valid_count += 1
                                source_stats[category_key]["lines"] += 1
                                source_stats[category_key]["chars"] += len(clean_line)
                                source_chars["book"] += len(clean_line)
                                if len(sample_outputs) < 5:
                                    sample_outputs.append(f"[{label}] {clean_line[:100]}...")
                        continue

                    # 普通 JSON
                    source_stats[category_key]["files"] += 1
                    with open(fp, 'r', encoding='utf-8') as f_json:
                        data = json.load(f_json)
                    records = data if isinstance(data, list) else [data]
                    for rec in records:
                        segs = []
                        if isinstance(rec, dict):
                            for k in ["instruction", "input", "output", "content", "text", "paragraph", "report"]:
                                if k in rec and isinstance(rec[k], str):
                                    segs.append(rec[k].strip())
                        elif isinstance(rec, str):
                            segs.append(rec.strip())
                        full_text = " ".join([s for s in segs if s])
                        if len(full_text) < MIN_TEXT_LEN:
                            continue
                        file_hash = get_content_hash(full_text)
                        if file_hash in unique_hashes:
                            duplicate_count += 1
                            continue
                        unique_hashes.add(file_hash)
                        clean_line = full_text.replace("\n", " ")
                        write_record(clean_line, category_key, f_out, split_handlers)
                        valid_count += 1
                        source_stats[category_key]["lines"] += 1
                        source_stats[category_key]["chars"] += len(clean_line)
                        source_stats[category_key]["files"] += 1
                        source_chars["book"] += len(clean_line)
                        if len(sample_outputs) < 5:
                            sample_outputs.append(f"[{label}] {clean_line[:100]}...")
                except Exception:
                    parse_error_count += 1
                    source_stats[category_key]["parse_errors"] += 1
                    continue

        # 旧书籍列表（兼容逻辑）
        process_file_list(BOOK_JSON_FILES, "book_old", "book_old")  # noqa: F821
        # 新的类别化文件
        process_file_list(MEDICAL_BOOK_FILES, "book_core", "book_core")
        process_file_list(MEDICAL_PAPER_FILES, "paper", "paper")
        process_file_list(GENERAL_CPT_FILES, "general", "general")
        process_file_list(SUPPLEMENTARY_FILES, "supplement", "supplement")

        # 关闭分类文件句柄
        for fh in split_handlers.values():
            fh.close()

    # 估算 token 数（优先使用指定 tokenizer）
    approx_tokens = None
    tokenizer_used = "chars (fallback)"
    try:
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_NAME)
        # 仅用长度和中文字符数量做粗估：单独跑 tokenizer 会很慢，这里用平均字符≈1 token 的近似；
        # 如果需要精确估计，可改为分批 tokenize。
        approx_tokens = source_chars["ocr"] + source_chars["book"]
        tokenizer_used = TOKENIZER_NAME + " (char≈token rough est)"
    except Exception:
        # 回退到字符数估计
        approx_tokens = source_chars["ocr"] + source_chars["book"]
        tokenizer_used = "chars (fallback)"

    print("\n" + "="*30)
    print("清洗与提取报告 (Final Report)")
    print("="*30)
    print(f"扫描文件总数: {total_files}")
    print(f"JSON解析失败: {parse_error_count}")
    print(f"发现重复内容: {duplicate_count}")
    print(f"重复率: {(duplicate_count / total_files * 100):.2f}%" if total_files > 0 else "0%")
    print(f"有效数据行数: {valid_count}")
    print(f"结果已保存至: {OUTPUT_FILE}")
    print("="*30)

    # 来源比例与 token 粗估
    total_lines_all = sum(v["lines"] for v in source_stats.values())
    total_chars_all = sum(v["chars"] for v in source_stats.values())
    print("来源行数/字符统计:")
    for k, v in source_stats.items():
        ratio = (v["lines"] / total_lines_all * 100) if total_lines_all else 0
        print(f"  - {k}: 行={v['lines']}, 字符={v['chars']}, 目录/文件数≈{v['files']}, 占比={ratio:.2f}%")
    print(f"总计: 行={total_lines_all}, 字符={total_chars_all}")
    print(f"Token 近似: {approx_tokens} (估算方式: {tokenizer_used})")
    
    print("\n=== 数据采样检查 (前 5 条) ===")
    for idx, sample in enumerate(sample_outputs):
        print(f"{idx+1}. {sample}")

if __name__ == "__main__":
    main()