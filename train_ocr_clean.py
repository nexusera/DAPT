import os
import re
from tokenizers import BertWordPieceTokenizer

# ================= 配置区域 =================
# 使用重采样后的语料
CORPUS_FILE = "train_balanced.txt"
OUTPUT_DIR = "./medical_vocab_ocr_only"

# 1. 训练参数
# 策略：先抓 80000 个，清洗掉垃圾后，可能剩 30000 个好词，这正好是我们想要的
VOCAB_SIZE = 80000  
MIN_FREQUENCY = 10
HANDLE_CHINESE_CHARS = False  # 保持 False 以挖掘长词

# 2. 清洗黑名单 (根据你的反馈定制)
# 常见姓氏：用于过滤 2-3 字的人名
# 注意：去掉了"高"、"方"、"马"等可能构成医疗词(高血压/方案/马凡氏)的姓，保留了明显的"吴、赵、钱、孙"等
SURNAMES = set("李王张刘陈杨黄赵周吴徐孙胡朱林何郭罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范石姚谭廖邹熊金陆郝孔白崔毛邱秦江史顾侯邵孟龙万段雷汤尹黎武乔贺赖龚文")

# OCR 粘连噪音
# 只要词以这些开头，或者包含这些结构，统统删掉
BAD_PREFIXES = ("右收到", "左收到", "告医", "送检", "操作", "录入", "审核", "报告", "采样", "接收", "诊断医师", "日期")
BAD_SUFFIXES = ("日", "月", "年", "医师", "医生", "时间")

# 数字+单位或占位符噪音（如 00mg, 01a, 123mm 等）
PAT_NOISY_NUM_UNIT = re.compile(r"^(0{1,3}[a-zA-Zµμ]{0,4}|[0-9]{1,3}[a-zA-Zµμ]{1,4})$")

# 混合数字/字母短噪音（如 0a0, 0b4, 0c5 等）
PAT_MIXED_SHORT = re.compile(r"^(?=.*\d)[0-9A-Za-zµμ]{2,6}$")

# 以 0 开头的短串（常见版面/占位噪音，如 00元, 0372马）
PAT_LEADING_ZERO = re.compile(r"^0[0-9A-Za-z\u4e00-\u9fffµμ]{1,7}$")

# 特定版面模式（如 06x10, 08行, 05为阳性）
PAT_LAYOUT = re.compile(r"^(?:[0-9]{1,3}x10|[0-9]{1,3}行|[0-9]{1,3}为阳性)$")

# 数字前缀 + 中文/英文词干（如 10中性粒细胞绝对值）：希望去掉前导数字保留词干
PAT_LEADING_DIGIT_WORD = re.compile(r"^\d{1,3}[\u4e00-\u9fffA-Za-z]")

# 长数字前缀混合（如 20251020g, 16696665788微信同号）：视为编码/电话，直接丢弃
PAT_LONG_DIGIT_MIX = re.compile(r"^\d{4,}[A-Za-z\u4e00-\u9fff].*")

# 字母开头长数字串（如 a02474970）：编码类，丢弃
PAT_LETTER_DIGIT_LONG = re.compile(r"^[A-Za-z]\d{4,}.*")

def is_garbage(token):
    """
    清洗核心逻辑
    """
    # 0. 单字符一律丢弃（中文/英文/符号）
    if len(token) == 1:
        return True
    # 1. 过滤单字 (官方词表都有，OCR挖出来的单字通常是生僻字或乱码)
    # 如果是纯中文且长度为1，删
    if len(token) == 1 and '\u4e00' <= token <= '\u9fff':
        return True

    # 2. 人名清洗
    # 逻辑：姓氏开头 + 长度为2或3 + 纯中文
    if 2 <= len(token) <= 3 and token[0] in SURNAMES:
        # 正则判断是否纯中文
        if re.match(r'^[\u4e00-\u9fa5]+$', token):
            return True

    # 3. OCR 结构性噪音
    for p in BAD_PREFIXES:
        if token.startswith(p):
            return True
    for s in BAD_SUFFIXES:
        # 比如 "2023年"，我们不需要这样的词
        if token.endswith(s) and len(token) > len(s):
            return True
            
    # 4. 纯数字/纯符号 (WordPiece 有时会保留纯数字)
    if re.match(r'^[0-9\W_]+$', token):
        return True

    # 5. 数字+单位或占位符（典型噪音: 00mg, 01a, 12mm, 5μmol 等）
    # 数值信息可以由“数字”+“单位”组成，不需要保留特定组合。
    if PAT_NOISY_NUM_UNIT.match(token):
        return True

    # 6. 混合数字/字母短噪音（如 0a0, 0b4, 0c5 等）
    if PAT_MIXED_SHORT.match(token):
        return True

    # 7. 以数字开头且很短的混合串（如 05为阳性，长度<=6）：多数为编码/占位，可丢弃
    if token[0].isdigit() and len(token) <= 6:
        return True

    # 8. 以 0 开头的短串（版面/占位噪音）
    if PAT_LEADING_ZERO.match(token):
        return True

    # 9. 特定版面模式（如 06x10, 08行, 05为阳性）
    if PAT_LAYOUT.match(token):
        return True

    # 10. 长数字前缀混合（电话/日期/编码）
    if PAT_LONG_DIGIT_MIX.match(token):
        return True

    # 11. 字母开头长数字编码
    if PAT_LETTER_DIGIT_LONG.match(token):
        return True

    # 12. 含“微信同号”直接丢弃
    if "微信同号" in token:
        return True

    return False

def main():
    print(f"🚀 [Step 1] 开始 WordPiece 训练 (挖掘模式)...")
    print(f"   目标: 先挖掘 {VOCAB_SIZE} 个候选词")
    
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=HANDLE_CHINESE_CHARS,
        strip_accents=True,
        lowercase=True
    )

    # 训练
    tokenizer.train(
        files=[CORPUS_FILE],
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=1000,
        wordpieces_prefix="##",
    )
    
    # 获取原始词表
    raw_vocab = tokenizer.get_vocab() # 这是一个 dict {token: id}
    # 按 ID 排序还原列表
    sorted_raw_vocab = sorted(raw_vocab.keys(), key=lambda k: raw_vocab[k])
    
    print(f"   ✅ 挖掘完成，原始词表大小: {len(sorted_raw_vocab)}")

    # --- 开始清洗 ---
    print(f"\n🚀 [Step 2] 开始执行清洗规则 (去除人名/OCR噪音)...")
    
    clean_vocab = []
    dropped_count = 0
    
    for token in sorted_raw_vocab:
        # 跳过特殊字符
        if token in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
            continue
            
        # 过滤 ## 开头的英文后缀 (通常医疗词不需要这个，除非你想保留英文术语后缀)
        if token.startswith("##"):
            continue

        # 去掉前导数字（如 10中性粒细胞绝对值 -> 中性粒细胞绝对值）再判定；
        # 若数字前缀>=4位且后续为字母/中文（多为电话/编码），直接丢弃
        normalized = token
        if PAT_LONG_DIGIT_MIX.match(token):
            dropped_count += 1
            continue
        if PAT_LEADING_DIGIT_WORD.match(token):
            normalized = re.sub(r"^\d{1,3}", "", token)

        if not normalized:  # 全是数字被剔除
            dropped_count += 1
            continue

        if is_garbage(normalized):
            dropped_count += 1
        else:
            clean_vocab.append(normalized)
            
    # 去重并排序 (简单的字典序)
    clean_vocab = sorted(list(set(clean_vocab)))

    print(f"   🗑️  过滤掉垃圾词: {dropped_count} 个")
    print(f"   💎 最终保留纯净词: {len(clean_vocab)} 个")

    # --- 保存 ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    output_path = os.path.join(OUTPUT_DIR, "vocab.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for token in clean_vocab:
            f.write(token + "\n")
            
    print(f"\n💾 结果已保存: {os.path.abspath(output_path)}")
    print("-" * 30)
    print("👀 抽查前 20 个结果 (看看还有没有人名):")
    print(clean_vocab[:20])
    print("-" * 30)
    print("👀 抽查后 20 个结果:")
    print(clean_vocab[-20:])

if __name__ == "__main__":
    main()