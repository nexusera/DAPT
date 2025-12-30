import os
import re
import random
from transformers import BertTokenizer

# ================= 🎛️ 精简版配置 =================
# 与 v9 逻辑一致，但显式去掉外部医疗词典 (medical_dict.txt)，避免词表过长。
BASE_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
OUTPUT_DIR = "./my-medical-tokenizer"

# 长度阈值 (超过这个长度的词/提取后的词，依然会被拦截)
MAX_TOKEN_LENGTH = 7

# 👑 VIP 白名单 (绝对保留)
VIP_TERMS = {
    "brca1基因", "brca2基因", "her2基因", "ki67", "fish检测",
    "er阳性", "pr阳性", "p53蛋白", "ptnm分期"
}

# 去掉 medical_dict.txt，仅保留 OCR 挖掘和业务 Keys
SOURCES = [
    {"path": "./medical_vocab_ocr_only/vocab.txt", "type": "dict"},
    {"path": "./keys.txt", "type": "key"}
]

def smart_extract_key_core(token):
    """
    🧠 核心逻辑：从复杂的 Key 中提取原子级的核心概念
    返回一个 list，包含提取出的核心词 (可能有多个)
    """
    extracted_candidates = set()

    cleaned = re.sub(r'[\(（][^\)）]+[\)）]', '', token)
    if len(cleaned) > 1 and len(cleaned) < len(token):
        extracted_candidates.add(cleaned.strip())

    chinese_parts = re.findall(r'[\u4e00-\u9fa5]+', token)
    for part in chinese_parts:
        if len(part) > 1:
            extracted_candidates.add(part)

    split_parts = re.split(r'[：:-]', token)
    if len(split_parts) > 1:
        for part in split_parts:
            clean_part = part.strip()
            if len(clean_part) > 1:
                extracted_candidates.add(clean_part)

    return list(extracted_candidates)

def main():
    print("🚀 精简版 Tokenizer 构建 (基于 v9, 移除 medical_dict.txt)...")

    try:
        tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_NAME)
    except Exception as e:
        print(f"❌ 基座加载失败: {e}")
        return

    candidates = set()
    vip_lower = {v.lower() for v in VIP_TERMS}

    for vip in VIP_TERMS:
        candidates.add(vip)

    stats = {
        "scanned": 0,
        "kept_direct": 0,
        "regex_extracted": 0,
        "dropped_too_long": 0
    }

    dropped_by_len = {}
    dropped_samples = {}
    MAX_DROPPED_SAMPLES = 5

    per_source_seen = {os.path.basename(s["path"]): 0 for s in SOURCES}
    per_source_added = {os.path.basename(s["path"]): 0 for s in SOURCES}

    def maybe_keep_cn_sample(length_dict, sample_dict, L, token):
        if not re.search(r"[\u4e00-\u9fff]", token):
            return
        sample_dict.setdefault(L, [])
        seen = length_dict.get(L, 0)
        bucket = sample_dict[L]
        if len(bucket) < MAX_DROPPED_SAMPLES:
            bucket.append(token)
        else:
            if random.randint(1, seen) <= MAX_DROPPED_SAMPLES:
                bucket[random.randrange(MAX_DROPPED_SAMPLES)] = token

    for src in SOURCES:
        fpath = src["path"]
        src_type = src["type"]
        src_name = os.path.basename(fpath)

        if not os.path.exists(fpath):
            print(f"⚠️ 跳过: 未找到 {fpath}")
            continue

        print(f"📖 读取: {fpath} (类型: {src_type})")

        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.strip()
                if not token or " " in token:
                    continue
                stats["scanned"] += 1
                per_source_seen[src_name] += 1

                if token.lower() in vip_lower:
                    if token not in candidates:
                        per_source_added[src_name] += 1
                    candidates.add(token)
                    continue

                if src_type == "key" and len(token) > MAX_TOKEN_LENGTH:
                    cores = smart_extract_key_core(token)
                    if cores:
                        for core in cores:
                            if len(core) <= MAX_TOKEN_LENGTH:
                                if core not in candidates:
                                    per_source_added[src_name] += 1
                                candidates.add(core)
                                stats["regex_extracted"] += 1
                            else:
                                stats["dropped_too_long"] += 1
                                L = len(core)
                                dropped_by_len[L] = dropped_by_len.get(L, 0) + 1
                                maybe_keep_cn_sample(dropped_by_len, dropped_samples, L, core)
                        continue

                if len(token) > MAX_TOKEN_LENGTH:
                    stats["dropped_too_long"] += 1
                    L = len(token)
                    dropped_by_len[L] = dropped_by_len.get(L, 0) + 1
                    maybe_keep_cn_sample(dropped_by_len, dropped_samples, L, token)
                    continue

                if len(token) > 1:
                    if token not in candidates:
                        per_source_added[src_name] += 1
                    candidates.add(token)
                    stats["kept_direct"] += 1

    base_vocab = tokenizer.get_vocab()
    final_tokens = [w for w in candidates if w not in base_vocab]

    print("-" * 30)
    print("📊 统计:")
    print(f"   - ✂️ 拦截过长垃圾: {stats['dropped_too_long']} 个")
    print(f"   - 🧪 从长Key中提纯: {stats['regex_extracted']} 个核心词")
    print(f"   - ✅ 最终新增注入: {len(final_tokens)} 个")
    print("-" * 30)

    print("📑 各源贡献 (已去重后新增):")
    for src_name, cnt in per_source_added.items():
        seen = per_source_seen.get(src_name, 0)
        print(f"   - {src_name}: 扫描 {seen} 行，新增 {cnt} 词")

    if dropped_by_len:
        print("🧹 过长词长度分布（仅中文样例，随机采样，最多 5 个/长度）：")
        for L in sorted(dropped_by_len.keys()):
            samples = dropped_samples.get(L, [])
            preview = " | ".join(samples)
            print(f"   len={L:>2}  count={dropped_by_len[L]:>6}  sample: {preview}")

    tokenizer.add_tokens(final_tokens)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 结果已保存: {OUTPUT_DIR}")

    print("\n🔍 复杂 Key 切分验证:")
    test_cases = [
        "既往史(PastHistory)",
        "临床资料（手术所见、影像学、相关检验等）",
        "PTNM分期(AJCC第8版)"
    ]

    for tc in test_cases:
        tokens = tokenizer.tokenize(tc)
        print(f"\n   输入: '{tc}'")
        print(f"   输出: {tokens}")
        core_words = ["既往史", "临床资料", "PTNM分期"]
        hit = any(cw in tokens or cw.lower() in [t.lower() for t in tokens] for cw in core_words)
        if hit:
            print("   ✅ 成功提取核心实体！")
        else:
            print("   ⚠️  未提取到核心实体，可能已被切碎。")

if __name__ == "__main__":
    main()
