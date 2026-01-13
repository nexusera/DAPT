import os
from pathlib import Path

# ================= 🎛️ 核心配置 =================
# 1. 输入：优先使用 LLM 精修后的 OCR 词表 kept_vocab.txt，若无则回退原始 OCR 词表；外加业务键名（纯键名，min5）
WORKSPACE_DIR = Path("/data/ocean/DAPT/workspace")
PROJECT_ROOT = Path("/data/ocean/DAPT")
OCR_VOCAB_MAIN = WORKSPACE_DIR / "kept_vocab.txt"
OCR_VOCAB_FALLBACK = WORKSPACE_DIR / "medical_vocab_ocr_only" / "vocab.txt"
KEYS_FILE = PROJECT_ROOT / "biaozhu_keys_only_min5.txt"  # 纯键名，频次>5

# 2. 输出: 给 Jieba 用的外挂词典
OUTPUT_FILE = "vocab_for_jieba.txt"

# 3. 过滤逻辑 (必须与 Tokenizer 合并时的逻辑保持一致)
MAX_LEN = 6  # 超过6字的不要 (太长的可能是垃圾句子)

# 4. 👑 VIP 白名单 (必须与 Tokenizer 保持一致)
# 这些词虽然可能超长，但必须保留
VIP_TERMS = {
    "brca1基因", "brca2基因", "her2基因", "fish检测",
    "er阳性", "pr阳性", "p53蛋白", "ptnm分期"
}

def main():
    print(f"🚀 开始生成 Jieba 专用词典 (源自 OCR 挖掘)...")

    sources = []
    if OCR_VOCAB_MAIN.exists():
        sources.append(OCR_VOCAB_MAIN)
    elif OCR_VOCAB_FALLBACK.exists():
        print(f"⚠️ 未找到 {OCR_VOCAB_MAIN}，回退使用 {OCR_VOCAB_FALLBACK}")
        sources.append(OCR_VOCAB_FALLBACK)
    else:
        print(f"❌ 未找到 OCR 词表: {OCR_VOCAB_MAIN} 或 {OCR_VOCAB_FALLBACK}")
        return

    if KEYS_FILE.exists():
        sources.append(KEYS_FILE)
    else:
        print(f"⚠️ 未找到业务键名文件: {KEYS_FILE}，将仅使用 OCR 词表")

    valid_words = set()
    stats = {
        "total": 0,
        "kept_vip": 0,
        "kept_normal": 0,
        "dropped_too_long": 0,
        "dropped_garbage": 0
    }

    # 预加载 VIP (确保 VIP 一定在词典里)
    for vip in VIP_TERMS:
        valid_words.add(vip)

    vip_lower = {v.lower() for v in VIP_TERMS}

    for src in sources:
        with open(src, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.strip()
                if not token:
                    continue
                stats["total"] += 1

                # 0. 基础清洗
                if " " in token:
                    stats["dropped_garbage"] += 1
                    continue
                
                # 1. VIP 检查 (绿色通道)
                if token.lower() in vip_lower:
                    valid_words.add(token)
                    stats["kept_vip"] += 1
                    continue

                # 2. 长度拦截
                if len(token) > MAX_LEN:
                    stats["dropped_too_long"] += 1
                    continue

                # 3. 保留常规词 (双字以上)
                if len(token) > 1:
                    valid_words.add(token)
                    stats["kept_normal"] += 1

    # 保存
    print(f"💾 正在保存 {len(valid_words)} 个词到 {OUTPUT_FILE} ...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for word in sorted(list(valid_words)):
            # Jieba 词典格式: 词语 词频(可选)
            # 我们给一个默认高频值，保证 Jieba 优先切分
            f.write(f"{word} 99999\n")

    print("\n" + "="*40)
    print(f"✅ Jieba 词典生成完成！")
    print(f"📊 统计:")
    print(f"   - 原始词数: {stats['total']}")
    print(f"   - ✂️ 剔除长句/垃圾: {stats['dropped_too_long'] + stats['dropped_garbage']}")
    print(f"   - 💎 最终保留: {len(valid_words)}")
    print("="*40)
    print("💡 下一步: 在 build_dataset.py 中，记得把这个文件也加入 load_userdict 列表。")

if __name__ == "__main__":
    main()
