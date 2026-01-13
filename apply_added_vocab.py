import os
from transformers import AutoModelForMaskedLM, AutoTokenizer

WORKSPACE_DIR = "/data/ocean/bpe_workspace"
BASE_MODEL = "hfl/chinese-roberta-wwm-ext"
# Tokenizer 统一指向 /data/ocean/DAPT/my-medical-tokenizer
ADDED_TOKENIZER_DIR = "/data/ocean/DAPT/my-medical-tokenizer"
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "output_medical_bert_add_vocab")


def main():
    tokenizer = AutoTokenizer.from_pretrained(ADDED_TOKENIZER_DIR)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)

    # 增加词表并扩展 embedding
    old_vocab_size = model.get_input_embeddings().weight.size(0)
    new_vocab_size = len(tokenizer)
    if new_vocab_size > old_vocab_size:
        model.resize_token_embeddings(new_vocab_size)
        print(f"Resized embeddings: {old_vocab_size} -> {new_vocab_size}")
    else:
        print("No new tokens to add.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    print("Saved model+tokenizer to", os.path.join(OUTPUT_DIR, "final_model"))


if __name__ == '__main__':
    main()
