import json
import os
import random
from functools import lru_cache
from typing import Dict, Any


@lru_cache(maxsize=1)
def _load_mapping() -> Dict[str, Any]:
    # 1) 环境变量覆盖
    cand = []
    env_path = os.environ.get("EBQA_STRUCT_PATH") or os.environ.get("STRUCT_PATH")
    if env_path:
        cand.append(env_path)

    # 2) 项目内默认路径
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # /data/ocean/DAPT
    cand.append(os.path.join(repo_root, "data", "kv_ner_prepared_comparison", "keys_v2.json"))
    cand.append(os.path.join(repo_root, "dapt_eval_package", "kv_ner_prepared_comparison", "keys_v2.json"))
    cand.append(os.path.join(repo_root, "keys", "keys_merged.json"))

    for p in cand:
        if p and os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return {}


def convert_key_to_question(category, key):
    mapping_json = _load_mapping()
    category_data = mapping_json.get(category, {}) if isinstance(mapping_json, dict) else {}
    key_data = category_data.get(key, {}) if isinstance(category_data, dict) else {}
    alias_str = ""
    
    if isinstance(key_data, dict):
        custom_question = key_data.get("Q", "")
        alias = key_data.get("别名", [])
        
        if alias:
            if len(alias) >= 2:
                alias_str = f",{key}别名有" + "、".join(random.sample(alias, 2)) + "等"
            elif len(alias) == 1:
                alias_str = f",{key}别名有{alias[0]}"
        
        if custom_question:
            return custom_question + alias_str
    
    return f"找到文本中的{key}" + alias_str


if __name__ == "__main__":
    print(convert_key_to_question("通用病历", "姓名"))