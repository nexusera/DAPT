#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return obj


def _dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _variant_final_model_dir(out_root: str, variant: str, seed: str) -> str:
    # Matches tokenizer ablation README naming.
    # OUT_ROOT=/data/ocean/DAPT/ablation/tokenizer
    # Full runs are stored under ${OUT_ROOT}/runs/t{n}_full_seed{seed}/final_staged_model
    return f"{out_root}/runs/{variant}_full_seed{seed}/final_staged_model"


def main() -> None:
    p = argparse.ArgumentParser(description="Generate KV-NER + EBQA configs for 4 tokenizer variants (T1~T4)")
    p.add_argument("--dapt_root", type=str, default="/data/ocean/DAPT", help="Repo root on the server")
    p.add_argument("--out_root", type=str, default="/data/ocean/DAPT/ablation/tokenizer", help="Tokenizer ablation OUT_ROOT")
    p.add_argument("--seed", type=str, default="42", help="Seed suffix used in run directory names")
    p.add_argument(
        "--query_set",
        type=str,
        default="/data/ocean/DAPT/dapt_eval_package/MedStruct-S-master/keys_merged_1027_cleaned.json",
        help="Query set / schema JSON used by downstream evaluation",
    )
    p.add_argument(
        "--kvner_template",
        type=str,
        default="/data/ocean/DAPT/dapt_eval_package/pre_struct/kv_ner/kv_ner_config.json",
        help="KV-NER base config to clone",
    )
    p.add_argument(
        "--ebqa_template",
        type=str,
        default="/data/ocean/DAPT/dapt_eval_package/pre_struct/ebqa/ebqa_config_macbert.json",
        help="EBQA base config to clone",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/data/ocean/DAPT/experiments/downstream_finetune/generated_configs",
        help="Where to write generated configs",
    )
    args = p.parse_args()

    dapt_root = str(args.dapt_root).rstrip("/")
    out_root = str(args.out_root).rstrip("/")
    seed = str(args.seed)

    variants = {
        "t1": "t1",
        "t2": "t2",
        "t3": "t3",
        "t4": "t4",
    }

    kvner_tpl = _load_json(Path(args.kvner_template))
    ebqa_tpl = _load_json(Path(args.ebqa_template))

    out_dir = Path(args.output_dir)

    for v in variants.values():
        model_dir = _variant_final_model_dir(out_root, v, seed)

        # --- KV-NER config ---
        kv_cfg = json.loads(json.dumps(kvner_tpl))  # deep copy
        kv_cfg["model_name_or_path"] = model_dir
        if not isinstance(kv_cfg.get("train"), dict):
            kv_cfg["train"] = {}
        kv_cfg["train"]["output_dir"] = f"{dapt_root}/runs/kv_ner_finetuned_{v}"
        kv_out = out_dir / f"kv_ner_config_{v}.json"
        _dump_json(kv_out, kv_cfg)

        # --- EBQA config ---
        ebqa_cfg = json.loads(json.dumps(ebqa_tpl))  # deep copy
        ebqa_cfg["report_struct_path"] = args.query_set
        ebqa_cfg["model_name_or_path"] = model_dir
        ebqa_cfg["tokenizer_name_or_path"] = model_dir
        ebqa_cfg["output_dir"] = f"{dapt_root}/runs/ebqa_{v}"
        ebqa_cfg["model_dir"] = f"{dapt_root}/runs/ebqa_{v}/best"
        if not isinstance(ebqa_cfg.get("train"), dict):
            ebqa_cfg["train"] = {}
        ebqa_cfg["train"]["data_path"] = f"{dapt_root}/data/kv_ner_prepared_comparison/ebqa_train_real_{v}.jsonl"
        ebqa_out = out_dir / f"ebqa_config_{v}.json"
        _dump_json(ebqa_out, ebqa_cfg)

    print(f"[OK] Generated configs under: {out_dir}")


if __name__ == "__main__":
    main()
