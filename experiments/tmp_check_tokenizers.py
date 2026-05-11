#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_PATHS = {
    "public_tokenizer": "/data/ocean/DAPT/my-medical-tokenizer",
    "noise_bucket_model": "/data/ocean/DAPT/workspace/output_ablation_noise_bucket/final_staged_model",
    "mlm_kvmlm_model": "/data/ocean/DAPT/workspace/output_ablation_mlm_kvmlm/final_staged_model",
    "nsp_ratio_1_1_model": "/data/ocean/DAPT/workspace/output_ablation_nsp_ratio_1_1/final_staged_model",
}

FILES_TO_CHECK = [
    "vocab.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
]

PROBE_TEXTS = [
    "肿瘤标志物",
    "白细胞",
    "既往史",
    "糖类抗原CA19-9",
]


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(value: str | None) -> str:
    return value[:12] if value else "MISSING"


def compare_dicts(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(
        b, sort_keys=True, ensure_ascii=False
    )


def inspect_tokenizer(name: str, path_str: str) -> dict:
    path = Path(path_str)
    print("=" * 100)
    print(f"[{name}] {path}")
    if not path.exists():
        print("  PATH_MISSING=True")
        return {"missing": True}

    file_hashes: dict[str, str | None] = {}
    print("  [file hashes]")
    for fn in FILES_TO_CHECK:
        hv = sha256_file(path / fn)
        file_hashes[fn] = hv
        print(f"    {fn:<24} {short_hash(hv)}")

    tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    added_vocab = tok.get_added_vocab()
    probes = {text: tok.tokenize(text) for text in PROBE_TEXTS}

    print("  [tokenizer summary]")
    print(f"    class={type(tok).__name__}")
    print(f"    is_fast={getattr(tok, 'is_fast', None)}")
    print(f"    vocab_size={len(tok)}")
    print(f"    added_vocab_size={len(added_vocab)}")
    print(
        "    specials="
        f"cls={tok.cls_token!r}, sep={tok.sep_token!r}, "
        f"unk={tok.unk_token!r}, mask={tok.mask_token!r}, pad={tok.pad_token!r}"
    )

    print("  [probe tokenization]")
    for text, pieces in probes.items():
        print(f"    {text!r} -> {pieces}")

    return {
        "missing": False,
        "path": str(path),
        "file_hashes": file_hashes,
        "class_name": type(tok).__name__,
        "is_fast": getattr(tok, "is_fast", None),
        "vocab_size": len(tok),
        "added_vocab": dict(sorted(added_vocab.items())),
        "specials": {
            "cls_token": tok.cls_token,
            "sep_token": tok.sep_token,
            "unk_token": tok.unk_token,
            "mask_token": tok.mask_token,
            "pad_token": tok.pad_token,
            "bos_token": tok.bos_token,
            "eos_token": tok.eos_token,
        },
        "probes": probes,
    }


def print_reference_compare(reference_name: str, all_info: dict[str, dict]) -> None:
    ref = all_info[reference_name]
    if ref.get("missing"):
        print("\n[WARN] reference tokenizer is missing, skip compare.")
        return

    print("\n" + "=" * 100)
    print(f"[COMPARE AGAINST {reference_name}]")
    header = (
        f"{'name':<22} | {'hashes':<7} | {'vocab':<7} | "
        f"{'added':<7} | {'specials':<8} | {'probes':<7}"
    )
    print(header)
    print("-" * len(header))
    for name, cur in all_info.items():
        if name == reference_name or cur.get("missing"):
            continue
        same_hashes = cur["file_hashes"] == ref["file_hashes"]
        same_vocab = cur["vocab_size"] == ref["vocab_size"]
        same_added = compare_dicts(cur["added_vocab"], ref["added_vocab"])
        same_specials = compare_dicts(cur["specials"], ref["specials"])
        same_probes = compare_dicts(cur["probes"], ref["probes"])
        print(
            f"{name:<22} | {str(same_hashes):<7} | {str(same_vocab):<7} | "
            f"{str(same_added):<7} | {str(same_specials):<8} | {str(same_probes):<7}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether several tokenizer directories are effectively the same."
    )
    parser.add_argument(
        "--reference",
        default="public_tokenizer",
        help="Reference tokenizer name for final comparison.",
    )
    parser.add_argument(
        "--paths_json",
        default=None,
        help="Optional JSON file mapping tokenizer names to directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = DEFAULT_PATHS
    if args.paths_json:
        with open(args.paths_json, "r", encoding="utf-8") as f:
            paths = json.load(f)

    info: dict[str, dict] = {}
    for name, path in paths.items():
        info[name] = inspect_tokenizer(name, path)

    if args.reference not in info:
        raise KeyError(f"Unknown reference name: {args.reference}")

    print_reference_compare(args.reference, info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
