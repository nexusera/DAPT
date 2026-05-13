#!/usr/bin/env python3
"""Run a minimal Tesseract 5 OCR smoke test on one image.

This helper:
1. runs Tesseract OCR on one image
2. saves plain OCR text next to the image
3. saves raw Tesseract TSV rows as json next to the image
4. saves a lightweight DAPT-style `ocr_raw.words_result` json next to the image

Notes
-----
- The exported `ocr_raw.words_result` is word-level because Tesseract TSV is
  word-oriented at `level == 5`.
- Tesseract TSV exposes one confidence per word, not per character. The
  exported `chars` list therefore preserves characters only, without per-char
  probabilities.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytesseract
from PIL import Image
from pytesseract import Output


DEFAULT_IMAGE = Path("/Users/wy/Documents/data/半结构化/病历/CT/record-image.jpeg")
DEFAULT_LANG = "chi_sim+eng"
DEFAULT_PSM = 6
DEFAULT_OEM = 3


def _resolve_tesseract_cmd() -> str:
    custom = os.environ.get("TESSERACT_CMD")
    if custom:
        return custom
    found = shutil.which("tesseract")
    if found:
        return found
    raise RuntimeError(
        "Tesseract executable not found. Activate the dapt environment or set "
        "TESSERACT_CMD explicitly."
    )


def _float_conf(raw: str) -> float | None:
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if value < 0:
        return None
    return value / 100.0


def _read_tsv(tsv_text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
    return [dict(row) for row in reader]


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return default


def _word_item_from_row(row: dict[str, str]) -> dict[str, Any] | None:
    text = (row.get("text") or "").strip()
    if not text:
        return None
    if _to_int(row.get("level")) != 5:
        return None

    conf = _float_conf(row.get("conf", ""))
    left = _to_int(row.get("left"))
    top = _to_int(row.get("top"))
    width = _to_int(row.get("width"))
    height = _to_int(row.get("height"))

    probability = {
        "average": float(conf) if conf is not None else 0.0,
        "min": float(conf) if conf is not None else 0.0,
        "variance": 0.0,
    }

    return {
        "words": text,
        "location": {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        },
        "probability": probability,
        "chars": [{"char": ch} for ch in text],
        "tesseract_meta": {
            "conf_0_100": _to_int(row.get("conf", ""), default=-1),
            "page_num": _to_int(row.get("page_num")),
            "block_num": _to_int(row.get("block_num")),
            "par_num": _to_int(row.get("par_num")),
            "line_num": _to_int(row.get("line_num")),
            "word_num": _to_int(row.get("word_num")),
        },
    }


def _to_dapt_ocr_raw(rows: list[dict[str, str]]) -> dict[str, Any]:
    words_result: list[dict[str, Any]] = []
    for row in rows:
        item = _word_item_from_row(row)
        if item is not None:
            words_result.append(item)
    return {
        "source": "tesseract_smoke_test",
        "words_result_num": len(words_result),
        "words_result": words_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal Tesseract 5 OCR smoke test")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--out-dir", type=Path, default=None, help="Defaults to the image directory")
    parser.add_argument("--lang", default=DEFAULT_LANG, help="Tesseract language list, e.g. chi_sim+eng")
    parser.add_argument("--psm", type=int, default=DEFAULT_PSM, help="Tesseract page segmentation mode")
    parser.add_argument("--oem", type=int, default=DEFAULT_OEM, help="Tesseract OCR engine mode")
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Optional DPI override passed through Tesseract config",
    )
    return parser.parse_args()


def _run_tesseract(image: Path, lang: str, psm: int, oem: int, dpi: int | None) -> tuple[str, str]:
    pytesseract.pytesseract.tesseract_cmd = _resolve_tesseract_cmd()
    config_parts = [f"--psm {psm}", f"--oem {oem}"]
    if dpi is not None:
        config_parts.append(f"-c user_defined_dpi={dpi}")
    config = " ".join(config_parts)
    pil_image = Image.open(image)
    text = pytesseract.image_to_string(pil_image, lang=lang, config=config)
    tsv_text = pytesseract.image_to_data(
        pil_image,
        lang=lang,
        config=config,
        output_type=Output.STRING,
    )
    return text, tsv_text


def main() -> int:
    args = parse_args()
    image = args.image.resolve()
    if not image.exists():
        print(f"[ERROR] image not found: {image}", file=sys.stderr)
        return 2

    out_dir = (args.out_dir or image.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image.stem

    try:
        text, tsv_text = _run_tesseract(
            image=image,
            lang=args.lang,
            psm=args.psm,
            oem=args.oem,
            dpi=args.dpi,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 3

    rows = _read_tsv(tsv_text)
    raw_pred = {
        "engine": "tesseract",
        "engine_version": 5,
        "image": str(image),
        "lang": args.lang,
        "psm": args.psm,
        "oem": args.oem,
        "dpi": args.dpi,
        "rows": rows,
    }
    ocr_raw = _to_dapt_ocr_raw(rows)

    raw_pred_path = out_dir / f"{stem}.tesseract.pred.json"
    text_path = out_dir / f"{stem}.tesseract.txt"
    ocr_raw_path = out_dir / f"{stem}.tesseract.ocr_raw.json"

    raw_pred_path.write_text(json.dumps(raw_pred, ensure_ascii=False, indent=2), encoding="utf-8")
    text_path.write_text(text, encoding="utf-8")
    ocr_raw_path.write_text(json.dumps(ocr_raw, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "text_path": str(text_path),
            "raw_pred_path": str(raw_pred_path),
            "ocr_raw_path": str(ocr_raw_path),
            "words_result_count": ocr_raw["words_result_num"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
