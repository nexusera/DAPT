#!/usr/bin/env python3
"""Run a small Tesseract 5 rescue grid on one image.

This script is meant for fast triage:
1. preprocess the image with a small set of OCR-oriented variants
2. run Tesseract with `tessdata_best` and a PSM grid
3. save per-run text / TSV / lightweight ocr_raw artifacts
4. emit a ranked summary so we can decide whether Tesseract is still viable
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageOps
import pytesseract
from pytesseract import Output
from skimage.filters import threshold_sauvola


DEFAULT_IMAGE = Path("/Users/wy/Documents/data/半结构化/病历/CT/record-image.jpeg")
DEFAULT_TESSDATA_DIR = Path("/Users/wy/Documents/code/DAPT/.cache/tessdata_best")
DEFAULT_LANG = "chi_sim+eng"
DEFAULT_PSMS = [3, 4, 6, 11, 12]
DEFAULT_ANCHORS = [
    "天津海滨人民医院",
    "CT诊断报告单",
    "申请科室",
    "检查方法",
    "影像所见",
    "检查结果",
    "报告医生",
    "审核医生",
]


@dataclass
class Variant:
    name: str
    fn: Callable[[Image.Image], Image.Image]
    dpi: int | None = None


def _resolve_tesseract_cmd() -> str:
    custom = os.environ.get("TESSERACT_CMD")
    if custom:
        return custom
    found = shutil.which("tesseract")
    if found:
        return found
    raise RuntimeError("Tesseract executable not found")


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _gray(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(_pil_to_bgr(image), cv2.COLOR_BGR2GRAY)


def _crop_page(image: Image.Image) -> Image.Image:
    bgr = _pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    h, w = gray.shape[:2]
    page = None
    best_area = 0
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if area > best_area and cw > w * 0.6 and ch > h * 0.6:
            page = (x, y, cw, ch)
            best_area = area
    if page is None:
        return image
    x, y, cw, ch = page
    pad = 8
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + cw + pad)
    y1 = min(h, y + ch + pad)
    return image.crop((x0, y0, x1, y1))


def _adaptive_bin(image: Image.Image) -> Image.Image:
    gray = _gray(image)
    out = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    return Image.fromarray(out)


def _sauvola_bin(image: Image.Image) -> Image.Image:
    gray = _gray(image)
    thresh = threshold_sauvola(gray, window_size=31, k=0.15)
    out = np.where(gray > thresh, 255, 0).astype(np.uint8)
    return Image.fromarray(out)


def _deskew(image: Image.Image) -> Image.Image:
    bgr = _pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    angle = -angle
    if abs(angle) < 0.2:
        return image
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        bgr,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return _bgr_to_pil(rotated)


def _trim_and_pad(image: Image.Image, pad: int = 10) -> Image.Image:
    gray = _gray(image)
    mask = gray < 245
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return ImageOps.expand(image, border=pad, fill="white")
    x0, x1 = max(0, xs.min() - 2), min(gray.shape[1], xs.max() + 3)
    y0, y1 = max(0, ys.min() - 2), min(gray.shape[0], ys.max() + 3)
    cropped = image.crop((x0, y0, x1, y1))
    return ImageOps.expand(cropped, border=pad, fill="white")


def _upscale_300(image: Image.Image) -> Image.Image:
    w, h = image.size
    scale = 2.0
    return image.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)


def _combined_best(image: Image.Image) -> Image.Image:
    out = _crop_page(image)
    out = _deskew(out)
    out = _trim_and_pad(out, pad=10)
    out = _upscale_300(out)
    out = _sauvola_bin(out)
    return out


VARIANTS: list[Variant] = [
    Variant("original", lambda img: img),
    Variant("crop_page", _crop_page),
    Variant("adaptive_bin", _adaptive_bin),
    Variant("sauvola_bin", _sauvola_bin),
    Variant("deskew", _deskew),
    Variant("trim_pad", _trim_and_pad),
    Variant("upscale_300", _upscale_300, dpi=300),
    Variant("combined_best", _combined_best, dpi=300),
]


def _read_tsv(tsv_text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
    return [dict(row) for row in reader]


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except Exception:
        return default


def _float_conf(raw: str) -> float | None:
    value = _to_float(raw, default=-1.0)
    if value < 0:
        return None
    return value / 100.0


def _word_item_from_row(row: dict[str, str]) -> dict[str, Any] | None:
    text = (row.get("text") or "").strip()
    if not text or _to_int(row.get("level")) != 5:
        return None
    conf = _float_conf(row.get("conf", ""))
    return {
        "words": text,
        "location": {
            "left": _to_int(row.get("left")),
            "top": _to_int(row.get("top")),
            "width": _to_int(row.get("width")),
            "height": _to_int(row.get("height")),
        },
        "probability": {
            "average": float(conf) if conf is not None else 0.0,
            "min": float(conf) if conf is not None else 0.0,
            "variance": 0.0,
        },
        "chars": [{"char": ch} for ch in text],
        "tesseract_meta": {
            "conf_0_100": _to_int(row.get("conf"), default=-1),
            "page_num": _to_int(row.get("page_num")),
            "block_num": _to_int(row.get("block_num")),
            "par_num": _to_int(row.get("par_num")),
            "line_num": _to_int(row.get("line_num")),
            "word_num": _to_int(row.get("word_num")),
        },
    }


def _to_dapt_ocr_raw(rows: list[dict[str, str]]) -> dict[str, Any]:
    words_result = []
    for row in rows:
        item = _word_item_from_row(row)
        if item is not None:
            words_result.append(item)
    return {
        "source": "tesseract_rescue_grid",
        "words_result_num": len(words_result),
        "words_result": words_result,
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _metrics(text: str, rows: list[dict[str, str]], anchors: list[str]) -> dict[str, Any]:
    words = [r for r in rows if (r.get("text") or "").strip() and _to_int(r.get("level")) == 5]
    confs = [
        _to_float(r.get("conf"), -1.0)
        for r in words
        if _to_float(r.get("conf"), -1.0) >= 0
    ]
    avg_conf = sum(confs) / len(confs) if confs else 0.0

    stripped = _normalize_text(text)
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", stripped))
    ascii_letters = len(re.findall(r"[A-Za-z]", stripped))
    digits = len(re.findall(r"\d", stripped))
    total = max(1, len(stripped))
    anchor_hits = sum(1 for anchor in anchors if _normalize_text(anchor) in stripped)

    # Biased toward Chinese structured report quality.
    score = (
        anchor_hits * 20.0
        + (avg_conf / 100.0) * 10.0
        + (cjk_chars / total) * 10.0
        - (ascii_letters / total) * 12.0
        + min(digits, 40) * 0.05
    )
    return {
        "word_count": len(words),
        "avg_conf_0_100": avg_conf,
        "cjk_chars": cjk_chars,
        "ascii_letters": ascii_letters,
        "digits": digits,
        "anchor_hits": anchor_hits,
        "score": round(score, 4),
    }


def _run_tesseract(
    image: Path,
    lang: str,
    psm: int,
    oem: int,
    tessdata_dir: Path,
    dpi: int | None,
) -> tuple[str, str]:
    pytesseract.pytesseract.tesseract_cmd = _resolve_tesseract_cmd()
    config_parts = [f"--tessdata-dir {tessdata_dir}", f"--oem {oem}", f"--psm {psm}"]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Tesseract 5 rescue grid on one image")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--out-dir", type=Path, default=Path("/Users/wy/Documents/code/DAPT/outputs/tesseract_rescue"))
    parser.add_argument("--lang", default=DEFAULT_LANG)
    parser.add_argument("--oem", type=int, default=1)
    parser.add_argument("--tessdata-dir", type=Path, default=DEFAULT_TESSDATA_DIR)
    parser.add_argument("--psm", type=int, nargs="+", default=DEFAULT_PSMS)
    parser.add_argument("--anchors", nargs="*", default=DEFAULT_ANCHORS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image.resolve()
    if not image_path.exists():
        print(f"[ERROR] image not found: {image_path}", file=sys.stderr)
        return 2
    tessdata_dir = args.tessdata_dir.resolve()
    if not tessdata_dir.exists():
        print(f"[ERROR] tessdata_dir not found: {tessdata_dir}", file=sys.stderr)
        return 3

    source_image = Image.open(image_path).convert("RGB")
    stem = image_path.stem
    run_root = (args.out_dir / stem).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        variant_dir = run_root / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        processed = variant.fn(source_image.copy())
        image_out = variant_dir / f"{stem}.{variant.name}.png"
        processed.save(image_out)

        for psm in args.psm:
            run_name = f"{variant.name}.psm{psm}"
            try:
                text, tsv_text = _run_tesseract(
                    image=image_out,
                    lang=args.lang,
                    psm=psm,
                    oem=args.oem,
                    tessdata_dir=tessdata_dir,
                    dpi=variant.dpi,
                )
                rows = _read_tsv(tsv_text)
                ocr_raw = _to_dapt_ocr_raw(rows)
                metrics = _metrics(text, rows, args.anchors)
                record = {
                    "variant": variant.name,
                    "psm": psm,
                    "oem": args.oem,
                    "lang": args.lang,
                    "tessdata_dir": str(tessdata_dir),
                    "image_path": str(image_out),
                    "text_path": str(variant_dir / f"{run_name}.txt"),
                    "tsv_json_path": str(variant_dir / f"{run_name}.pred.json"),
                    "ocr_raw_path": str(variant_dir / f"{run_name}.ocr_raw.json"),
                    **metrics,
                }
                (variant_dir / f"{run_name}.txt").write_text(text, encoding="utf-8")
                (variant_dir / f"{run_name}.pred.json").write_text(
                    json.dumps({"rows": rows, **record}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (variant_dir / f"{run_name}.ocr_raw.json").write_text(
                    json.dumps(ocr_raw, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                summary_rows.append(record)
            except Exception as exc:
                summary_rows.append(
                    {
                        "variant": variant.name,
                        "psm": psm,
                        "oem": args.oem,
                        "lang": args.lang,
                        "tessdata_dir": str(tessdata_dir),
                        "image_path": str(image_out),
                        "error": str(exc),
                        "score": -999.0,
                    }
                )

    summary_rows.sort(key=lambda row: row.get("score", -999.0), reverse=True)
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "summary_path": str(summary_path),
            "top_runs": summary_rows[:5],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
