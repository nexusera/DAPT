"""Small data-loading helpers used by KV-LLM training scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from torch.utils.data import Dataset


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]
    raise ValueError(f"Unsupported JSON payload in {path}")


def _record_text(item: dict[str, Any]) -> str | None:
    for key in ("text", "ocr_text", "content", "report_text"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def iter_texts_from_records(records: Iterable[dict[str, Any]]) -> Iterator[str]:
    for item in records:
        text = _record_text(item)
        if text:
            yield text


class TextFileDataset(Dataset):
    """Text dataset for CLM/span-corruption experiments.

    Supports `.txt`, `.json`, and `.jsonl`. JSON records are expected to contain
    one of: `text`, `ocr_text`, `content`, or `report_text`.
    """

    def __init__(self, path: str | Path, *, max_samples: int | None = None) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        self.records: list[dict[str, Any]] = []
        if path.is_dir():
            files = sorted(
                x for x in path.iterdir() if x.is_file() and x.suffix.lower() in {".txt", ".json", ".jsonl"}
            )
            if not files:
                raise ValueError(f"No supported text files found in directory: {path}")
            for file_path in files:
                self.records.extend(self._load_one(file_path))
        else:
            self.records.extend(self._load_one(path))
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]
        texts = [r["text"] for r in self.records]
        if not texts:
            raise ValueError(f"No usable text found in {path}")
        self.texts = texts

    @staticmethod
    def _load_one(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if path.suffix.lower() == ".txt":
            with path.open("r", encoding="utf-8") as f:
                rows.extend({"text": line.strip()} for line in f if line.strip())
            return rows
        for item in read_json_or_jsonl(path):
            text = _record_text(item)
            if not text:
                continue
            rec = {"text": text}
            if "noise_values" in item:
                rec["noise_values"] = item["noise_values"]
            rows.append(rec)
        return rows

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.records[idx]


def find_json_files(path: str | Path) -> Sequence[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.suffix.lower() in {".json", ".jsonl"}])
    raise FileNotFoundError(p)
