from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import yaml


def read_jsonl(path: Path, *, skip_invalid: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_bytes().decode('utf-8').splitlines(), start=1):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                preview = line[:160]
                message = (
                    f"Invalid JSONL in {path} at line {line_no}, "
                    f"column {exc.colno}: {exc.msg}. "
                    f"Line preview: {preview!r}"
                )
                if not skip_invalid:
                    raise ValueError(message) from exc
                warnings.warn(f"{message}; skipping line", RuntimeWarning)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def load_benchmark_spec(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Benchmark spec must be a mapping: {path}')
    return data


def normalize_dataset_name(name: str) -> str:
    return name.strip().lower().replace(' ', '_')
