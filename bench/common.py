from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _escape_newlines_in_strings(text: str) -> str:
    """Escape literal newlines/carriage-returns that appear inside JSON strings.

    Valid JSON forbids unescaped control characters in strings (RFC 7159 §7).
    If a writer emitted them anyway, splitlines() breaks the record across
    multiple file lines and json.loads fails.  This pass fixes the raw text
    before parsing so the rest of read_jsonl can stay simple.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
        elif ch == '\\' and in_string:
            out.append(ch)
            escaped = True
        elif ch == '"':
            in_string = not in_string
            out.append(ch)
        elif in_string and ch == '\n':
            out.append('\\n')
        elif in_string and ch == '\r':
            out.append('\\r')
        else:
            out.append(ch)
    return ''.join(out)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes().decode('utf-8', errors='replace')
    fixed = _escape_newlines_in_strings(raw)
    rows: list[dict[str, Any]] = []
    for line in fixed.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
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
