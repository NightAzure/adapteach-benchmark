from __future__ import annotations

import argparse
import sys
import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BENCH_DIR = ROOT / 'bench'
MANIFEST_PATH = BENCH_DIR / 'datasets_manifest.lock.json'

from bench.common import read_jsonl, sha256_file, write_jsonl

CORE_TOPICS = ['variables', 'loops', 'conditionals', 'functions']
TOPIC_KEYWORDS: dict[str, list[str]] = {
    'variables': ['variable', 'variables', 'assign', 'assignment', 'scope', 'name', 'type', 'mutable', 'immutable'],
    'loops': ['loop', 'loops', 'for ', 'while ', 'iterate', 'iteration', 'range', 'break', 'continue', 'nested loop'],
    'conditionals': ['if ', 'elif', 'else', 'condition', 'conditional', 'boolean', 'compare', 'comparison', 'truthy', 'falsy'],
    'functions': ['function', 'functions', 'def ', 'return', 'parameter', 'argument', 'call', 'lambda', 'recursion', 'method'],
}
QUESTION_OPENERS = re.compile(r'^(how|what|why|when|where|which|can|could|should|do|does|did|is|are|was|were|will|would)\b', re.IGNORECASE)
NOISE_RE = re.compile(
    r'(\bhubo\b|\btask\s*\d|\blab\s*\d|\bhw\s*\d|\bassignment\b|\bpiazza\b|\bcanvas\b|\bgrader\b|\bsubmission\b|\bprofessor\b|\bta\b)',
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')

DATASET_SOURCES: dict[str, dict[str, Any]] = {
    'custom': {
        'path': BENCH_DIR / 'queries_custom.jsonl',
        'origin': 'frozen_local',
        'refreshable': False,
    },
    'cs1qa': {
        'path': BENCH_DIR / 'queries_cs1qa.jsonl',
        'origin': 'official_github',
        'refreshable': True,
        'urls': [
            'https://raw.githubusercontent.com/cyoon47/CS1QA/main/data/final/cleaned/equal/train_cleaned.jsonl',
            'https://raw.githubusercontent.com/cyoon47/CS1QA/main/data/final/cleaned/equal/dev_cleaned.jsonl',
            'https://raw.githubusercontent.com/cyoon47/CS1QA/main/data/final/cleaned/equal/test_cleaned.jsonl',
        ],
    },
    'mbpp': {
        'path': BENCH_DIR / 'queries_mbpp.jsonl',
        'origin': 'official_github',
        'refreshable': True,
        'urls': [
            'https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl',
        ],
    },
    'staqc': {
        'path': BENCH_DIR / 'queries_staqc.jsonl',
        'origin': 'frozen_local',
        'refreshable': False,
        'note': 'The repo ships a frozen StaQC query extract. Refresh is intentionally disabled because upstream formats vary and often break reproducibility.',
    },
}


def stable_id(prefix: str, text: str) -> str:
    return f"{prefix}-{hashlib.md5(text.encode('utf-8')).hexdigest()[:10]}"


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={'User-Agent': 'AdapTeachBenchmark/2.0'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode('utf-8')


def classify_topic(text: str) -> str | None:
    lower = text.lower()
    scores: dict[str, int] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score:
            scores[topic] = score
    if not scores:
        return None
    return max(scores, key=scores.get)


def classify_category(text: str, raw: str | None = None) -> str:
    raw_lower = (raw or '').lower()
    lower = text.lower()
    if any(key in raw_lower for key in ('error', 'logical', 'fix', 'bug')) or any(key in lower for key in ('error', 'bug', 'fix', 'traceback', 'exception')):
        return 'debugging'
    if any(key in raw_lower for key in ('misconception', 'wrong')) or 'misconception' in lower:
        return 'misconception'
    return 'concept_explanation'


def quality_query(text: str, require_question_opener: bool = False) -> bool:
    if len(text.strip()) < 20:
        return False
    if NOISE_RE.search(text):
        return False
    if require_question_opener and not QUESTION_OPENERS.match(text.strip()):
        return False
    return classify_topic(text) is not None


def refresh_cs1qa(max_items: int = 437) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for url in DATASET_SOURCES['cs1qa']['urls']:
        raw = fetch_text(url)
        for line in raw.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            text = (item.get('question') or '').strip()
            if not quality_query(text, require_question_opener=True):
                continue
            topic = classify_topic(text)
            if topic is None:
                continue
            qid = stable_id('cs1qa', text)
            if qid in seen:
                continue
            seen.add(qid)
            rows.append({
                'id': qid,
                'category': classify_category(text, str(item.get('questionType', ''))),
                'topic': topic,
                'query': text,
            })
            if len(rows) >= max_items:
                return rows
    return rows


def refresh_mbpp(max_items: int = 500) -> list[dict[str, Any]]:
    raw = fetch_text(DATASET_SOURCES['mbpp']['urls'][0])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        text = (item.get('text') or '').strip()
        if len(text) < 10:
            continue
        topic = classify_topic(text) or 'functions'
        qtext = f'Write a Python function to: {text}'
        qid = stable_id('mbpp', qtext)
        if qid in seen:
            continue
        seen.add(qid)
        rows.append({'id': qid, 'category': 'concept_explanation', 'topic': topic, 'query': qtext})
        if len(rows) >= max_items:
            break
    return rows


def build_manifest(selected: list[str]) -> dict[str, Any]:
    datasets = []
    for name in selected:
        meta = DATASET_SOURCES[name]
        path = Path(meta['path'])
        rows = read_jsonl(path)
        datasets.append({
            'name': name,
            'path': path.as_posix(),
            'origin': meta.get('origin', ''),
            'refreshable': bool(meta.get('refreshable', False)),
            'count': len(rows),
            'sha256': sha256_file(path),
            'sample_ids': [row['id'] for row in rows[:3]],
            'sample_topics': sorted({row.get('topic', '') for row in rows[:10]}),
            'urls': meta.get('urls', []),
            'note': meta.get('note', ''),
        })
    return {'datasets': datasets}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate or refresh benchmark query datasets.')
    parser.add_argument('--mode', choices=['validate', 'refresh'], default='validate')
    parser.add_argument('--sources', nargs='+', default=['custom', 'cs1qa', 'mbpp', 'staqc'])
    parser.add_argument('--manifest-out', default=str(MANIFEST_PATH))
    parser.add_argument('--backup-dir', default='bench/legacy/frozen_queries')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = []
    for name in args.sources:
        if name not in DATASET_SOURCES:
            raise SystemExit(f'Unknown dataset source: {name}')
        selected.append(name)

    if args.mode == 'refresh':
        backup_dir = ROOT / args.backup_dir
        backup_dir.mkdir(parents=True, exist_ok=True)
        for name in selected:
            meta = DATASET_SOURCES[name]
            path = Path(meta['path'])
            if path.exists():
                shutil.copy2(path, backup_dir / path.name)
            if not meta.get('refreshable', False):
                print(f'[skip] {name}: refresh disabled; using shipped frozen file.')
                continue
            if name == 'cs1qa':
                rows = refresh_cs1qa()
            elif name == 'mbpp':
                rows = refresh_mbpp()
            else:
                raise SystemExit(f'Refresh is not implemented for {name}.')
            write_jsonl(path, rows)
            print(f'[ok] refreshed {name}: {len(rows)} rows -> {path.as_posix()}')

    manifest = build_manifest(selected)
    manifest_path = Path(args.manifest_out)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
