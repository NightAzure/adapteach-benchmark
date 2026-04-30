from __future__ import annotations

import argparse
import sys
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.common import load_benchmark_spec, read_jsonl

TOKEN_RE = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'to', 'of', 'in', 'for', 'on', 'with', 'is', 'are', 'as', 'by', 'from', 'that',
    'this', 'it', 'be', 'at', 'not', 'do', 'if', 'we', 'you', 'can', 'has', 'have', 'will', 'was', 'but', 'all',
    'so', 'no', 'when', 'what', 'how', 'which', 'their', 'each', 'than', 'its', 'also', 'into', 'just', 'about',
    'would', 'should', 'could', 'then', 'these', 'those', 'them', 'they', 'been', 'were', 'being', 'had', 'did',
    'does', 'may', 'might', 'must', 'shall', 'our', 'my', 'your', 'his', 'her', 'who', 'one', 'two', 'short',
    'python', 'snippet', 'code', 'example', 'examples', 'learner', 'beginner', 'show', 'explain', 'write', 'function',
}
DEBUG_TERMS = {'bug', 'fix', 'error', 'traceback', 'exception', 'wrong', 'incorrect', 'debug'}
MISCONCEPTION_TERMS = {'misconception', 'myth', 'incorrect', 'wrong', 'belief'}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def keywords(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2}


def lexical_overlap_score(query_terms: set[str], text_terms: set[str]) -> float:
    if not query_terms or not text_terms:
        return 0.0
    inter = len(query_terms & text_terms)
    if inter == 0:
        return 0.0
    precision = inter / len(query_terms)
    recall = inter / len(text_terms)
    return (2 * precision * recall) / max(1e-9, precision + recall)


def category_bonus(category: str, text_terms: set[str]) -> float:
    if category == 'debugging':
        return 0.20 if DEBUG_TERMS & text_terms else 0.0
    if category == 'misconception':
        return 0.15 if MISCONCEPTION_TERMS & text_terms else 0.0
    return 0.0


def chunk_length_bonus(content: str) -> float:
    length = len(tokenize(content))
    if 40 <= length <= 250:
        return 0.10
    if 20 <= length < 40 or 250 < length <= 400:
        return 0.05
    return 0.0


def chunk_type_bonus(chunker: str) -> float:
    if chunker.startswith('ast'):
        return 0.04
    return 0.0


def has_code_bonus(content: str, query: str) -> float:
    if ('`' in query or 'snippet' in query.lower() or 'code' in query.lower()) and '```' in content:
        return 0.08
    return 0.0


def build_chunk_index(chunk_manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(chunk_manifest_path.read_text(encoding='utf-8'))
    index: list[dict[str, Any]] = []
    for chunk in manifest.get('chunks', []):
        content = chunk.get('content', '')
        title = chunk.get('metadata', {}).get('title', '')
        concept_tags = {str(t).lower() for t in chunk.get('concept_tags', [])}
        text_terms = keywords(content)
        title_terms = keywords(title)
        index.append({
            'chunk_id': chunk['chunk_id'],
            'doc_id': chunk.get('doc_id', ''),
            'chunker': chunk.get('chunker', ''),
            'concept_tags': concept_tags,
            'content': content,
            'text_terms': text_terms,
            'title_terms': title_terms,
            'title': title,
        })
    return index


def score_chunk(query_row: dict[str, Any], chunk: dict[str, Any]) -> float:
    query_text = query_row['query']
    q_terms = keywords(query_text)
    topic = str(query_row.get('topic', '')).lower().strip()
    category = str(query_row.get('category', '')).lower().strip()
    topic_match = 1.0 if topic and topic in chunk['concept_tags'] else 0.0
    text_overlap = lexical_overlap_score(q_terms, chunk['text_terms'])
    title_overlap = lexical_overlap_score(q_terms, chunk['title_terms'])
    score = 0.0
    score += 0.55 * text_overlap
    score += 0.25 * title_overlap
    score += 0.35 * topic_match
    score += category_bonus(category, chunk['text_terms'])
    score += chunk_length_bonus(chunk['content'])
    score += chunk_type_bonus(chunk['chunker'])
    score += has_code_bonus(chunk['content'], query_text)
    return round(score, 8)


def label_query(query_row: dict[str, Any], chunks: list[dict[str, Any]], high_k: int, mid_k: int, min_score: float) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    topic = str(query_row.get('topic', '')).lower().strip()
    # Two-stage filtering keeps runtime low and avoids obviously irrelevant chunks.
    for chunk in chunks:
        if topic and topic not in chunk['concept_tags']:
            # allow fallback only if title or lexical overlap is non-trivial
            if lexical_overlap_score(keywords(query_row['query']), chunk['title_terms']) < 0.05:
                continue
        score = score_chunk(query_row, chunk)
        if score >= min_score:
            scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)

    rows: list[dict[str, Any]] = []
    for rank, (score, chunk) in enumerate(scored[: high_k + mid_k], start=1):
        relevance = 2 if rank <= high_k else 1
        rows.append({
            'query_id': query_row['id'],
            'query': query_row['query'],
            'chunk_id': chunk['chunk_id'],
            'doc_id': chunk['doc_id'],
            'relevance': relevance,
            'silver_score': score,
            'label_source': 'offline_topic_lexical_judge_v1',
            'notes': f"topic={query_row.get('topic', '')}; title={chunk['title'][:80]}",
        })
    return rows


def infer_dataset_name(path: Path) -> str:
    stem = path.stem.lower()
    if stem.startswith('queries_'):
        return stem.replace('queries_', '')
    return stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build deterministic silver relevance labels for Obj1 retrieval benchmarking.')
    parser.add_argument('--benchmark', default='', help='Optional YAML benchmark spec listing datasets.')
    parser.add_argument('--query-files', nargs='*', default=[])
    parser.add_argument('--chunk-manifest', default='data/corpus_meta/chunk_manifest.json')
    parser.add_argument('--labels-dir', default='bench/labels')
    parser.add_argument('--high-k', type=int, default=2)
    parser.add_argument('--mid-k', type=int, default=4)
    parser.add_argument('--min-score', type=float, default=0.20)
    parser.add_argument('--combined-out', default='')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_index = build_chunk_index(ROOT / args.chunk_manifest)
    dataset_paths: list[Path] = []
    if args.benchmark:
        spec = load_benchmark_spec(ROOT / args.benchmark)
        for item in spec.get('datasets', []):
            dataset_paths.append(ROOT / item['query_file'])
    for value in args.query_files:
        dataset_paths.append(ROOT / value)
    if not dataset_paths:
        raise SystemExit('No query files supplied. Use --benchmark or --query-files.')

    labels_dir = ROOT / args.labels_dir
    labels_dir.mkdir(parents=True, exist_ok=True)
    combined_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {'datasets': []}

    for dataset_path in dataset_paths:
        query_rows = read_jsonl(dataset_path)
        label_rows: list[dict[str, Any]] = []
        for row in query_rows:
            label_rows.extend(label_query(row, chunk_index, args.high_k, args.mid_k, args.min_score))
        dataset_name = infer_dataset_name(dataset_path)
        out_path = labels_dir / f'silver_labels_{dataset_name}.csv'
        with out_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['query_id', 'query', 'chunk_id', 'doc_id', 'relevance', 'silver_score', 'label_source', 'notes'])
            writer.writeheader()
            writer.writerows(label_rows)
        combined_rows.extend(label_rows)
        summary['datasets'].append({'dataset': dataset_name, 'queries': len(query_rows), 'labels': len(label_rows), 'out': out_path.as_posix()})

    if args.combined_out:
        combined_path = ROOT / args.combined_out
        with combined_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['query_id', 'query', 'chunk_id', 'doc_id', 'relevance', 'silver_score', 'label_source', 'notes'])
            writer.writeheader()
            writer.writerows(combined_rows)
        summary['combined_out'] = combined_path.as_posix()

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
