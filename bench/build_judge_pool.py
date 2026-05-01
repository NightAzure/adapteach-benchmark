"""
Build the deduplicated LLM judging pool from four sources.

Sources (in priority order for deduplication):
  1. silver_positive  — pairs scored >= min_score by the lexical scorer
  2. hard_negative    — topic-matching pairs scored < min_score (sampled n_hard_neg per query)
  3. retrieved        — all unique chunks retrieved by any config in the latest run file
  4. random_negative  — randomly sampled chunks with no connection to the query

Output: bench/judge_pool/pool.csv
Columns: query_id, query_text, dataset, chunk_id, doc_id, chunk_text, source, silver_score

Usage:
    python bench/build_judge_pool.py
    python bench/build_judge_pool.py --min-score 0.60 --n-hard-neg 5
    python bench/build_judge_pool.py --run-file bench/runs/run_20240501.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.build_silver_labels import (
    build_chunk_index,
    score_chunk,
    keywords,
    lexical_overlap_score,
    infer_dataset_name,
)
from bench.common import load_benchmark_spec, read_jsonl

MAX_CHUNK_CHARS = 400

# Priority order: lower index = higher priority.
SOURCE_PRIORITY = ["silver_positive", "hard_negative", "retrieved", "random_negative"]


# ---------------------------------------------------------------------------
# Run file helpers
# ---------------------------------------------------------------------------

def latest_run(runs_dir: Path) -> Path | None:
    files = sorted(runs_dir.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_retrieved_chunks(run_file: Path) -> dict[str, set[tuple[str, str]]]:
    """Return {query_id: set of (chunk_id, doc_id)} retrieved across all configs."""
    rows = read_jsonl(run_file, skip_invalid=True)
    result: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        qid = row.get("query_id", "")
        if not qid:
            continue
        context = row.get("response", {}).get("context", [])
        for item in context:
            chunk_id = item.get("chunk_id", "")
            doc_id = item.get("doc_id", "") or item.get("provenance", {}).get("doc_id", "")
            if chunk_id:
                result[qid].add((chunk_id, doc_id))
    return result


# ---------------------------------------------------------------------------
# Chunk content lookup (text chunks via fixed-split, AST via manifest)
# ---------------------------------------------------------------------------

CHUNK_SIZE = 450


def _hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _fixed_split_ids(content: str, doc_id: str) -> dict[str, str]:
    lookup: dict[str, str] = {}
    start = 0
    while start < len(content):
        end = min(start + CHUNK_SIZE, len(content))
        if end < len(content):
            boundary = content.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        piece = content[start:end].strip()
        if piece:
            chunk_id = "text-" + _hash(doc_id + str(start) + piece)
            lookup[chunk_id] = piece
        start = end
    return lookup


def build_chunk_content_lookup(corpus_dir: Path, manifest_path: Path) -> dict[str, str]:
    """Build {chunk_id: content} for text chunks and AST chunks."""
    lookup: dict[str, str] = {}
    for doc_path in sorted(corpus_dir.glob("*.json")):
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        doc_id = doc_path.stem
        lookup.update(_fixed_split_ids(doc.get("content", ""), doc_id))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for chunk in manifest.get("chunks", []):
            cid = chunk.get("chunk_id", "")
            text = chunk.get("content", "")
            if cid and text:
                lookup[cid] = text
    return lookup


def truncate(text: str, max_chars: int = MAX_CHUNK_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


# ---------------------------------------------------------------------------
# Pool building logic
# ---------------------------------------------------------------------------

def _passes_topic_filter(query_row: dict, chunk: dict) -> bool:
    """Return True if chunk passes the two-stage topic filter used by build_silver_labels."""
    topic = str(query_row.get("topic", "")).lower().strip()
    if not topic:
        return True
    if topic in chunk["concept_tags"]:
        return True
    return lexical_overlap_score(keywords(query_row["query"]), chunk["title_terms"]) >= 0.05


def build_pool_for_dataset(
    dataset_name: str,
    query_rows: list[dict],
    chunk_index: list[dict],
    content_lookup: dict[str, str],
    retrieved_by_query: dict[str, set[tuple[str, str]]],
    min_score: float,
    n_hard_neg: int,
    n_random_neg: int,
    rng: random.Random,
) -> list[dict]:
    """Build pool rows for one dataset. Returns list of pool row dicts."""

    # Pre-build index from chunk_id for fast lookup
    chunk_by_id: dict[str, dict] = {c["chunk_id"]: c for c in chunk_index}

    pool_rows: list[dict] = []

    for query_row in query_rows:
        qid = query_row["id"]
        query_text = query_row["query"]

        # Accumulators keyed by chunk_id, tracking (source, silver_score)
        # We collect candidates per source, then dedup by priority.
        candidates: dict[str, dict] = {}  # chunk_id -> pool row dict

        def _upsert(chunk_id: str, doc_id: str, source: str, silver_score: float) -> None:
            """Insert or upgrade a candidate using source priority."""
            content = content_lookup.get(chunk_id, "")
            if not content:
                return
            new_priority = SOURCE_PRIORITY.index(source)
            if chunk_id in candidates:
                existing_priority = SOURCE_PRIORITY.index(candidates[chunk_id]["source"])
                if new_priority >= existing_priority:
                    return  # existing source has equal or higher priority
            candidates[chunk_id] = {
                "query_id": qid,
                "query_text": query_text,
                "dataset": dataset_name,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_text": truncate(content),
                "source": source,
                "silver_score": silver_score,
            }

        # --- Source 1 & 2: score topic-filtered chunks ---
        positives: list[tuple[float, dict]] = []
        hard_negatives: list[tuple[float, dict]] = []

        for chunk in chunk_index:
            if not _passes_topic_filter(query_row, chunk):
                continue
            sc = score_chunk(query_row, chunk)
            if sc >= min_score:
                positives.append((sc, chunk))
            else:
                hard_negatives.append((sc, chunk))

        for sc, chunk in positives:
            _upsert(chunk["chunk_id"], chunk["doc_id"], "silver_positive", sc)

        # Sample hard negatives (highest scoring below threshold)
        hard_negatives.sort(key=lambda t: t[0], reverse=True)
        sampled_hard = hard_negatives[:n_hard_neg]
        for sc, chunk in sampled_hard:
            _upsert(chunk["chunk_id"], chunk["doc_id"], "hard_negative", sc)

        # --- Source 3: retrieved chunks ---
        for chunk_id, doc_id in retrieved_by_query.get(qid, set()):
            chunk = chunk_by_id.get(chunk_id)
            sc = score_chunk(query_row, chunk) if chunk else 0.0
            _upsert(chunk_id, doc_id, "retrieved", sc)

        # --- Source 4: random negatives ---
        # Exclude chunk_ids already in candidates
        excluded = set(candidates.keys())
        eligible = [c for c in chunk_index if c["chunk_id"] not in excluded]
        n_sample = min(n_random_neg, len(eligible))
        for chunk in rng.sample(eligible, n_sample):
            sc = score_chunk(query_row, chunk)
            _upsert(chunk["chunk_id"], chunk["doc_id"], "random_negative", sc)

        pool_rows.extend(candidates.values())

    return pool_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deduplicated LLM judging pool from four sources."
    )
    parser.add_argument(
        "--benchmark",
        default="bench/benchmarks/obj1_full.yaml",
        help="YAML benchmark spec (default: bench/benchmarks/obj1_full.yaml)",
    )
    parser.add_argument(
        "--run-file",
        default="",
        help="Run JSONL to source retrieved chunks from (default: latest run_*.jsonl in bench/runs/)",
    )
    parser.add_argument(
        "--chunk-manifest",
        default="data/corpus_meta/chunk_manifest.json",
        help="Path to chunk_manifest.json (default: data/corpus_meta/chunk_manifest.json)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.55,
        help="Minimum silver score for silver_positive source (default: 0.55)",
    )
    parser.add_argument(
        "--n-hard-neg",
        type=int,
        default=3,
        help="Hard negatives sampled per query (default: 3)",
    )
    parser.add_argument(
        "--n-random-neg",
        type=int,
        default=2,
        help="Random negatives sampled per query (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--out",
        default="bench/judge_pool/pool.csv",
        help="Output pool CSV path (default: bench/judge_pool/pool.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    manifest_path = ROOT / args.chunk_manifest
    corpus_dir = ROOT / "data" / "corpus"
    runs_dir = ROOT / "bench" / "runs"
    out_path = ROOT / args.out

    # Resolve run file
    if args.run_file:
        run_file: Path | None = ROOT / args.run_file
        if not run_file.exists():
            raise SystemExit(f"ERROR: Run file not found: {run_file}")
    else:
        run_file = latest_run(runs_dir)
        if run_file is None:
            print("Warning: No run_*.jsonl found in bench/runs/ — retrieved source will be empty.")

    print("Loading chunk index from manifest...")
    chunk_index = build_chunk_index(manifest_path)
    print(f"  {len(chunk_index):,} chunks indexed")

    print("Building chunk content lookup...")
    content_lookup = build_chunk_content_lookup(corpus_dir, manifest_path)
    print(f"  {len(content_lookup):,} chunk contents loaded")

    print("Loading run file for retrieved chunks...")
    if run_file:
        print(f"  Using: {run_file.name}")
        retrieved_by_query = load_retrieved_chunks(run_file)
        n_queries_with_retrieval = sum(1 for v in retrieved_by_query.values() if v)
        total_retrieved = sum(len(v) for v in retrieved_by_query.values())
        print(f"  {n_queries_with_retrieval} queries have retrieved chunks ({total_retrieved} total chunk-query pairs)")
    else:
        retrieved_by_query = {}

    # Load benchmark spec for datasets
    spec = load_benchmark_spec(ROOT / args.benchmark)
    datasets = spec.get("datasets", [])
    if not datasets:
        raise SystemExit(f"ERROR: No datasets found in benchmark spec: {args.benchmark}")

    all_pool_rows: list[dict] = []

    for ds in datasets:
        query_file = ROOT / ds["query_file"]
        dataset_name = ds.get("name") or infer_dataset_name(query_file)

        if not query_file.exists():
            print(f"  WARNING: Query file not found, skipping dataset '{dataset_name}': {query_file}")
            continue

        query_rows = read_jsonl(query_file)
        print(f"\nDataset '{dataset_name}': {len(query_rows)} queries...")

        pool_rows = build_pool_for_dataset(
            dataset_name=dataset_name,
            query_rows=query_rows,
            chunk_index=chunk_index,
            content_lookup=content_lookup,
            retrieved_by_query=retrieved_by_query,
            min_score=args.min_score,
            n_hard_neg=args.n_hard_neg,
            n_random_neg=args.n_random_neg,
            rng=rng,
        )
        print(f"  {len(pool_rows):,} pool rows generated")
        all_pool_rows.extend(pool_rows)

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["query_id", "query_text", "dataset", "chunk_id", "doc_id", "chunk_text", "source", "silver_score"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_pool_rows)

    # Summary breakdown by source
    source_counts: dict[str, int] = defaultdict(int)
    for row in all_pool_rows:
        source_counts[row["source"]] += 1

    print(f"\nPool written to: {out_path}")
    print(f"Total pool size: {len(all_pool_rows):,} rows")
    print("\nBreakdown by source:")
    for source in SOURCE_PRIORITY:
        count = source_counts.get(source, 0)
        print(f"  {source:<20} {count:>6,}")

    print("\nNext step: python bench/run_llm_judge.py --pool-file bench/judge_pool/pool.csv")


if __name__ == "__main__":
    main()
