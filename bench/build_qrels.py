"""
Merge LLM judge scores with the judging pool to produce qrels CSVs.

Reads:
  - bench/judge_pool/pool.csv         — the full judging pool
  - bench/llm_judge/scores/judge_scores_batch_*.csv  — LLM scores (row_id,llm_score)

Outputs one file per dataset:
  bench/labels/qrels_<dataset>.csv

Schema (matches silver_labels_*.csv):
  query_id, query, chunk_id, doc_id, relevance, silver_score, label_source, notes

Only pairs with relevance > 0 are written (zeros are implicit negatives).

Usage:
    python bench/build_qrels.py
    python bench/build_qrels.py --pool-file bench/judge_pool/pool.csv
    python bench/build_qrels.py --scores-dir bench/llm_judge/scores --labels-dir bench/labels
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABEL_SOURCE = "llm_judge_qrels_v1"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pool(pool_path: Path) -> list[dict]:
    """Load pool.csv into a list of dicts."""
    if not pool_path.exists():
        raise SystemExit(f"ERROR: Pool file not found: {pool_path}\n  Run bench/build_judge_pool.py first.")
    rows: list[dict] = []
    with pool_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_scores(scores_dir: Path) -> dict[str, int]:
    """Load all judge_scores_batch_*.csv files and return {row_id: llm_score}."""
    scores: dict[str, int] = {}
    score_files = sorted(scores_dir.glob("judge_scores_batch_*.csv"))
    if not score_files:
        print(f"Warning: No score files found in {scores_dir}")
        return scores

    for score_path in score_files:
        with score_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("row_id"):
                    continue
                parts = line.split(",", 1)
                if len(parts) < 2:
                    continue
                row_id = parts[0].strip()
                try:
                    score = int(parts[1].strip())
                    if score in (0, 1, 2):
                        # Last write wins if the same row_id appears in multiple batches
                        scores[row_id] = score
                except ValueError:
                    continue

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LLM judge scores with the judging pool to produce qrels CSVs."
    )
    parser.add_argument(
        "--pool-file",
        default="bench/judge_pool/pool.csv",
        help="Path to pool.csv (default: bench/judge_pool/pool.csv)",
    )
    parser.add_argument(
        "--scores-dir",
        default="bench/llm_judge/scores",
        help="Directory containing judge_scores_batch_*.csv (default: bench/llm_judge/scores)",
    )
    parser.add_argument(
        "--labels-dir",
        default="bench/labels",
        help="Output directory for qrels_<dataset>.csv files (default: bench/labels)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pool_path = ROOT / args.pool_file
    scores_dir = ROOT / args.scores_dir
    labels_dir = ROOT / args.labels_dir

    print(f"Loading pool from: {pool_path}")
    pool = load_pool(pool_path)
    print(f"  {len(pool):,} pool rows loaded")

    print(f"Loading LLM scores from: {scores_dir}")
    scores = load_scores(scores_dir)
    print(f"  {len(scores):,} scored pairs loaded")

    # Group pool rows by dataset
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        by_dataset[row["dataset"]].append(row)

    labels_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["query_id", "query", "chunk_id", "doc_id", "relevance", "silver_score", "label_source", "notes"]

    total_pairs = 0
    total_positive = 0

    dataset_summary: list[dict] = []

    for dataset in sorted(by_dataset.keys()):
        pool_rows = by_dataset[dataset]
        qrels_rows: list[dict] = []

        for pr in pool_rows:
            query_id = pr["query_id"]
            chunk_id = pr["chunk_id"]
            row_id = f"{query_id}__{chunk_id}"
            relevance = scores.get(row_id, 0)

            # Only include pairs with relevance > 0 (zeros are implicit negatives)
            if relevance <= 0:
                continue

            qrels_rows.append({
                "query_id": query_id,
                "query": pr["query_text"],
                "chunk_id": chunk_id,
                "doc_id": pr["doc_id"],
                "relevance": relevance,
                "silver_score": pr.get("silver_score", ""),
                "label_source": LABEL_SOURCE,
                "notes": pr.get("source", ""),
            })

        out_path = labels_dir / f"qrels_{dataset}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(qrels_rows)

        n_pool = len(pool_rows)
        n_pos = len(qrels_rows)
        total_pairs += n_pool
        total_positive += n_pos

        dataset_summary.append({
            "dataset": dataset,
            "pool_rows": n_pool,
            "positive_pairs": n_pos,
            "out": out_path.name,
        })

        print(f"  [{dataset}] {n_pool:,} pool rows → {n_pos:,} positive qrel pairs → {out_path.name}")

    # Print summary
    print(f"\nTotal pool pairs:            {total_pairs:,}")
    print(f"Pairs with LLM score loaded: {len(scores):,}")
    print(f"Pairs with relevance > 0:    {total_positive:,}")
    print(f"\nDataset breakdown:")
    for ds in dataset_summary:
        print(f"  {ds['dataset']:<16}  {ds['pool_rows']:>5} pool  →  {ds['positive_pairs']:>4} positive  →  {ds['out']}")

    print(f"\nQrels written to: {labels_dir}")
    print("\nNext step: python bench/score_obj1.py --labels-dir bench/labels")


if __name__ == "__main__":
    main()
