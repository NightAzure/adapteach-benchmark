"""
auto_label.py — Automated relevance labeling for Objective 1 benchmark.

Scores each (query, chunk) pair in a labeling sheet CSV using cosine similarity
from sentence-transformers/all-MiniLM-L6-v2 (same model used in the pipeline).

Relevance scale:
  2 — Highly relevant: similarity >= HIGH_THRESHOLD
  1 — Partially relevant: similarity >= LOW_THRESHOLD
  0 — Not relevant: similarity < LOW_THRESHOLD

Thresholds are calibrated for MiniLM cosine similarity:
  HIGH_THRESHOLD = 0.50  (directly addresses the query concept)
  LOW_THRESHOLD  = 0.32  (tangentially related but has useful content)

Usage:
    py bench/auto_label.py \
        --sheet bench/labels/labeling_sheet.csv \
        --queries bench/queries_obj1_test.jsonl \
        --out bench/labels/labels.csv
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

HIGH_THRESHOLD = 0.50
LOW_THRESHOLD = 0.32


def _load_queries(path: Path) -> dict[str, dict[str, str]]:
    queries: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            queries[row["id"]] = row
    return queries


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _cosine(a: Any, b: Any) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _score(sim: float) -> int:
    if sim >= HIGH_THRESHOLD:
        return 2
    if sim >= LOW_THRESHOLD:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-label relevance using MiniLM cosine similarity")
    p.add_argument("--sheet", default="bench/labels/labeling_sheet.csv")
    p.add_argument("--queries", default="bench/queries_obj1_test.jsonl")
    p.add_argument("--out", default="bench/labels/labels.csv")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--high-threshold", type=float, default=HIGH_THRESHOLD)
    p.add_argument("--low-threshold", type=float, default=LOW_THRESHOLD)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sheet_path = Path(args.sheet)
    queries_path = Path(args.queries)
    out_path = Path(args.out)

    if not sheet_path.exists():
        raise SystemExit(f"Sheet not found: {sheet_path}")
    if not queries_path.exists():
        raise SystemExit(f"Queries not found: {queries_path}")

    print(f"Loading model: {args.model}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    queries = _load_queries(queries_path)
    sheet_rows = _read_csv(sheet_path)

    # Collect all unique texts to embed in one batch pass (fast)
    query_texts: dict[str, str] = {}
    chunk_texts: dict[str, str] = {}  # chunk_id -> preview text

    for row in sheet_rows:
        qid = row["query_id"]
        cid = row["chunk_id"]
        # Use full query text from queries file if available, else fall back to sheet column
        q_text = queries.get(qid, {}).get("query", row.get("query", ""))
        query_texts[qid] = q_text
        chunk_texts[cid] = row.get("chunk_preview", "")

    unique_qids = list(query_texts.keys())
    unique_cids = list(chunk_texts.keys())

    print(f"Embedding {len(unique_qids)} queries + {len(unique_cids)} chunks...")
    q_embeddings = model.encode([query_texts[qid] for qid in unique_qids], batch_size=32, show_progress_bar=True)
    c_embeddings = model.encode([chunk_texts[cid] for cid in unique_cids], batch_size=32, show_progress_bar=True)

    q_emb_map = {qid: q_embeddings[i] for i, qid in enumerate(unique_qids)}
    c_emb_map = {cid: c_embeddings[i] for i, cid in enumerate(unique_cids)}

    # Score every row
    labeled: list[dict[str, Any]] = []
    score_dist = {0: 0, 1: 0, 2: 0}

    for row in sheet_rows:
        qid = row["query_id"]
        cid = row["chunk_id"]
        if qid not in q_emb_map or cid not in c_emb_map:
            sim, rel = 0.0, 0
        else:
            sim = _cosine(q_emb_map[qid], c_emb_map[cid])
            rel = _score(sim)
        score_dist[rel] += 1
        labeled.append({
            "query_id": qid,
            "query": query_texts.get(qid, row.get("query", "")),
            "chunk_id": cid,
            "chunk_title": row.get("chunk_title", ""),
            "chunk_preview": row.get("chunk_preview", ""),
            "seen_in_configs": row.get("seen_in_configs", ""),
            "best_retrieval_score": row.get("best_retrieval_score", ""),
            "cosine_similarity": f"{sim:.4f}",
            "relevance": rel,
            "notes": f"auto:minilm sim={sim:.4f}",
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["query_id", "query", "chunk_id", "chunk_title", "chunk_preview",
                  "seen_in_configs", "best_retrieval_score", "cosine_similarity", "relevance", "notes"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeled)

    total = len(labeled)
    print(json.dumps({
        "total_rows": total,
        "score_0_not_relevant": score_dist[0],
        "score_1_partial": score_dist[1],
        "score_2_highly_relevant": score_dist[2],
        "pct_relevant": round((score_dist[1] + score_dist[2]) / total * 100, 1) if total else 0,
        "thresholds": {"high": args.high_threshold, "low": args.low_threshold},
        "out": str(out_path),
    }, indent=2))


if __name__ == "__main__":
    main()
