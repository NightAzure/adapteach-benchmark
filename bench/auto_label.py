"""
auto_label.py — Automated relevance labeling for Objective 1 benchmark.

Two modes:

  --gemini   Use Gemini 2.5 Flash as an LLM judge (more accurate, recommended).
             Batches multiple (query, chunk) pairs per API call to stay within
             free-tier limits (10 RPM / 250 RPD). Resumable — already-labeled
             rows in --out are skipped automatically.

  (default)  Use sentence-transformers/all-MiniLM-L6-v2 cosine similarity.
             Fast, offline, no API quota.

Relevance scale (both modes):
  2 — Highly relevant: directly addresses the query concept
  1 — Partially relevant: somewhat related but does not directly answer
  0 — Not relevant: off-topic or unrelated

Usage:
    # Gemini mode (recommended)
    py bench/auto_label.py \\
        --gemini \\
        --sheet bench/labels/sheet_custom.csv \\
        --queries bench/queries_custom.jsonl \\
        --out bench/labels/labels_custom.csv \\
        --api-key $GEMINI_API_KEY \\
        --batch-size 20 \\
        --delay 7

    # MiniLM mode (offline fallback)
    py bench/auto_label.py \\
        --sheet bench/labels/labeling_sheet.csv \\
        --queries bench/queries_obj1_test.jsonl \\
        --out bench/labels/labels.csv
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

HIGH_THRESHOLD = 0.50
LOW_THRESHOLD = 0.32

GEMINI_PROMPT_TEMPLATE = """\
You are a relevance judge for a Python programming tutoring system.

For each numbered (Query, Chunk) pair, rate how relevant the chunk is for \
answering the student's question.

Scale:
  2 = Highly relevant — directly addresses the concept in the query
  1 = Partially relevant — somewhat related but does not directly answer
  0 = Not relevant — off-topic or unrelated

Return ONLY a JSON array of integers, one per pair, in the same order.
Example for 3 pairs: [2, 0, 1]
No explanation. No markdown. Just the array.

=== Pairs ===
{pairs_block}"""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_existing_labels(out_path: Path) -> set[tuple[str, str]]:
    """Return set of (query_id, chunk_id) already labeled in the output file."""
    if not out_path.exists():
        return set()
    try:
        rows = _read_csv(out_path)
        return {(r["query_id"], r["chunk_id"]) for r in rows if r.get("query_id") and r.get("chunk_id")}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Gemini mode
# ---------------------------------------------------------------------------

def _parse_gemini_ratings(text: str, expected: int) -> list[int]:
    """Extract a JSON integer array from Gemini response. Falls back to 0 on failure."""
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"```[a-z]*\n?", "", text).strip()
    try:
        ratings = json.loads(text)
        if isinstance(ratings, list) and len(ratings) == expected:
            return [max(0, min(2, int(r))) for r in ratings]
    except Exception:
        pass
    # Try to extract array substring
    m = re.search(r"\[[\d,\s]+\]", text)
    if m:
        try:
            ratings = json.loads(m.group())
            if isinstance(ratings, list) and len(ratings) == expected:
                return [max(0, min(2, int(r))) for r in ratings]
        except Exception:
            pass
    print(f"  [warn] Could not parse Gemini response (expected {expected} ratings): {text[:120]}")
    return [0] * expected


def _gemini_label_batch(client: Any, pairs: list[tuple[str, str, str, str]]) -> list[int]:
    """
    Call Gemini once for a batch of (query_id, chunk_id, query_text, chunk_preview) pairs.
    Returns a list of int ratings (0/1/2), one per pair.
    """
    lines = []
    for i, (_, _, q_text, chunk_preview) in enumerate(pairs, start=1):
        lines.append(f"[{i}] Query: {q_text.strip()}")
        lines.append(f"     Chunk: {chunk_preview.strip()[:350]}")
    pairs_block = "\n".join(lines)
    prompt = GEMINI_PROMPT_TEMPLATE.format(pairs_block=pairs_block)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return _parse_gemini_ratings(response.text, len(pairs))
    except Exception as e:
        print(f"  [error] Gemini API call failed: {e}")
        return [0] * len(pairs)


def label_with_gemini(
    sheet_rows: list[dict[str, Any]],
    query_texts: dict[str, str],
    out_path: Path,
    api_key: str,
    batch_size: int,
    delay: float,
) -> list[dict[str, Any]]:
    try:
        from google import genai
    except ImportError:
        raise SystemExit("google-genai not installed. Run: pip install google-genai")

    client = genai.Client(api_key=api_key)

    # Resume: skip already-labeled pairs
    done = _load_existing_labels(out_path)
    if done:
        print(f"Resuming — {len(done)} pairs already labeled, skipping.")

    # Filter to unlabeled rows
    pending = [
        r for r in sheet_rows
        if (r["query_id"], r["chunk_id"]) not in done
    ]
    total_pending = len(pending)
    total_batches = (total_pending + batch_size - 1) // batch_size
    print(f"Pending: {total_pending} pairs → {total_batches} batches (size={batch_size})")
    estimated_calls = total_batches
    estimated_min = round(estimated_calls * delay / 60, 1)
    print(f"Estimated API calls: {estimated_calls}  |  Est. time: {estimated_min} min at {delay}s delay")

    fieldnames = ["query_id", "query", "chunk_id", "chunk_title", "chunk_preview",
                  "seen_in_configs", "best_retrieval_score", "cosine_similarity", "relevance", "notes"]

    # Open output in append mode so resume works
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_path.exists() or out_path.stat().st_size == 0
    f_out = out_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if is_new:
        writer.writeheader()

    score_dist = {0: 0, 1: 0, 2: 0}
    newly_labeled: list[dict[str, Any]] = []

    for batch_i in range(0, total_pending, batch_size):
        batch = pending[batch_i: batch_i + batch_size]
        pairs = [
            (r["query_id"], r["chunk_id"],
             query_texts.get(r["query_id"], r.get("query", "")),
             r.get("chunk_preview", ""))
            for r in batch
        ]

        ratings = _gemini_label_batch(client, pairs)

        for row, rating in zip(batch, ratings):
            score_dist[rating] += 1
            out_row = {
                "query_id": row["query_id"],
                "query": query_texts.get(row["query_id"], row.get("query", "")),
                "chunk_id": row["chunk_id"],
                "chunk_title": row.get("chunk_title", ""),
                "chunk_preview": row.get("chunk_preview", ""),
                "seen_in_configs": row.get("seen_in_configs", ""),
                "best_retrieval_score": row.get("best_retrieval_score", ""),
                "cosine_similarity": "",
                "relevance": rating,
                "notes": "auto:gemini-2.5-flash",
            }
            writer.writerow(out_row)
            newly_labeled.append(out_row)

        f_out.flush()
        batch_num = batch_i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches} done — ratings: {ratings}")

        if delay > 0 and batch_i + batch_size < total_pending:
            time.sleep(delay)

    f_out.close()

    total_new = len(newly_labeled)
    total_all = len(done) + total_new
    print(json.dumps({
        "newly_labeled": total_new,
        "total_labeled": total_all,
        "score_0_not_relevant": score_dist[0],
        "score_1_partial": score_dist[1],
        "score_2_highly_relevant": score_dist[2],
        "pct_relevant": round((score_dist[1] + score_dist[2]) / total_new * 100, 1) if total_new else 0,
        "provider": "gemini-2.5-flash",
        "out": str(out_path),
    }, indent=2))

    return newly_labeled


# ---------------------------------------------------------------------------
# MiniLM mode
# ---------------------------------------------------------------------------

def _cosine(a: Any, b: Any) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _score(sim: float, high: float, low: float) -> int:
    if sim >= high:
        return 2
    if sim >= low:
        return 1
    return 0


def label_with_minilm(
    sheet_rows: list[dict[str, Any]],
    query_texts: dict[str, str],
    out_path: Path,
    model_name: str,
    high_threshold: float,
    low_threshold: float,
) -> list[dict[str, Any]]:
    print(f"Loading model: {model_name}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    chunk_texts: dict[str, str] = {r["chunk_id"]: r.get("chunk_preview", "") for r in sheet_rows}

    unique_qids = list(query_texts.keys())
    unique_cids = list(chunk_texts.keys())

    print(f"Embedding {len(unique_qids)} queries + {len(unique_cids)} chunks...")
    q_embeddings = model.encode([query_texts[qid] for qid in unique_qids], batch_size=32, show_progress_bar=True)
    c_embeddings = model.encode([chunk_texts[cid] for cid in unique_cids], batch_size=32, show_progress_bar=True)

    q_emb_map = {qid: q_embeddings[i] for i, qid in enumerate(unique_qids)}
    c_emb_map = {cid: c_embeddings[i] for i, cid in enumerate(unique_cids)}

    labeled: list[dict[str, Any]] = []
    score_dist = {0: 0, 1: 0, 2: 0}

    for row in sheet_rows:
        qid = row["query_id"]
        cid = row["chunk_id"]
        if qid not in q_emb_map or cid not in c_emb_map:
            sim, rel = 0.0, 0
        else:
            sim = _cosine(q_emb_map[qid], c_emb_map[cid])
            rel = _score(sim, high_threshold, low_threshold)
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

    _write_csv(out_path, labeled, fieldnames=[
        "query_id", "query", "chunk_id", "chunk_title", "chunk_preview",
        "seen_in_configs", "best_retrieval_score", "cosine_similarity", "relevance", "notes",
    ])

    total = len(labeled)
    print(json.dumps({
        "total_rows": total,
        "score_0_not_relevant": score_dist[0],
        "score_1_partial": score_dist[1],
        "score_2_highly_relevant": score_dist[2],
        "pct_relevant": round((score_dist[1] + score_dist[2]) / total * 100, 1) if total else 0,
        "thresholds": {"high": high_threshold, "low": low_threshold},
        "out": str(out_path),
    }, indent=2))

    return labeled


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-label relevance for Objective 1 benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sheet", default="bench/labels/labeling_sheet.csv",
                   help="Labeling sheet CSV from export_labeling_sheet.py")
    p.add_argument("--queries", default="bench/queries_obj1_test.jsonl",
                   help="Queries JSONL (for full query text lookup)")
    p.add_argument("--out", default="bench/labels/labels.csv",
                   help="Output labels CSV")

    # Gemini mode
    p.add_argument("--gemini", action="store_true",
                   help="Use Gemini 2.5 Flash as LLM judge (more accurate, needs API key)")
    p.add_argument("--api-key",
                   default=os.environ.get("GEMINI_API_KEY", os.environ.get("ADAPTEACH_LLM_GEMINI_API_KEY", "")),
                   help="Google AI Studio API key (or set GEMINI_API_KEY env var)")
    p.add_argument("--batch-size", type=int, default=20,
                   help="Pairs per Gemini call (default 20). Higher = fewer API calls.")
    p.add_argument("--delay", type=float, default=7.0,
                   help="Seconds between Gemini calls (default 7 for free-tier 10 RPM)")

    # MiniLM mode
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Sentence-transformers model (MiniLM mode only)")
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

    queries = _load_queries(queries_path)
    sheet_rows = _read_csv(sheet_path)

    # Build query text lookup (full text preferred over sheet preview)
    query_texts: dict[str, str] = {}
    for row in sheet_rows:
        qid = row["query_id"]
        query_texts[qid] = queries.get(qid, {}).get("query", row.get("query", ""))

    if args.gemini:
        if not args.api_key:
            raise SystemExit("--api-key required for Gemini mode (or set GEMINI_API_KEY)")
        label_with_gemini(
            sheet_rows=sheet_rows,
            query_texts=query_texts,
            out_path=out_path,
            api_key=args.api_key,
            batch_size=args.batch_size,
            delay=args.delay,
        )
    else:
        label_with_minilm(
            sheet_rows=sheet_rows,
            query_texts=query_texts,
            out_path=out_path,
            model_name=args.model,
            high_threshold=args.high_threshold,
            low_threshold=args.low_threshold,
        )


if __name__ == "__main__":
    main()
