"""
Silver label cross-validation — Option B (semantic signal).

For each query in the custom dataset, takes the chunks retrieved by Config B,
computes cosine similarity between the query embedding and each chunk embedding
using all-MiniLM-L6-v2, then measures Spearman rank correlation between the
semantic ranking and the silver label ranking.

A mean rho > 0.60 indicates the lexical labels are not badly misaligned with
independent semantic signals, partially defusing the circularity concern.

Usage:
    python bench/validate_silver_labels.py
    python bench/validate_silver_labels.py --run-file bench/runs/run_<ts>.jsonl
    python bench/validate_silver_labels.py --config B --k 10 --out bench/results/silver_validation.csv

Install:
    pip install sentence-transformers scipy
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_silver_labels(labels_dir: Path) -> dict[str, dict[str, int]]:
    """Returns {query_id: {chunk_id: relevance}}."""
    labels: dict[str, dict[str, int]] = defaultdict(dict)
    for csv_path in sorted(labels_dir.glob("silver_labels_*.csv")):
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qid = row["query_id"]
                cid = row.get("chunk_id") or row.get("doc_id", "")
                if cid:
                    labels[qid][cid] = int(row["relevance"])
    return labels


def _latest_run(runs_dir: Path) -> Path:
    files = sorted(runs_dir.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No run files found in {runs_dir}")
    return files[0]


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def _spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation between two equal-length sequences."""
    from scipy.stats import spearmanr
    rho, _ = spearmanr(xs, ys)
    return float(rho) if rho == rho else 0.0  # guard NaN


def validate(
    run_file: Path,
    labels_dir: Path,
    config: str = "B",
    k: int = 10,
    out_path: Path | None = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, Any]:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        raise SystemExit(
            "Required packages missing. Run:\n"
            "  pip install sentence-transformers scipy"
        )

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Loading run file: {run_file}")
    all_rows = _read_jsonl(run_file)
    config_rows = [r for r in all_rows if r.get("config") == config]

    if not config_rows:
        raise ValueError(f"No rows found for config '{config}' in {run_file}")

    print(f"Loading silver labels from: {labels_dir}")
    silver = _load_silver_labels(labels_dir)

    per_query_results: list[dict[str, Any]] = []
    skipped = 0

    for row in config_rows:
        qid = row.get("query_id") or row.get("id", "")
        query_text = row.get("query", "")
        response = row.get("response") or {}
        context_items = response.get("context") or []

        chunks = [
            {
                "chunk_id": c.get("chunk_id", ""),
                "text": c.get("text") or c.get("content") or "",
            }
            for c in context_items[:k]
            if c.get("chunk_id")
        ]

        if len(chunks) < 2:
            skipped += 1
            continue

        # Only keep chunks that have a silver label for this query
        query_labels = silver.get(qid, {})
        labeled_chunks = [c for c in chunks if c["chunk_id"] in query_labels]

        if len(labeled_chunks) < 2:
            skipped += 1
            continue

        # Embed query and chunks together (single batch call is faster)
        texts = [query_text] + [c["text"] for c in labeled_chunks]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        import numpy as np
        query_vec = embeddings[0]
        chunk_vecs = embeddings[1:]

        # Cosine similarity: since MiniLM outputs are L2-normalised, dot product = cosine sim
        norms = np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        chunk_vecs_norm = chunk_vecs / norms
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        cosine_sims = (chunk_vecs_norm @ query_norm).tolist()

        # Silver label scores for same chunks
        label_scores = [query_labels[c["chunk_id"]] for c in labeled_chunks]

        rho = _spearman_rho(cosine_sims, label_scores)

        per_query_results.append({
            "query_id": qid,
            "n_chunks": len(labeled_chunks),
            "spearman_rho": round(rho, 4),
        })

    if not per_query_results:
        raise ValueError("No queries had enough labeled chunks to compute correlation.")

    rhos = [r["spearman_rho"] for r in per_query_results]
    summary = {
        "config": config,
        "n_queries_evaluated": len(per_query_results),
        "n_queries_skipped": skipped,
        "mean_rho": round(mean(rhos), 4),
        "median_rho": round(median(rhos), 4),
        "stdev_rho": round(stdev(rhos), 4) if len(rhos) > 1 else 0.0,
        "pct_above_0_60": round(sum(r > 0.60 for r in rhos) / len(rhos), 4),
        "pct_above_0_40": round(sum(r > 0.40 for r in rhos) / len(rhos), 4),
    }

    print("\n── Silver Label Validation Results ──")
    print(f"  Queries evaluated : {summary['n_queries_evaluated']}")
    print(f"  Queries skipped   : {summary['n_queries_skipped']}")
    print(f"  Mean Spearman ρ   : {summary['mean_rho']}")
    print(f"  Median ρ          : {summary['median_rho']}")
    print(f"  Std dev ρ         : {summary['stdev_rho']}")
    print(f"  Queries ρ > 0.60  : {summary['pct_above_0_60']:.0%}")
    print(f"  Queries ρ > 0.40  : {summary['pct_above_0_40']:.0%}")

    if summary["mean_rho"] >= 0.60:
        print("\n  ✓ Mean ρ ≥ 0.60 — labels show good alignment with semantic signal.")
    elif summary["mean_rho"] >= 0.40:
        print("\n  ~ Mean ρ 0.40–0.60 — moderate alignment; acknowledge limitation in paper.")
    else:
        print("\n  ✗ Mean ρ < 0.40 — labels may be poorly aligned; consider revision.")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "n_chunks", "spearman_rho"])
            writer.writeheader()
            writer.writerows(per_query_results)

        summary_path = out_path.parent / "silver_validation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n  Per-query results : {out_path}")
        print(f"  Summary           : {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validate silver labels against semantic similarity.")
    parser.add_argument("--run-file", type=Path, default=None,
                        help="Path to run JSONL (default: latest in bench/runs/)")
    parser.add_argument("--labels-dir", type=Path, default=ROOT / "bench" / "labels",
                        help="Directory containing silver_labels_*.csv files")
    parser.add_argument("--config", default="B",
                        help="Pipeline config to validate against (default: B)")
    parser.add_argument("--k", type=int, default=10,
                        help="Max chunks per query to consider (default: 10)")
    parser.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "silver_validation_per_query.csv",
                        help="Output CSV path for per-query Spearman rho values")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-transformers model name")
    args = parser.parse_args()

    run_file = args.run_file or _latest_run(ROOT / "bench" / "runs")
    print(f"Using run file: {run_file}")

    validate(
        run_file=run_file,
        labels_dir=args.labels_dir,
        config=args.config,
        k=args.k,
        out_path=args.out,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
