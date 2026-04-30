"""
Import LLM judge scores and compute Cohen's kappa against silver labels.

Reads all judge_scores_batch_*.csv files from bench/llm_judge/scores/,
merges with silver labels, and reports inter-rater agreement statistics.

Usage:
    python bench/import_llm_scores.py
    python bench/import_llm_scores.py --scores-dir bench/llm_judge/scores
    python bench/import_llm_scores.py --out bench/results/llm_judge_summary.json

Install:
    pip install scikit-learn
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS_DIR = ROOT / "bench" / "labels"
SCORES_DIR = ROOT / "bench" / "llm_judge" / "scores"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_silver_labels() -> dict[str, int]:
    """Returns {row_id: relevance} where row_id = query_id__chunk_id."""
    silver: dict[str, int] = {}
    for csv_path in sorted(LABELS_DIR.glob("silver_labels_*.csv")):
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qid = row["query_id"]
                cid = row.get("chunk_id", "")
                if cid:
                    row_id = f"{qid}__{cid}"
                    silver[row_id] = int(row["relevance"])
    return silver


def load_llm_scores(scores_dir: Path) -> dict[str, int]:
    """Returns {row_id: llm_score} merged from all batch score files."""
    scores: dict[str, int] = {}
    batch_files = sorted(scores_dir.glob("judge_scores_batch_*.csv"))
    if not batch_files:
        # Also accept a single merged file
        merged = scores_dir / "judge_scores.csv"
        if merged.exists():
            batch_files = [merged]
    if not batch_files:
        raise FileNotFoundError(
            f"No score files found in {scores_dir}\n"
            "Expected: judge_scores_batch_01.csv, judge_scores_batch_02.csv, ..."
        )
    for path in batch_files:
        with path.open(encoding="utf-8") as f:
            content = f.read().strip()
        # Support with or without header
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        for line in lines:
            if line.startswith("row_id"):  # header
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                row_id = parts[0].strip()
                try:
                    score = int(parts[1].strip())
                    scores[row_id] = score
                except ValueError:
                    continue
    return scores


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _cohen_kappa(y1: list[int], y2: list[int]) -> float:
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(y1, y2))
    except ImportError:
        # Manual computation
        labels = sorted(set(y1) | set(y2))
        n = len(y1)
        # Observed agreement
        p_o = sum(a == b for a, b in zip(y1, y2)) / n
        # Expected agreement
        count1 = Counter(y1)
        count2 = Counter(y2)
        p_e = sum((count1[k] / n) * (count2[k] / n) for k in labels)
        return (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 0.0


def _percent_exact(y1: list[int], y2: list[int]) -> float:
    return sum(a == b for a, b in zip(y1, y2)) / len(y1)


def _percent_within_one(y1: list[int], y2: list[int]) -> float:
    return sum(abs(a - b) <= 1 for a, b in zip(y1, y2)) / len(y1)


def _disagreement_table(y1: list[int], y2: list[int]) -> dict[str, int]:
    table: dict[str, int] = defaultdict(int)
    for a, b in zip(y1, y2):
        if a != b:
            table[f"silver={a} llm={b}"] += 1
    return dict(sorted(table.items()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(scores_dir: Path, out_path: Path | None) -> None:
    print("Loading silver labels...")
    silver = load_silver_labels()
    print(f"  {len(silver):,} labeled pairs")

    print(f"Loading LLM scores from {scores_dir}...")
    llm = load_llm_scores(scores_dir)
    print(f"  {len(llm):,} scored pairs")

    # Intersect on row_id
    common_ids = sorted(set(silver) & set(llm))
    if not common_ids:
        raise ValueError(
            "No row_ids matched between silver labels and LLM scores.\n"
            "Check that row_id format matches: query_id__chunk_id"
        )

    missing_in_llm = len(silver) - len(common_ids)
    if missing_in_llm:
        print(f"  Warning: {missing_in_llm} silver pairs have no LLM score (excluded)")

    y_silver = [silver[rid] for rid in common_ids]
    y_llm    = [llm[rid]    for rid in common_ids]

    kappa = _cohen_kappa(y_silver, y_llm)
    pct_exact = _percent_exact(y_silver, y_llm)
    pct_within1 = _percent_within_one(y_silver, y_llm)
    disagree = _disagreement_table(y_silver, y_llm)

    # Distribution comparison
    silver_dist = Counter(y_silver)
    llm_dist    = Counter(y_llm)

    print("\n── LLM Judge vs Silver Label Agreement ──")
    print(f"  Pairs evaluated       : {len(common_ids)}")
    print(f"  Cohen's κ             : {kappa:.4f}")
    print(f"  Exact agreement       : {pct_exact:.1%}")
    print(f"  Within-1 agreement    : {pct_within1:.1%}")
    print()
    print("  Score distribution:")
    for s in [0, 1, 2]:
        print(f"    Score {s} — silver: {silver_dist[s]:4d}  llm: {llm_dist[s]:4d}")
    print()
    print("  Disagreement breakdown (silver→llm):")
    for k, v in disagree.items():
        print(f"    {k}: {v}")

    if kappa >= 0.60:
        verdict = "STRONG — labels have substantial independent validation."
    elif kappa >= 0.40:
        verdict = "MODERATE — acknowledge limitation; labels show partial agreement."
    else:
        verdict = "WEAK — labels may be poorly aligned; consider revising."

    print(f"\n  Verdict (κ={kappa:.3f}): {verdict}")

    summary = {
        "n_pairs": len(common_ids),
        "n_missing_llm_score": missing_in_llm,
        "cohens_kappa": round(kappa, 4),
        "pct_exact_agreement": round(pct_exact, 4),
        "pct_within_one": round(pct_within1, 4),
        "score_distribution": {
            "silver": dict(silver_dist),
            "llm": dict(llm_dist),
        },
        "disagreement_breakdown": disagree,
        "verdict": verdict,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n  Summary saved → {out_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Import LLM judge scores and compute Cohen's kappa.")
    parser.add_argument("--scores-dir", type=Path, default=SCORES_DIR,
                        help="Directory containing judge_scores_batch_*.csv files")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "bench" / "results" / "llm_judge_summary.json",
                        help="Output path for summary JSON")
    args = parser.parse_args()
    run(args.scores_dir, args.out)


if __name__ == "__main__":
    main()
