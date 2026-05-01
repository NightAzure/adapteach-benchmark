"""
Export silver-labeled query-chunk pairs for manual LLM judging.

Generates batched CSV files (80 rows each) in bench/llm_judge/input/
and a PROMPT.txt file to paste into ChatGPT or Gemini alongside each batch.

Usage:
    python bench/export_for_llm_judge.py
    python bench/export_for_llm_judge.py --batch-size 100 --max-chunk-chars 300
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"
MANIFEST_PATH = ROOT / "data" / "corpus_meta" / "chunk_manifest.json"
LABELS_DIR = ROOT / "bench" / "labels"
OUT_DIR = ROOT / "bench" / "llm_judge" / "input"

CHUNK_SIZE = 450


# ---------------------------------------------------------------------------
# Chunk content lookup
# ---------------------------------------------------------------------------

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _fixed_split_ids(content: str, doc_id: str) -> dict[str, str]:
    """Regenerate text chunk IDs and return {chunk_id: content}."""
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


def build_chunk_lookup() -> dict[str, str]:
    """Build {chunk_id: content} for all text and ast chunks."""
    lookup: dict[str, str] = {}

    # Text chunks — regenerated from corpus docs
    for doc_path in sorted(CORPUS_DIR.glob("*.json")):
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        doc_id = doc_path.stem
        content = doc.get("content", "")
        lookup.update(_fixed_split_ids(content, doc_id))

    # AST chunks — from manifest
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for chunk in manifest.get("chunks", []):
            cid = chunk.get("chunk_id", "")
            text = chunk.get("content", "")
            if cid and text:
                lookup[cid] = text

    return lookup


# ---------------------------------------------------------------------------
# Load silver labels
# ---------------------------------------------------------------------------

def load_silver_labels() -> list[dict]:
    rows = []
    for csv_path in sorted(LABELS_DIR.glob("silver_labels_*.csv")):
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(batch_size: int, max_chunk_chars: int) -> None:
    print("Building chunk lookup (regenerating text chunks)...")
    chunk_lookup = build_chunk_lookup()
    print(f"  {len(chunk_lookup):,} chunks indexed")

    print("Loading silver labels...")
    labels = load_silver_labels()
    print(f"  {len(labels):,} labeled pairs")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict] = []
    missing = 0

    for label in labels:
        qid = label["query_id"]
        query = label["query"]
        chunk_id = label.get("chunk_id", "")
        relevance = label.get("relevance", "")

        content = chunk_lookup.get(chunk_id)
        if not content:
            missing += 1
            continue

        truncated = content[:max_chunk_chars]
        if len(content) > max_chunk_chars:
            truncated += "…"

        rows_out.append({
            "row_id": f"{qid}__{chunk_id}",
            "query_id": qid,
            "query_text": query,
            "chunk_id": chunk_id,
            "chunk_text": truncated,
        })

    if missing:
        print(f"  Warning: {missing} pairs had no chunk content (skipped)")

    # Write batches
    total = len(rows_out)
    n_batches = (total + batch_size - 1) // batch_size
    print(f"\nWriting {total} rows across {n_batches} batch file(s) → {OUT_DIR}")

    fieldnames = ["row_id", "query_id", "query_text", "chunk_id", "chunk_text"]

    for i in range(n_batches):
        batch = rows_out[i * batch_size : (i + 1) * batch_size]
        out_path = OUT_DIR / f"judge_input_batch_{i+1:02d}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(batch)
        print(f"  Batch {i+1:02d}: {len(batch)} rows → {out_path.name}")

    # Write prompt file
    prompt_path = OUT_DIR.parent / "PROMPT.txt"
    prompt_path.write_text(PROMPT_TEXT.strip() + "\n", encoding="utf-8")
    print(f"\nPrompt file → {prompt_path}")
    print("\nDone. Next steps:")
    print("  1. Upload judge_input_batch_XX.csv to ChatGPT / Gemini")
    print("  2. Paste the contents of PROMPT.txt as your message")
    print("  3. Save the response as judge_scores_batch_XX.csv in bench/llm_judge/scores/")
    print("  4. Run: python bench/import_llm_scores.py")


PROMPT_TEXT = """
TASK: Score each row in the attached CSV for relevance. Output ONLY a CSV — no intro, no explanation, no summary.

=== INPUT FORMAT ===
The CSV has these columns:
  row_id        — unique identifier, copy this exactly into your output
  query_text    — a learner's Python programming question
  chunk_text    — a passage from an educational document

=== SCORING RUBRIC ===
  0 = Not relevant     — chunk does not address the query topic at all
  1 = Partially relevant — related topic but does not directly answer the query
  2 = Directly relevant  — chunk answers or strongly supports the query

=== OUTPUT FORMAT ===
Return a code block containing ONLY these two columns, one row per input row, NO header line:

```csv
row_id,llm_score
```

Example (your entire response should look exactly like this):
```csv
obj1-dev-001__text-abc123,2
obj1-dev-001__text-def456,0
obj1-dev-002__text-ghi789,1
```

STRICT RULES:
- Every row_id from the input must appear exactly once in your output
- Scores must be exactly 0, 1, or 2 — no decimals, no other values
- Do NOT include a header row
- Do NOT write anything outside the code block
- Do NOT summarize, explain, or comment
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export silver labels for LLM judging.")
    parser.add_argument("--batch-size", type=int, default=80,
                        help="Rows per batch CSV file (default: 80)")
    parser.add_argument("--max-chunk-chars", type=int, default=300,
                        help="Truncate chunk text to this many characters (default: 300)")
    args = parser.parse_args()
    export(args.batch_size, args.max_chunk_chars)


if __name__ == "__main__":
    main()
