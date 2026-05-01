"""
Automated LLM judge for silver label cross-validation.

Calls a local Ollama model to score query-chunk pairs for relevance (0/1/2),
then saves results directly to bench/llm_judge/scores/ for import_llm_scores.py.

Recommended models (ollama pull first):
  qwen3.5:27b   — ~17GB VRAM, best quality/speed balance
  qwen3.5:9b    — ~7GB VRAM, fast, still accurate
  qwen3.6:27b   — ~17GB VRAM, latest Qwen release
  llama3.3:70b  — ~43GB VRAM, highest accuracy

Usage:
    python bench/run_llm_judge.py
    python bench/run_llm_judge.py --model qwen3.5:27b
    python bench/run_llm_judge.py --model qwen3.6:27b --batch-size 50
    python bench/run_llm_judge.py --resume   # skip already-scored batches
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS_DIR = ROOT / "bench" / "labels"
CORPUS_DIR = ROOT / "data" / "corpus"
MANIFEST_PATH = ROOT / "data" / "corpus_meta" / "chunk_manifest.json"
SCORES_DIR = ROOT / "bench" / "llm_judge" / "scores"

CHUNK_SIZE = 450
DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_BATCH_SIZE = 30
MAX_CHUNK_CHARS = 400


# ---------------------------------------------------------------------------
# Chunk content lookup (mirrors export_for_llm_judge.py)
# ---------------------------------------------------------------------------

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


def build_chunk_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for doc_path in sorted(CORPUS_DIR.glob("*.json")):
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        doc_id = doc_path.stem
        lookup.update(_fixed_split_ids(doc.get("content", ""), doc_id))
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for chunk in manifest.get("chunks", []):
            cid = chunk.get("chunk_id", "")
            text = chunk.get("content", "")
            if cid and text:
                lookup[cid] = text
    return lookup


def load_silver_labels() -> list[dict]:
    rows = []
    for csv_path in sorted(LABELS_DIR.glob("silver_labels_*.csv")):
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a relevance judge. You will be given query-chunk pairs. "
    "For each pair, output ONLY a score: 0, 1, or 2. Nothing else."
)

JUDGE_PROMPT_TEMPLATE = """\
Score each row for relevance. Return ONLY a CSV block — no intro, no explanation, no summary.

Scoring:
  0 = Not relevant
  1 = Partially relevant
  2 = Directly relevant

Return format (no header, inside a code block):
```csv
row_id,llm_score
```

Input rows:
{rows_csv}
"""


def call_ollama(base_url: str, model: str, prompt: str, timeout: int = 120) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "system": JUDGE_SYSTEM,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return str(body.get("response", "")).strip()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama unreachable at {base_url} — is it running? ({e})") from e


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_scores(response: str, expected_ids: list[str], debug: bool = False) -> dict[str, int]:
    """Extract row_id,score pairs from model response."""
    # Strip Qwen3 thinking blocks <think>...</think>
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    if debug:
        print(f"\n--- RAW RESPONSE ---\n{response[:1000]}\n--- END ---", file=sys.stderr)

    # Prefer content inside ```csv ... ``` or ``` ... ``` block
    block_match = re.search(r"```(?:csv)?\s*\n(.*?)```", cleaned, re.DOTALL)
    raw = block_match.group(1) if block_match else cleaned

    scores: dict[str, int] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("row_id"):
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            row_id = parts[0].strip()
            try:
                score = int(parts[-1].strip())
                if score in (0, 1, 2):
                    scores[row_id] = score
            except ValueError:
                continue

    missing = [rid for rid in expected_ids if rid not in scores]
    if missing:
        print(f"\n  Warning: {len(missing)} row_ids missing from model response", file=sys.stderr)

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    model: str,
    base_url: str,
    batch_size: int,
    resume: bool,
    timeout: int,
    debug: bool = False,
) -> None:
    print("Building chunk lookup...")
    chunk_lookup = build_chunk_lookup()
    print(f"  {len(chunk_lookup):,} chunks indexed")

    print("Loading silver labels...")
    labels = load_silver_labels()
    print(f"  {len(labels):,} labeled pairs")

    # Build rows to score (same logic as export_for_llm_judge.py)
    rows_to_score: list[dict] = []
    missing = 0
    for label in labels:
        qid = label["query_id"]
        query = label["query"]
        chunk_id = label.get("chunk_id", "")
        content = chunk_lookup.get(chunk_id)
        if not content:
            missing += 1
            continue
        truncated = content[:MAX_CHUNK_CHARS]
        if len(content) > MAX_CHUNK_CHARS:
            truncated += "…"
        rows_to_score.append({
            "row_id": f"{qid}__{chunk_id}",
            "query_text": query,
            "chunk_text": truncated,
        })

    if missing:
        print(f"  Warning: {missing} pairs had no chunk content (skipped)")

    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    total = len(rows_to_score)
    n_batches = (total + batch_size - 1) // batch_size
    print(f"\nScoring {total} pairs in {n_batches} batches (model={model})\n")

    all_scores: dict[str, int] = {}
    start_time = time.perf_counter()

    for batch_idx in range(n_batches):
        batch_num = batch_idx + 1
        out_path = SCORES_DIR / f"judge_scores_batch_{batch_num:02d}.csv"

        if resume and out_path.exists():
            print(f"  Batch {batch_num:02d}/{n_batches} — skipped (already exists)")
            # Load existing scores into all_scores
            with out_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("row_id"):
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        try:
                            all_scores[parts[0].strip()] = int(parts[1].strip())
                        except ValueError:
                            pass
            continue

        batch = rows_to_score[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        expected_ids = [r["row_id"] for r in batch]

        # Build CSV input for the model
        rows_csv_lines = ["row_id,query_text,chunk_text"]
        for r in batch:
            query_safe = r["query_text"].replace('"', '""').replace("\n", " ")
            chunk_safe = r["chunk_text"].replace('"', '""').replace("\n", " ")
            rows_csv_lines.append(f'"{r["row_id"]}","{query_safe}","{chunk_safe}"')
        rows_csv = "\n".join(rows_csv_lines)

        prompt = JUDGE_PROMPT_TEMPLATE.format(rows_csv=rows_csv)

        elapsed = time.perf_counter() - start_time
        done = batch_idx * batch_size
        eta = (elapsed / done * (total - done)) if done > 0 else 0
        print(
            f"  Batch {batch_num:02d}/{n_batches} "
            f"[{batch_idx * batch_size + 1}–{min((batch_idx + 1) * batch_size, total)}/{total}] "
            f"elapsed={elapsed:.0f}s eta={eta:.0f}s ...",
            end=" ", flush=True,
        )

        try:
            response = call_ollama(base_url, model, prompt, timeout=timeout)
        except RuntimeError as e:
            print(f"\nFATAL: {e}")
            sys.exit(1)

        scores = parse_scores(response, expected_ids, debug=debug)
        all_scores.update(scores)

        # Write batch score file
        with out_path.open("w", newline="", encoding="utf-8") as f:
            for row_id in expected_ids:
                score = scores.get(row_id, "")
                f.write(f"{row_id},{score}\n")

        print(f"scored {len(scores)}/{len(expected_ids)}")

    total_scored = sum(1 for v in all_scores.values() if isinstance(v, int))
    print(f"\nDone. {total_scored}/{total} pairs scored.")
    print(f"Scores saved to: {SCORES_DIR}")
    print("\nNext step: python bench/import_llm_scores.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated LLM judge for silver label cross-validation.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama base URL (default: http://localhost:11434)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Pairs per Ollama call (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Seconds to wait per Ollama call (default: 180)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip batches whose output file already exists")
    parser.add_argument("--debug", action="store_true",
                        help="Print raw model response for the first batch (for diagnosing parse failures)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        model=args.model,
        base_url=args.ollama_url,
        batch_size=args.batch_size,
        resume=args.resume,
        timeout=args.timeout,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
