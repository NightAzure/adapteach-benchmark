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
SCORES_DIR = ROOT / "bench" / "llm_judge" / "scores"

DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_BATCH_SIZE = 30


def load_from_pool(pool_file: Path) -> list[dict]:
    rows: list[dict] = []
    with pool_file.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            query_id = row.get("query_id", "")
            chunk_id = row.get("chunk_id", "")
            if not query_id or not chunk_id:
                continue
            rows.append({
                "row_id": f"{query_id}__{chunk_id}",
                "query_text": row.get("query_text", ""),
                "chunk_text": row.get("chunk_text", ""),
            })
    return rows


JUDGE_SYSTEM = (
    "You are a relevance scoring tool. "
    "You output ONLY CSV lines in the format: row_id,score - nothing else. "
    "Score: 0=not relevant, 1=partially relevant, 2=directly relevant. "
    "No explanations. No headers. No markdown. Just the CSV lines."
)

JUDGE_PROMPT_TEMPLATE = """\
Below is a table of query-chunk pairs to score for relevance.
Score: 0=not relevant, 1=partially relevant, 2=directly relevant.

### INPUT (do NOT echo this back)
{rows_csv}

### OUTPUT INSTRUCTIONS
For every row_id above, output exactly one line: row_id,score
Do NOT include the query or chunk text. Do NOT add headers. Example:
some-row-id__chunk-abc,2
another-row-id__chunk-def,0
"""


def call_ollama(base_url: str, model: str, prompt: str, timeout: int = 120) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
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
        text = body.get("message", {}).get("content", "")
        return str(text).strip()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama unreachable at {base_url} - is it running? ({e})") from e


def parse_scores(response: str, expected_ids: list[str], debug: bool = False) -> dict[str, int]:
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    if debug:
        preview = response[:2000] if response else "(empty)"
        print(f"\n--- RAW RESPONSE ({len(response)} chars) ---\n{preview}\n--- END ---", file=sys.stderr)

    block_match = re.search(r"```(?:csv)?\s*\n(.*?)```", cleaned, re.DOTALL)
    raw = block_match.group(1) if block_match else cleaned

    expected_set = set(expected_ids)
    scores: dict[str, int] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("row_id"):
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            row_id = parts[0].strip().strip('"')
            if row_id not in expected_set:
                continue
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


def run(
    model: str,
    base_url: str,
    batch_size: int,
    resume: bool,
    timeout: int,
    pool_file: Path,
    debug: bool = False,
    retries: int = 2,
) -> None:
    if not pool_file.exists():
        raise SystemExit(f"Pool file not found: {pool_file}\nRun bench/build_judge_pool.py first.")

    print(f"Loading judging pool from: {pool_file}")
    rows_to_score = load_from_pool(pool_file)
    print(f"  {len(rows_to_score):,} pairs loaded from pool")

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
            print(f"  Batch {batch_num:02d}/{n_batches} - skipped (already exists)")
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

        rows_csv_lines = ["row_id,query_text,chunk_text"]
        for r in batch:
            query_safe = r["query_text"].replace('"', '""').replace("\n", " ")
            chunk_safe = r["chunk_text"].replace('"', '""').replace("\n", " ")
            rows_csv_lines.append(f'"{r["row_id"]}","{query_safe}","{chunk_safe}"')
        prompt = JUDGE_PROMPT_TEMPLATE.format(rows_csv="\n".join(rows_csv_lines))

        elapsed = time.perf_counter() - start_time
        done = batch_idx * batch_size
        eta = (elapsed / done * (total - done)) if done > 0 else 0
        print(
            f"  Batch {batch_num:02d}/{n_batches} "
            f"[{batch_idx * batch_size + 1}-{min((batch_idx + 1) * batch_size, total)}/{total}] "
            f"elapsed={elapsed:.0f}s eta={eta:.0f}s ...",
            end=" ", flush=True,
        )

        try:
            response = call_ollama(base_url, model, prompt, timeout=timeout)
        except RuntimeError as e:
            print(f"\nFATAL: {e}")
            sys.exit(1)

        scores = parse_scores(response, expected_ids, debug=debug)

        missing_rows = [r for r in batch if r["row_id"] not in scores]
        for attempt in range(retries):
            if not missing_rows:
                break
            print(f"\n    retrying {len(missing_rows)} missing rows (attempt {attempt + 1}/{retries}) ...", end=" ", flush=True)
            retry_ids = [r["row_id"] for r in missing_rows]
            retry_csv_lines = ["row_id,query_text,chunk_text"]
            for r in missing_rows:
                query_safe = r["query_text"].replace('"', '""').replace("\n", " ")
                chunk_safe = r["chunk_text"].replace('"', '""').replace("\n", " ")
                retry_csv_lines.append(f'"{r["row_id"]}","{query_safe}","{chunk_safe}"')
            retry_prompt = JUDGE_PROMPT_TEMPLATE.format(rows_csv="\n".join(retry_csv_lines))
            try:
                retry_response = call_ollama(base_url, model, retry_prompt, timeout=timeout)
            except RuntimeError as e:
                print(f"\nFATAL: {e}")
                sys.exit(1)
            retry_scores = parse_scores(retry_response, retry_ids, debug=debug)
            scores.update(retry_scores)
            newly_recovered = sum(1 for rid in retry_ids if rid in retry_scores)
            missing_rows = [r for r in missing_rows if r["row_id"] not in scores]
            print(f"recovered {newly_recovered}/{len(retry_ids)}", end="")

        all_scores.update(scores)

        with out_path.open("w", newline="", encoding="utf-8") as f:
            for row_id in expected_ids:
                score = scores.get(row_id, "")
                f.write(f"{row_id},{score}\n")

        print(f"\n  scored {len(scores)}/{len(expected_ids)}")

    total_scored = sum(1 for v in all_scores.values() if isinstance(v, int))
    print(f"\nDone. {total_scored}/{total} pairs scored.")
    print(f"Scores saved to: {SCORES_DIR}")
    print("\nNext step: python bench/build_qrels.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated LLM judge for qrels construction.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama base URL (default: http://localhost:11434)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Pairs per Ollama call (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Seconds to wait per Ollama call (default: 600)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip batches whose output file already exists")
    parser.add_argument("--retries", type=int, default=2,
                        help="Times to retry rows missing from a batch response (default: 2)")
    parser.add_argument("--debug", action="store_true",
                        help="Print raw model response for the first batch (for diagnosing parse failures)")
    parser.add_argument("--pool-file", default="bench/judge_pool/pool.csv",
                        help="Path to pool.csv from build_judge_pool.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool_file = Path(args.pool_file)
    if not pool_file.is_absolute():
        pool_file = ROOT / pool_file
    run(
        model=args.model,
        base_url=args.ollama_url,
        batch_size=args.batch_size,
        resume=args.resume,
        timeout=args.timeout,
        pool_file=pool_file,
        debug=args.debug,
        retries=args.retries,
    )


if __name__ == "__main__":
    main()
