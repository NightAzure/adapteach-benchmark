import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build labels CSV template from a benchmark run JSONL file")
    p.add_argument("--run-file", required=True, help="Path to bench/runs/run_*.jsonl")
    p.add_argument("--out-csv", default="bench/labels/labels.csv")
    p.add_argument("--raters", default="rater_a,rater_b", help="Comma-separated rater ids")
    p.add_argument("--max-chunks-per-query", type=int, default=0, help="0 means keep all retrieved chunks")
    return p.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    run_file = Path(args.run_file)
    if not run_file.exists():
        raise SystemExit(f"run file not found: {run_file.as_posix()}")

    raters = [r.strip() for r in args.raters.split(",") if r.strip()]
    if not raters:
        raise SystemExit("at least one rater is required")

    rows = _read_jsonl(run_file)
    qid_to_chunks: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        qid = str(row.get("query_id", "")).strip()
        if not qid:
            continue
        context = row.get("response", {}).get("context", [])
        for c in context:
            chunk_id = str(c.get("chunk_id", "")).strip()
            if chunk_id and chunk_id not in qid_to_chunks[qid]:
                qid_to_chunks[qid].append(chunk_id)

    max_chunks = int(args.max_chunks_per_query)
    out_rows: list[dict[str, str]] = []
    for qid in sorted(qid_to_chunks.keys()):
        chunk_ids = qid_to_chunks[qid]
        if max_chunks > 0:
            chunk_ids = chunk_ids[:max_chunks]
        for chunk_id in chunk_ids:
            for rater in raters:
                out_rows.append(
                    {
                        "query_id": qid,
                        "rater": rater,
                        "chunk_id": chunk_id,
                        "relevance": "",
                        "notes": "",
                    }
                )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "rater", "chunk_id", "relevance", "notes"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(
        json.dumps(
            {
                "run_file": run_file.as_posix(),
                "out_csv": out_csv.as_posix(),
                "queries": len(qid_to_chunks),
                "rows": len(out_rows),
                "raters": raters,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
