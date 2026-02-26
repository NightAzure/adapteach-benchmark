import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _print_summary(rows: list[dict[str, Any]], limit: int) -> None:
    print("query_id | config | status | context_n | top_chunks | retrieval_ms")
    print("-" * 90)
    shown = 0
    for row in rows:
        response = row.get("response", {})
        record = row.get("record", {})
        context = response.get("context", []) or []
        top = ",".join(c.get("chunk_id", "") for c in context[:3] if c.get("chunk_id"))
        retrieval_ms = _to_float(record.get("stage_timings_ms", {}).get("retrieval", 0.0))
        print(
            f"{row.get('query_id','')} | {row.get('config','')} | {response.get('status','')} | "
            f"{len(context)} | {top} | {retrieval_ms:.3f}"
        )
        shown += 1
        if 0 < limit <= shown:
            break


def _print_query(rows: list[dict[str, Any]], query_id: str) -> None:
    matched = [r for r in rows if r.get("query_id") == query_id]
    if not matched:
        print(f"No rows found for query_id={query_id}")
        return

    for row in matched:
        response = row.get("response", {})
        record = row.get("record", {})
        context = response.get("context", []) or []
        print("=" * 100)
        print(f"query_id: {row.get('query_id')} | config: {row.get('config')} | status: {response.get('status')}")
        print(f"query: {row.get('query')}")
        print(f"context count: {len(context)}")
        print(f"retrieval ms: {_to_float(record.get('stage_timings_ms', {}).get('retrieval', 0.0)):.3f}")
        print("-" * 100)
        for i, chunk in enumerate(context, start=1):
            cid = chunk.get("chunk_id", "")
            score = _to_float(chunk.get("score", 0.0))
            title = chunk.get("provenance", {}).get("title", "")
            text_preview = str(chunk.get("text", "")).replace("\n", " ")[:180]
            print(f"[{i}] {cid} | score={score:.4f} | title={title}")
            print(f"    {text_preview}")
        print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect benchmark run JSONL in a readable view")
    p.add_argument("--run-file", required=True, help="Path to bench/runs/run_*.jsonl")
    p.add_argument("--query-id", default="", help="If set, print detailed rows only for this query_id")
    p.add_argument("--config", default="", help="Optional config filter (A/B/C/D/E)")
    p.add_argument("--limit", type=int, default=25, help="Summary row limit (0 = all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_file = Path(args.run_file)
    if not run_file.exists():
        raise SystemExit(f"run file not found: {run_file.as_posix()}")

    rows = _read_jsonl(run_file)
    if args.config:
        rows = [r for r in rows if r.get("config") == args.config]

    if args.query_id:
        _print_query(rows, args.query_id)
    else:
        _print_summary(rows, args.limit)


if __name__ == "__main__":
    main()
