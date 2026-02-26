import argparse
import csv
import json
from collections import defaultdict
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a non-technical labeling sheet from benchmark run JSONL")
    p.add_argument("--run-file", required=True)
    p.add_argument("--out-csv", default="bench/labels/labels_sheet.csv")
    p.add_argument(
        "--configs",
        default="B,C,D,E",
        help="Comma-separated configs to include. Default excludes A because A has no retrieval chunks.",
    )
    p.add_argument("--max-chunks-per-query-per-config", type=int, default=5)
    p.add_argument("--preview-chars", type=int, default=220)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_file = Path(args.run_file)
    if not run_file.exists():
        raise SystemExit(f"run file not found: {run_file.as_posix()}")

    selected_configs = [c.strip().upper() for c in args.configs.split(",") if c.strip()]
    if not selected_configs:
        raise SystemExit("at least one config must be selected")

    rows = _read_jsonl(run_file)
    rows = [r for r in rows if str(r.get("config", "")).upper() in selected_configs]

    # Group by (query_id, chunk_id), then track where chunk appears.
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    query_texts: dict[str, str] = {}
    skipped_rows = 0

    per_query_cfg_count: dict[tuple[str, str], int] = defaultdict(int)
    max_per_cfg = int(args.max_chunks_per_query_per_config)

    for row in rows:
        qid = str(row.get("query_id", "")).strip()
        query = str(row.get("query", "")).strip()
        cfg = str(row.get("config", "")).upper().strip()
        if not qid or not cfg:
            continue
        query_texts[qid] = query

        context = row.get("response", {}).get("context", []) or []
        for chunk in context:
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            if not chunk_id:
                continue
            key_cfg = (qid, cfg)
            if max_per_cfg > 0 and per_query_cfg_count[key_cfg] >= max_per_cfg:
                skipped_rows += 1
                continue
            per_query_cfg_count[key_cfg] += 1

            key = (qid, chunk_id)
            if key not in grouped:
                grouped[key] = {
                    "query_id": qid,
                    "query": query,
                    "chunk_id": chunk_id,
                    "chunk_title": str(chunk.get("provenance", {}).get("title", "")).strip(),
                    "chunk_preview": str(chunk.get("text", "")).replace("\n", " ").strip()[: args.preview_chars],
                    "seen_in_configs": set(),
                    "best_retrieval_score": _to_float(chunk.get("score", 0.0)),
                }
            grouped[key]["seen_in_configs"].add(cfg)
            grouped[key]["best_retrieval_score"] = max(grouped[key]["best_retrieval_score"], _to_float(chunk.get("score", 0.0)))

    out_rows: list[dict[str, str]] = []
    for (qid, _), data in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        out_rows.append(
            {
                "query_id": qid,
                "query": data["query"] or query_texts.get(qid, ""),
                "chunk_id": data["chunk_id"],
                "chunk_title": data["chunk_title"],
                "chunk_preview": data["chunk_preview"],
                "seen_in_configs": ",".join(sorted(data["seen_in_configs"])),
                "best_retrieval_score": f"{data['best_retrieval_score']:.6f}",
                "relevance_rater_a": "",
                "relevance_rater_b": "",
                "notes": "",
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "query",
                "chunk_id",
                "chunk_title",
                "chunk_preview",
                "seen_in_configs",
                "best_retrieval_score",
                "relevance_rater_a",
                "relevance_rater_b",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(
        json.dumps(
            {
                "run_file": run_file.as_posix(),
                "out_csv": out_csv.as_posix(),
                "selected_configs": selected_configs,
                "rows_for_labeling": len(out_rows),
                "queries_found": len({r['query_id'] for r in out_rows}),
                "skipped_due_to_max_per_config": skipped_rows,
                "note": "Config A intentionally excluded by default because it has no retrieval chunks.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
