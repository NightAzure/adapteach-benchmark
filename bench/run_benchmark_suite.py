import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.runner import run_pipeline
from src.utils.config import load_app_config, load_pipeline_config


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run benchmark suite across configs on a query set")
    p.add_argument("--query-set", default="bench/queries_dev.jsonl")
    p.add_argument("--configs", default="A,B,C,D,E")
    p.add_argument("--artifact-type", default="flashcard", choices=["flashcard", "parsons", "tracing", "mutation"])
    p.add_argument("--out-dir", default="bench/runs")
    p.add_argument("--dry-run", default="none", choices=["none", "retrieval", "graph"])
    p.add_argument("--provider", default="", choices=["", "mock", "gemini", "ollama"])
    p.add_argument("--model", default="")
    p.add_argument("--tag", default="",
                   help="Optional label appended to the output filename, e.g. 'custom', 'cs1qa'")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_rows = _read_jsonl(Path(args.query_set))
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    app_cfg = load_app_config()
    if args.provider:
        app_cfg["llm"]["provider"] = args.provider
    if args.model:
        app_cfg["llm"]["model"] = args.model
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag_suffix = f"_{args.tag}" if args.tag else ""
    out_file = out_dir / f"run_{timestamp}{tag_suffix}.jsonl"

    with out_file.open("w", encoding="utf-8") as handle:
        for cfg_name in configs:
            p_cfg = load_pipeline_config(cfg_name)
            for q in query_rows:
                query = q["query"]
                request = {
                    "query": query,
                    "artifact_type": args.artifact_type,
                    "query_id": q["id"],
                    "category": q.get("category", ""),
                    "topic": q.get("topic", ""),
                }
                response, record = run_pipeline(request=request, app_config=app_cfg, pipeline_config=p_cfg, dry_run=args.dry_run)
                run_row = {
                    "timestamp": timestamp,
                    "dataset": args.tag or "unknown",
                    "query_id": q["id"],
                    "query": query,
                    "config": cfg_name,
                    "artifact_type": args.artifact_type,
                    "dry_run": args.dry_run,
                    "response": response,
                    "record": record,
                }
                handle.write(json.dumps(run_row, ensure_ascii=True) + "\n")

    print(json.dumps({"query_count": len(query_rows), "configs": configs, "output": str(out_file.as_posix())}, indent=2))


if __name__ == "__main__":
    main()
