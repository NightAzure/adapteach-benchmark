import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.runner import run_pipeline
from src.utils.config import load_app_config, load_pipeline_config


def _read_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ablation experiments")
    p.add_argument("--query-set", default="bench/queries_dev.jsonl")
    p.add_argument("--out-csv", default="bench/runs/ablation_results.csv")
    p.add_argument("--dry-run", default="none", choices=["none", "retrieval", "graph"])
    return p.parse_args()


def _run_config(config_name: str, queries: list[dict], app_cfg: dict, dry_run: str) -> dict:
    p_cfg = load_pipeline_config(config_name)
    success = 0
    avg_context = 0.0
    for q in queries:
        response, _ = run_pipeline(
            {"query": q["query"], "artifact_type": "flashcard", "topic": q.get("topic", "")},
            app_cfg,
            p_cfg,
            dry_run=dry_run,
        )
        if response["status"] in {"ok", "ok_with_fallback"}:
            success += 1
        avg_context += len(response.get("context", []))
    n = len(queries)
    return {"config": config_name, "success_rate": round(success / n, 6) if n else 0.0, "avg_context_items": round(avg_context / n, 6) if n else 0.0}


def main() -> None:
    args = parse_args()
    queries = list(_read_jsonl(Path(args.query_set)))
    app = load_app_config()
    rows = []

    # chunking ablation
    rows.append({"ablation": "chunking", "variant": "fixed(B)", **_run_config("B", queries, app, args.dry_run)})
    rows.append({"ablation": "chunking", "variant": "ast(D)", **_run_config("D", queries, app, args.dry_run)})

    # retrieval ablation
    rows.append({"ablation": "retrieval", "variant": "dense(B)", **_run_config("B", queries, app, args.dry_run)})
    rows.append({"ablation": "retrieval", "variant": "hybrid(C)", **_run_config("C", queries, app, args.dry_run)})
    rows.append({"ablation": "retrieval", "variant": "hybrid+graph(E)", **_run_config("E", queries, app, args.dry_run)})

    # graph ablation
    rows.append({"ablation": "graph", "variant": "no-graph(C)", **_run_config("C", queries, app, args.dry_run)})
    rows.append({"ablation": "graph", "variant": "dual-graph(E)", **_run_config("E", queries, app, args.dry_run)})

    # rerank weight sweep for config E
    for retrieval_w, graph_w in [(0.9, 0.1), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]:
        app_mod = json.loads(json.dumps(app))
        app_mod["retrieval"]["rerank_weights"] = {"retrieval": retrieval_w, "graph": graph_w}
        result = _run_config("E", queries, app_mod, args.dry_run)
        rows.append(
            {
                "ablation": "rerank_weight_sweep",
                "variant": f"E_r{retrieval_w}_g{graph_w}",
                **result,
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["ablation", "variant", "config", "success_rate", "avg_context_items"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(out_path.as_posix()), "timestamp": datetime.now(timezone.utc).isoformat()}, indent=2))


if __name__ == "__main__":
    main()
