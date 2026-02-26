import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.runner import run_pipeline
from src.utils.config import load_app_config, load_pipeline_config


def ensure_snapshot_exists(snapshot_id: str, snapshots_root: Path = Path("data/snapshots")) -> Path:
    snapshot_manifest = snapshots_root / snapshot_id / "manifest.json"
    if not snapshot_manifest.exists():
        raise FileNotFoundError(
            f"Snapshot '{snapshot_id}' not found. Create one with: "
            "py -m src.indexing.snapshot_tool create"
        )
    return snapshot_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark runner with snapshot freeze enforcement")
    parser.add_argument("--config", required=True, choices=["A", "B", "C", "D", "E", "F"])
    parser.add_argument("--query", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--dry-run", choices=["none", "retrieval", "graph"], default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        ensure_snapshot_exists(args.snapshot_id)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    app_config = load_app_config()
    app_config["pipelines"]["active"] = args.config
    app_config["runtime"]["dry_run"] = args.dry_run
    app_config["retrieval"]["corpus_snapshot"] = args.snapshot_id

    pipeline_config = load_pipeline_config(args.config)
    response, record = run_pipeline(
        request={"query": args.query},
        app_config=app_config,
        pipeline_config=pipeline_config,
        dry_run=args.dry_run,
    )
    output = {"response": response, "log_record_trace_id": record["trace_id"]}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
