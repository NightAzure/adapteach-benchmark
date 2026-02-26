import argparse
import json

from src.pipelines.runner import run_pipeline
from src.utils.config import load_app_config, load_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdapTeach Phase 1 pipeline runner")
    parser.add_argument("--config", required=True, choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--query", required=True)
    parser.add_argument("--dry-run", choices=["none", "retrieval", "graph"], default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_config = load_app_config()
    app_config["pipelines"]["active"] = args.config
    app_config["runtime"]["dry_run"] = args.dry_run
    pipeline_config = load_pipeline_config(args.config)

    response, _ = run_pipeline(
        request={"query": args.query},
        app_config=app_config,
        pipeline_config=pipeline_config,
        dry_run=args.dry_run,
    )
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
