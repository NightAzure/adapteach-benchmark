from typing import Any

from src.pipelines.runner import run_pipeline
from src.utils.config import load_app_config, load_pipeline_config


def run_request(query: str, config_name: str, dry_run: str = "none") -> dict[str, Any]:
    app_config = load_app_config()
    app_config["pipelines"]["active"] = config_name
    pipeline_config = load_pipeline_config(config_name)
    request = {"query": query}
    response, _ = run_pipeline(
        request=request,
        app_config=app_config,
        pipeline_config=pipeline_config,
        dry_run=dry_run,
    )
    return response
