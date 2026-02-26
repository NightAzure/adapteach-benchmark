import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.runner import run_pipeline
from src.utils.config import load_app_config, load_pipeline_config


def _load_with_temp_log_dir() -> dict:
    config = load_app_config()
    tmp = tempfile.mkdtemp(prefix="adapteach-logs-")
    config["runtime"]["log_dir"] = tmp
    return config


def _assert_output_schema(response: dict) -> None:
    expected_keys = {"trace_id", "config", "mode", "status", "answer", "context", "validation"}
    assert set(response.keys()) == expected_keys


def test_pipeline_switch_keeps_output_schema_stable() -> None:
    app_config = _load_with_temp_log_dir()
    for name in ["A", "B", "C", "D", "E"]:
        pipeline = load_pipeline_config(name)
        response, _ = run_pipeline(
            request={"query": "schema check"},
            app_config=app_config,
            pipeline_config=pipeline,
            dry_run="none",
        )
        assert response["config"] == name
        _assert_output_schema(response)


def test_dry_run_modes() -> None:
    app_config = _load_with_temp_log_dir()
    pipeline = load_pipeline_config("E")
    retrieval_response, _ = run_pipeline(
        request={"query": "dry retrieval"},
        app_config=app_config,
        pipeline_config=pipeline,
        dry_run="retrieval",
    )
    graph_response, _ = run_pipeline(
        request={"query": "dry graph"},
        app_config=app_config,
        pipeline_config=pipeline,
        dry_run="graph",
    )
    assert retrieval_response["mode"] == "dry_run_retrieval"
    assert graph_response["mode"] == "dry_run_graph"
    _assert_output_schema(retrieval_response)
    _assert_output_schema(graph_response)


def test_trace_id_propagates_to_stage_events() -> None:
    app_config = _load_with_temp_log_dir()
    pipeline = load_pipeline_config("C")
    response, record = run_pipeline(
        request={"query": "trace check"},
        app_config=app_config,
        pipeline_config=pipeline,
        dry_run="none",
    )
    assert response["trace_id"] == record["trace_id"]
    assert record["stage_events"]
    assert all(event["trace_id"] == record["trace_id"] for event in record["stage_events"])


def test_env_override_applies() -> None:
    os.environ["ADAPTEACH__retrieval__k"] = "9"
    try:
        config = load_app_config()
        assert config["retrieval"]["k"] == 9
    finally:
        del os.environ["ADAPTEACH__retrieval__k"]
