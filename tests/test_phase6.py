import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.runner import run_pipeline
from src.sandbox.harness import SandboxPolicy, execute_in_sandbox, static_blocklist_check
from src.utils.config import load_app_config, load_pipeline_config
from src.validation.validator import validate_artifact


def _app_config_with_temp_logs() -> dict:
    cfg = load_app_config()
    cfg["runtime"]["log_dir"] = tempfile.mkdtemp(prefix="phase6-logs-")
    return cfg


def test_schema_validation_and_unsafe_content_blocked() -> None:
    payload = {
        "artifact_type": "flashcard",
        "question": "visit https://example.com",
        "answer": "Use API_KEY=abc",
        "concept": "loops",
    }
    result = validate_artifact(payload, artifact_type="flashcard")
    assert result["valid"] is False
    assert result["unsafe_content"]["safe"] is False


def test_pipeline_uses_validator_fallback_and_case_bundle() -> None:
    app = _app_config_with_temp_logs()
    pipeline = load_pipeline_config("A")
    # Link in query is propagated into flashcard payload and should trigger unsafe validation fallback.
    response, record = run_pipeline(
        request={"query": "Explain loops from https://example.com", "artifact_type": "flashcard"},
        app_config=app,
        pipeline_config=pipeline,
        dry_run="none",
    )
    assert response["status"] == "ok_with_fallback"
    fallback = record["stage_outputs"]["validation"].get("fallback", {})
    assert "case_bundle" in fallback
    assert Path(fallback["case_bundle"]).exists()


def test_sandbox_policy_static_checks_and_no_exec_default() -> None:
    blocked = static_blocklist_check("import os\nprint('x')")
    assert blocked["valid"] is False

    result = execute_in_sandbox("print('hello')", policy=SandboxPolicy(allow_execution=False))
    assert result["executed"] is False
    assert result["reason"] == "policy_disallow_execution"
