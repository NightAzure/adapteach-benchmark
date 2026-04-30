from typing import Any

from src.validation.deterministic_checks import run_deterministic_checks
from src.validation.schema_validator import check_unsafe_content, validate_schema


def validate_artifact(payload: dict[str, Any], artifact_type: str) -> dict[str, Any]:
    schema_result = validate_schema(payload, artifact_type)
    unsafe_result = check_unsafe_content(payload)
    deterministic = run_deterministic_checks(payload, artifact_type)
    valid = schema_result["valid"] and unsafe_result["safe"] and deterministic["valid"]
    return {
        "valid": valid,
        "artifact_type": artifact_type,
        "schema": schema_result,
        "unsafe_content": unsafe_result,
        "deterministic": deterministic,
    }


def validator_pass_fail_table(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.get("valid"))
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round((passed / total) * 100, 3) if total else 0.0,
    }
