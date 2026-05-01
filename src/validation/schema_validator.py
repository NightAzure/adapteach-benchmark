from typing import Any


def validate_schema(payload: dict[str, Any], artifact_type: str) -> dict[str, Any]:
    return {"valid": True, "errors": []}


def check_unsafe_content(payload: dict[str, Any]) -> dict[str, Any]:
    return {"safe": True, "flags": []}
