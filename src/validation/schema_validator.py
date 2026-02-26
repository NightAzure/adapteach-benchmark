import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


SCHEMA_DIR = Path(__file__).parent / "schemas"
SCHEMA_FILES = {
    "parsons": "parsons.schema.json",
    "tracing": "tracing.schema.json",
    "mutation": "mutation.schema.json",
    "flashcard": "flashcard.schema.json",
}

UNSAFE_PATTERNS = [
    re.compile(r"https?://", re.IGNORECASE),  # links
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # api key style
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
]


def _load_schema(artifact_type: str) -> dict[str, Any]:
    if artifact_type not in SCHEMA_FILES:
        raise ValueError(f"Unsupported artifact_type: {artifact_type}")
    schema_path = SCHEMA_DIR / SCHEMA_FILES[artifact_type]
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _iter_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)


def check_unsafe_content(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    for text in _iter_strings(payload):
        for pat in UNSAFE_PATTERNS:
            if pat.search(text):
                violations.append(pat.pattern)
    return {"safe": len(violations) == 0, "violations": sorted(set(violations))}


def validate_schema(payload: dict[str, Any], artifact_type: str) -> dict[str, Any]:
    schema = _load_schema(artifact_type)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        return {"valid": False, "errors": [e.message for e in errors]}
    return {"valid": True, "errors": []}
