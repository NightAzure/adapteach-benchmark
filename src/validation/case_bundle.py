import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_case_bundle(
    trace_id: str,
    request: dict[str, Any],
    artifact_type: str,
    context: list[dict[str, Any]],
    generated_payload: dict[str, Any],
    validation_result: dict[str, Any],
    out_dir: Path = Path("logs/case_bundles"),
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "artifact_type": artifact_type,
        "request": request,
        "context": context,
        "generated_payload": generated_payload,
        "validation_result": validation_result,
    }
    out_path = out_dir / f"{trace_id}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
