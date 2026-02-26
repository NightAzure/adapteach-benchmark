import ast
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BLOCKED_IMPORTS = {"os", "subprocess", "socket", "requests", "http", "pathlib", "shutil"}


@dataclass
class SandboxPolicy:
    allow_execution: bool = False
    no_network: bool = True
    cpu_time_limit_seconds: int = 2
    memory_limit_mb: int = 128
    non_root_required: bool = True
    file_system_restricted: bool = True


def static_blocklist_check(code: str) -> dict[str, Any]:
    violations: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"valid": False, "violations": [f"syntax_error:{exc}"]}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BLOCKED_IMPORTS:
                    violations.append(f"import:{root}")
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BLOCKED_IMPORTS:
                violations.append(f"from_import:{root}")
    return {"valid": len(violations) == 0, "violations": sorted(set(violations))}


def execute_in_sandbox(
    code: str,
    policy: SandboxPolicy = SandboxPolicy(),
    telemetry_path: Path = Path("logs/sandbox_telemetry.jsonl"),
) -> dict[str, Any]:
    start = time.perf_counter()
    static = static_blocklist_check(code)
    if not policy.allow_execution:
        result = {
            "executed": False,
            "reason": "policy_disallow_execution",
            "static_check": static,
            "timeout": False,
            "exception": None,
            "resource_usage": {"elapsed_ms": round((time.perf_counter() - start) * 1000, 3)},
        }
        _append_telemetry(telemetry_path, result)
        return result
    if not static["valid"]:
        result = {
            "executed": False,
            "reason": "blocked_import",
            "static_check": static,
            "timeout": False,
            "exception": None,
            "resource_usage": {"elapsed_ms": round((time.perf_counter() - start) * 1000, 3)},
        }
        _append_telemetry(telemetry_path, result)
        return result

    try:
        completed = subprocess.run(
            ["py", "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=policy.cpu_time_limit_seconds,
            check=False,
        )
        timeout = False
        exc = None
        stdout = completed.stdout[:2000]
        stderr = completed.stderr[:2000]
    except subprocess.TimeoutExpired as exc_timeout:
        timeout = True
        exc = str(exc_timeout)
        stdout = ""
        stderr = ""

    elapsed = round((time.perf_counter() - start) * 1000, 3)
    result = {
        "executed": not timeout,
        "reason": "ok" if not timeout else "timeout",
        "static_check": static,
        "timeout": timeout,
        "exception": exc,
        "stdout": stdout,
        "stderr": stderr,
        "resource_usage": {"elapsed_ms": elapsed},
    }
    _append_telemetry(telemetry_path, result)
    return result


def _append_telemetry(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
