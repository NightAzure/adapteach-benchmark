import ast
from typing import Any


def _ast_parsable(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def check_tracing(payload: dict[str, Any]) -> dict[str, Any]:
    code = payload.get("code", "")
    steps = payload.get("trace_steps", [])
    ok = True
    errors = []
    if not _ast_parsable(code):
        ok = False
        errors.append("Code is not parsable.")
    if not isinstance(steps, list) or len(steps) == 0:
        ok = False
        errors.append("Trace steps missing.")
    if isinstance(steps, list) and len(steps) > 200:
        ok = False
        errors.append("Trace steps too many.")
    if isinstance(steps, list):
        for s in steps:
            if not isinstance(s, dict) or "line" not in s or "state" not in s:
                ok = False
                errors.append("Invalid trace step shape.")
                break
    return {"valid": ok, "errors": errors}


def check_mutation(payload: dict[str, Any]) -> dict[str, Any]:
    original_code = payload.get("original_code", "")
    mutated_code = payload.get("mutated_code", "")
    correct_fix = payload.get("correct_fix", "")
    ok = True
    errors = []
    if original_code == mutated_code:
        ok = False
        errors.append("Mutation is not real.")
    if original_code.count("\n") > 0 and mutated_code.count("\n") > 0:
        diff_len = abs(original_code.count("\n") - mutated_code.count("\n"))
        if diff_len > 20:
            ok = False
            errors.append("Mutation not localized.")
    if correct_fix and not _ast_parsable(correct_fix):
        ok = False
        errors.append("Correct fix is not parsable.")
    return {"valid": ok, "errors": errors}


def check_parsons(payload: dict[str, Any]) -> dict[str, Any]:
    lines = payload.get("lines", [])
    order = payload.get("solution_order", [])
    distractors = payload.get("distractors", [])
    ok = True
    errors = []
    if not lines or not order:
        return {"valid": False, "errors": ["Parsons lines/order missing."]}
    if len(set(order)) != len(order):
        ok = False
        errors.append("Solution order has duplicates.")
    if max(order, default=-1) >= len(lines):
        ok = False
        errors.append("Solution order index out of range.")
    if distractors and len(set(distractors)) != len(distractors):
        ok = False
        errors.append("Distractors are ambiguous duplicates.")
    try:
        ordered = "\n".join(lines[i] for i in order)
        ast.parse(ordered)
    except Exception:
        ok = False
        errors.append("Ordered Parsons solution is not parsable.")
    return {"valid": ok, "errors": errors}


def _coerce_str(v: Any) -> str:
    """Normalize a value to string — handles cases where the LLM returns a list."""
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v) if v is not None else ""


def check_flashcard(payload: dict[str, Any]) -> dict[str, Any]:
    q = _coerce_str(payload.get("question", "")).strip()
    a = _coerce_str(payload.get("answer", "")).strip()
    c = _coerce_str(payload.get("concept", "")).strip()
    ok = bool(q and a and c)
    return {"valid": ok, "errors": [] if ok else ["Flashcard fields missing."]}


def check_code_parsability(payload: dict[str, Any], artifact_type: str) -> dict[str, Any]:
    if artifact_type == "tracing":
        return {"valid": _ast_parsable(payload.get("code", "")), "errors": []}
    if artifact_type == "mutation":
        return {"valid": _ast_parsable(payload.get("mutated_code", "")), "errors": []}
    if artifact_type == "parsons":
        lines = payload.get("lines", [])
        order = payload.get("solution_order", [])
        try:
            code = "\n".join(lines[i] for i in order)
        except Exception:
            return {"valid": False, "errors": ["Invalid order."]}
        return {"valid": _ast_parsable(code), "errors": []}
    return {"valid": True, "errors": []}


def run_deterministic_checks(payload: dict[str, Any], artifact_type: str) -> dict[str, Any]:
    parsability = check_code_parsability(payload, artifact_type)
    if artifact_type == "tracing":
        detail = check_tracing(payload)
    elif artifact_type == "mutation":
        detail = check_mutation(payload)
    elif artifact_type == "parsons":
        detail = check_parsons(payload)
    elif artifact_type == "flashcard":
        detail = check_flashcard(payload)
    else:
        return {"valid": False, "errors": [f"Unsupported artifact type: {artifact_type}"], "parsability": parsability}
    valid = parsability["valid"] and detail["valid"]
    return {
        "valid": valid,
        "parsability": parsability,
        "detail": detail,
        "errors": parsability.get("errors", []) + detail.get("errors", []),
    }
