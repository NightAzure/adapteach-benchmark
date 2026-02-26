from typing import Any


def retrieval_fallback_response(query: str, context: list[dict[str, Any]]) -> dict[str, Any] | None:
    if context:
        return None
    return {
        "mode": "retrieval_weak",
        "answer": (
            "I need a bit more detail to answer accurately. "
            "Could you clarify the specific concept (loops, conditionals, functions, or variables)?"
        ),
    }


def graph_fallback_context(expanded: list[dict[str, Any]], base_candidates: list[dict[str, Any]], cap: int = 12) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(expanded) > cap:
        return base_candidates[:cap], {"degraded": True, "reason": "graph_expansion_capped", "cap": cap}
    return expanded, {"degraded": False}


def validator_failure_response(artifact_type: str) -> str:
    return (
        f"Could not produce a fully validated {artifact_type} artifact safely. "
        "Here is a conservative explanation instead: review the core concept step-by-step and avoid executing untrusted code."
    )


def deterministic_regenerate_payload(payload: dict[str, Any], attempt: int) -> dict[str, Any]:
    regenerated = dict(payload)
    suffix = f" [regen_attempt_{attempt}]"
    if "answer" in regenerated and isinstance(regenerated["answer"], str):
        regenerated["answer"] += suffix
    if "question" in regenerated and isinstance(regenerated["question"], str):
        regenerated["question"] += suffix
    if "prompt" in regenerated and isinstance(regenerated["prompt"], str):
        regenerated["prompt"] += suffix
    return regenerated
