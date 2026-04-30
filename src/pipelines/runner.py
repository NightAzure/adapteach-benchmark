from pathlib import Path
from typing import Any

from src.generation.providers import generate_with_provider
from src.graphs.graph_context_extractor import extract_graph_metadata
from src.retrieval.engine import expand_with_graphs, rerank_with_graph_signal, retrieve_candidates
from src.utils.run_logging import RunLogger
from src.validation.case_bundle import write_case_bundle
from src.validation.fallbacks import (
    deterministic_regenerate_payload,
    graph_fallback_context,
    retrieval_fallback_response,
    validator_failure_response,
)
from src.validation.validator import validate_artifact


def _assemble_context(candidates: list[dict[str, Any]], max_items: int = 6) -> list[dict[str, Any]]:
    return candidates[:max_items]


def _generate_answer(
    config_name: str,
    query: str,
    context: list[dict[str, Any]],
    app_config: dict[str, Any],
    graph_metadata: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    return generate_with_provider(
        query=query,
        context=context,
        config_name=config_name,
        app_config=app_config,
        graph_metadata=graph_metadata,
    )


def _validate(answer: str) -> dict[str, Any]:
    valid = bool(answer.strip())
    return {"valid": valid, "reason": "ok" if valid else "empty"}


def _extract_json_from_response(answer: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from the LLM response (may be wrapped in markdown)."""
    import json as _json
    import re as _re
    # Try direct parse first
    stripped = answer.strip()
    try:
        return _json.loads(stripped)
    except (ValueError, _json.JSONDecodeError):
        pass
    # Try extracting from ```json ... ``` block
    match = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, _re.DOTALL)
    if match:
        try:
            return _json.loads(match.group(1))
        except (ValueError, _json.JSONDecodeError):
            pass
    # Try finding first { ... } block
    brace_start = stripped.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(stripped)):
            if stripped[i] == "{":
                depth += 1
            elif stripped[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return _json.loads(stripped[brace_start : i + 1])
                    except (ValueError, _json.JSONDecodeError):
                        break
    return None


def _extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from markdown-formatted text."""
    import re as _re
    blocks = _re.findall(r"```python\s*(.*?)```", text, _re.DOTALL | _re.IGNORECASE)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    # Fallback: look for indented code blocks
    lines = text.split("\n")
    code_lines = [l for l in lines if l.startswith("    ") or l.startswith("\t")]
    if code_lines:
        return ["\n".join(l.lstrip() for l in code_lines)]
    return []


def _artifact_payload_from_response(
    artifact_type: str,
    query: str,
    answer: str,
    context: list[dict[str, Any]],
) -> dict[str, Any]:
    concept = "python-basics"
    if context and context[0].get("provenance", {}).get("title"):
        concept = context[0]["provenance"]["title"].lower().replace(" ", "-")

    # Try to parse structured JSON from LLM response
    parsed = _extract_json_from_response(answer)
    if parsed and isinstance(parsed, dict):
        # Ensure artifact_type and concept are set
        parsed.setdefault("artifact_type", artifact_type)
        parsed.setdefault("concept", concept)
        if artifact_type == "flashcard":
            parsed.setdefault("question", query)
            parsed.setdefault("answer", answer)
        elif artifact_type == "parsons":
            parsed.setdefault("prompt", query)
        elif artifact_type == "mutation":
            parsed.setdefault("prompt", query)
        elif artifact_type == "tracing":
            parsed.setdefault("prompt", query)
        return parsed

    # Fallback: construct payload from raw LLM text
    code_blocks = _extract_code_blocks(answer)

    if artifact_type == "flashcard":
        return {"artifact_type": "flashcard", "question": query, "answer": answer, "concept": concept}

    if artifact_type == "parsons":
        if code_blocks:
            lines = code_blocks[0].split("\n")
        else:
            lines = [l for l in answer.split("\n") if l.strip()]
        return {
            "artifact_type": "parsons",
            "prompt": query,
            "lines": lines if lines else [answer],
            "solution_order": list(range(len(lines))) if lines else [0],
            "concept": concept,
        }

    if artifact_type == "mutation":
        original = code_blocks[0] if len(code_blocks) >= 1 else ""
        mutated = code_blocks[1] if len(code_blocks) >= 2 else original
        return {
            "artifact_type": "mutation",
            "prompt": query,
            "original_code": original or answer,
            "mutated_code": mutated or answer,
            "correct_fix": original or answer,
            "explanation": answer,
            "concept": concept,
        }

    if artifact_type == "tracing":
        code = code_blocks[0] if code_blocks else answer
        # Simple trace: one step per line of code
        code_lines = [l for l in code.split("\n") if l.strip()]
        trace_steps = [{"line": i + 1, "state": {}} for i in range(len(code_lines))]
        if not trace_steps:
            trace_steps = [{"line": 1, "state": {}}]
        return {
            "artifact_type": "tracing",
            "prompt": query,
            "code": code,
            "trace_steps": trace_steps,
            "concept": concept,
        }

    return {"artifact_type": "flashcard", "question": query, "answer": answer, "concept": concept}


def run_pipeline(
    request: dict[str, Any],
    app_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    dry_run: str = "none",
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_name = pipeline_config["pipeline"]["name"]
    logger = RunLogger(log_dir=Path(app_config["runtime"]["log_dir"]), app_config=app_config)
    query = request["query"]

    retrieval_cfg = app_config["retrieval"]
    k = int(retrieval_cfg["k"])
    chunking_mode = pipeline_config["pipeline"]["chunking"]
    hybrid = bool(pipeline_config["pipeline"]["hybrid"])
    graph_enabled = bool(pipeline_config["pipeline"]["graph_expansion"])
    graph_context_enabled = bool(pipeline_config["pipeline"].get("graph_context", False))
    retrieval_enabled = bool(pipeline_config["pipeline"]["retrieval_enabled"])

    retrieval_start = logger.stage_start("retrieval")
    if retrieval_enabled:
        candidates, retrieval_debug = retrieve_candidates(
            query=query,
            k=k,
            chunking_mode=chunking_mode,
            hybrid=hybrid,
            query_topic=request.get("topic", ""),
            meta_dir=Path("data/corpus_meta"),
            indexes_dir=Path("indexes"),
            index_hash=retrieval_cfg.get("index_hash"),
        )
    else:
        candidates, retrieval_debug = [], {"retrieval_mode": "none", "top_k": 0, "index_found": False}

    # Config D needs parse fallback visibility in logs.
    if config_name == "D":
        parse_backends = {}
        fallback_count = 0
        for c in candidates:
            backend = c.get("provenance", {}).get("chunker", "")
            parse_backends[backend] = parse_backends.get(backend, 0) + 1
            if backend.startswith("fixed"):
                fallback_count += 1
        retrieval_debug["parser_fallback"] = {"chunker_counts": parse_backends, "fallback_chunks": fallback_count}

    logger.stage_end(
        "retrieval",
        retrieval_start,
        {"count": len(candidates), "items": candidates, "debug": retrieval_debug},
    )

    # For retrieval dry-run, still run graph expansion + rerank when enabled
    # so that Config E's full retrieval pipeline is evaluated.
    if dry_run == "retrieval":
        if graph_enabled:
            expanded, _g_debug = expand_with_graphs(query=query, candidates=candidates, graphs_dir=Path("graphs"), meta_dir=Path("data/corpus_meta"))
            reranked, _rr_debug = rerank_with_graph_signal(expanded, rerank_weights=retrieval_cfg.get("rerank_weights", {}))
            context_out = reranked[:k]
        else:
            context_out = candidates
        response = {
            "trace_id": logger.trace_id,
            "config": config_name,
            "mode": "dry_run_retrieval",
            "status": "ok",
            "answer": None,
            "context": context_out,
            "validation": None,
        }
        record = logger.write_final_record(request=request, response=response)
        return response, record

    weak = retrieval_fallback_response(query=query, context=candidates)
    if weak and retrieval_enabled:
        response = {
            "trace_id": logger.trace_id,
            "config": config_name,
            "mode": "full",
            "status": "ok_with_fallback",
            "answer": weak["answer"],
            "context": candidates,
            "validation": {"valid": True, "reason": "retrieval_weak_fallback"},
        }
        record = logger.write_final_record(request=request, response=response, token_counts={"prompt": 1, "response": len(response["answer"].split())})
        return response, record

    graph_start = logger.stage_start("graph_expansion")
    if graph_enabled:
        expanded, graph_debug = expand_with_graphs(
            query=query,
            candidates=candidates,
            query_topic=request.get("topic", ""),
            graphs_dir=Path("graphs"),
            meta_dir=Path("data/corpus_meta"),
        )
        reranked, rerank_debug = rerank_with_graph_signal(expanded, rerank_weights=retrieval_cfg.get("rerank_weights", {}))
        expanded = reranked
        graph_debug["rerank"] = rerank_debug
        expanded, graph_fallback_debug = graph_fallback_context(expanded, candidates, cap=12)
        graph_debug["fallback"] = graph_fallback_debug
    else:
        expanded = candidates
        graph_debug = {"graph_used": False}

    logger.stage_end("graph_expansion", graph_start, {"count": len(expanded), "items": expanded, "debug": graph_debug})

    # Config F: extract graph metadata (concepts + pitfalls + CPG deps) to inject into prompt.
    # This runs AFTER graph_expansion (so expanded candidates are available) but graph_context
    # and graph_expansion are mutually exclusive by design — F sets graph_expansion=false.
    graph_context_start = logger.stage_start("graph_context")
    graph_metadata: dict[str, Any] | None = None
    if graph_context_enabled:
        graph_metadata = extract_graph_metadata(
            candidates=expanded,
            query=query,
            graphs_dir=Path("graphs"),
        )
    logger.stage_end(
        "graph_context",
        graph_context_start,
        graph_metadata if graph_metadata is not None else {"graph_context_enabled": False},
    )

    if dry_run == "graph":
        response = {
            "trace_id": logger.trace_id,
            "config": config_name,
            "mode": "dry_run_graph",
            "status": "ok",
            "answer": None,
            "context": expanded,
            "validation": None,
        }
        record = logger.write_final_record(request=request, response=response)
        return response, record

    assemble_start = logger.stage_start("context_assembly")
    context = _assemble_context(expanded, max_items=k)
    logger.stage_end(
        "context_assembly",
        assemble_start,
        {
            "count": len(context),
            "items": context,
            "token_budget_policy": {"max_items": k},
        },
    )

    generation_start = logger.stage_start("generation")
    answer, generation_debug = _generate_answer(
        config_name=config_name,
        query=query,
        context=context,
        app_config=app_config,
        graph_metadata=graph_metadata,
    )
    logger.stage_end("generation", generation_start, {"answer_preview": answer[:120], "debug": generation_debug})

    validation_start = logger.stage_start("validation")
    artifact_type = request.get("artifact_type", "flashcard")
    payload = _artifact_payload_from_response(artifact_type=artifact_type, query=query, answer=answer, context=context)
    validation = validate_artifact(payload, artifact_type=artifact_type)

    max_regen_attempts = 2
    attempt = 0
    while not validation["valid"] and attempt < max_regen_attempts:
        attempt += 1
        payload = deterministic_regenerate_payload(payload, attempt=attempt)
        validation = validate_artifact(payload, artifact_type=artifact_type)

    if not validation["valid"]:
        case_path = write_case_bundle(
            trace_id=logger.trace_id,
            request=request,
            artifact_type=artifact_type,
            context=context,
            generated_payload=payload,
            validation_result=validation,
        )
        answer = validator_failure_response(artifact_type=artifact_type)
        validation["fallback"] = {"case_bundle": str(case_path.as_posix()), "attempts": attempt}

    basic_answer_check = _validate(answer)
    validation["answer_nonempty"] = basic_answer_check
    logger.stage_end("validation", validation_start, validation)

    response = {
        "trace_id": logger.trace_id,
        "config": config_name,
        "mode": "full",
        "status": "ok" if validation.get("valid", False) else "ok_with_fallback",
        "answer": answer,
        "context": context,
        "validation": validation,
    }
    token_counts = {"prompt": max(1, len(query.split())), "response": max(1, len(answer.split()))}
    record = logger.write_final_record(request=request, response=response, token_counts=token_counts)
    return response, record
