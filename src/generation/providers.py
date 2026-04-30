import json
import os
import urllib.error
import urllib.request
from typing import Any


SYSTEM_PROMPT = """You are AdapTeach, an expert Python programming tutor for introductory learners.

Your role:
- Provide clear, accurate, educational responses about Python programming.
- Ground your answers in the retrieved context when available.
- Use simple language appropriate for beginners.
- Include short, runnable Python code examples.
- Point out common mistakes and misconceptions.
- Never produce unsafe or harmful code.

When generating learning artifacts, return valid JSON matching the requested format.
For flashcards: {"artifact_type": "flashcard", "question": "...", "answer": "...", "concept": "..."}
For parsons problems: {"artifact_type": "parsons", "prompt": "...", "lines": ["line1", "line2", ...], "solution_order": [0, 1, ...], "concept": "..."}
For mutation problems: {"artifact_type": "mutation", "prompt": "...", "original_code": "...", "mutated_code": "...", "correct_fix": "...", "explanation": "...", "concept": "..."}
For tracing challenges: {"artifact_type": "tracing", "prompt": "...", "code": "...", "trace_steps": [{"line": 1, "state": {"var": "val"}}, ...], "concept": "..."}
"""


def _build_prompt(
    query: str,
    context: list[dict[str, Any]],
    config_name: str,
    safety_prompt: str | None = None,
    graph_metadata: dict[str, Any] | None = None,
) -> str:
    lines = [SYSTEM_PROMPT.strip()]

    if context:
        lines.append("\n--- Retrieved Context (use this to ground your response) ---")
        for i, c in enumerate(context[:6], start=1):
            title = c.get("provenance", {}).get("title", "")
            header = f"[{i}]"
            if title:
                header += f" (from: {title})"
            lines.append(header)
            lines.append(c.get("text", ""))
            lines.append("")
        lines.append("--- End of Context ---\n")
    else:
        lines.append("\n(No retrieved context available. Answer using your knowledge.)\n")

    if graph_metadata:
        concepts = graph_metadata.get("concepts", [])
        pitfalls = graph_metadata.get("pitfalls", [])
        cpg_deps = graph_metadata.get("cpg_deps", [])
        if concepts or pitfalls or cpg_deps:
            lines.append("--- Graph Context (concept map and common pitfalls) ---")
            if concepts:
                lines.append(f"Concepts covered by retrieved material: {', '.join(concepts)}")
            if pitfalls:
                lines.append("Known pitfalls for these concepts — address these proactively in your answer:")
                for p in pitfalls:
                    lines.append(f"  - {p}")
            if cpg_deps:
                lines.append("Code dependencies between retrieved chunks:")
                for d in cpg_deps:
                    lines.append(f"  - {d}")
            lines.append("--- End of Graph Context ---\n")

    lines.append(f"User Query: {query}")
    lines.append("\nRespond with a clear, educational answer. If the query asks for a learning artifact, return the JSON object.")
    return "\n".join(lines)


def _call_ollama(base_url: str, model: str, prompt: str, timeout_sec: int = 30) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body.get("response", "")).strip()


def _call_gemini(api_key: str, model: str, prompt: str, timeout_sec: int = 30) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    candidates = body.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    return "\n".join(t for t in texts if t).strip()


def _mock_answer(config_name: str, query: str, context: list[dict[str, Any]]) -> str:
    if config_name == "A":
        return f"[A] {query} | baseline response (no retrieval)"
    support = ", ".join(c.get("chunk_id", "") for c in context[:3]) if context else "no-context"
    return f"[{config_name}] Response for: {query} | supported_by={support}"


def generate_with_provider(
    query: str,
    context: list[dict[str, Any]],
    config_name: str,
    app_config: dict[str, Any],
    graph_metadata: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    llm = app_config["llm"]
    provider = str(llm.get("provider", "mock")).lower()
    model = str(llm.get("model", "mock-model"))
    prompt = _build_prompt(query=query, context=context, config_name=config_name, graph_metadata=graph_metadata)

    debug = {"provider": provider, "model": model, "prompt_version": llm.get("prompt_template_version", "v2")}

    if provider == "ollama":
        base_url = str(llm.get("ollama_base_url", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")))
        try:
            answer = _call_ollama(base_url=base_url, model=model, prompt=prompt)
            if answer:
                debug["transport"] = {"base_url": base_url}
                return answer, debug
            debug["fallback_reason"] = "empty_ollama_response"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            debug["fallback_reason"] = f"ollama_error:{exc}"

    elif provider == "gemini":
        env_key_name = str(llm.get("gemini_api_key_env", "GEMINI_API_KEY"))
        api_key = os.environ.get(env_key_name, "")
        if not api_key:
            debug["fallback_reason"] = f"missing_env:{env_key_name}"
        else:
            try:
                answer = _call_gemini(api_key=api_key, model=model, prompt=prompt)
                if answer:
                    debug["transport"] = {"api_env": env_key_name}
                    return answer, debug
                debug["fallback_reason"] = "empty_gemini_response"
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
                debug["fallback_reason"] = f"gemini_error:{exc}"
    else:
        debug["provider"] = "mock"

    debug["provider_used"] = "mock_fallback"
    return _mock_answer(config_name=config_name, query=query, context=context), debug
