import json
import re
from pathlib import Path
from typing import Any

from src.indexing.build_indexes import embed_text

TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
KEYWORD_MAP = {
    "loop": "loops",
    "for": "loops",
    "while": "loops",
    "if": "conditionals",
    "else": "conditionals",
    "function": "functions",
    "def": "functions",
    "variable": "variables",
    "assignment": "variables",
}

# Short/common words that should not trigger classifier matches by themselves.
# These exist as CKG concept labels but match too many unrelated queries.
_STOPWORD_CONCEPTS = frozenset({
    "and", "or", "not", "if", "else", "for", "while", "all", "any",
    "get", "len", "def", "in", "is", "as", "of", "the", "a", "an",
    "to", "with", "from", "by", "on", "at", "it", "be", "do",
    "return", "none", "print", "input", "match",
})


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _concepts_from_ckg(ckg_path: Path) -> list[str]:
    graph = json.loads(ckg_path.read_text(encoding="utf-8"))
    return sorted([n["label"] for n in graph.get("nodes", []) if n.get("type") == "concept"])


def map_query_to_concepts(
    query: str,
    ckg_path: Path = Path("graphs/ckg.json"),
    top_k: int = 3,
) -> dict[str, Any]:
    concepts = _concepts_from_ckg(ckg_path)
    query_tokens = [t.lower() for t in TOKEN_RE.findall(query)]

    # ── Rule-based mapping ────────────────────────────────────────────
    rule_scores = {c: 0.0 for c in concepts}
    rule_hits: dict[str, list[str]] = {c: [] for c in concepts}
    for token in query_tokens:
        mapped = KEYWORD_MAP.get(token)
        if mapped and mapped in rule_scores:
            rule_scores[mapped] += 1.0
            rule_hits[mapped].append(token)

    # ── Semantic embedding similarity (MiniLM) ────────────────────────
    query_emb = embed_text(query)
    embed_scores: dict[str, float] = {}
    for c in concepts:
        cvec = embed_text(c.replace("-", " "))
        if len(cvec) == len(query_emb):
            embed_scores[c] = max(0.0, _dot(query_emb, cvec))
        else:
            embed_scores[c] = 0.0

    # ── Classifier: exact-token match with stopword guard ─────────────
    classifier_scores = {c: 0.0 for c in concepts}
    for c in concepts:
        c_toks = [t for t in c.split("-") if t not in _STOPWORD_CONCEPTS]
        if not c_toks:
            continue
        if any(tok in query_tokens for tok in c_toks):
            classifier_scores[c] = 1.0

    combined = {}
    for c in concepts:
        score = 0.5 * rule_scores[c] + 0.35 * embed_scores[c] + 0.15 * classifier_scores[c]
        combined[c] = score

    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    max_score = ranked[0][1] if ranked else 0.0
    mapped_results = []
    for label, score in ranked:
        confidence = (score / max_score) if max_score > 0 else 0.0
        explanation = {
            "rule_keywords": sorted(set(rule_hits[label])),
            "embedding_similarity": round(embed_scores[label], 6),
            "classifier_vote": classifier_scores[label],
        }
        mapped_results.append({"concept": label, "confidence": round(confidence, 6), "score": round(score, 6), "explanation": explanation})

    return {"query": query, "top_concepts": mapped_results}
