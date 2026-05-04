from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "with", "is", "are", "as", "by", "from", "that",
    "this", "it", "be", "at", "not", "do", "if", "we", "you", "can", "has", "have", "will", "was", "but", "all",
    "so", "no", "when", "what", "how", "which", "their", "each", "than", "its", "also", "into", "just", "about",
    "would", "should", "could", "then", "these", "those", "them", "they", "been", "were", "being", "had", "did",
    "does", "may", "might", "must", "shall", "our", "my", "your", "his", "her", "who", "one", "two", "short",
    "python", "snippet", "code", "example", "examples", "learner", "beginner", "show", "explain", "write", "function",
}
DEBUG_TERMS = {"bug", "fix", "error", "traceback", "exception", "wrong", "incorrect", "debug"}
MISCONCEPTION_TERMS = {"misconception", "myth", "incorrect", "wrong", "belief"}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def keywords(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2}


def lexical_overlap_score(query_terms: set[str], text_terms: set[str]) -> float:
    if not query_terms or not text_terms:
        return 0.0
    inter = len(query_terms & text_terms)
    if inter == 0:
        return 0.0
    precision = inter / len(query_terms)
    recall = inter / len(text_terms)
    return (2 * precision * recall) / max(1e-9, precision + recall)


def category_bonus(category: str, text_terms: set[str]) -> float:
    if category == "debugging":
        return 0.20 if DEBUG_TERMS & text_terms else 0.0
    if category == "misconception":
        return 0.15 if MISCONCEPTION_TERMS & text_terms else 0.0
    return 0.0


def chunk_length_bonus(content: str) -> float:
    length = len(tokenize(content))
    if 40 <= length <= 250:
        return 0.10
    if 20 <= length < 40 or 250 < length <= 400:
        return 0.05
    return 0.0


def chunk_type_bonus(chunker: str) -> float:
    if chunker.startswith("ast"):
        return 0.04
    return 0.0


def has_code_bonus(content: str, query: str) -> float:
    if ("`" in query or "snippet" in query.lower() or "code" in query.lower()) and "```" in content:
        return 0.08
    return 0.0


def build_chunk_index(chunk_manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
    index: list[dict[str, Any]] = []
    for chunk in manifest.get("chunks", []):
        content = chunk.get("content", "")
        title = chunk.get("metadata", {}).get("title", "")
        concept_tags = {str(t).lower() for t in chunk.get("concept_tags", [])}
        index.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk.get("doc_id", ""),
            "chunker": chunk.get("chunker", ""),
            "concept_tags": concept_tags,
            "content": content,
            "text_terms": keywords(content),
            "title_terms": keywords(title),
            "title": title,
        })
    return index


def score_chunk(query_row: dict[str, Any], chunk: dict[str, Any]) -> float:
    query_text = query_row["query"]
    q_terms = keywords(query_text)
    topic = str(query_row.get("topic", "")).lower().strip()
    category = str(query_row.get("category", "")).lower().strip()
    topic_match = 1.0 if topic and topic in chunk["concept_tags"] else 0.0
    text_overlap = lexical_overlap_score(q_terms, chunk["text_terms"])
    title_overlap = lexical_overlap_score(q_terms, chunk["title_terms"])
    score = 0.0
    score += 0.55 * text_overlap
    score += 0.25 * title_overlap
    score += 0.35 * topic_match
    score += category_bonus(category, chunk["text_terms"])
    score += chunk_length_bonus(chunk["content"])
    score += chunk_type_bonus(chunk["chunker"])
    score += has_code_bonus(chunk["content"], query_text)
    return round(score, 8)


def infer_dataset_name(path: Path) -> str:
    stem = path.stem.lower()
    if stem.startswith("queries_"):
        return stem.replace("queries_", "")
    return stem
