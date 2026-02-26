import hashlib
from typing import Any


def _stable_chunk_id(doc_id: str, chunk_size: int, overlap: int, start: int, end: int, text: str) -> str:
    basis = f"{doc_id}|fixed|{chunk_size}|{overlap}|{start}|{end}|{text}"
    return "fixed-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _code_text_type(text: str) -> str:
    lowered = text.lower()
    if "```python" in lowered or "def " in lowered or "class " in lowered:
        return "code"
    return "text"


def chunk_fixed(
    doc: dict[str, Any],
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[dict[str, Any]]:
    content = str(doc.get("content", "")).strip()
    if not content:
        return []

    chunks: list[dict[str, Any]] = []
    cursor = 0
    length = len(content)
    while cursor < length:
        end = min(length, cursor + chunk_size)
        chunk_text = content[cursor:end]
        chunk_id = _stable_chunk_id(doc["doc_id"], chunk_size, overlap, cursor, end, chunk_text)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc["doc_id"],
                "chunker": "fixed-v1",
                "start_char": cursor,
                "end_char": end,
                "content": chunk_text,
                "concept_tags": list(doc.get("concept_tags", [])),
                "content_type": _code_text_type(chunk_text),
                "metadata": {
                    "title": doc.get("title", ""),
                    "type": doc.get("type", ""),
                },
            }
        )
        if end == length:
            break
        cursor = max(0, end - overlap)
    return chunks
