"""
Build chunk_manifest.json from corpus_clean documents.

Per document:
  - Always produces fixed-v1 chunks (used by configs A–C).
  - Extracts ```python``` fenced blocks and tries ast.parse() on each.
      * If any block parses cleanly → ast-v2 chunks for that code (used by D–F).
      * If none parse → text-v1 chunks of the full content (used by D–F).

This means every doc is retrievable in every config.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

CHUNK_SIZE = 450
CHUNKER_VERSION = "ast-v1"
CORPUS_CLEAN_DIR = Path("data/corpus_clean")
META_DIR = Path("data/corpus_meta")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _fixed_split(content: str, doc: dict, tag: str) -> list[dict]:
    """Split content into fixed-size chunks at word boundaries."""
    chunks = []
    start = 0
    while start < len(content):
        end = min(start + CHUNK_SIZE, len(content))
        if end < len(content):
            boundary = content.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        piece = content[start:end].strip()
        if piece:
            chunk_id = f"{tag}-" + _hash(doc["doc_id"] + str(start) + piece)
            chunks.append({
                "chunk_id": chunk_id,
                "chunker": f"{tag}-v1",
                "concept_tags": doc.get("concept_tags", []),
                "content": piece,
                "content_type": "text",
                "doc_id": doc["doc_id"],
                "start_char": start,
                "end_char": end,
                "metadata": {
                    "title": doc.get("title", ""),
                    "type": doc.get("type", ""),
                },
            })
        start = end
    return chunks


def _extract_code_blocks(content: str) -> list[tuple[str, int]]:
    """Return (code_text, start_pos_in_content) for each ```python block."""
    results = []
    for m in re.finditer(r"```python\s*\n(.*?)```", content, re.DOTALL):
        code = m.group(1).rstrip()
        if code.strip():
            results.append((code, m.start()))
    return results


def _ast_chunks_from_block(code: str, doc: dict, block_start: int) -> list[dict]:
    """
    Walk the top-level AST of a code block and produce ast-v2 chunks.
    Functions and classes each get their own chunk; statements are batched.
    Returns [] if code fails ast.parse().
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    lines = code.splitlines(keepends=True)

    def line_offset(lineno: int) -> int:
        return sum(len(l) for l in lines[: lineno - 1])

    def node_src(node: ast.AST) -> str:
        s = node.lineno - 1
        e = getattr(node, "end_lineno", node.lineno) - 1
        return "".join(lines[s : e + 1]).rstrip()

    def make_chunk(content_body: str, node_type: str, scope: str, char_start: int) -> dict:
        full = f"# scope: {scope}\n{content_body}"
        chunk_id = "ast-" + _hash(doc["doc_id"] + full)
        abs_start = block_start + char_start
        return {
            "chunk_id": chunk_id,
            "chunker": "ast-v2",
            "concept_tags": doc.get("concept_tags", []),
            "content": full,
            "content_type": "code",
            "doc_id": doc["doc_id"],
            "start_char": abs_start,
            "end_char": abs_start + len(full),
            "metadata": {
                "node_type": node_type,
                "parse_backend": "ast",
                "scope": scope,
                "title": doc.get("title", ""),
                "type": doc.get("type", ""),
            },
        }

    chunks: list[dict] = []
    pending: list[str] = []
    pending_start: int = 0

    def flush(scope: str = "module") -> None:
        if not pending:
            return
        chunks.append(make_chunk("\n".join(pending), "statement", scope, pending_start))
        pending.clear()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            flush()
            src = node_src(node)
            chunks.append(make_chunk(src, "function", node.name, line_offset(node.lineno)))
        elif isinstance(node, ast.ClassDef):
            flush()
            src = node_src(node)
            chunks.append(make_chunk(src, "class", node.name, line_offset(node.lineno)))
        else:
            src = node_src(node)
            if not pending:
                pending_start = line_offset(node.lineno)
            pending.append(src)
            if sum(len(p) for p in pending) >= CHUNK_SIZE:
                flush()

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Per-document chunking
# ---------------------------------------------------------------------------

def chunk_document(doc: dict) -> tuple[list[dict], list[dict], list[dict], bool]:
    """
    Returns (fixed_chunks, ast_chunks, text_chunks, ast_success).
    fixed_chunks  → always produced, for configs A–C
    ast_chunks    → produced when code blocks parse, for D–F
    text_chunks   → fallback when no code parses, for D–F
    """
    content = doc.get("content", "")

    fixed_chunks = _fixed_split(content, doc, "fixed")

    code_blocks = _extract_code_blocks(content)
    ast_chunks: list[dict] = []
    for code, start_pos in code_blocks:
        ast_chunks.extend(_ast_chunks_from_block(code, doc, start_pos))

    if ast_chunks:
        return fixed_chunks, ast_chunks, [], True
    else:
        text_chunks = _fixed_split(content, doc, "text")
        return fixed_chunks, [], text_chunks, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)

    doc_files = sorted(CORPUS_CLEAN_DIR.glob("*.json"))
    if not doc_files:
        print(f"No documents found in {CORPUS_CLEAN_DIR}", file=sys.stderr)
        sys.exit(1)

    all_chunks: list[dict] = []
    parsed_docs: list[str] = []
    failed_docs: list[dict] = []
    content_type_counts: dict[str, int] = {"code": 0, "text": 0}

    for doc_file in doc_files:
        doc = json.loads(doc_file.read_text(encoding="utf-8"))
        fixed, ast_c, text_c, ok = chunk_document(doc)

        all_chunks.extend(fixed)
        all_chunks.extend(ast_c)
        all_chunks.extend(text_c)

        for c in ast_c:
            content_type_counts["code"] += 1
        for c in fixed + text_c:
            content_type_counts["text"] += 1

        if ok:
            parsed_docs.append(doc["doc_id"])
        else:
            failed_docs.append({
                "doc_id": doc["doc_id"],
                "parsed": False,
                "reason": "fallback_text",
                "parse_backend": "fixed",
                "ast_reason": "syntax_error_ast",
                "tree_sitter_reason": "not_used",
                "fallback_chunk_count": len(fixed),
            })

    doc_count = len(doc_files)
    parse_rate = round(100 * len(parsed_docs) / doc_count, 1) if doc_count else 0.0

    manifest = {
        "chunk_size": CHUNK_SIZE,
        "chunker": CHUNKER_VERSION,
        "chunks": all_chunks,
    }
    (META_DIR / "chunk_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    char_counts = [len(c["content"]) for c in all_chunks]
    stats = {
        "chunk_count": len(all_chunks),
        "chunker": CHUNKER_VERSION,
        "content_type_counts": content_type_counts,
        "doc_count": doc_count,
        "chunk_size_stats": {
            "count": len(char_counts),
            "min_chars": min(char_counts) if char_counts else 0,
            "max_chars": max(char_counts) if char_counts else 0,
            "mean_chars": round(sum(char_counts) / len(char_counts), 3) if char_counts else 0,
            "median_chars": sorted(char_counts)[len(char_counts) // 2] if char_counts else 0,
        },
    }
    (META_DIR / "chunk_stats_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    failure_report = {
        "chunker": CHUNKER_VERSION,
        "doc_count": doc_count,
        "parsed_docs": len(parsed_docs),
        "failure_count": len(failed_docs),
        "parse_rate": parse_rate,
        "failure_examples": failed_docs[:10],
    }
    (META_DIR / "parser_failure_report.json").write_text(
        json.dumps(failure_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"Chunked {doc_count} docs → {len(all_chunks)} chunks "
        f"({len(parsed_docs)} AST-parsed, {len(failed_docs)} text-fallback)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
