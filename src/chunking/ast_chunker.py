import ast
import hashlib
import re
from dataclasses import dataclass
from typing import Any

from src.chunking.fixed_chunker import chunk_fixed

try:
    import tree_sitter as ts  # type: ignore
    import tree_sitter_python as tspython  # type: ignore

    TREE_SITTER_AVAILABLE = True
except Exception:
    TREE_SITTER_AVAILABLE = False


CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*.*?```", re.DOTALL | re.IGNORECASE)


def _stable_id(doc_id: str, start: int, end: int, scope: str, text: str) -> str:
    basis = f"{doc_id}|ast|{start}|{end}|{scope}|{text}"
    return "ast-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _text_chunk_id(doc_id: str, start: int, end: int, text: str) -> str:
    basis = f"{doc_id}|text|{start}|{end}|{text}"
    return "text-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _extract_python_source(text: str) -> str:
    blocks = CODE_BLOCK_RE.findall(text)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks if block.strip())
    return text


def _extract_text_segments(
    content: str,
    doc: dict[str, Any],
    max_chars: int = 450,
    min_chars: int = 40,
) -> list[dict[str, Any]]:
    """Extract prose/explanatory text between code blocks as separate text chunks."""
    # Remove all code fences to get only the prose
    parts = CODE_FENCE_RE.split(content)
    chunks: list[dict[str, Any]] = []
    offset = 0
    for part in parts:
        idx = content.find(part, offset)
        if idx == -1:
            idx = offset
        text = part.strip()
        # Strip markdown heading markers but keep the text
        text = re.sub(r"^---+\s*", "", text, flags=re.MULTILINE).strip()
        if len(text) < min_chars:
            offset = idx + len(part)
            continue
        # Split large text segments
        if len(text) <= max_chars:
            segments = [text]
        else:
            segments = []
            cursor = 0
            while cursor < len(text):
                end = min(len(text), cursor + max_chars)
                # Try to break at sentence boundary
                if end < len(text):
                    last_period = text.rfind(". ", cursor, end)
                    last_newline = text.rfind("\n", cursor, end)
                    break_at = max(last_period + 2, last_newline + 1)
                    if break_at > cursor + min_chars:
                        end = break_at
                segments.append(text[cursor:end].strip())
                cursor = end
        for seg in segments:
            if len(seg) < min_chars:
                continue
            chunk_id = _text_chunk_id(doc["doc_id"], idx, idx + len(seg), seg)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc["doc_id"],
                    "chunker": "ast-v2",
                    "start_char": idx,
                    "end_char": idx + len(seg),
                    "content": seg,
                    "concept_tags": list(doc.get("concept_tags", [])),
                    "content_type": "text",
                    "metadata": {
                        "title": doc.get("title", ""),
                        "type": doc.get("type", ""),
                        "node_type": "prose",
                        "scope": "document",
                        "parse_backend": "text_extraction",
                    },
                }
            )
        offset = idx + len(part)
    return chunks


@dataclass
class Candidate:
    start_char: int
    end_char: int
    node_type: str
    scope: str


def _split_large_candidate(source: str, candidate: Candidate, max_chars: int) -> list[Candidate]:
    span = candidate.end_char - candidate.start_char
    if span <= max_chars:
        return [candidate]
    segments: list[Candidate] = []
    cursor = candidate.start_char
    while cursor < candidate.end_char:
        end = min(candidate.end_char, cursor + max_chars)
        segments.append(Candidate(start_char=cursor, end_char=end, node_type=candidate.node_type, scope=candidate.scope))
        cursor = end
    return segments


def _merge_tiny_segments(candidates: list[Candidate], min_chars: int) -> list[Candidate]:
    if not candidates:
        return []
    merged: list[Candidate] = []
    pending = candidates[0]
    for current in candidates[1:]:
        pending_span = pending.end_char - pending.start_char
        if pending_span < min_chars and pending.scope == current.scope:
            pending = Candidate(
                start_char=pending.start_char,
                end_char=current.end_char,
                node_type=pending.node_type,
                scope=pending.scope,
            )
        else:
            merged.append(pending)
            pending = current
    merged.append(pending)
    return merged


def _offsets_by_line(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _line_span_to_char(offsets: list[int], lineno: int, end_lineno: int) -> tuple[int, int]:
    start_char = offsets[max(0, lineno - 1)]
    end_char = offsets[max(0, end_lineno)]
    return start_char, end_char


class _PyAstCollector(ast.NodeVisitor):
    def __init__(self, offsets: list[int]) -> None:
        self.offsets = offsets
        self.scope_stack: list[str] = ["module"]
        self.candidates: list[Candidate] = []

    def _record(self, node: ast.AST, node_type: str) -> None:
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return
        start_char, end_char = _line_span_to_char(self.offsets, node.lineno, node.end_lineno)
        self.candidates.append(Candidate(start_char=start_char, end_char=end_char, node_type=node_type, scope="::".join(self.scope_stack)))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._record(node, "function")
        self.scope_stack.append(f"function:{node.name}")
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._record(node, "function")
        self.scope_stack.append(f"function:{node.name}")
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._record(node, "class")
        self.scope_stack.append(f"class:{node.name}")
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_For(self, node: ast.For) -> Any:
        self._record(node, "for_block")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> Any:
        self._record(node, "while_block")
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> Any:
        self._record(node, "if_block")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        self._record(node, "statement")
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> Any:
        self._record(node, "statement")
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> Any:
        self._record(node, "statement")
        self.generic_visit(node)


def _collect_with_python_ast(source: str) -> tuple[list[Candidate], dict[str, Any]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [], {"parsed": False, "reason": "syntax_error_ast", "error": str(exc), "parse_backend": "ast"}

    offsets = _offsets_by_line(source)
    collector = _PyAstCollector(offsets)
    collector.visit(tree)
    return sorted(collector.candidates, key=lambda c: (c.start_char, c.end_char, c.node_type)), {
        "parsed": True,
        "reason": "ok",
        "parse_backend": "ast",
    }


TS_NODE_MAP = {
    "function_definition": "function",
    "class_definition": "class",
    "for_statement": "for_block",
    "while_statement": "while_block",
    "if_statement": "if_block",
    "assignment": "statement",
    "expression_statement": "statement",
    "return_statement": "statement",
}


def _byte_to_char_map(source: str) -> dict[int, int]:
    src_bytes = source.encode("utf-8")
    mapping = {0: 0, len(src_bytes): len(source)}
    for i, ch in enumerate(source, start=1):
        mapping[len(source[:i].encode("utf-8"))] = i
    return mapping


def _ts_node_name(node: Any) -> str | None:
    try:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        return name_node.text.decode("utf-8").strip()
    except Exception:
        return None


def _collect_with_tree_sitter(source: str) -> tuple[list[Candidate], dict[str, Any]]:
    if not TREE_SITTER_AVAILABLE:
        return [], {"parsed": False, "reason": "tree_sitter_unavailable", "parse_backend": "tree_sitter"}
    try:
        parser = ts.Parser(ts.Language(tspython.language()))
        tree = parser.parse(source.encode("utf-8"))
        root = tree.root_node
        if root is None:
            return [], {"parsed": False, "reason": "tree_sitter_parse_error", "parse_backend": "tree_sitter"}

        def has_error(node: Any) -> bool:
            if node.type == "ERROR":
                return True
            return any(has_error(child) for child in node.children)

        if has_error(root):
            return [], {"parsed": False, "reason": "tree_sitter_error_nodes", "parse_backend": "tree_sitter"}
        char_map = _byte_to_char_map(source)
        candidates: list[Candidate] = []

        def walk(node: Any, scope_stack: list[str]) -> None:
            node_type = TS_NODE_MAP.get(node.type)
            next_scope = list(scope_stack)
            if node.type in {"function_definition", "class_definition"}:
                nm = _ts_node_name(node) or "anonymous"
                prefix = "function" if node.type == "function_definition" else "class"
                next_scope = scope_stack + [f"{prefix}:{nm}"]
            if node_type:
                start_char = char_map.get(node.start_byte, 0)
                end_char = char_map.get(node.end_byte, len(source))
                if end_char > start_char:
                    candidates.append(
                        Candidate(
                            start_char=start_char,
                            end_char=end_char,
                            node_type=node_type,
                            scope="::".join(scope_stack),
                        )
                    )
            for child in node.children:
                walk(child, next_scope)

        walk(root, ["module"])
        candidates = sorted(candidates, key=lambda c: (c.start_char, c.end_char, c.node_type))
        if not candidates:
            return [], {"parsed": False, "reason": "no_candidates_tree_sitter", "parse_backend": "tree_sitter"}
        return candidates, {"parsed": True, "reason": "ok", "parse_backend": "tree_sitter"}
    except Exception as exc:
        return [], {"parsed": False, "reason": "tree_sitter_runtime_error", "error": str(exc), "parse_backend": "tree_sitter"}


def _candidates_to_chunks(
    doc: dict[str, Any],
    source: str,
    candidates: list[Candidate],
    max_chars: int,
    min_merge_chars: int,
    parse_backend: str,
) -> list[dict[str, Any]]:
    expanded: list[Candidate] = []
    for candidate in candidates:
        expanded.extend(_split_large_candidate(source, candidate, max_chars=max_chars))
    merged = _merge_tiny_segments(expanded, min_chars=min_merge_chars)

    chunks: list[dict[str, Any]] = []
    for seg in merged:
        text = source[seg.start_char : seg.end_char].strip()
        if not text:
            continue
        scope_header = f"# scope: {seg.scope}\n"
        chunk_text = scope_header + text
        chunk_id = _stable_id(doc["doc_id"], seg.start_char, seg.end_char, seg.scope, chunk_text)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc["doc_id"],
                "chunker": "ast-v2",
                "start_char": seg.start_char,
                "end_char": seg.end_char,
                "content": chunk_text,
                "concept_tags": list(doc.get("concept_tags", [])),
                "content_type": "code",
                "metadata": {
                    "title": doc.get("title", ""),
                    "type": doc.get("type", ""),
                    "node_type": seg.node_type,
                    "scope": seg.scope,
                    "parse_backend": parse_backend,
                },
            }
        )
    return chunks


def chunk_ast(
    doc: dict[str, Any],
    max_chars: int = 450,
    min_merge_chars: int = 120,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    content = str(doc.get("content", "")).strip()
    if not content:
        return [], {"parsed": False, "reason": "empty", "parse_backend": None}

    # Extract text (prose) chunks from non-code portions of the document.
    text_chunks = _extract_text_segments(content, doc, max_chars=max_chars)

    source = _extract_python_source(content)

    # Preferred parser backend: tree-sitter (cAST-aligned), fallback to Python AST.
    ts_candidates, ts_meta = _collect_with_tree_sitter(source)
    if ts_meta.get("parsed"):
        # Tree-sitter can recover from malformed code; enforce strict Python syntax validity
        # before treating the parse as successful for this pipeline.
        try:
            ast.parse(source)
            strict_python_ok = True
        except SyntaxError:
            strict_python_ok = False
        if not strict_python_ok:
            ts_meta = {
                "parsed": False,
                "reason": "tree_sitter_recovered_but_python_invalid",
                "parse_backend": "tree_sitter",
            }
        else:
            code_chunks = _candidates_to_chunks(doc, source, ts_candidates, max_chars=max_chars, min_merge_chars=min_merge_chars, parse_backend="tree_sitter")
            if code_chunks:
                all_chunks = code_chunks + text_chunks
                return all_chunks, {
                    "parsed": True,
                    "reason": "ok",
                    "parse_backend": "tree_sitter",
                    "candidate_count": len(ts_candidates),
                    "chunk_count": len(all_chunks),
                    "code_chunks": len(code_chunks),
                    "text_chunks": len(text_chunks),
                }

    ast_candidates, ast_meta = _collect_with_python_ast(source)
    if ast_meta.get("parsed"):
        code_chunks = _candidates_to_chunks(doc, source, ast_candidates, max_chars=max_chars, min_merge_chars=min_merge_chars, parse_backend="ast")
        if code_chunks:
            all_chunks = code_chunks + text_chunks
            return all_chunks, {
                "parsed": True,
                "reason": "ok",
                "parse_backend": "ast",
                "candidate_count": len(ast_candidates),
                "chunk_count": len(all_chunks),
                "code_chunks": len(code_chunks),
                "text_chunks": len(text_chunks),
                "tree_sitter_reason": ts_meta.get("reason"),
            }

    fallback_chunks = chunk_fixed(doc, chunk_size=400, overlap=80)
    return (
        fallback_chunks,
        {
            "parsed": False,
            "reason": "fallback_fixed",
            "parse_backend": "fixed",
            "tree_sitter_reason": ts_meta.get("reason"),
            "ast_reason": ast_meta.get("reason"),
            "fallback_chunk_count": len(fallback_chunks),
        },
    )
