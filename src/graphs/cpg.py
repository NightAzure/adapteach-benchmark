import ast
import hashlib
import json
from pathlib import Path
from typing import Any


def _stable_id(kind: str, basis: str) -> str:
    return f"{kind}-{hashlib.sha256(basis.encode('utf-8')).hexdigest()[:14]}"


def _strip_scope_header(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# scope: "):
        return "\n".join(lines[1:])
    return text


def _node(kind: str, basis: str, **kwargs: Any) -> dict[str, Any]:
    node_id = _stable_id(kind, basis)
    payload = {"id": node_id, "type": kind}
    payload.update(kwargs)
    return payload


def _edge(kind: str, src: str, dst: str, basis: str, **kwargs: Any) -> dict[str, Any]:
    edge_id = _stable_id(f"edge-{kind}", basis)
    payload = {"id": edge_id, "type": kind, "source": src, "target": dst}
    payload.update(kwargs)
    return payload


class _StmtCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.statement_nodes: list[tuple[str, ast.AST]] = []
        self.functions: list[ast.FunctionDef] = []
        self.classes: list[ast.ClassDef] = []
        self.calls: list[tuple[str, str]] = []  # (caller_name, callee_name)
        self.def_use: list[tuple[str, str, str]] = []  # (scope_name, def_name, use_name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.functions.append(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.classes.append(node)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        name = getattr(node.targets[0], "id", None) if node.targets else None
        if isinstance(name, str):
            self.statement_nodes.append((f"assign:{name}:{node.lineno}", node))
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> Any:
        self.statement_nodes.append((f"expr:{node.lineno}", node))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        callee = None
        if isinstance(node.func, ast.Name):
            callee = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee = node.func.attr
        if callee:
            self.calls.append((f"line:{node.lineno}", callee))
        self.generic_visit(node)


def build_cpg(
    chunk_manifest_path: Path = Path("data/corpus_meta/chunk_manifest.json"),
    out_dir: Path = Path("graphs"),
) -> dict[str, Any]:
    manifest = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
    chunks = manifest.get("chunks", [])
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    function_name_to_ids: dict[str, list[str]] = {}
    call_collision_log: list[dict[str, Any]] = []

    for chunk in sorted(chunks, key=lambda c: c["chunk_id"]):
        chunk_id = chunk["chunk_id"]
        chunk_node = _node("chunk", chunk_id, chunk_id=chunk_id, doc_id=chunk["doc_id"])
        nodes[chunk_node["id"]] = chunk_node

        if chunk.get("content_type") != "code":
            placeholder = _node("statement", f"{chunk_id}|statement|non-code", label="non-code", chunk_id=chunk_id, lineno=None)
            nodes[placeholder["id"]] = placeholder
            e = _edge("CONTAINS", chunk_node["id"], placeholder["id"], f"{chunk_node['id']}->{placeholder['id']}")
            edges[e["id"]] = e
            continue

        source = _strip_scope_header(chunk.get("content", ""))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            placeholder = _node("statement", f"{chunk_id}|statement|parse-error", label="parse-error", chunk_id=chunk_id, lineno=None)
            nodes[placeholder["id"]] = placeholder
            e = _edge("CONTAINS", chunk_node["id"], placeholder["id"], f"{chunk_node['id']}->{placeholder['id']}")
            edges[e["id"]] = e
            continue

        collector = _StmtCollector()
        collector.visit(tree)

        class_nodes: list[dict[str, Any]] = []
        function_nodes: list[dict[str, Any]] = []
        stmt_nodes: list[dict[str, Any]] = []

        for cls in collector.classes:
            cls_basis = f"{chunk_id}|class|{cls.name}|{cls.lineno}"
            cls_node = _node("class", cls_basis, name=cls.name, chunk_id=chunk_id, lineno=cls.lineno)
            nodes[cls_node["id"]] = cls_node
            class_nodes.append(cls_node)
            e = _edge("CONTAINS", chunk_node["id"], cls_node["id"], f"{chunk_node['id']}->{cls_node['id']}")
            edges[e["id"]] = e

        for fn in collector.functions:
            fn_basis = f"{chunk_id}|function|{fn.name}|{fn.lineno}"
            fn_node = _node("function", fn_basis, name=fn.name, chunk_id=chunk_id, lineno=fn.lineno)
            nodes[fn_node["id"]] = fn_node
            function_nodes.append(fn_node)
            function_name_to_ids.setdefault(fn.name, []).append(fn_node["id"])
            e = _edge("CONTAINS", chunk_node["id"], fn_node["id"], f"{chunk_node['id']}->{fn_node['id']}")
            edges[e["id"]] = e

        for label, stmt in collector.statement_nodes:
            st_basis = f"{chunk_id}|statement|{label}"
            st_node = _node("statement", st_basis, label=label, chunk_id=chunk_id, lineno=getattr(stmt, "lineno", None))
            nodes[st_node["id"]] = st_node
            stmt_nodes.append(st_node)
            e = _edge("CONTAINS", chunk_node["id"], st_node["id"], f"{chunk_node['id']}->{st_node['id']}")
            edges[e["id"]] = e

        if not class_nodes and not function_nodes and not stmt_nodes:
            placeholder = _node("statement", f"{chunk_id}|statement|raw-block", label="raw-block", chunk_id=chunk_id, lineno=None)
            nodes[placeholder["id"]] = placeholder
            stmt_nodes.append(placeholder)
            e = _edge("CONTAINS", chunk_node["id"], placeholder["id"], f"{chunk_node['id']}->{placeholder['id']}")
            edges[e["id"]] = e

        stmt_by_line = {s.get("lineno"): s["id"] for s in stmt_nodes if isinstance(s.get("lineno"), int)}
        defs_by_name: dict[str, str] = {}
        for label, stmt in collector.statement_nodes:
            if label.startswith("assign:"):
                parts = label.split(":")
                if len(parts) >= 2:
                    defs_by_name[parts[1]] = stmt_by_line.get(getattr(stmt, "lineno", None), "")

        # DEF-USE within local chunk scope, name-based conservative
        for label, stmt in collector.statement_nodes:
            use_names = []
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    use_names.append(node.id)
            for name in sorted(set(use_names)):
                def_node = defs_by_name.get(name)
                use_node = stmt_by_line.get(getattr(stmt, "lineno", None))
                if def_node and use_node and def_node != use_node:
                    e = _edge("DEF-USE", def_node, use_node, f"{def_node}->{use_node}:{name}", symbol=name)
                    edges[e["id"]] = e

        # CALLS conservative: only when callee resolvable by unique function name
        for caller_line, callee_name in collector.calls:
            callees = function_name_to_ids.get(callee_name, [])
            if len(callees) == 1:
                caller_stmt = None
                if caller_line.startswith("line:"):
                    try:
                        ln = int(caller_line.split(":")[1])
                        caller_stmt = stmt_by_line.get(ln)
                    except ValueError:
                        caller_stmt = None
                if caller_stmt:
                    e = _edge("CALLS", caller_stmt, callees[0], f"{caller_stmt}->{callees[0]}:{callee_name}", callee=callee_name)
                    edges[e["id"]] = e
            elif len(callees) > 1:
                call_collision_log.append({"callee_name": callee_name, "candidate_count": len(callees)})

        # Class->function containment when function declared after class header within same chunk
        for cls_node in class_nodes:
            cls_line = cls_node.get("lineno") or 0
            for fn_node in function_nodes:
                fn_line = fn_node.get("lineno") or 0
                if fn_line > cls_line:
                    e = _edge("CONTAINS", cls_node["id"], fn_node["id"], f"{cls_node['id']}->{fn_node['id']}:nested")
                    edges[e["id"]] = e

    node_list = sorted(nodes.values(), key=lambda n: (n["type"], n["id"]))
    edge_list = sorted(edges.values(), key=lambda e: (e["type"], e["id"]))
    node_ids = {n["id"] for n in node_list}
    connected = {e["source"] for e in edge_list} | {e["target"] for e in edge_list}
    orphans = sorted(node_ids - connected)

    edge_type_counts: dict[str, int] = {}
    for e in edge_list:
        edge_type_counts[e["type"]] = edge_type_counts.get(e["type"], 0) + 1

    # Bounded sanity: edges should not blow up compared to node count
    bounded = len(edge_list) <= max(1, len(node_list) * 12)
    sanity = {
        "node_count": len(node_list),
        "edge_count": len(edge_list),
        "edge_type_counts": edge_type_counts,
        "orphan_node_count": len(orphans),
        "orphan_node_ids": orphans[:50],
        "bounded": bounded,
        "call_collision_log": call_collision_log[:100],
    }

    graph = {"graph_type": "CPG", "node_types": ["chunk", "function", "class", "statement"], "edge_types": ["CONTAINS", "CALLS", "DEF-USE"], "nodes": node_list, "edges": edge_list, "sanity": sanity}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cpg.json").write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "cpg_sanity_report.json").write_text(json.dumps(sanity, indent=2, sort_keys=True), encoding="utf-8")
    return graph
