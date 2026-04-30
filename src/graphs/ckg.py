import hashlib
import json
from pathlib import Path
from typing import Any


PREREQ_ORDER = ["variables", "loops", "conditionals", "functions"]


def _stable_id(kind: str, value: str) -> str:
    return f"{kind}-{hashlib.sha256(f'{kind}:{value}'.encode('utf-8')).hexdigest()[:12]}"


def _normalize_label(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _load_manifest(meta_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((meta_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    docs = manifest.get("documents", [])
    if not isinstance(docs, list):
        raise ValueError("corpus_manifest.json is invalid")
    return docs


def _concept_nodes(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    concepts = sorted(
        {
            _normalize_label(tag)
            for doc in docs
            for tag in doc.get("concept_tags", [])
            if str(tag).strip()
        }
    )
    return [{"id": _stable_id("concept", c), "label": c, "type": "concept"} for c in concepts]


def _subskill_nodes(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes = []
    for c in concepts:
        label = f"{c['label']}-core"
        nodes.append({"id": _stable_id("subskill", label), "label": label, "type": "subskill"})
    return nodes


def _misconception_nodes(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes = []
    for c in concepts:
        label = f"misconception-{c['label']}"
        nodes.append({"id": _stable_id("misconception", label), "label": label, "type": "misconception"})
    return nodes


def _resource_nodes(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for doc in sorted(docs, key=lambda d: d["doc_id"]):
        label = doc["title"]
        nodes.append({"id": _stable_id("resource", doc["doc_id"]), "label": label, "type": "resource", "doc_id": doc["doc_id"]})
    return nodes


def _concept_id_map(concepts: list[dict[str, Any]]) -> dict[str, str]:
    return {c["label"]: c["id"] for c in concepts}


def _resource_id_map(resources: list[dict[str, Any]]) -> dict[str, str]:
    return {r["doc_id"]: r["id"] for r in resources}


def _build_edges(
    docs: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    subskills: list[dict[str, Any]],
    misconceptions: list[dict[str, Any]],
    resources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    concept_ids = _concept_id_map(concepts)
    resource_ids = _resource_id_map(resources)
    subskill_ids = {n["label"]: n["id"] for n in subskills}
    misconception_ids = {n["label"]: n["id"] for n in misconceptions}

    # prerequisite edges from predefined progression order
    for i in range(len(PREREQ_ORDER) - 1):
        src = PREREQ_ORDER[i]
        dst = PREREQ_ORDER[i + 1]
        if src in concept_ids and dst in concept_ids:
            edges.append(
                {
                    "id": _stable_id("edge-prerequisite", f"{src}->{dst}"),
                    "type": "prerequisite",
                    "source": concept_ids[src],
                    "target": concept_ids[dst],
                }
            )

    # related edges via co-occurrence in the same resource
    related_pairs = set()
    for doc in docs:
        tags = sorted({_normalize_label(t) for t in doc.get("concept_tags", [])})
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                if tags[i] in concept_ids and tags[j] in concept_ids:
                    related_pairs.add((tags[i], tags[j]))
    for a, b in sorted(related_pairs):
        edges.append(
            {
                "id": _stable_id("edge-related", f"{a}<->{b}"),
                "type": "related",
                "source": concept_ids[a],
                "target": concept_ids[b],
            }
        )

    # subskill and misconception attachments
    for concept in concepts:
        subskill_label = f"{concept['label']}-core"
        edges.append(
            {
                "id": _stable_id("edge-subskill", f"{concept['label']}->{subskill_label}"),
                "type": "related",
                "source": concept["id"],
                "target": subskill_ids[subskill_label],
            }
        )

        misconception_label = f"misconception-{concept['label']}"
        edges.append(
            {
                "id": _stable_id("edge-misconception", f"{misconception_label}->{concept['label']}"),
                "type": "misconception-of",
                "source": misconception_ids[misconception_label],
                "target": concept["id"],
            }
        )

    # resource addressing concept links
    for doc in docs:
        resource_id = resource_ids[doc["doc_id"]]
        for tag in sorted({_normalize_label(t) for t in doc.get("concept_tags", [])}):
            if tag in concept_ids:
                edges.append(
                    {
                        "id": _stable_id("edge-addresses", f"{doc['doc_id']}->{tag}"),
                        "type": "addresses",
                        "source": resource_id,
                        "target": concept_ids[tag],
                    }
                )
    return sorted(edges, key=lambda e: (e["type"], e["id"]))


def _has_cycle(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> bool:
    concept_ids = {n["id"] for n in nodes if n["type"] == "concept"}
    adj = {node_id: [] for node_id in concept_ids}
    for edge in edges:
        if edge["type"] == "prerequisite" and edge["source"] in concept_ids and edge["target"] in concept_ids:
            adj[edge["source"]].append(edge["target"])

    visited: dict[str, int] = {}

    def dfs(node: str) -> bool:
        state = visited.get(node, 0)
        if state == 1:
            return True
        if state == 2:
            return False
        visited[node] = 1
        for nxt in adj[node]:
            if dfs(nxt):
                return True
        visited[node] = 2
        return False

    return any(dfs(n) for n in adj if visited.get(n, 0) == 0)


def build_ckg(meta_dir: Path = Path("data/corpus_meta"), out_dir: Path = Path("graphs")) -> dict[str, Any]:
    docs = _load_manifest(meta_dir)
    concepts = _concept_nodes(docs)
    subskills = _subskill_nodes(concepts)
    misconceptions = _misconception_nodes(concepts)
    resources = _resource_nodes(docs)
    nodes = sorted(concepts + subskills + misconceptions + resources, key=lambda n: (n["type"], n["id"]))
    edges = _build_edges(docs, concepts, subskills, misconceptions, resources)

    concept_ids = {n["id"] for n in concepts}
    has_resource_edge = {cid: False for cid in concept_ids}
    for e in edges:
        if e["type"] == "addresses" and e["target"] in concept_ids:
            has_resource_edge[e["target"]] = True

    validation = {
        "prerequisite_cycle": _has_cycle(nodes, edges),
        "concepts_without_resource": sorted([cid for cid, ok in has_resource_edge.items() if not ok]),
    }

    graph = {"graph_type": "CKG", "node_types": ["concept", "subskill", "misconception", "resource"], "nodes": nodes, "edges": edges, "validation": validation}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ckg.json").write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "ckg_validation_report.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    return graph
