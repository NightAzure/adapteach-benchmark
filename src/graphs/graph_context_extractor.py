"""
Graph-as-Context extractor for Config F.

Instead of adding graph-discovered chunks to the retrieval pool (Config E's approach),
this module annotates the already-retrieved chunks with:
  - Which CKG concept nodes they relate to
  - Pedagogical pitfall warnings from those concepts
  - CPG variable/call dependencies between the retrieved chunks

The output is a dict that gets injected as a structured "Graph Context" section
in the LLM prompt, without competing with the primary retrieval results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _concept_labels_for_chunks(
    chunk_ids: set[str],
    ckg: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Return (concept_labels, pitfalls) linked to the given chunk_ids via CKG."""
    # Map doc_id → chunk_ids (CKG links resources to docs, not chunk_ids directly)
    # We match by doc_id: resource nodes carry doc_id, edges link resource → concept.
    resource_nodes = {n["id"]: n for n in ckg.get("nodes", []) if n.get("type") == "resource"}
    concept_nodes = {n["id"]: n for n in ckg.get("nodes", []) if n.get("type") == "concept"}

    # We need doc_ids from the candidate chunks — stored in the chunk objects.
    # The caller passes pre-resolved chunk objects from the candidates list.
    # Here we resolve via CKG resource nodes that carry doc_ids.
    # Strategy: collect all concept IDs that are linked to any resource whose
    # doc_id appears in the retrieved candidates.
    candidate_doc_ids: set[str] = set()
    for node in resource_nodes.values():
        # We'll populate this after receiving doc_ids from candidates in the outer function.
        pass  # filled below

    # Build resource_id → concept_ids map from "addresses" edges
    resource_to_concepts: dict[str, set[str]] = {}
    for e in ckg.get("edges", []):
        if e.get("type") == "addresses":
            src = e.get("source", "")
            tgt = e.get("target", "")
            if src not in resource_to_concepts:
                resource_to_concepts[src] = set()
            resource_to_concepts[src].add(tgt)

    return resource_to_concepts, concept_nodes


def extract_graph_metadata(
    candidates: list[dict[str, Any]],
    query: str,
    graphs_dir: Path = Path("graphs"),
    max_pitfalls: int = 4,
    max_deps: int = 5,
) -> dict[str, Any]:
    """
    Given the top-k retrieved candidates, return structured graph metadata:

    {
        "concepts": ["loops", "iteration"],
        "pitfalls": ["Off-by-one: range(n) stops at n-1", ...],
        "cpg_deps": ["chunk-abc calls chunk-def  (CALLS edge)", ...],
    }
    """
    ckg_path = graphs_dir / "ckg.json"
    cpg_path = graphs_dir / "cpg.json"

    concepts: list[str] = []
    pitfalls: list[str] = []
    cpg_deps: list[str] = []

    # ── CKG: concept labels + pitfalls ──────────────────────────────────────
    if ckg_path.exists():
        ckg = _load_json(ckg_path)

        resource_nodes = {n["id"]: n for n in ckg.get("nodes", []) if n.get("type") == "resource"}
        concept_nodes  = {n["id"]: n for n in ckg.get("nodes", []) if n.get("type") == "concept"}

        # Collect doc_ids from retrieved candidates
        candidate_doc_ids = {c.get("doc_id") for c in candidates if c.get("doc_id")}

        # Find resource node IDs whose doc_id is in our retrieved set
        relevant_resource_ids = {
            rid for rid, rnode in resource_nodes.items()
            if rnode.get("doc_id") in candidate_doc_ids
        }

        # Collect concept IDs linked to those resources via "addresses" edges
        linked_concept_ids: set[str] = set()
        for e in ckg.get("edges", []):
            if e.get("type") == "addresses" and e.get("source") in relevant_resource_ids:
                linked_concept_ids.add(e.get("target"))

        # Also expand one hop via "related" / "prerequisite" / "misconception-of" edges
        # so we surface slightly broader pedagogical context
        expanded_concept_ids = set(linked_concept_ids)
        for e in ckg.get("edges", []):
            etype = e.get("type", "")
            if etype in {"related", "prerequisite"}:
                if e.get("source") in linked_concept_ids:
                    expanded_concept_ids.add(e.get("target"))

        # Resolve labels and pitfalls, sorted by label for determinism
        seen_labels: set[str] = set()
        seen_pitfalls: set[str] = set()
        # Primary concepts first (directly linked), then expanded
        for cid in sorted(linked_concept_ids) + sorted(expanded_concept_ids - linked_concept_ids):
            node = concept_nodes.get(cid)
            if not node:
                continue
            label = node.get("label", "")
            if not label or label in seen_labels:
                continue
            seen_labels.add(label)
            concepts.append(label)
            for p in node.get("pitfalls", []):
                if p not in seen_pitfalls and len(pitfalls) < max_pitfalls:
                    seen_pitfalls.add(p)
                    pitfalls.append(p)

    # ── CPG: call / def-use dependencies between retrieved chunks ────────────
    if cpg_path.exists():
        cpg = _load_json(cpg_path)

        candidate_chunk_ids = {c["chunk_id"] for c in candidates if c.get("chunk_id")}

        # Map CPG node id → chunk_id
        node_to_chunk: dict[str, str] = {
            n["id"]: n["chunk_id"]
            for n in cpg.get("nodes", [])
            if n.get("chunk_id")
        }
        # Map chunk_id → short label (first 60 chars of first line)
        chunk_labels: dict[str, str] = {}
        for c in candidates:
            cid = c.get("chunk_id", "")
            text = (c.get("text") or "").strip()
            first_line = text.split("\n")[0][:60].strip() if text else ""
            chunk_labels[cid] = first_line or cid[:16]

        # Find CPG node IDs for our candidate chunks
        candidate_node_ids = {
            nid for nid, cid in node_to_chunk.items()
            if cid in candidate_chunk_ids
        }

        seen_deps: set[str] = set()
        for e in cpg.get("edges", []):
            etype = e.get("type", "")
            if etype not in {"CALLS", "DEF-USE"}:
                continue
            src_node = e.get("source")
            tgt_node = e.get("target")
            # Both endpoints must be in our candidate set (inter-chunk relationship)
            if src_node not in candidate_node_ids or tgt_node not in candidate_node_ids:
                continue
            src_cid = node_to_chunk.get(src_node, "")
            tgt_cid = node_to_chunk.get(tgt_node, "")
            if not src_cid or not tgt_cid or src_cid == tgt_cid:
                continue
            src_label = chunk_labels.get(src_cid, src_cid[:16])
            tgt_label = chunk_labels.get(tgt_cid, tgt_cid[:16])
            dep_str = f'"{src_label[:50]}" {etype} "{tgt_label[:50]}"'
            if dep_str not in seen_deps and len(cpg_deps) < max_deps:
                seen_deps.add(dep_str)
                cpg_deps.append(dep_str)

    return {
        "concepts": concepts,
        "pitfalls": pitfalls,
        "cpg_deps": cpg_deps,
    }
