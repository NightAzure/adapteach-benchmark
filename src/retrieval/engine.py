import json
import math
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.graphs.query_mapper import map_query_to_concepts
from src.indexing.build_indexes import embed_text

CORE_TOPICS = {"variables", "loops", "conditionals", "functions"}
TOPIC_KEYWORDS = {
    "variable": "variables",
    "variables": "variables",
    "assignment": "variables",
    "scope": "variables",
    "loop": "loops",
    "loops": "loops",
    "for": "loops",
    "while": "loops",
    "conditional": "conditionals",
    "conditionals": "conditionals",
    "if": "conditionals",
    "elif": "conditionals",
    "else": "conditionals",
    "function": "functions",
    "functions": "functions",
    "def": "functions",
    "parameter": "functions",
    "return": "functions",
}
_EMBED_CACHE: dict[str, list[float]] = {}


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _tokenize(query: str) -> list[str]:
    return [token.lower() for token in query.replace("\n", " ").split() if token.strip()]


def _embed_cached(text: str) -> list[float]:
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    cached = _EMBED_CACHE.get(key)
    if cached is not None:
        return cached
    vec = embed_text(text)
    _EMBED_CACHE[key] = vec
    return vec


def _cheap_text_match_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = {t.strip("`'\",.:;()[]{}").lower() for t in text.split() if t.strip()}
    if not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap / max(1, len(query_tokens))


def _prefilter_graph_candidates(query: str, candidates: list[dict[str, Any]], limit: int = 48) -> list[dict[str, Any]]:
    if len(candidates) <= limit:
        return candidates
    query_tokens = set(_tokenize(query))
    ranked = []
    for c in candidates:
        topic_bonus = float(c.get("topic_overlap", 0.0))
        lexical = _cheap_text_match_score(query_tokens, c.get("text", ""))
        rank_score = (0.6 * topic_bonus) + (0.4 * lexical)
        ranked.append((rank_score, c))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:limit]]


def _infer_query_topics(query: str, explicit_topic: str | None = None) -> set[str]:
    topics: set[str] = set()
    if explicit_topic:
        t = explicit_topic.strip().lower()
        if t in CORE_TOPICS:
            topics.add(t)
    for token in _tokenize(query):
        mapped = TOPIC_KEYWORDS.get(token)
        if mapped:
            topics.add(mapped)
    return topics


def _topic_overlap(tags: list[str], intent_topics: set[str]) -> float:
    if not intent_topics:
        return 0.0
    norm = {str(t).strip().lower() for t in tags}
    return 1.0 if (norm & intent_topics) else 0.0


def _apply_topic_boost(
    rows: list[dict[str, Any]],
    chunk_map: dict[str, dict[str, Any]],
    intent_topics: set[str],
    boost: float = 0.18,
) -> list[dict[str, Any]]:
    if not rows:
        return rows
    adjusted: list[dict[str, Any]] = []
    for row in rows:
        chunk = chunk_map.get(row["chunk_id"], {})
        overlap = _topic_overlap(chunk.get("concept_tags", []), intent_topics)
        score = float(row.get("score", 0.0))
        if overlap > 0:
            score = score * (1.0 + boost)
        new_row = dict(row)
        new_row["score"] = score
        new_row["topic_overlap"] = overlap
        adjusted.append(new_row)
    adjusted.sort(key=lambda x: x["score"], reverse=True)
    for i, row in enumerate(adjusted, start=1):
        row["rank"] = i
    return adjusted


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_index_dir(indexes_dir: Path) -> Path | None:
    dirs = [p for p in indexes_dir.glob("*") if p.is_dir() and (p / "manifest.json").exists()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def _resolve_index_dir(indexes_dir: Path, index_hash: str | None = None) -> Path | None:
    if index_hash:
        prefix = index_hash[:16]
        candidate = indexes_dir / prefix
        if candidate.exists() and (candidate / "manifest.json").exists():
            return candidate
    return _latest_index_dir(indexes_dir)


def _load_chunk_manifest(meta_dir: Path) -> dict[str, Any]:
    chunk_manifest_path = meta_dir / "chunk_manifest.json"
    if not chunk_manifest_path.exists():
        return {"chunks": [], "chunker": "unknown"}
    return _load_json(chunk_manifest_path)


def _chunks_by_id(meta_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = _load_chunk_manifest(meta_dir)
    return {c["chunk_id"]: c for c in manifest.get("chunks", [])}


def _filter_by_chunking(chunks: list[dict[str, Any]], chunking_mode: str) -> list[dict[str, Any]]:
    if chunking_mode == "ast":
        # Include both ast-produced code chunks and text chunks from the same pipeline
        return [c for c in chunks if c.get("chunker", "").startswith(("ast", "text"))]
    if chunking_mode == "fixed":
        return [c for c in chunks if c.get("chunker", "").startswith("fixed")]
    return chunks


def _dense_rank(
    query: str,
    vectors: list[dict[str, Any]],
    chunk_map: dict[str, dict[str, Any]],
    chunking_mode: str,
    intent_topics: set[str] | None = None,
) -> list[dict[str, Any]]:
    # Detect index dimension from first vector to match embedding method
    index_dim = 64
    if vectors and vectors[0].get("vector"):
        index_dim = len(vectors[0]["vector"])
    qvec = embed_text(query, dim=index_dim)
    # If dimensions don't match (e.g. semantic model unavailable at query time
    # but was used at index time), fall back gracefully
    if vectors and vectors[0].get("vector") and len(qvec) != len(vectors[0]["vector"]):
        from src.indexing.build_indexes import embed_text_hash
        qvec = embed_text_hash(query, dim=index_dim)
    allowed_chunks = set(c["chunk_id"] for c in _filter_by_chunking(list(chunk_map.values()), chunking_mode))
    rows = []
    for row in vectors:
        chunk_id = row["chunk_id"]
        if allowed_chunks and chunk_id not in allowed_chunks:
            continue
        score = _dot(qvec, row["vector"])
        rows.append({"chunk_id": chunk_id, "score": float(score), "method": "dense"})
    rows.sort(key=lambda x: x["score"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return _apply_topic_boost(rows, chunk_map=chunk_map, intent_topics=intent_topics or set())


def _bm25_rank(
    query: str,
    bm25_index: dict[str, Any],
    chunk_map: dict[str, dict[str, Any]],
    chunking_mode: str,
    intent_topics: set[str] | None = None,
) -> list[dict[str, Any]]:
    doc_tokens = bm25_index.get("doc_tokens", {})
    df = bm25_index.get("document_frequency", {})
    k1 = bm25_index.get("bm25_params", {}).get("k1", 1.2)
    b = bm25_index.get("bm25_params", {}).get("b", 0.75)
    avgdl = max(1e-9, float(bm25_index.get("avgdl", 1.0)))
    n_docs = max(1, int(bm25_index.get("doc_count", len(doc_tokens))))
    query_terms = _tokenize(query)

    allowed_chunks = set(c["chunk_id"] for c in _filter_by_chunking(list(chunk_map.values()), chunking_mode))
    rows: list[dict[str, Any]] = []
    for chunk_id, tokens in doc_tokens.items():
        if allowed_chunks and chunk_id not in allowed_chunks:
            continue
        tf: dict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        dl = max(1, len(tokens))
        score = 0.0
        for q in query_terms:
            f = tf.get(q, 0)
            if f == 0:
                continue
            dfi = int(df.get(q, 0))
            idf = math.log(1 + (n_docs - dfi + 0.5) / (dfi + 0.5))
            denom = f + k1 * (1 - b + b * dl / avgdl)
            score += idf * ((f * (k1 + 1)) / max(1e-9, denom))
        rows.append({"chunk_id": chunk_id, "score": float(score), "method": "bm25"})
    rows.sort(key=lambda x: x["score"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return _apply_topic_boost(rows, chunk_map=chunk_map, intent_topics=intent_topics or set())


def _rrf_fuse(
    dense_rows: list[dict[str, Any]],
    bm25_rows: list[dict[str, Any]],
    k: int,
    rrf_k: int = 60,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dense_rank = {r["chunk_id"]: r["rank"] for r in dense_rows}
    bm25_rank = {r["chunk_id"]: r["rank"] for r in bm25_rows}
    dense_topic = {r["chunk_id"]: float(r.get("topic_overlap", 0.0)) for r in dense_rows}
    bm25_topic = {r["chunk_id"]: float(r.get("topic_overlap", 0.0)) for r in bm25_rows}
    chunk_ids = sorted(set(dense_rank) | set(bm25_rank))
    fused = []
    debug_rows = []
    for cid in chunk_ids:
        d = dense_rank.get(cid)
        b = bm25_rank.get(cid)
        score = 0.0
        if d is not None:
            score += 1.0 / (rrf_k + d)
        if b is not None:
            score += 1.0 / (rrf_k + b)
        # Preserve topic overlap so downstream reranking can use query-topic intent.
        topic_overlap = max(dense_topic.get(cid, 0.0), bm25_topic.get(cid, 0.0))
        fused.append(
            {
                "chunk_id": cid,
                "score": score,
                "method": "hybrid_rrf",
                "topic_overlap": topic_overlap,
            }
        )
        debug_rows.append(
            {
                "chunk_id": cid,
                "dense_rank": d,
                "bm25_rank": b,
                "topic_overlap": round(topic_overlap, 6),
                "fused_score": round(score, 10),
            }
        )

    fused.sort(key=lambda x: x["score"], reverse=True)
    for i, row in enumerate(fused, start=1):
        row["rank"] = i
    return fused[:k], {"rrf_k": rrf_k, "rows": debug_rows[: max(25, k * 3)]}


def _attach_chunk_data(rows: list[dict[str, Any]], chunk_map: dict[str, dict[str, Any]], include_method_debug: bool = False) -> list[dict[str, Any]]:
    attached = []
    for row in rows:
        chunk = chunk_map.get(row["chunk_id"])
        if not chunk:
            continue
        item = {
            "chunk_id": row["chunk_id"],
            "score": round(float(row["score"]), 8),
            "text": chunk.get("content", ""),
            "doc_id": chunk.get("doc_id"),
            "provenance": {
                "doc_id": chunk.get("doc_id"),
                "title": chunk.get("metadata", {}).get("title", ""),
                "chunker": chunk.get("chunker", ""),
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
            },
            "evidence": [],
            "topic_overlap": float(row.get("topic_overlap", 0.0)),
        }
        if include_method_debug:
            item["retrieval"] = {"method": row.get("method"), "rank": row.get("rank")}
        attached.append(item)
    return attached


def retrieve_candidates(
    query: str,
    k: int,
    chunking_mode: str,
    hybrid: bool,
    query_topic: str | None = None,
    meta_dir: Path = Path("data/corpus_meta"),
    indexes_dir: Path = Path("indexes"),
    index_hash: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunk_map = _chunks_by_id(meta_dir)
    intent_topics = _infer_query_topics(query, explicit_topic=query_topic)
    index_dir = _resolve_index_dir(indexes_dir, index_hash=index_hash)
    if index_dir is None:
        fallback = []
        for c in _filter_by_chunking(list(chunk_map.values()), chunking_mode)[:k]:
            fallback.append(
                {
                    "chunk_id": c["chunk_id"],
                    "score": 0.0,
                    "text": c["content"],
                    "doc_id": c["doc_id"],
                    "provenance": {
                        "doc_id": c["doc_id"],
                        "title": c.get("metadata", {}).get("title", ""),
                        "chunker": c.get("chunker", ""),
                        "start_char": c.get("start_char"),
                        "end_char": c.get("end_char"),
                    },
                    "evidence": [{"kind": "fallback", "detail": "index_missing"}],
                }
            )
        return fallback, {"index_found": False, "retrieval_mode": "fallback", "top_k": k}

    vector_index = _load_json(index_dir / "vector_index.json")
    bm25_index = _load_json(index_dir / "bm25_index.json")
    dense_rows = _dense_rank(
        query,
        vector_index.get("vectors", []),
        chunk_map,
        chunking_mode=chunking_mode,
        intent_topics=intent_topics,
    )
    if not hybrid:
        top_dense = dense_rows[:k]
        candidates = _attach_chunk_data(top_dense, chunk_map, include_method_debug=True)
        return candidates, {
            "index_found": True,
            "index_id": index_dir.name,
            "retrieval_mode": "dense",
            "top_k": k,
            "intent_topics": sorted(intent_topics),
            "dense_topk": [{"chunk_id": r["chunk_id"], "score": round(r["score"], 8), "rank": r["rank"]} for r in top_dense],
        }

    bm25_rows = _bm25_rank(
        query,
        bm25_index,
        chunk_map,
        chunking_mode=chunking_mode,
        intent_topics=intent_topics,
    )
    fused_rows, fusion_debug = _rrf_fuse(dense_rows, bm25_rows, k=k)
    candidates = _attach_chunk_data(fused_rows, chunk_map, include_method_debug=True)
    return candidates, {
        "index_found": True,
        "index_id": index_dir.name,
        "retrieval_mode": "hybrid",
        "top_k": k,
        "intent_topics": sorted(intent_topics),
        "dense_topk": [{"chunk_id": r["chunk_id"], "score": round(r["score"], 8), "rank": r["rank"]} for r in dense_rows[:k]],
        "bm25_topk": [{"chunk_id": r["chunk_id"], "score": round(r["score"], 8), "rank": r["rank"]} for r in bm25_rows[:k]],
        "fusion_debug": fusion_debug,
    }


def _score_graph_chunks(
    query: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score graph-expanded chunks against the query using semantic similarity.

    Instead of inserting graph-discovered chunks with score=0.0, we compute a
    real embedding similarity so they can compete fairly with retrieval results
    during reranking.
    """
    if not chunks:
        return chunks
    qvec = _embed_cached(query)
    scored = []
    for c in chunks:
        text = c.get("text", "") or ""
        if not text.strip():
            scored.append(c)
            continue
        # Keep scoring stable while limiting embedding cost for very long chunks.
        cvec = _embed_cached(text[:900])
        # Ensure dimensions match
        if len(cvec) == len(qvec):
            sim = max(0.0, _dot(qvec, cvec))
        else:
            sim = 0.0
        updated = dict(c)
        updated["score"] = float(sim)
        scored.append(updated)
    return scored


def expand_with_graphs(
    query: str,
    candidates: list[dict[str, Any]],
    max_extra: int = 5,
    query_topic: str | None = None,
    graphs_dir: Path = Path("graphs"),
    meta_dir: Path = Path("data/corpus_meta"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ckg_path = graphs_dir / "ckg.json"
    cpg_path = graphs_dir / "cpg.json"
    if not ckg_path.exists() or not cpg_path.exists():
        return candidates, {"graph_used": False, "reason": "graph_files_missing"}

    try:
        ckg = _load_json(ckg_path)
        cpg = _load_json(cpg_path)
    except Exception as exc:
        return candidates, {"graph_used": False, "reason": "graph_load_error", "error": str(exc)}
    chunk_map = _chunks_by_id(meta_dir)
    existing = {c["chunk_id"] for c in candidates}
    added: list[dict[str, Any]] = []
    evidence_paths: list[dict[str, Any]] = []

    mapped = map_query_to_concepts(query, ckg_path=ckg_path, top_k=5)
    mapped_concepts = [x["concept"] for x in mapped.get("top_concepts", [])]
    intent_topics = _infer_query_topics(query, explicit_topic=query_topic)
    # Keep expansion centered on the objective's core concepts to reduce graph drift.
    top_concepts = sorted(set(mapped_concepts) & CORE_TOPICS)
    if not top_concepts:
        top_concepts = sorted(intent_topics) if intent_topics else mapped_concepts[:3]

    concept_by_label = {n["label"]: n["id"] for n in ckg.get("nodes", []) if n.get("type") == "concept"}
    resource_nodes = {n["id"]: n for n in ckg.get("nodes", []) if n.get("type") == "resource"}
    target_concept_ids = {concept_by_label[c] for c in top_concepts if c in concept_by_label}

    # CKG expansion: direct concept resource links and one-hop related/prereq neighbors.
    neighbor_concepts = set(target_concept_ids)
    for e in ckg.get("edges", []):
        if e.get("type") in {"related", "prerequisite"}:
            if e.get("source") in target_concept_ids:
                neighbor_concepts.add(e.get("target"))
            if e.get("target") in target_concept_ids:
                neighbor_concepts.add(e.get("source"))

    ckg_doc_ids: set[str] = set()
    for e in ckg.get("edges", []):
        if e.get("type") == "addresses" and e.get("target") in neighbor_concepts:
            resource = resource_nodes.get(e.get("source"))
            if resource and resource.get("doc_id"):
                ckg_doc_ids.add(resource["doc_id"])

    # Collect all CKG candidate chunks, score semantically, pick best
    ckg_candidates: list[dict[str, Any]] = []
    for chunk in chunk_map.values():
        if chunk["doc_id"] in ckg_doc_ids and chunk["chunk_id"] not in existing:
            overlap = _topic_overlap(chunk.get("concept_tags", []), intent_topics)
            ckg_candidates.append({
                "chunk_id": chunk["chunk_id"],
                "score": 0.0,
                "text": chunk.get("content", ""),
                "doc_id": chunk.get("doc_id"),
                "provenance": {
                    "doc_id": chunk.get("doc_id"),
                    "title": chunk.get("metadata", {}).get("title", ""),
                    "chunker": chunk.get("chunker", ""),
                    "start_char": chunk.get("start_char"),
                    "end_char": chunk.get("end_char"),
                },
                "evidence": [{"kind": "ckg", "detail": "concept-neighbor-resource"}],
                "topic_overlap": overlap,
            })
    ckg_candidates = _prefilter_graph_candidates(query, ckg_candidates, limit=48)
    ckg_candidates = _score_graph_chunks(query, ckg_candidates)
    ckg_candidates.sort(key=lambda c: (c.get("topic_overlap", 0.0), c["score"]), reverse=True)
    # Gate: only admit CKG-expanded chunks that are also semantically relevant to the
    # query (cosine sim >= 0.35).  Without this gate, neighboring concept chunks get a
    # free +0.18 rerank bonus regardless of whether they actually answer the query,
    # which displaces high-precision retrieval results and hurts nDCG for single-concept
    # pedagogical queries (the dominant query type in introductory Python education).
    CKG_SIM_GATE = 0.42
    for item in ckg_candidates[:max_extra]:
        if float(item.get("score", 0.0)) < CKG_SIM_GATE:
            continue
        added.append(item)
        existing.add(item["chunk_id"])
        evidence_paths.append({"chunk_id": item["chunk_id"], "path": "query->concept->(related/prereq)->resource->chunk"})

    # CPG expansion: 1-hop from candidate chunk nodes via contains/calls/def-use to chunk-linked nodes.
    chunk_node_to_chunk_id = {n["id"]: n["chunk_id"] for n in cpg.get("nodes", []) if n.get("type") == "chunk" and n.get("chunk_id")}
    node_to_chunk_id = {n["id"]: n.get("chunk_id") for n in cpg.get("nodes", []) if n.get("chunk_id")}
    candidate_chunk_ids = {c["chunk_id"] for c in candidates}
    candidate_chunk_node_ids = {nid for nid, cid in chunk_node_to_chunk_id.items() if cid in candidate_chunk_ids}

    neighbor_node_ids: set[str] = set()
    for e in cpg.get("edges", []):
        if e.get("source") in candidate_chunk_node_ids:
            neighbor_node_ids.add(e.get("target"))
        if e.get("target") in candidate_chunk_node_ids:
            neighbor_node_ids.add(e.get("source"))
    # second step for 1-hop CALLS/DEF-USE from neighbor nodes
    one_hop_more: set[str] = set()
    for e in cpg.get("edges", []):
        if e.get("source") in neighbor_node_ids and e.get("type") in {"CALLS", "DEF-USE"}:
            one_hop_more.add(e.get("target"))
        if e.get("target") in neighbor_node_ids and e.get("type") in {"CALLS", "DEF-USE"}:
            one_hop_more.add(e.get("source"))
    neighbor_node_ids |= one_hop_more

    cpg_chunk_ids = {cid for nid, cid in node_to_chunk_id.items() if nid in neighbor_node_ids and cid}
    cpg_candidates: list[dict[str, Any]] = []
    for cid in cpg_chunk_ids:
        if cid in existing or cid not in chunk_map:
            continue
        chunk = chunk_map[cid]
        overlap = _topic_overlap(chunk.get("concept_tags", []), intent_topics)
        cpg_candidates.append({
            "chunk_id": cid,
            "score": 0.0,
            "text": chunk.get("content", ""),
            "doc_id": chunk.get("doc_id"),
            "provenance": {
                "doc_id": chunk.get("doc_id"),
                "title": chunk.get("metadata", {}).get("title", ""),
                "chunker": chunk.get("chunker", ""),
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
            },
            "evidence": [{"kind": "cpg", "detail": "contains/calls/def-use_1hop"}],
            "topic_overlap": overlap,
        })
    cpg_candidates = _prefilter_graph_candidates(query, cpg_candidates, limit=40)
    cpg_candidates = _score_graph_chunks(query, cpg_candidates)
    cpg_candidates.sort(key=lambda c: (c.get("topic_overlap", 0.0), c["score"]), reverse=True)
    # Same similarity gate for CPG-expanded chunks.
    CPG_SIM_GATE = 0.42
    for item in cpg_candidates[:max_extra]:
        if float(item.get("score", 0.0)) < CPG_SIM_GATE:
            continue
        added.append(item)
        existing.add(item["chunk_id"])
        evidence_paths.append({"chunk_id": item["chunk_id"], "path": "candidate_chunk->cpg_1hop->chunk"})

    expanded = candidates + added
    return expanded, {
        "graph_used": True,
        "intent_topics": sorted(intent_topics),
        "query_concepts": top_concepts,
        "added_count": len(added),
        "evidence_paths": evidence_paths[:50],
    }


def rerank_with_graph_signal(
    candidates: list[dict[str, Any]],
    rerank_weights: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    w_retrieval = float(rerank_weights.get("retrieval", 0.7))
    w_graph = float(rerank_weights.get("graph", 0.3))

    n = len(candidates)
    if n == 0:
        return [], {"weights": {"retrieval": w_retrieval, "graph": w_graph}, "candidate_count": 0}

    raw_vals = [float(c.get("score", 0.0)) for c in candidates]
    raw_min, raw_max = min(raw_vals), max(raw_vals)
    raw_range = raw_max - raw_min if raw_max - raw_min > 1e-12 else 1.0

    rows = []
    for c in candidates:
        raw = float(c.get("score", 0.0))
        retrieval_norm = (raw - raw_min) / raw_range
        evidence_kinds = {e.get("kind") for e in c.get("evidence", [])}
        graph_signal = 0.0
        if "ckg" in evidence_kinds:
            graph_signal += 0.6
        if "cpg" in evidence_kinds:
            graph_signal += 0.4
        graph_signal = min(graph_signal, 1.0)
        topic_signal = float(c.get("topic_overlap", 0.0))
        pre_score = (w_retrieval * retrieval_norm) + (w_graph * graph_signal) + (0.15 * topic_signal)

        row = dict(c)
        row["score"] = round(pre_score, 8)
        row["rerank"] = {
            "raw": round(raw, 8),
            "retrieval_norm": round(retrieval_norm, 8),
            "graph_signal": round(graph_signal, 8),
            "topic_signal": round(topic_signal, 8),
            "pre_score": round(pre_score, 8),
            "weights": {"retrieval": w_retrieval, "graph": w_graph},
        }
        rows.append(row)

    # Apply diversity penalty after pre-score sort to avoid order-dependent artifacts.
    rows.sort(key=lambda x: x["score"], reverse=True)
    seen_docs: dict[str, int] = defaultdict(int)
    for row in rows:
        doc = row.get("doc_id") or ""
        diversity_penalty = 0.04 * seen_docs[doc]
        seen_docs[doc] += 1
        row["score"] = round(float(row["score"]) - diversity_penalty, 8)
        row["rerank"]["diversity_penalty"] = round(diversity_penalty, 8)
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows, {
        "weights": {"retrieval": w_retrieval, "graph": w_graph},
        "candidate_count": len(rows),
    }
