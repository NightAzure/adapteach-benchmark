import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "is",
    "are",
    "as",
    "by",
    "from",
    "that",
    "this",
    "it",
    "be",
    "at",
    "not",
    "do",
    "if",
    "we",
    "you",
    "can",
    "has",
    "have",
    "will",
    "was",
    "but",
    "all",
    "so",
    "no",
    "when",
    "what",
    "how",
    "which",
    "their",
    "an",
    "each",
    "than",
    "its",
    "also",
    "into",
    "just",
    "about",
    "would",
    "should",
    "could",
    "then",
    "these",
    "those",
    "them",
    "they",
    "been",
    "were",
    "being",
    "had",
    "did",
    "does",
    "may",
    "might",
    "must",
    "shall",
    "our",
    "my",
    "your",
    "his",
    "her",
    "who",
}


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _tokens_no_stopwords(text: str) -> list[str]:
    return [t for t in _tokens(text) if t not in STOPWORDS]


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

# Semantic embedding model (sentence-transformers)
_ST_MODEL = None
_ST_DIM = 384  # all-MiniLM-L6-v2 dimension


def _load_st_model() -> Any:
    global _ST_MODEL
    if _ST_MODEL is not None:
        return _ST_MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _ST_MODEL
    except ImportError:
        return None


def embed_text_semantic(text: str) -> list[float]:
    """Embed using sentence-transformers (all-MiniLM-L6-v2, 384-dim)."""
    model = _load_st_model()
    if model is None:
        raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")
    vec = model.encode(text, normalize_embeddings=True)
    return [round(float(v), 8) for v in vec]


def embed_batch_semantic(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Batch embed for efficiency."""
    model = _load_st_model()
    if model is None:
        raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True)
    return [[round(float(v), 8) for v in vec] for vec in vecs]


# Hash-based fallback (deterministic, no dependencies, but low quality)
def _hash_float(token: str, i: int, dim: int) -> float:
    h = hashlib.sha256(f"{token}|{i}|{dim}".encode("utf-8")).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0


def embed_text_hash(text: str, dim: int = 64) -> list[float]:
    vec = [0.0] * dim
    toks = _tokens(text)
    if not toks:
        return vec
    for token in toks:
        for i in range(dim):
            vec[i] += _hash_float(token, i, dim)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [round(v / norm, 8) for v in vec]


def embed_text(text: str, dim: int = 64) -> list[float]:
    """Default embed function — tries semantic, falls back to hash."""
    try:
        return embed_text_semantic(text)
    except (RuntimeError, Exception):
        return embed_text_hash(text, dim=dim)


def build_vector_index(chunks: list[dict[str, Any]], embedder_version: str, batch_size: int = 32) -> dict[str, Any]:
    # Try batch semantic embedding first (much faster)
    use_semantic = False
    dim = 64
    try:
        _load_st_model()
        if _ST_MODEL is not None:
            use_semantic = True
            dim = _ST_DIM
    except Exception:
        pass

    if use_semantic:
        print(f"  Embedding {len(chunks)} chunks with sentence-transformers (batch_size={batch_size})...", file=sys.stderr)
        texts = [chunk["content"] for chunk in chunks]
        all_vecs = embed_batch_semantic(texts, batch_size=batch_size)
        vectors = []
        for chunk, vec in zip(chunks, all_vecs):
            vectors.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "vector": vec,
                }
            )
    else:
        print(f"  Embedding {len(chunks)} chunks with hash-embed fallback...", file=sys.stderr)
        vectors = []
        for chunk in chunks:
            vectors.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "vector": embed_text_hash(chunk["content"], dim=dim),
                }
            )

    return {
        "embedder_version": embedder_version,
        "batch_size": batch_size,
        "normalized": True,
        "dimension": dim,
        "vectors": vectors,
    }


def build_bm25_index(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    doc_tokens: dict[str, list[str]] = {}
    df: dict[str, int] = {}
    total_len = 0
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        toks = _tokens_no_stopwords(chunk["content"])
        doc_tokens[chunk_id] = toks
        total_len += len(toks)
        for term in set(toks):
            df[term] = df.get(term, 0) + 1

    avgdl = (total_len / len(chunks)) if chunks else 0.0
    return {
        "tokenization": "regex_identifier_lower",
        "stopword_policy": "fixed_english_minimal",
        "bm25_params": {"k1": 1.2, "b": 0.75},
        "avgdl": round(avgdl, 6),
        "doc_count": len(chunks),
        "doc_tokens": doc_tokens,
        "document_frequency": dict(sorted(df.items())),
    }


def _index_hash_payload(snapshot_id: str, chunker_version: str, embedder_version: str) -> dict[str, str]:
    return {
        "snapshot_id": snapshot_id,
        "chunker_version": chunker_version,
        "embedder_version": embedder_version,
    }


def build_indexes(
    snapshot_id: str,
    meta_dir: Path = Path("data/corpus_meta"),
    indexes_dir: Path = Path("indexes"),
    embedder_version: str = "minilm-v2",
    batch_size: int = 32,
) -> dict[str, Any]:
    chunk_manifest_path = meta_dir / "chunk_manifest.json"
    if not chunk_manifest_path.exists():
        raise FileNotFoundError("chunk_manifest.json is missing. Run chunk build first.")
    chunk_manifest = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
    chunks = chunk_manifest.get("chunks", [])
    chunker_version = chunk_manifest.get("chunker", "unknown")

    vector_index = build_vector_index(chunks, embedder_version=embedder_version, batch_size=batch_size)
    bm25_index = build_bm25_index(chunks)

    hash_payload = _index_hash_payload(snapshot_id, chunker_version, embedder_version)
    index_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True).encode("utf-8")).hexdigest()
    index_id = index_hash[:16]
    index_dir = indexes_dir / index_id
    index_dir.mkdir(parents=True, exist_ok=True)

    (index_dir / "vector_index.json").write_text(json.dumps(vector_index, indent=2, sort_keys=True), encoding="utf-8")
    (index_dir / "bm25_index.json").write_text(json.dumps(bm25_index, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "index_id": index_id,
        "index_hash": index_hash,
        "snapshot_id": snapshot_id,
        "chunker_version": chunker_version,
        "embedder_version": embedder_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vector_count": len(vector_index["vectors"]),
        "bm25_doc_count": bm25_index["doc_count"],
        "tokenization": bm25_index["tokenization"],
        "stopword_policy": bm25_index["stopword_policy"],
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    build_log = {
        "status": "ok",
        "inputs": hash_payload,
        "outputs": {
            "index_dir": str(index_dir.as_posix()),
            "vector_file": str((index_dir / "vector_index.json").as_posix()),
            "bm25_file": str((index_dir / "bm25_index.json").as_posix()),
            "manifest_file": str((index_dir / "manifest.json").as_posix()),
        },
    }
    (index_dir / "build_log.json").write_text(json.dumps(build_log, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic vector+BM25 indexes")
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--meta-dir", default="data/corpus_meta")
    parser.add_argument("--indexes-dir", default="indexes")
    parser.add_argument("--embedder-version", default="minilm-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = build_indexes(
        snapshot_id=args.snapshot_id,
        meta_dir=Path(args.meta_dir),
        indexes_dir=Path(args.indexes_dir),
        embedder_version=args.embedder_version,
        batch_size=args.batch_size,
    )
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
