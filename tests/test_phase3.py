import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.ast_chunker import chunk_ast
from src.chunking.build_chunks import build_chunks
from src.chunking.fixed_chunker import chunk_fixed
from src.indexing.build_indexes import build_indexes
from src.indexing.corpus_pipeline import CorpusPaths, build_corpus


def _write_raw_doc(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prepare_clean_docs(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    clean = tmp_path / "clean"
    meta = tmp_path / "meta"
    raw.mkdir(parents=True)
    _write_raw_doc(
        raw / "code.json",
        {
            "title": "Code Doc",
            "content": "```python\nclass A:\n    def m(self):\n        if True:\n            return 1\n```\n",
            "type": "tutorial",
            "concept_tags": ["functions", "conditionals"],
        },
    )
    _write_raw_doc(
        raw / "bad_code.json",
        {
            "title": "Bad Code Doc",
            "content": "```python\ndef broken(:\n    pass\n```",
            "type": "example",
            "concept_tags": ["functions"],
        },
    )
    build_corpus(CorpusPaths(raw_dir=raw, clean_dir=clean, meta_dir=meta, thin_threshold=1))
    return clean, meta


def test_fixed_chunker_stable_ids_and_metadata() -> None:
    doc = {
        "doc_id": "doc-1",
        "title": "T",
        "type": "tutorial",
        "concept_tags": ["loops"],
        "content": "for i in range(10): print(i)\n" * 50,
    }
    a = chunk_fixed(doc, chunk_size=100, overlap=20)
    b = chunk_fixed(doc, chunk_size=100, overlap=20)
    assert [x["chunk_id"] for x in a] == [x["chunk_id"] for x in b]
    assert all("doc_id" in c and "start_char" in c and "end_char" in c for c in a)


def test_ast_chunker_fallback_on_parse_error() -> None:
    doc = {
        "doc_id": "doc-bad",
        "title": "Bad",
        "type": "example",
        "concept_tags": ["functions"],
        "content": "```python\ndef broken(:\n    pass\n```",
    }
    chunks, meta = chunk_ast(doc)
    assert meta["parsed"] is False
    assert meta["reason"] == "fallback_fixed"
    assert meta["parse_backend"] == "fixed"
    assert len(chunks) > 0


def test_ast_chunker_reports_backend_on_success() -> None:
    doc = {
        "doc_id": "doc-good",
        "title": "Good",
        "type": "example",
        "concept_tags": ["functions"],
        "content": "```python\ndef add(a, b):\n    return a + b\n```",
    }
    chunks, meta = chunk_ast(doc)
    assert meta["parsed"] is True
    assert meta["parse_backend"] in {"tree_sitter", "ast"}
    assert len(chunks) > 0


def test_chunk_build_reports_and_index_hash_deterministic(tmp_path: Path) -> None:
    clean, meta = _prepare_clean_docs(tmp_path)
    out = build_chunks(clean_dir=clean, meta_dir=meta, chunker="ast", chunk_size=220, overlap=60)
    assert Path(out["chunk_manifest"]).exists()
    assert Path(out["chunk_stats_report"]).exists()
    assert Path(out["parser_failure_report"]).exists()

    indexes_dir = tmp_path / "indexes"
    first = build_indexes(snapshot_id="snap-123", meta_dir=meta, indexes_dir=indexes_dir, embedder_version="hash-embed-v1")
    second = build_indexes(snapshot_id="snap-123", meta_dir=meta, indexes_dir=indexes_dir, embedder_version="hash-embed-v1")
    assert first["index_hash"] == second["index_hash"]
    assert first["index_id"] == second["index_id"]
