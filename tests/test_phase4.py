import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.build_chunks import build_chunks
from src.graphs.build_graphs import main as _unused_main  # import smoke for module path
from src.graphs.ckg import build_ckg
from src.graphs.cpg import build_cpg
from src.graphs.query_mapper import map_query_to_concepts
from src.indexing.corpus_pipeline import CorpusPaths, build_corpus


def _write_raw_doc(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prepare_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw = tmp_path / "raw"
    clean = tmp_path / "clean"
    meta = tmp_path / "meta"
    graphs = tmp_path / "graphs"
    raw.mkdir(parents=True)
    _write_raw_doc(
        raw / "loops.json",
        {
            "title": "Loops Lesson",
            "content": "```python\nfor i in range(3):\n    print(i)\n```",
            "type": "tutorial",
            "concept_tags": ["variables", "loops"],
        },
    )
    _write_raw_doc(
        raw / "functions.json",
        {
            "title": "Functions Lesson",
            "content": "```python\ndef add(a,b):\n    c = a + b\n    return c\n```",
            "type": "tutorial",
            "concept_tags": ["functions", "conditionals"],
        },
    )
    build_corpus(CorpusPaths(raw_dir=raw, clean_dir=clean, meta_dir=meta, thin_threshold=1))
    build_chunks(clean_dir=clean, meta_dir=meta, chunker="ast", chunk_size=220, overlap=40)
    return meta, graphs, clean


def test_ckg_build_and_query_mapping(tmp_path: Path) -> None:
    meta, graphs, _ = _prepare_inputs(tmp_path)
    ckg = build_ckg(meta_dir=meta, out_dir=graphs)

    assert any(n["type"] == "concept" for n in ckg["nodes"])
    edge_types = {e["type"] for e in ckg["edges"]}
    assert {"prerequisite", "related", "addresses", "misconception-of"}.issubset(edge_types)
    assert ckg["validation"]["prerequisite_cycle"] is False

    mapped = map_query_to_concepts("How do for loops work with variables?", ckg_path=graphs / "ckg.json")
    assert mapped["top_concepts"]
    top = mapped["top_concepts"][0]
    assert "confidence" in top and "explanation" in top


def test_cpg_build_sanity_and_determinism(tmp_path: Path) -> None:
    meta, graphs, _ = _prepare_inputs(tmp_path)
    cpg_first = build_cpg(chunk_manifest_path=meta / "chunk_manifest.json", out_dir=graphs)
    cpg_second = build_cpg(chunk_manifest_path=meta / "chunk_manifest.json", out_dir=graphs)

    assert len(cpg_first["nodes"]) > 0
    assert len(cpg_first["edges"]) > 0
    assert cpg_first["sanity"]["bounded"] is True
    assert cpg_first["sanity"]["orphan_node_count"] == 0

    ids1 = [n["id"] for n in cpg_first["nodes"]]
    ids2 = [n["id"] for n in cpg_second["nodes"]]
    assert ids1 == ids2
