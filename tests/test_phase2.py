import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.run_bench import ensure_snapshot_exists
from src.indexing.corpus_pipeline import CorpusPaths, build_corpus
from src.indexing.snapshot_tool import compute_snapshot_materials, create_snapshot, verify_snapshot


def _write_raw_doc(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_corpus_build_is_deterministic(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    clean = tmp_path / "clean"
    meta = tmp_path / "meta"
    raw.mkdir(parents=True)

    _write_raw_doc(
        raw / "loops_a.json",
        {
            "title": "Loops A",
            "content": "Home\nLoops repeat.\n``` \nfor i in range(2):\n    print(i)\n```\nNext",
            "type": "tutorial",
            "concept_tags": ["loops"],
            "difficulty": "intro",
            "provenance": {"url": "u1", "license": "L1", "date": "d1", "author": "a1"},
        },
    )
    _write_raw_doc(
        raw / "loops_dup.json",
        {
            "title": "Loops Duplicate",
            "content": "Loops repeat.\n```python\nfor i in range(2):\n    print(i)\n```",
            "type": "tutorial",
            "concept_tags": ["loops"],
            "difficulty": "intro",
            "provenance": {"url": "u2", "license": "L1", "date": "d2", "author": "a2"},
        },
    )
    _write_raw_doc(
        raw / "if_a.json",
        {
            "title": "If Intro",
            "content": "If chooses paths.",
            "type": "tutorial",
            "concept_tags": ["conditionals"],
            "difficulty": "intro",
            "provenance": {"url": "u3", "license": "L1", "date": "d3", "author": "a3"},
        },
    )

    out1 = build_corpus(CorpusPaths(raw_dir=raw, clean_dir=clean, meta_dir=meta, thin_threshold=2))
    manifest1 = (meta / "corpus_manifest.json").read_text(encoding="utf-8")
    qc1 = (meta / "qc_report.json").read_text(encoding="utf-8")

    out2 = build_corpus(CorpusPaths(raw_dir=raw, clean_dir=clean, meta_dir=meta, thin_threshold=2))
    manifest2 = (meta / "corpus_manifest.json").read_text(encoding="utf-8")
    qc2 = (meta / "qc_report.json").read_text(encoding="utf-8")

    assert out1["kept_docs"] == out2["kept_docs"]
    assert manifest1 == manifest2
    assert qc1 == qc2
    assert out1["removed_duplicates"] >= 1


def test_snapshot_id_stable_for_same_material(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    clean = tmp_path / "clean"
    meta = tmp_path / "meta"
    snapshots = tmp_path / "snapshots"
    raw.mkdir(parents=True)

    _write_raw_doc(
        raw / "one.json",
        {
            "title": "One",
            "content": "Simple content",
            "type": "tutorial",
            "concept_tags": ["loops"],
            "difficulty": "intro",
            "provenance": {"url": "", "license": "", "date": "", "author": ""},
        },
    )
    build_corpus(CorpusPaths(raw_dir=raw, clean_dir=clean, meta_dir=meta, thin_threshold=1))
    a = compute_snapshot_materials(meta, clean)
    b = compute_snapshot_materials(meta, clean)
    assert a["snapshot_id"] == b["snapshot_id"]

    created = create_snapshot(meta, clean, snapshots)
    verification = verify_snapshot(created["snapshot_id"], snapshots)
    assert verification["valid"] is True


def test_ai_generated_flag_and_exclude_ai_snapshot(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    clean = tmp_path / "clean"
    meta = tmp_path / "meta"
    snapshots = tmp_path / "snapshots"
    raw.mkdir(parents=True)

    _write_raw_doc(
        raw / "human_doc.json",
        {
            "title": "Human Doc",
            "content": "Human authored content",
            "type": "tutorial",
            "concept_tags": ["loops"],
            "ai_generated": False,
        },
    )
    _write_raw_doc(
        raw / "ai_doc.json",
        {
            "title": "AI Doc",
            "content": "AI generated content",
            "type": "tutorial",
            "concept_tags": ["loops"],
            "ai_generated": True,
        },
    )

    build_corpus(CorpusPaths(raw_dir=raw, clean_dir=clean, meta_dir=meta, thin_threshold=1))
    manifest = json.loads((meta / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert any(bool(doc.get("ai_generated", False)) for doc in manifest["documents"])

    all_docs = create_snapshot(meta, clean, snapshots, exclude_ai=False)
    human_only = create_snapshot(meta, clean, snapshots, exclude_ai=True)
    assert all_docs["snapshot_id"] != human_only["snapshot_id"]


def test_benchmark_requires_snapshot_id(tmp_path: Path) -> None:
    missing_id = "missing123"
    try:
        ensure_snapshot_exists(missing_id, snapshots_root=tmp_path / "snapshots")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass
