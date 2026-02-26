import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _clean_doc_files(clean_dir: Path) -> list[Path]:
    return sorted([p for p in clean_dir.glob("*.json") if p.is_file()], key=lambda p: p.name)


def compute_snapshot_materials(meta_dir: Path, clean_dir: Path, exclude_ai: bool = False) -> dict[str, Any]:
    manifest_path = meta_dir / "corpus_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs = manifest.get("documents", [])
    if not isinstance(docs, list):
        raise ValueError("Invalid manifest: documents must be a list")
    if exclude_ai:
        docs = [doc for doc in docs if not bool(doc.get("ai_generated", False))]

    clean_files = _clean_doc_files(clean_dir)
    clean_hashes = []
    allowed_doc_ids = {str(doc.get("doc_id")) for doc in docs}
    for file in clean_files:
        doc_id = file.stem
        if exclude_ai and doc_id not in allowed_doc_ids:
            continue
        clean_hashes.append(
            {
                "file": file.name,
                "sha256": _sha256_bytes(file.read_bytes()),
            }
        )

    coverage_counts: dict[str, int] = {}
    for doc in docs:
        for concept in doc.get("concept_tags", []):
            concept = str(concept)
            coverage_counts[concept] = coverage_counts.get(concept, 0) + 1

    chunk_count = None
    chunk_manifest = meta_dir / "chunk_manifest.json"
    if chunk_manifest.exists():
        chunk_data = json.loads(chunk_manifest.read_text(encoding="utf-8"))
        if isinstance(chunk_data, dict) and isinstance(chunk_data.get("chunks"), list):
            chunk_count = len(chunk_data["chunks"])

    material = {
        "manifest": {"documents": docs},
        "clean_hashes": clean_hashes,
        "concept_coverage_counts": dict(sorted(coverage_counts.items())),
        "chunk_count": chunk_count,
        "exclude_ai": exclude_ai,
    }
    snapshot_hash = _sha256_bytes(_canonical_json_bytes(material))
    snapshot_id = snapshot_hash[:16]
    return {"snapshot_id": snapshot_id, "snapshot_hash": snapshot_hash, "material": material}


def create_snapshot(
    meta_dir: Path = Path("data/corpus_meta"),
    clean_dir: Path = Path("data/corpus_clean"),
    snapshots_dir: Path = Path("data/snapshots"),
    exclude_ai: bool = False,
) -> dict[str, Any]:
    result = compute_snapshot_materials(meta_dir=meta_dir, clean_dir=clean_dir, exclude_ai=exclude_ai)
    snapshot_id = result["snapshot_id"]
    snapshot_dir = snapshots_dir / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    manifest_doc = {
        "snapshot_id": snapshot_id,
        "snapshot_hash": result["snapshot_hash"],
        "created_from": {
            "meta_dir": str(meta_dir.as_posix()),
            "clean_dir": str(clean_dir.as_posix()),
        },
        "exclude_ai": exclude_ai,
        "concept_coverage_counts": result["material"]["concept_coverage_counts"],
        "chunk_count": result["material"]["chunk_count"],
        "clean_files": result["material"]["clean_hashes"],
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest_doc, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (snapshot_dir / "material.json").write_text(
        json.dumps(result["material"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_doc


def verify_snapshot(snapshot_id: str, snapshots_dir: Path = Path("data/snapshots")) -> dict[str, Any]:
    snapshot_dir = snapshots_dir / snapshot_id
    manifest_file = snapshot_dir / "manifest.json"
    material_file = snapshot_dir / "material.json"
    if not manifest_file.exists() or not material_file.exists():
        raise FileNotFoundError(f"Missing snapshot files under {snapshot_dir}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    material = json.loads(material_file.read_text(encoding="utf-8"))
    recomputed_hash = _sha256_bytes(_canonical_json_bytes(material))
    return {
        "snapshot_id": snapshot_id,
        "expected_hash": manifest.get("snapshot_hash"),
        "recomputed_hash": recomputed_hash,
        "valid": manifest.get("snapshot_hash") == recomputed_hash,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create/verify immutable corpus snapshots")
    parser.add_argument("command", choices=["create", "verify"])
    parser.add_argument("--snapshot-id", default="")
    parser.add_argument("--meta-dir", default="data/corpus_meta")
    parser.add_argument("--clean-dir", default="data/corpus_clean")
    parser.add_argument("--snapshots-dir", default="data/snapshots")
    parser.add_argument("--exclude-ai", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "create":
        output = create_snapshot(
            meta_dir=Path(args.meta_dir),
            clean_dir=Path(args.clean_dir),
            snapshots_dir=Path(args.snapshots_dir),
            exclude_ai=args.exclude_ai,
        )
    else:
        if not args.snapshot_id:
            raise SystemExit("--snapshot-id is required for verify")
        output = verify_snapshot(args.snapshot_id, snapshots_dir=Path(args.snapshots_dir))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
