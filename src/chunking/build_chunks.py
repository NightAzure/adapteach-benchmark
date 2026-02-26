import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.chunking.ast_chunker import chunk_ast
from src.chunking.fixed_chunker import chunk_fixed


def _load_clean_docs(clean_dir: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for file in sorted(clean_dir.glob("*.json"), key=lambda p: p.name):
        docs.append(json.loads(file.read_text(encoding="utf-8")))
    return docs


def _chunk_doc(doc: dict[str, Any], chunker: str, chunk_size: int, overlap: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if chunker == "fixed":
        chunks = chunk_fixed(doc, chunk_size=chunk_size, overlap=overlap)
        return chunks, {"parsed": None, "reason": "fixed"}
    return chunk_ast(doc, max_chars=chunk_size, min_merge_chars=max(40, overlap // 2))


def _chunk_stats(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = sorted(len(chunk["content"]) for chunk in chunks)
    if not lengths:
        return {"count": 0, "min_chars": 0, "max_chars": 0, "mean_chars": 0, "median_chars": 0}
    return {
        "count": len(lengths),
        "min_chars": lengths[0],
        "max_chars": lengths[-1],
        "mean_chars": round(mean(lengths), 3),
        "median_chars": median(lengths),
    }


def build_chunks(
    clean_dir: Path = Path("data/corpus_clean"),
    meta_dir: Path = Path("data/corpus_meta"),
    chunker: str = "fixed",
    chunk_size: int = 400,
    overlap: int = 80,
) -> dict[str, Any]:
    if chunker not in {"fixed", "ast"}:
        raise ValueError("chunker must be one of: fixed, ast")

    docs = _load_clean_docs(clean_dir)
    all_chunks: list[dict[str, Any]] = []
    parser_failures: list[dict[str, Any]] = []
    parsed_count = 0

    for doc in docs:
        chunks, parse_meta = _chunk_doc(doc, chunker=chunker, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        if chunker == "ast":
            if parse_meta.get("parsed"):
                parsed_count += 1
            else:
                parser_failures.append({"doc_id": doc["doc_id"], **parse_meta})

    all_chunks = sorted(all_chunks, key=lambda c: c["chunk_id"])
    chunk_manifest = {
        "chunker": f"{chunker}-v1",
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunks": all_chunks,
    }

    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "chunk_manifest.json").write_text(json.dumps(chunk_manifest, indent=2, sort_keys=True), encoding="utf-8")

    stats = {
        "chunker": f"{chunker}-v1",
        "doc_count": len(docs),
        "chunk_count": len(all_chunks),
        "chunk_size_stats": _chunk_stats(all_chunks),
        "content_type_counts": {
            "code": sum(1 for c in all_chunks if c["content_type"] == "code"),
            "text": sum(1 for c in all_chunks if c["content_type"] == "text"),
        },
    }
    (meta_dir / "chunk_stats_report.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

    parse_report = {
        "chunker": f"{chunker}-v1",
        "doc_count": len(docs),
        "parsed_docs": parsed_count if chunker == "ast" else None,
        "parse_rate": round((parsed_count / len(docs)) * 100, 3) if chunker == "ast" and docs else None,
        "failure_count": len(parser_failures),
        "failure_examples": parser_failures[:10],
    }
    (meta_dir / "parser_failure_report.json").write_text(json.dumps(parse_report, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "chunk_manifest": str((meta_dir / "chunk_manifest.json").as_posix()),
        "chunk_stats_report": str((meta_dir / "chunk_stats_report.json").as_posix()),
        "parser_failure_report": str((meta_dir / "parser_failure_report.json").as_posix()),
        "chunk_count": len(all_chunks),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic chunk manifest")
    parser.add_argument("--clean-dir", default="data/corpus_clean")
    parser.add_argument("--meta-dir", default="data/corpus_meta")
    parser.add_argument("--chunker", default="fixed", choices=["fixed", "ast"])
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_chunks(
        clean_dir=Path(args.clean_dir),
        meta_dir=Path(args.meta_dir),
        chunker=args.chunker,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
