import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


RAW_EXTENSIONS = {".json", ".md", ".txt"}
BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*(home|next|previous|table of contents)\s*$", re.IGNORECASE),
    re.compile(r"copyright", re.IGNORECASE),
]


@dataclass(frozen=True)
class CorpusPaths:
    raw_dir: Path = Path("data/corpus_raw")
    clean_dir: Path = Path("data/corpus_clean")
    meta_dir: Path = Path("data/corpus_meta")
    thin_threshold: int = 2


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slug(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    return lowered.strip("-") or "untitled"


def _strip_boilerplate(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in BOILERPLATE_PATTERNS):
            continue
        cleaned_lines.append(line.rstrip())
    return "\n".join(cleaned_lines).strip()


def _normalize_code_blocks(text: str) -> str:
    # Convert untyped fenced code blocks to python for consistency in this code-focused corpus.
    normalized = re.sub(r"```[ \t]*\n", "```python\n", text)
    # Ensure closing fences are standalone.
    normalized = re.sub(r"[ \t]*```", "```", normalized)
    return normalized


def _normalize_content(text: str) -> str:
    text = _strip_boilerplate(text)
    text = _normalize_code_blocks(text)
    # Collapse 3+ blank lines to 2 for consistent formatting.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _parse_json_doc(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def _parse_text_doc(path: Path) -> dict[str, Any]:
    title = path.stem.replace("_", " ").replace("-", " ").strip().title()
    content = path.read_text(encoding="utf-8")
    return {
        "title": title,
        "content": content,
        "type": "tutorial",
        "concept_tags": ["python-basics"],
        "difficulty": "intro",
        "provenance": {
            "url": "",
            "license": "unknown",
            "date": "",
            "author": "unknown",
            "source_file": str(path.as_posix()),
        },
    }


def _ingest_raw_doc(path: Path) -> dict[str, Any]:
    parsed = _parse_json_doc(path) if path.suffix.lower() == ".json" else _parse_text_doc(path)

    title = str(parsed.get("title", path.stem)).strip()
    content = str(parsed.get("content", "")).strip()
    doc_type = str(parsed.get("type", "tutorial")).strip()
    concept_tags = parsed.get("concept_tags", [])
    if isinstance(concept_tags, str):
        concept_tags = [concept_tags]
    concept_tags = sorted({str(tag).strip() for tag in concept_tags if str(tag).strip()})

    difficulty = parsed.get("difficulty")
    difficulty = str(difficulty).strip() if difficulty not in (None, "") else None
    ai_generated = bool(parsed.get("ai_generated", False))

    provenance = parsed.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    provenance = {
        "url": str(provenance.get("url", "")),
        "license": str(provenance.get("license", "")),
        "date": str(provenance.get("date", "")),
        "author": str(provenance.get("author", "")),
        "source_file": str(path.as_posix()),
    }

    content_norm = _normalize_content(content)
    raw_id_basis = f"{path.as_posix()}::{title}::{doc_type}"
    doc_id = f"doc-{_slug(title)}-{_stable_hash(raw_id_basis)[:10]}"

    return {
        "doc_id": doc_id,
        "title": title,
        "content": content_norm,
        "type": doc_type,
        "concept_tags": concept_tags,
        "difficulty": difficulty,
        "ai_generated": ai_generated,
        "provenance": provenance,
    }


def _all_raw_docs(raw_dir: Path) -> list[Path]:
    files = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in RAW_EXTENSIONS]
    return sorted(files, key=lambda p: str(p.as_posix()))


def _is_near_exact_duplicate(a: str, b: str) -> bool:
    # Deterministic near-exact check using normalized hashes and size ratio.
    if _stable_hash(a) == _stable_hash(b):
        return True
    if not a or not b:
        return False
    shorter, longer = sorted((a, b), key=len)
    if len(shorter) / len(longer) < 0.97:
        return False
    return shorter in longer


def deduplicate_docs(docs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    content_index: list[tuple[str, str, str]] = []

    for doc in sorted(docs, key=lambda d: d["doc_id"]):
        content = doc["content"]
        content_hash = _stable_hash(content)
        duplicate_reason = None
        duplicate_of = None

        for keep_hash, keep_id, keep_content in content_index:
            if keep_hash == content_hash or _is_near_exact_duplicate(content, keep_content):
                duplicate_reason = "exact_or_near_exact"
                duplicate_of = keep_id
                break

        if duplicate_reason:
            removed.append(
                {
                    "doc_id": doc["doc_id"],
                    "duplicate_of": duplicate_of,
                    "reason": duplicate_reason,
                }
            )
            continue

        kept.append(doc)
        content_index.append((content_hash, doc["doc_id"], content))

    return kept, removed


def _doc_word_count(content: str) -> int:
    return len(content.split())


def build_qc_report(docs: list[dict[str, Any]], thin_threshold: int) -> dict[str, Any]:
    lengths = sorted(_doc_word_count(doc["content"]) for doc in docs)
    concept_counts: dict[str, int] = {}
    for doc in docs:
        for concept in doc["concept_tags"]:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1

    thin_areas = sorted(
        [{"concept": concept, "count": count} for concept, count in concept_counts.items() if count < thin_threshold],
        key=lambda x: (x["count"], x["concept"]),
    )

    if lengths:
        distribution = {
            "count": len(lengths),
            "min": lengths[0],
            "max": lengths[-1],
            "mean": round(mean(lengths), 3),
            "median": median(lengths),
            "p90": lengths[min(len(lengths) - 1, int(round(0.9 * (len(lengths) - 1))))],
        }
    else:
        distribution = {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "p90": 0}

    return {
        "length_distribution_words": distribution,
        "per_concept_coverage": dict(sorted(concept_counts.items())),
        "thin_areas": thin_areas,
        "thin_threshold": thin_threshold,
    }


def _write_manifest_json(path: Path, docs: list[dict[str, Any]]) -> None:
    payload = {"documents": docs}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_manifest_csv(path: Path, docs: list[dict[str, Any]]) -> None:
    fields = [
        "doc_id",
        "title",
        "type",
        "difficulty",
        "ai_generated",
        "concept_tags",
        "source_file",
        "url",
        "license",
        "date",
        "author",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for doc in docs:
            prov = doc["provenance"]
            writer.writerow(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "type": doc["type"],
                    "difficulty": doc["difficulty"] or "",
                    "ai_generated": str(bool(doc.get("ai_generated", False))).lower(),
                    "concept_tags": "|".join(doc["concept_tags"]),
                    "source_file": prov["source_file"],
                    "url": prov["url"],
                    "license": prov["license"],
                    "date": prov["date"],
                    "author": prov["author"],
                }
            )


def _write_clean_docs(clean_dir: Path, docs: list[dict[str, Any]]) -> None:
    clean_dir.mkdir(parents=True, exist_ok=True)
    for old_file in clean_dir.glob("*.json"):
        old_file.unlink()
    for doc in sorted(docs, key=lambda d: d["doc_id"]):
        out_file = clean_dir / f"{doc['doc_id']}.json"
        out_file.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")


def build_corpus(paths: CorpusPaths) -> dict[str, Any]:
    paths.clean_dir.mkdir(parents=True, exist_ok=True)
    paths.meta_dir.mkdir(parents=True, exist_ok=True)
    raw_files = _all_raw_docs(paths.raw_dir)

    docs = [_ingest_raw_doc(path) for path in raw_files]
    docs_kept, dedup_removed = deduplicate_docs(docs)
    docs_kept = sorted(docs_kept, key=lambda d: d["doc_id"])

    qc_report = build_qc_report(docs_kept, thin_threshold=paths.thin_threshold)

    _write_clean_docs(paths.clean_dir, docs_kept)
    _write_manifest_json(paths.meta_dir / "corpus_manifest.json", docs_kept)
    _write_manifest_csv(paths.meta_dir / "corpus_manifest.csv", docs_kept)
    (paths.meta_dir / "qc_report.json").write_text(
        json.dumps(qc_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (paths.meta_dir / "dedup_report.json").write_text(
        json.dumps({"removed": dedup_removed, "removed_count": len(dedup_removed)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "raw_files": len(raw_files),
        "kept_docs": len(docs_kept),
        "removed_duplicates": len(dedup_removed),
        "manifest_json": str((paths.meta_dir / "corpus_manifest.json").as_posix()),
        "manifest_csv": str((paths.meta_dir / "corpus_manifest.csv").as_posix()),
        "qc_report": str((paths.meta_dir / "qc_report.json").as_posix()),
        "dedup_report": str((paths.meta_dir / "dedup_report.json").as_posix()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized corpus + QC outputs")
    parser.add_argument("--raw-dir", default="data/corpus_raw")
    parser.add_argument("--clean-dir", default="data/corpus_clean")
    parser.add_argument("--meta-dir", default="data/corpus_meta")
    parser.add_argument("--thin-threshold", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_corpus(
        CorpusPaths(
            raw_dir=Path(args.raw_dir),
            clean_dir=Path(args.clean_dir),
            meta_dir=Path(args.meta_dir),
            thin_threshold=args.thin_threshold,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
