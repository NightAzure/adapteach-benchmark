import argparse
import csv
import json
from pathlib import Path
from typing import Any


QUERY_CATEGORIES = [
    ("concept_explanation", "Explain the concept of {topic} with one short Python example."),
    ("debugging", "Find and fix the bug in this {topic} snippet, then explain why."),
    ("artifact_generation", "Generate a {artifact} exercise for {topic} at {difficulty} level."),
    ("misconception", "Correct this misconception about {topic}: '{misconception}'."),
]

TOPICS = ["variables", "loops", "conditionals", "functions"]
ARTIFACTS = ["parsons", "tracing", "mutation", "flashcard"]
DIFFICULTIES = ["easy", "moderate", "hard"]
MISCONCEPTIONS = [
    "a for-loop always runs at least once",
    "if statements do not need indentation",
    "functions cannot return values",
    "variables store only numbers",
]


def build_queries(count: int) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    idx = 1
    while len(queries) < count:
        for category, tmpl in QUERY_CATEGORIES:
            topic = TOPICS[(idx - 1) % len(TOPICS)]
            artifact = ARTIFACTS[(idx - 1) % len(ARTIFACTS)]
            difficulty = DIFFICULTIES[(idx - 1) % len(DIFFICULTIES)]
            misconception = MISCONCEPTIONS[(idx - 1) % len(MISCONCEPTIONS)]
            text = tmpl.format(topic=topic, artifact=artifact, difficulty=difficulty, misconception=misconception)
            queries.append({"id": f"q-{idx:03d}", "category": category, "topic": topic, "query": text})
            idx += 1
            if len(queries) >= count:
                break
    return queries


def write_queries(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_label_template(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "rater", "chunk_id", "relevance", "notes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({"query_id": row["id"], "rater": "rater_a", "chunk_id": "", "relevance": "", "notes": ""})
            writer.writerow({"query_id": row["id"], "rater": "rater_b", "chunk_id": "", "relevance": "", "notes": ""})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare benchmark query sets and label template")
    p.add_argument("--dev-count", type=int, default=30)
    p.add_argument("--test-count", type=int, default=30)
    p.add_argument("--dev-out", default="bench/queries_dev.jsonl")
    p.add_argument("--test-out", default="bench/queries_test.jsonl")
    p.add_argument("--labels-template", default="bench/labels/labels_template.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dev = build_queries(args.dev_count)
    test = build_queries(args.test_count)
    write_queries(Path(args.dev_out), dev)
    write_queries(Path(args.test_out), test)
    write_label_template(Path(args.labels_template), dev + test)
    print(
        json.dumps(
            {
                "dev_queries": len(dev),
                "test_queries": len(test),
                "dev_out": args.dev_out,
                "test_out": args.test_out,
                "labels_template": args.labels_template,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
