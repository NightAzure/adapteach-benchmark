import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return 0.0
    categories = sorted(set(labels_a) | set(labels_b))
    p0 = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    pa = {c: labels_a.count(c) / n for c in categories}
    pb = {c: labels_b.count(c) / n for c in categories}
    pe = sum(pa[c] * pb[c] for c in categories)
    if pe == 1:
        return 1.0
    return (p0 - pe) / (1 - pe)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Cohen's kappa from labels CSV")
    p.add_argument("--labels-csv", default="bench/labels/labels.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.labels_csv)
    if not path.exists():
        raise SystemExit(f"labels file not found: {path}")
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    by_key: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for row in rows:
        qid = row.get("query_id", "")
        chunk = row.get("chunk_id", "")
        rater = row.get("rater", "")
        rel = row.get("relevance", "")
        if not qid or not chunk or not rater or rel == "":
            continue
        try:
            v = int(rel)
        except ValueError:
            continue
        by_key[(qid, chunk)][rater] = v

    labels_a: list[int] = []
    labels_b: list[int] = []
    for _, ratings in by_key.items():
        if "rater_a" in ratings and "rater_b" in ratings:
            labels_a.append(ratings["rater_a"])
            labels_b.append(ratings["rater_b"])

    kappa = cohen_kappa(labels_a, labels_b)
    print(json.dumps({"pairs": len(labels_a), "cohen_kappa": round(kappa, 6)}, indent=2))


if __name__ == "__main__":
    main()
