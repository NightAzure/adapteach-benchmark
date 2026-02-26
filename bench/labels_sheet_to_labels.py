import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert non-technical labels sheet to labels.csv format")
    p.add_argument("--sheet-csv", default="bench/labels/labels_sheet.csv")
    p.add_argument("--out-csv", default="bench/labels/labels.csv")
    return p.parse_args()


def _norm_label(value: str) -> str:
    v = str(value).strip()
    if v not in {"0", "1", "2", "3"}:
        return ""
    return v


def main() -> None:
    args = parse_args()
    sheet = Path(args.sheet_csv)
    if not sheet.exists():
        raise SystemExit(f"sheet file not found: {sheet.as_posix()}")

    rows = list(csv.DictReader(sheet.open("r", encoding="utf-8")))
    out_rows: list[dict[str, str]] = []
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        chunk_id = str(row.get("chunk_id", "")).strip()
        notes = str(row.get("notes", "")).strip()
        if not query_id or not chunk_id:
            continue
        out_rows.append(
            {
                "query_id": query_id,
                "rater": "rater_a",
                "chunk_id": chunk_id,
                "relevance": _norm_label(row.get("relevance_rater_a", "")),
                "notes": notes,
            }
        )
        out_rows.append(
            {
                "query_id": query_id,
                "rater": "rater_b",
                "chunk_id": chunk_id,
                "relevance": _norm_label(row.get("relevance_rater_b", "")),
                "notes": notes,
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "rater", "chunk_id", "relevance", "notes"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(
        json.dumps(
            {
                "sheet_csv": sheet.as_posix(),
                "out_csv": out_csv.as_posix(),
                "sheet_rows": len(rows),
                "labels_rows": len(out_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
