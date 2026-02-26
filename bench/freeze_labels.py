import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze benchmark labels and query sets with hashes")
    p.add_argument("--labels-csv", default="bench/labels/labels.csv")
    p.add_argument("--queries-dev", default="bench/queries_dev.jsonl")
    p.add_argument("--queries-test", default="bench/queries_test.jsonl")
    p.add_argument("--out", default="bench/labels/freeze_manifest.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = Path(args.labels_csv)
    queries_dev = Path(args.queries_dev)
    queries_test = Path(args.queries_test)
    for required in [labels, queries_dev, queries_test]:
        if not required.exists():
            raise SystemExit(f"Missing required file: {required.as_posix()}")

    payload = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels_csv": labels.as_posix(),
        "labels_sha256": _sha256_file(labels),
        "queries_dev": queries_dev.as_posix(),
        "queries_dev_sha256": _sha256_file(queries_dev),
        "queries_test": queries_test.as_posix(),
        "queries_test_sha256": _sha256_file(queries_test),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": out.as_posix(), "labels_sha256": payload["labels_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
