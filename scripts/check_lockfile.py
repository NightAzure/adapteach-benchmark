from pathlib import Path


def main() -> None:
    lockfile = Path("requirements.lock")
    if not lockfile.exists():
        raise SystemExit("requirements.lock is missing")

    lines = [
        line.strip()
        for line in lockfile.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        raise SystemExit("requirements.lock is empty")

    for line in lines:
        if "==" not in line:
            raise SystemExit(f"Unpinned dependency in requirements.lock: {line}")

    print(f"Lockfile check passed: {len(lines)} pinned dependencies")


if __name__ == "__main__":
    main()
