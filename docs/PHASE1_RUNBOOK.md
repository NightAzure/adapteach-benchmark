# Phase 1 Runbook

This runbook explains how to operate the current Phase 1 scaffold.

## 1. Prerequisites

- Python 3.12 (the project was validated with Python 3.12.3).
- PowerShell or shell with Python available.
- Optional: `make` if you want to use Makefile commands.

## 2. Install dependencies

With `make`:

```bash
make setup
```

Without `make` (Windows-friendly):

```bash
py -m pip install -r requirements.lock
```

## 3. Verify setup

With `make`:

```bash
make test
```

Without `make`:

```bash
py -m pytest -q -p no:cacheprovider tests/test_phase1.py
```

Expected result: all tests in `tests/test_phase1.py` pass.

## 4. Run pipelines A-E

CLI syntax:

```bash
py -m src.cli --config <A|B|C|D|E> --query "<your query>" [--dry-run none|retrieval|graph]
```

Examples:

```bash
py -m src.cli --config A --query "Explain for loops"
py -m src.cli --config B --query "Explain for loops"
py -m src.cli --config E --query "Explain for loops" --dry-run retrieval
py -m src.cli --config E --query "Explain for loops" --dry-run graph
```

## 5. Understand outputs

Response fields are stable across configs:

- `trace_id`
- `config`
- `mode`
- `status`
- `answer`
- `context`
- `validation`

Dry-run behavior:

- `--dry-run retrieval`: returns retrieval stage output only, no generation.
- `--dry-run graph`: returns graph-expanded context, no generation.

## 6. Logs and tracing

- Run records are written to `logs/runs.jsonl`.
- Log schema reference is `logs/run_log.schema.json`.
- Example record is `logs/example_run.jsonl`.

Each record includes:

- model/provider and prompt template version
- retrieval config, corpus snapshot, and index hash
- stage timings and stage outputs
- stage events with the same `trace_id`
- token counts

## 7. Config management

Core files:

- `config.yaml`: runtime, llm, retrieval defaults
- `config.schema.json`: validates config structure and types
- `configs/A.yaml` ... `configs/E.yaml`: pipeline-specific toggles

Environment override pattern:

- Prefix: `ADAPTEACH__`
- Nested keys separated by `__`

PowerShell example:

```bash
$env:ADAPTEACH__retrieval__k = "9"
py -m src.cli --config C --query "override test"
```

## 8. Task commands reference

- `make setup`: install pinned dependencies from `requirements.lock`
- `make index`: smoke command for retrieval-only run
- `make bench`: smoke command for full benchmark-style run
- `make test`: run tests
- `make check-lock`: verify all lockfile deps are pinned

If `make` is unavailable, run equivalent `py -m ...` commands directly.

## 9. Docker baseline

Build:

```bash
docker build -t adapteach-phase1 .
```

Run:

```bash
docker run --rm adapteach-phase1
```

This executes the default command in `Dockerfile` for a smoke run.

## 10. Typical workflow

1. Install deps (`make setup` or `py -m pip install -r requirements.lock`).
2. Run tests (`make test`).
3. Run one pipeline command with your query.
4. Inspect `logs/runs.jsonl` for traceable stage metadata.
5. Adjust `config.yaml` or env overrides.
6. Re-run and compare outputs/logs.
