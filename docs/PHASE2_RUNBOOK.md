# Phase 2 Runbook (Simplified)

This is a lightweight Phase 2 procedure focused on:

- building a usable corpus from `data/corpus_raw`
- generating a quick QC summary
- freezing an immutable snapshot for benchmarks
- keeping reproducibility without heavy validation workflows

## 1. Goal

Build deterministic corpus artifacts from `data/corpus_raw`, then freeze them as an immutable snapshot for benchmark use.

## 2. Raw doc format

Minimal JSON fields:

- `title` (string)
- `content` (string)
- `type` (string)
- `concept_tags` (array of strings)

Optional fields:

- `difficulty` (string)
- `provenance` object: `url`, `license`, `date`, `author`
- `ai_generated` (boolean)

Files are read recursively from `data/corpus_raw`.

## 3. Build corpus

```bash
make corpus
```

Equivalent:

```bash
py -m src.indexing.corpus_pipeline
```

Pipeline behavior:

1. Ingest docs into a common output schema.
2. Normalize content lightly (boilerplate strip + code fence normalization).
3. Deduplicate only easy cases:
   - exact duplicates (hash)
   - near-exact duplicates (basic similarity)
4. Produce lightweight QC output.

Outputs:

- `data/corpus_clean/*.json`
- `data/corpus_meta/corpus_manifest.json`
- `data/corpus_meta/corpus_manifest.csv`
- `data/corpus_meta/qc_report.json`
- `data/corpus_meta/dedup_report.json`

## 4. Quick QC

Use `data/corpus_meta/qc_report.json` to check:

- per-concept coverage
- thin concepts
- suspiciously short/long docs

Report keys:

- `length_distribution_words`
- `per_concept_coverage`
- `thin_areas`
- `thin_threshold`

Optional threshold tuning:

```bash
py -m src.indexing.corpus_pipeline --thin-threshold 3
```

## 5. Freeze snapshot

```bash
make snapshot
```

Equivalent:

```bash
py -m src.indexing.snapshot_tool create
```

Snapshot ID is based on:

- manifest content
- normalized clean-file hashes
- coverage metadata
- optional chunk count (if available)

Outputs:

- `data/snapshots/<snapshot_id>/manifest.json`
- `data/snapshots/<snapshot_id>/material.json`

Optional AI-exclusion snapshot:

```bash
make snapshot EXCLUDE_AI=1
```

Equivalent:

```bash
py -m src.indexing.snapshot_tool create --exclude-ai
```

## 6. Verify snapshot

```bash
py -m src.indexing.snapshot_tool verify --snapshot-id <snapshot_id>
```

Expected: `"valid": true`.

## 7. Benchmark freeze switch

Bench runs require a snapshot ID.

```bash
make bench SNAPSHOT_ID=<snapshot_id>
```

Direct:

```bash
py bench/run_bench.py --config E --query "bench smoke test" --snapshot-id <snapshot_id>
```

Missing snapshots stop execution.

## 8. Determinism check

1. Run `make corpus`.
2. Run `make snapshot` and save ID.
3. Run `make corpus` again without changing raw files.
4. Run `make snapshot` again.
5. Confirm IDs match.
