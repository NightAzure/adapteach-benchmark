# Phase 2 Runbook (Simplified Replacement)

This is a **lightweight** Phase 2 procedure for an undergraduate thesis. It keeps the essentials:
- build a usable corpus from `data/corpus_raw`
- generate a quick QC summary
- freeze an immutable snapshot for benchmarks
- keep everything reproducible **without** heavy validation or formatting work

> This replaces the more complicated Phase 2. It intentionally avoids extensive schema enforcement, deep provenance validation, and advanced dedup/verification workflows.

---

## 1) Goal

Build deterministic corpus artifacts from `data/corpus_raw`, then freeze them as an immutable snapshot for benchmark use.  
(Your benchmarks should always reference a snapshot ID so results are comparable over time.)  

---

## 2) What you need to put into `data/corpus_raw`

### 2.1 Minimal raw doc format (keep it simple)

Each raw JSON file should contain at least:

- `title` (string)
- `content` (string)
- `type` (string: tutorial/example/misconception/exercise/etc.)
- `concept_tags` (array of strings)

Optional (recommended, but not required for undergrad pace):
- `difficulty` (string)
- `provenance`:
  - `url` (string)
  - `license` (string)
  - `date` (string)
  - `author` (string)

Files are read from `data/corpus_raw` recursively.

### 2.2 If the content is AI-generated (simple rule)

If a doc is AI-generated, add **one** optional flag:

- `ai_generated: true`

That’s all.  
No forced reviewer workflow. If you later decide to exclude AI docs from benchmarks, you can do it by filtering on this flag.

---

## 3) Build normalized corpus (one command)

### Command

```bash
make corpus
```

Equivalent direct command:

```bash
py -m src.indexing.corpus_pipeline
```

### What happens (simplified expectations)

1. Ingest raw docs into a common output format.
2. Normalize content lightly (e.g., normalize code fences, strip obvious boilerplate).
3. Deduplicate **only what’s easy**:
   - exact duplicates (hash)
   - near-exact duplicates (basic similarity)
4. Generate a lightweight QC report.

### Outputs (expected)

- `data/corpus_clean/*.json`
- `data/corpus_meta/corpus_manifest.json`
- `data/corpus_meta/corpus_manifest.csv`
- `data/corpus_meta/qc_report.json`
- `data/corpus_meta/dedup_report.json`

---

## 4) Quick QC (don’t overthink)

Open:

- `data/corpus_meta/qc_report.json`

Use it only to answer these questions:

- Do we have enough documents per concept?
- Are any concepts “thin” (under-covered)?
- Are there extremely short or extremely long docs that look wrong?

`qc_report.json` includes:
- `length_distribution_words`
- `per_concept_coverage`
- `thin_areas` (concepts below threshold)
- `thin_threshold`

### Adjust thin-area sensitivity (optional)

If “thin areas” is too strict/too loose:

```bash
py -m src.indexing.corpus_pipeline --thin-threshold 3
```

Recommended undergrad behavior:
- Only change this if your coverage report is obviously not helpful.

---

## 5) Freeze an immutable snapshot (required for benchmarking)

### Command

```bash
make snapshot
```

Equivalent:

```bash
py -m src.indexing.snapshot_tool create
```

### What snapshot ID is based on

Snapshot ID is computed from:
- corpus manifest content
- normalized clean-file hashes
- coverage metadata
- optional chunk count (when chunk manifest exists)

### Outputs

- `data/snapshots/<snapshot_id>/manifest.json`
- `data/snapshots/<snapshot_id>/material.json`

---

## 6) Verify snapshot integrity (recommended)

```bash
py -m src.indexing.snapshot_tool verify --snapshot-id <snapshot_id>
```

Expected: `"valid": true`.

---

## 7) Benchmark freeze switch (required)

Bench runs must include a snapshot ID.

With Make:

```bash
make bench SNAPSHOT_ID=<snapshot_id>
```

Direct:

```bash
py bench/run_bench.py --config E --query "bench smoke test" --snapshot-id <snapshot_id>
```

If snapshot is missing, benchmark execution should stop with an error.

---

## 8) Determinism check (simple reproducibility test)

Do this once before “final results”:

1. Run `make corpus`.
2. Run `make snapshot`, capture ID.
3. Run `make corpus` again **without changing raw files**.
4. Run `make snapshot` again.
5. Snapshot IDs should match.

If IDs don’t match:
- check whether raw files changed
- check whether your pipeline injects timestamps or non-deterministic ordering

---

## 9) Recommended thesis-friendly policy for AI-generated docs (optional)

Pick **one** of these, and state it clearly in your thesis:

### Option A (simplest): include AI-generated docs in corpus
- You allow `ai_generated=true` docs to be indexed and retrieved.
- You acknowledge in the thesis that parts of the corpus are AI-generated.

### Option B (more defensible): exclude AI-generated docs from benchmark snapshots
- You still keep `ai_generated=true` docs for development/demo.
- But for the final benchmark snapshot, you generate a snapshot from only non-AI docs (implementation-dependent).

If you can’t easily filter at pipeline level, just do Option A and be transparent.

---

## 10) “Done” definition for Phase 2

You are done with Phase 2 when:

- `make corpus` runs successfully
- `make snapshot` produces a snapshot ID
- `snapshot_tool verify` returns valid
- `make bench SNAPSHOT_ID=...` runs without errors

That’s sufficient to proceed to Phase 3 (chunking + index building) and Phase 7 (benchmarking).
