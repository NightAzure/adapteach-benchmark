# AdapTeach Technical Foundation

Phase 1 scaffold for the AdapTeach technical progression checklist.

## What this includes (Phase 1)

- Repo/task foundations (`Makefile`, locked deps, Docker baseline)
- Config system (`config.yaml` + `config.schema.json` + env overrides)
- Pipeline toggles (`--config A|B|C|D|E`) with stable output shape
- Dry-run modes (`--dry-run retrieval|graph`)
- Structured run logging with trace IDs (`logs/runs.jsonl`)
- Basic tests for Phase 1 gates (`tests/test_phase1.py`)

## What this includes (Phase 2)

- Corpus ingestion into a common schema (`src/indexing/corpus_pipeline.py`)
- Normalization: boilerplate stripping + code block normalization
- Duplicate removal: exact/near-exact duplicates only
- QC outputs: length distribution, per-concept coverage, thin-area report
- Immutable snapshot tooling (`src/indexing/snapshot_tool.py`)
- Optional `ai_generated` flag support in corpus and snapshot filtering
- Benchmark freeze enforcement (requires snapshot ID in `bench/run_bench.py`)

## What this includes (Phase 3)

- Fixed chunker baseline with deterministic chunk IDs
- AST/cAST-style chunker with:
  - function/class/block/statement boundaries
  - scope header injection
  - split/merge heuristics
  - fixed-chunker fallback on parse failures
- Chunk stats and parser failure reports
- Deterministic vector + BM25 index builder with versioned index hash

## What this includes (Phase 4)

- CKG builder with concept/subskill/misconception/resource nodes
- CKG edges for prerequisite/related/addresses/misconception-of
- Query-to-concept mapper with confidence + explanation
- Lightweight CPG with CONTAINS/CALLS/DEF-USE edges
- Graph validation and sanity reports

## What this includes (Phase 5)

- Config A baseline mode with safety prompt logging
- Config B dense retrieval with inspectable top-k output
- Config C hybrid retrieval + RRF fusion debug
- Config D AST/cAST retrieval mode + parser fallback visibility
- Config E graph expansion (CKG/CPG) + rerank + evidence-path debug

## What this includes (Phase 6)

- Artifact JSON schemas (Parsons/Tracing/Mutation/Flashcard)
- Strict schema + unsafe-content validation
- Deterministic artifact correctness checks
- Validation fallback with deterministic regeneration
- Case bundle logging for failed validations
- Sandbox harness and policy docs

## What this includes (Phase 7)

- Benchmark query-set preparation (`bench/prepare_benchmark.py`)
- Cross-config benchmark runner (`bench/run_benchmark_suite.py`)
- Label agreement calculation (Cohen's kappa) (`bench/compute_kappa.py`)
- Label/query freeze manifest with hashes (`bench/freeze_labels.py`)
- Metrics pipeline from run logs (`bench/compute_metrics.py`)
- Ablation runner (`bench/run_ablations.py`)

## Quick start

### Option A: with `make`

```bash
make setup
make test
make bench
```

### Option B: Windows without `make`

```bash
py -m pip install -r requirements.lock
py -m pytest -q -p no:cacheprovider tests/test_phase1.py
py -m src.cli --config E --query "Explain loops"
```

## How to run the pipeline

```bash
# baseline (no retrieval)
py -m src.cli --config A --query "Explain Python loops"

# retrieval only dry run (no generation call)
py -m src.cli --config C --query "Explain Python loops" --dry-run retrieval

# graph-expansion dry run (no generation call)
py -m src.cli --config E --query "Explain Python loops" --dry-run graph
```

Each run appends a JSONL record to `logs/runs.jsonl`.

## Phase 2 corpus workflow

```bash
# 1) Build normalized corpus + manifest + QC outputs
make corpus

# 2) Create immutable snapshot from manifest + clean corpus
make snapshot

# Optional: create benchmark snapshot excluding ai_generated docs
make snapshot EXCLUDE_AI=1

# 3) Run benchmark smoke command with required snapshot id
make bench SNAPSHOT_ID=<snapshot_id>
```

Outputs are created in:

- `data/corpus_clean/`
- `data/corpus_meta/corpus_manifest.json`
- `data/corpus_meta/corpus_manifest.csv`
- `data/corpus_meta/qc_report.json`
- `data/corpus_meta/dedup_report.json`
- `data/snapshots/<snapshot_id>/manifest.json`

## Phase 3 chunking and indexing

```bash
# Build fixed chunks
py -m src.chunking.build_chunks --chunker fixed

# Build AST/cAST chunks
py -m src.chunking.build_chunks --chunker ast

# Build indexes from chunk manifest and snapshot
py -m src.indexing.build_indexes --snapshot-id <snapshot_id>
```

Makefile path:

```bash
make index SNAPSHOT_ID=<snapshot_id> CHUNKER=ast
```

## Phase 4 graph build

```bash
# Prereq: corpus + chunk manifest should already exist
make graphs
```

Outputs:

- `graphs/ckg.json`
- `graphs/cpg.json`
- `graphs/ckg_validation_report.json`
- `graphs/cpg_sanity_report.json`

## Phase 5 pipeline configs

```bash
py -m src.cli --config A --query "Explain loops"
py -m src.cli --config B --query "Explain loops"
py -m src.cli --config C --query "Explain loops"
py -m src.cli --config D --query "Explain loops"
py -m src.cli --config E --query "Explain loops"
```

## Phase 6 reliability layer

Validation modules:

- `src/validation/validator.py`
- `src/validation/deterministic_checks.py`
- `src/validation/schema_validator.py`
- `src/validation/fallbacks.py`

Policy docs:

- `docs/PHASE6_RUNBOOK.md`
- `docs/SANDBOX_POLICY.md`
- `docs/FALLBACK_POLICY.md`

## Config and overrides

- Main app config: `config.yaml`
- Validation schema: `config.schema.json`
- Per-pipeline toggles: `configs/A.yaml` to `configs/E.yaml`
- Shared defaults reference: `configs/defaults.yaml`

Env override format:

```bash
# Example: override retrieval k from config.yaml
# PowerShell:
$env:ADAPTEACH__retrieval__k = "9"
py -m src.cli --config B --query "test"
```

Provider configuration notes:

- `llm.provider`: `mock`, `gemini`, or `ollama`
- `llm.model`: provider model id/tag
- `llm.gemini_api_key_env`: env var name that stores Gemini key (default `GEMINI_API_KEY`)
- `llm.ollama_base_url`: Ollama base URL (default `http://localhost:11434`)

## Project docs

- Operational runbook: `docs/PHASE1_RUNBOOK.md`
- Corpus/snapshot runbook: `docs/PHASE2_RUNBOOK.md`
- Chunking/indexing runbook: `docs/PHASE3_RUNBOOK.md`
- Graphs runbook: `docs/PHASE4_RUNBOOK.md`
- Pipelines runbook: `docs/PHASE5_RUNBOOK.md`
- Validators/reliability runbook: `docs/PHASE6_RUNBOOK.md`
- Benchmarking runbook: `docs/PHASE7_RUNBOOK.md`
