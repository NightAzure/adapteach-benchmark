# Phase 3 Runbook

This runbook covers chunking and index build.

## 1. Scope

Phase 3 adds:

- deterministic fixed chunking baseline
- AST/cAST-style chunking with fallback
- deterministic vector + BM25 index build
- index versioning tied to snapshot/chunker/embedder versions

## 2. Chunking options

### Fixed chunking (baseline)

```bash
py -m src.chunking.build_chunks --chunker fixed --chunk-size 400 --overlap 80
```

### AST/cAST chunking

```bash
py -m src.chunking.build_chunks --chunker ast --chunk-size 400 --overlap 80
```

Implementation choice:

- Preferred parser: Tree-sitter Python (cAST-aligned).
- Fallback parser: Python `ast` when Tree-sitter runtime is unavailable.
- Final fallback: fixed chunking when both parser paths fail.
- Reference baseline: cAST paper implementation available locally in `astchunk/` (used as behavior reference for split/merge/context concepts).

AST chunking behavior:

- chunks on function/class boundaries
- chunks blocks (`for`, `while`, `if`)
- includes statement-level chunks for smaller snippets
- injects context header (`# scope: ...`)
- splits large segments and merges tiny adjacent segments
- falls back to fixed chunking on parse failure

Outputs:

- `data/corpus_meta/chunk_manifest.json`
- `data/corpus_meta/chunk_stats_report.json`
- `data/corpus_meta/parser_failure_report.json`

## 3. Parser failure policy

On AST parse failure:

- document is chunked via fixed chunker
- failure reason and sample errors are logged to `parser_failure_report.json`

Gate check target:

- parse rate should be high for code docs; failures must fall back cleanly.

## 4. Build indexes

Requires a frozen snapshot ID from Phase 2.

```bash
py -m src.indexing.build_indexes --snapshot-id <snapshot_id>
```

Outputs:

- `indexes/<index_id>/vector_index.json`
- `indexes/<index_id>/bm25_index.json`
- `indexes/<index_id>/manifest.json`
- `indexes/<index_id>/build_log.json`

## 5. Index versioning and hash

`index_hash` is computed from:

- `snapshot_id`
- `chunker_version` (from chunk manifest)
- `embedder_version`

This makes index IDs deterministic for identical inputs.

## 6. Makefile workflow

```bash
# Phase 2 prereqs
make corpus
make snapshot

# Chunking
make chunk-fixed
make chunk-ast

# Full Phase 3 index build (set snapshot + chunker)
make index SNAPSHOT_ID=<snapshot_id> CHUNKER=ast
```

## 7. BM25 policy

- Tokenization: lowercase regex identifier tokens
- Stopword policy: fixed minimal English stopword set

## 8. Determinism check

1. Run chunk build with same inputs twice.
2. Confirm chunk IDs remain the same.
3. Build indexes twice with same `snapshot_id`, chunk manifest, and embedder version.
4. Confirm `index_id` and `index_hash` match.
