# Phase 5 Runbook

This runbook covers Retrieval Pipelines A-E.

## 1. Scope

Phase 5 provides switchable configs:

- A: Base LLM (no retrieval)
- B: Dense-only RAG
- C: Hybrid RAG (BM25 + dense + RRF fusion)
- D: AST/cAST retrieval mode with parser fallback logging
- E: Dual-graph GraphRAG (CKG + CPG expansion + rerank)

## 2. Prerequisites

1. Build corpus: `make corpus`
2. Build chunks: `make chunk-ast` (or `make chunk-fixed`)
3. Create snapshot: `make snapshot`
4. Build indexes: `make index SNAPSHOT_ID=<snapshot_id> CHUNKER=ast`
5. Build graphs: `make graphs`

## 3. Running configs

```bash
py -m src.cli --config A --query "Explain loops"
py -m src.cli --config B --query "Explain loops"
py -m src.cli --config C --query "Explain loops"
py -m src.cli --config D --query "Explain loops"
py -m src.cli --config E --query "Explain loops"
```

## 4. Config behavior summary

### Config A

- No retrieval path
- Baseline response
- Safety system prompt logged in generation stage

### Config B

- Dense top-k retrieval only
- Retrieval debug includes inspectable top-k list

### Config C

- Dense + BM25 retrieval
- RRF fusion
- Fusion debug includes component ranks + fused scores

### Config D

- Retrieval uses AST/cAST chunking preference
- Parser fallback visibility logged in retrieval debug

### Config E

- Starts from hybrid retrieval
- CKG expansion (concept neighbor resources)
- CPG expansion (CONTAINS/CALLS/DEF-USE 1-hop)
- Weighted rerank with graph bonus + diversity penalty
- Graph evidence paths included in debug output

## 5. Dry-run modes

```bash
py -m src.cli --config E --query "Explain loops" --dry-run retrieval
py -m src.cli --config E --query "Explain loops" --dry-run graph
```

## 6. Inspection points in logs

All runs write to `logs/runs.jsonl`.

Key stage debug fields:

- retrieval stage:
  - dense top-k (B)
  - fusion debug (C)
  - parser fallback stats (D)
- graph expansion stage:
  - query concept mapping
  - evidence paths
  - rerank weights/details (E)

## 7. Gate checks for Phase 5

- Config A runnable and logged
- Config B top-k retrieval inspectable
- Config C fused ranking inspectable
- Config D AST-mode retrieval + fallback logging visible
- Config E explains why chunks are included (retrieval + graph rationale)
