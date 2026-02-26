# Phase 4 Runbook

This runbook covers graph construction for GraphRAG foundations.

## 1. Scope

Phase 4 adds:

- Concept Knowledge Graph (CKG)
- Query-to-concept mapper
- Lightweight Code Property Graph (CPG)
- Graph sanity and validation reports

## 2. Build prerequisites

1. Build corpus (`make corpus`)
2. Build chunk manifest (`make chunk-ast` recommended for code structure)

## 3. Build graphs

```bash
make graphs
```

Equivalent:

```bash
py -m src.graphs.build_graphs
```

Outputs:

- `graphs/ckg.json`
- `graphs/ckg_validation_report.json`
- `graphs/cpg.json`
- `graphs/cpg_sanity_report.json`

## 4. CKG details

Node types:

- concept
- subskill
- misconception
- resource

Edge types:

- prerequisite
- related
- addresses
- misconception-of

Validation checks:

- prerequisite cycle detection (DAG check)
- concepts without connected resources

## 5. Query-to-concept mapping

Module: `src/graphs/query_mapper.py`

Signals combined:

- rule-based keyword mapping
- embedding similarity over concept labels
- constrained classifier vote (taxonomy-only labels)

Output includes:

- top concepts
- confidence
- explanation per concept

## 6. CPG details

Node types:

- chunk
- function
- class
- statement

Edge types:

- CONTAINS (chunk/class containment)
- CALLS (resolvable function calls only)
- DEF-USE (name-based local conservative links)

Sanity checks:

- orphan node count
- edge type counts
- bounded edge growth
- call-collision log

## 7. Determinism check

Run graph build twice with unchanged inputs.

- node IDs and edge IDs should remain stable
- sanity report should remain consistent
