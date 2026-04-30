# Benchmark folder layout

## Core benchmark path

- `bench/benchmarks/obj1_primary.yaml` — default high-signal Objective 1 benchmark (custom query set only)
- `bench/benchmarks/obj1_full.yaml` — full four-dataset Objective 1 benchmark
- `bench/fetch_datasets.py` — dataset validation or refresh
- `bench/build_silver_labels.py` — offline deterministic label generation
- `bench/run_obj1.py` — Objective 1 runner
- `bench/score_obj1.py` — Objective 1 scoring and significance
- `bench/run_obj2.py` — Objective 2 wrapper
- `bench/REPRODUCE_BENCHMARK.md` — detailed commands and expected outputs

## Frozen query sets used by default

- `queries_custom.jsonl`
- `queries_cs1qa.jsonl`
- `queries_mbpp.jsonl`
- `queries_staqc.jsonl`

## Legacy scripts

Anything in `bench/legacy/` is retained only for reference and should not be used for the cleaned benchmark workflow.
