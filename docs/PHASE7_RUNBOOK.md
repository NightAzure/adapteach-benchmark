# Phase 7 Runbook: Benchmark Harness and Evaluation

This runbook covers the technical workflow for benchmark data prep, run execution, labeling, agreement checks, metrics, and ablations.

## 1) Prerequisites

- Python dependencies installed:
  - `py -m pip install -r requirements.lock`
- Corpus, chunks, indexes, and graphs already built (Phase 2 to Phase 4).

## 2) Prepare Query Sets and Label Template

Generate dev/test query sets and a starter label template:

```powershell
py bench/prepare_benchmark.py --dev-count 30 --test-count 30
```

Outputs:

- `bench/queries_dev.jsonl`
- `bench/queries_test.jsonl`
- `bench/labels/labels_template.csv`

## 3) Run Benchmark Suite Across Configs A-E

Run all configs on the dev set:

```powershell
py bench/run_benchmark_suite.py --query-set bench/queries_dev.jsonl --configs A,B,C,D,E
```

Fast retrieval-only mode (no LLM generation, recommended for Objective 1 iteration):

```powershell
py bench/run_benchmark_suite.py --query-set bench/queries_dev.jsonl --configs A,B,C,D,E --dry-run retrieval
```

Optional provider override:

```powershell
py bench/run_benchmark_suite.py --query-set bench/queries_dev.jsonl --configs A,B,C,D,E --provider ollama --model llama3
py bench/run_benchmark_suite.py --query-set bench/queries_dev.jsonl --configs A,B,C,D,E --provider gemini --model gemini-1.5-flash
```

Output:

- `bench/runs/run_<timestamp>.jsonl`

## 4) Labeling Workflow

1. Copy `bench/labels/labels_template.csv` to `bench/labels/labels.csv`.
2. Fill in `chunk_id` and `relevance` values.
3. Recommended relevance scale:
   - `0` = not relevant
   - `1` = somewhat relevant
   - `2` = relevant
- `3` = highly relevant

Non-technical workflow (recommended):

1. Export a readable deduplicated sheet from a run:

```powershell
py bench/export_labeling_sheet.py --run-file bench/runs/run_<timestamp>.jsonl --configs B,C,D,E --out-csv bench/labels/labels_sheet.csv
```

2. Open `bench/labels/labels_sheet.csv` and fill:
   - `relevance_rater_a`
   - `relevance_rater_b`

3. Convert sheet to metrics-ready `labels.csv`:

```powershell
py bench/labels_sheet_to_labels.py --sheet-csv bench/labels/labels_sheet.csv --out-csv bench/labels/labels.csv
```

## 5) Inter-Rater Agreement (Cohen's kappa)

```powershell
py bench/compute_kappa.py --labels-csv bench/labels/labels.csv
```

## 6) Freeze Labels and Query Sets (Hash Manifest)

```powershell
py bench/freeze_labels.py --labels-csv bench/labels/labels.csv --queries-dev bench/queries_dev.jsonl --queries-test bench/queries_test.jsonl
```

Output:

- `bench/labels/freeze_manifest.json`

## 7) Compute Metrics

Compute metrics from the latest run:

```powershell
py bench/compute_metrics.py --labels-csv bench/labels/labels.csv
```

Output:

- `bench/runs/metrics_latest.csv`

Notes:

- If labels are provided for a query, retrieval metrics use labeled relevance.
- If labels are missing for a query, retrieval metrics fall back to proxy scoring.
- Generation metrics are currently proxy metrics (`faithfulness_proxy`, `answer_relevance_proxy`).

## 8) Run Ablation Experiments

```powershell
py bench/run_ablations.py --query-set bench/queries_dev.jsonl
```

Fast retrieval-only ablations:

```powershell
py bench/run_ablations.py --query-set bench/queries_dev.jsonl --dry-run retrieval
```

Output:

- `bench/runs/ablation_results.csv`

## 9) Makefile Shortcuts

```powershell
make bench-prepare
make bench-run
make bench-metrics
make bench-ablations
make bench-kappa
make bench
```

`make bench` runs suite + metrics + ablations on the dev set.
It now uses retrieval-only dry-run mode by default for speed.
