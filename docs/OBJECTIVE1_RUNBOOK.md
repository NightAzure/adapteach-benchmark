# Objective 1 Runbook (Step-by-Step, Copy-Paste Ready)

Use this when you want to complete **Objective 1** end-to-end without guessing commands.

Objective 1 focus:
- Iteratively design a structure-aware retrieval framework
- Compare retrieval framework components across configs A-E
- Primary comparison set: `B, C, D, E`
- `A` is baseline (no retrieval, LLM only)

| Config | Chunking | Retrieval | Graph | Rerank |
|--------|----------|-----------|-------|--------|
| A | — | none | no | no |
| B | fixed | dense | no | no |
| C | fixed | hybrid (dense + BM25 + RRF) | no | no |
| D | AST | dense | no | no |
| E | AST | hybrid (dense + BM25 + RRF) | yes (CKG + CPG) | yes |

---

## 0) Prerequisites

**Environment setup** (from project root `C:\Users\Nokie\Desktop\Thesis_Proj`):

```powershell
py -m pip install -r requirements.lock
```

**Set Gemini API key** (required for full generation runs):

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
```

Verify:
```powershell
echo $env:GEMINI_API_KEY
```

**Suppress TensorFlow warnings** (optional, reduces noise):

```powershell
$env:TF_ENABLE_ONEDNN_OPTS=0
```

---

## 1) Build corpus

```powershell
py -m src.indexing.corpus_pipeline
```

- Reads raw documents from `data/corpus_raw/`
- Cleans and writes to `data/corpus_clean/`
- 59 documents expected

---

## 2) Create corpus snapshot

```powershell
py -m src.indexing.snapshot_tool create
```

Output: snapshot ID (e.g. `48b71a4bc4e67618`). Save this for later steps.

---

## 3) Build chunks (AST chunker)

```powershell
py -m src.chunking.build_chunks --chunker ast --chunk-size 450 --overlap 80
```

Expected output:
- ~1,460 chunks (code + text)
- Stats saved to `data/corpus_meta/chunk_stats_report.json`
- Code chunks: ~1,140 | Text chunks: ~320

Verify:
```powershell
py -c "import json; d=json.load(open('data/corpus_meta/chunk_stats_report.json')); print(f'Total: {d[\"chunk_count\"]}  Code: {d[\"content_type_counts\"].get(\"code\",0)}  Text: {d[\"content_type_counts\"].get(\"text\",0)}')"
```

---

## 4) Build indexes (semantic embeddings)

```powershell
py -m src.indexing.build_indexes --snapshot-id <SNAPSHOT_ID> --embedder-version minilm-v2
```

Replace `<SNAPSHOT_ID>` with the value from step 2 (e.g. `48b71a4bc4e67618`).

Output: new `index_id` (e.g. `18a68eaa927189c0`). This uses `all-MiniLM-L6-v2` (384-dim).

**Update `config.yaml`** with the new values:

```yaml
retrieval:
  chunking: ast
  corpus_snapshot: 48b71a4bc4e67618   # from step 2
  index_hash: 18a68eaa927189c0         # from this step
```

Verify index manifest:
```powershell
py -c "import json; d=json.load(open('indexes/<INDEX_ID>/manifest.json')); print(json.dumps(d, indent=2))"
```

---

## 5) Build graphs (CKG + CPG)

```powershell
py -m src.graphs.build_graphs
```

- Builds Concept Knowledge Graph (CKG) and Code Property Graph (CPG)
- Required for Config E (graph expansion + rerank)

---

## 6) Smoke test retrieval

Quick sanity check that retrieval is working:

```powershell
py -c "from src.retrieval.engine import retrieve; from src.utils.config import load_app_config; r=retrieve('How do for loops work in Python?', load_app_config()); print(f'Chunks: {len(r[\"ranked\"])}'); [print(f'  [{c[\"score\"]:.3f}] {c[\"title\"]}') for c in r['ranked'][:5]]" 2>NUL
```

Expected: 5 ranked chunks with relevant titles and non-zero scores.

---

## 7) Prepare query sets

Use the Objective 1-tuned query sets (already provided):

- `bench/queries_obj1_dev.jsonl` — 30 queries for development iteration
- `bench/queries_obj1_test.jsonl` — held-out test queries

If you need to regenerate generic query sets:

```powershell
py bench/prepare_benchmark.py --dev-count 30 --test-count 30
```

---

## 8) Run benchmark suite

### Option A (recommended for iteration): retrieval-only, fast

```powershell
py bench/run_benchmark_suite.py --query-set bench/queries_obj1_dev.jsonl --configs A,B,C,D,E --dry-run retrieval
```

### Option B (full run with Gemini): slower, includes generation

```powershell
py bench/run_benchmark_suite.py --query-set bench/queries_obj1_dev.jsonl --configs A,B,C,D,E --dry-run none --provider gemini --model gemini-3-flash-preview
```

Output: `bench/runs/run_<timestamp>.jsonl`

Find latest run:
```powershell
Get-ChildItem bench/runs/run_*.jsonl | Sort-Object LastWriteTime -Descending | Select-Object -First 1 Name,LastWriteTime
```

Inspect specific results:
```powershell
py bench/inspect_run.py --run-file bench/runs/run_<timestamp>.jsonl --config E --limit 10
```

---

## 9) Create labeling sheet (for raters)

```powershell
py bench/export_labeling_sheet.py --run-file bench/runs/run_<timestamp>.jsonl --configs B,C,D,E --out-csv bench/labels/labels_sheet_nontech.csv
```

What this gives:
- Query text, chunk title + preview, `seen_in_configs` column
- Two columns to fill: `relevance_rater_a`, `relevance_rater_b`

Open and label using rubric at `bench/labels/rubric.md`:

| Score | Meaning |
|-------|---------|
| 0 | Not relevant |
| 1 | Partially relevant |
| 2 | Relevant |
| 3 | Highly relevant |

---

## 10) Convert sheet to labels + compute agreement

### Convert ratings to labels.csv

```powershell
py bench/labels_sheet_to_labels.py --sheet-csv bench/labels/labels_sheet_nontech.csv --out-csv bench/labels/labels.csv
```

### Inter-rater agreement (Cohen's kappa)

```powershell
py bench/compute_kappa.py --labels-csv bench/labels/labels.csv
```

---

## 11) Compute metrics

```powershell
py bench/compute_metrics.py --run-file bench/runs/run_<timestamp>.jsonl --labels-csv bench/labels/labels.csv --out-csv bench/runs/metrics_objective1.csv
```

Metrics computed: MRR@10, nDCG, context precision, faithfulness, answer relevance, parsability.

Output: `bench/runs/metrics_objective1.csv`

---

## 12) Run ablations

### Fast retrieval-only ablations

```powershell
py bench/run_ablations.py --query-set bench/queries_obj1_dev.jsonl --dry-run retrieval --out-csv bench/runs/ablation_results_objective1.csv
```

### Full ablations (if needed)

```powershell
py bench/run_ablations.py --query-set bench/queries_obj1_dev.jsonl --dry-run none --out-csv bench/runs/ablation_results_objective1.csv
```

Output: `bench/runs/ablation_results_objective1.csv`

---

## 13) Freeze labels and query sets

```powershell
py bench/freeze_labels.py --labels-csv bench/labels/labels.csv --queries-dev bench/queries_obj1_dev.jsonl --queries-test bench/queries_obj1_test.jsonl --out bench/labels/freeze_manifest_objective1.json
```

Output: `bench/labels/freeze_manifest_objective1.json` (SHA256 hashes for reproducibility)

---

## 14) Deliverables checklist

Required files for Objective 1 submission:

- [ ] Run logs: `bench/runs/run_<timestamp>.jsonl`
- [ ] Metrics table: `bench/runs/metrics_objective1.csv`
- [ ] Ablation results: `bench/runs/ablation_results_objective1.csv`
- [ ] Human labels: `bench/labels/labels.csv`
- [ ] Freeze manifest: `bench/labels/freeze_manifest_objective1.json`
- [ ] Inter-rater agreement (kappa) reported

Interpretation:
- Use the metrics table to identify which config performs best across retrieval quality measures
- Use the ablation table to isolate which component (chunking, retrieval mode, graph, rerank) drives gains
- Config A serves as no-retrieval baseline; improvements from B-E show retrieval value

---

## 15) Architecture quick reference

```
config.yaml
  retrieval.index_hash ──► indexes/<hash>/manifest.json
  retrieval.corpus_snapshot ──► snapshots/<hash>/
  retrieval.chunking ──► ast | fixed
  pipelines.active ──► A | B | C | D | E

configs/config_<X>.yaml  ──► per-pipeline flags
  retrieval_enabled, retrieval_mode, chunking,
  graph_expansion, rerank_enabled
```

Current active values:
- Snapshot: `48b71a4bc4e67618`
- Index: `18a68eaa927189c0` (MiniLM-v2, 384-dim, ~1,460 vectors)
- Embedder: `sentence-transformers/all-MiniLM-L6-v2`
- Active config: `E`

---

## 16) Common issues

### `answer` is null in run file
You ran `--dry-run retrieval` (expected). For full answers, use `--dry-run none`.

### Gemini key not set
```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
echo $env:GEMINI_API_KEY
```

### `index_found: false` in run output
`config.yaml` `index_hash` does not match any directory under `indexes/`. Verify:
```powershell
Get-ChildItem indexes/ -Directory | Select-Object Name
```
Then update `config.yaml` `retrieval.index_hash` to match.

### TensorFlow/oneDNN warnings flooding output
```powershell
$env:TF_ENABLE_ONEDNN_OPTS=0
```
Or append `2>NUL` to commands.

### Permission error writing CSV
File is open in Excel/editor. Close it and rerun, or write to a new filename.

### Metrics show `labeled_rows = 0`
`labels.csv` may be empty or `query_id`/`chunk_id` values don't align with the run file. Re-export the labeling sheet from the correct run file and re-convert.

### `py` command not found
Use `python` instead of `py`, or verify Python launcher is installed.
