# Reproducing the AdapTeach Benchmark

Complete step-by-step guide for reproducing all benchmark results reported in the thesis.
Covers Objective 1 (retrieval evaluation, configs A–F) and Objective 2 (RAGAS generation quality).

---

## Prerequisites

### Software

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Tested on 3.10 and 3.11 |
| Ollama | Any recent | Must be installed and running as a local server |
| LLM model | `mistral` (default) | Pull with `ollama pull mistral` before running |
| Gemini API key | — | Optional; only needed if you choose `--provider gemini` for Obj2 |

Install Ollama from https://ollama.com, then pull the model:

```bash
ollama pull mistral
```

Confirm Ollama is serving at `http://localhost:11434` (default). You can verify with:

```bash
curl http://localhost:11434/api/tags
```

### Hardware

- **RAM:** 8 GB minimum; 16 GB recommended (sentence-transformers loads a 90 MB model)
- **Disk:** ~500 MB for indexes + run outputs
- **GPU:** Not required; all inference runs on CPU via Ollama

### Python Dependencies

Core dependencies (Objective 1):

```bash
pip install -r requirements.lock
```

Additional dependencies for Objective 2 (RAGAS evaluation) — **not in requirements.lock**:

```bash
# Only needed if using --provider gemini:
pip install langchain-google-genai google-genai
```

---

## Environment Setup

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
# GEMINI_API_KEY=your_key_here   # Objective 2 only
```

To override any `config.yaml` value without editing the file, use the `ADAPTEACH__` prefix
with double-underscores as separators:

```bash
export ADAPTEACH__LLM__PROVIDER=ollama
export ADAPTEACH__RETRIEVAL__K=5
export ADAPTEACH__LLM__MODEL=mistral
```

Config precedence (highest to lowest): shell env vars > `ADAPTEACH__` overrides > `config.yaml`.

---

## Pipeline Configurations

Six ablation configs are evaluated. Each is defined in `configs/<letter>.yaml`.

| Config | Retrieval | Chunking | Hybrid (BM25) | Graph Expansion | Graph Context | Purpose |
|--------|-----------|----------|---------------|-----------------|---------------|---------|
| **A** | Disabled | Fixed | No | No | No | No-retrieval baseline |
| **B** | Dense only | Fixed | No | No | No | Dense vector reference |
| **C** | Dense + BM25 | Fixed | Yes (RRF) | No | No | Hybrid on fixed chunks |
| **D** | Dense only | AST | No | No | No | Dense on code-aware chunks |
| **E** | Dense + BM25 | AST | Yes (RRF) | Yes | No | Hybrid + graph expansion |
| **F** | Dense only | AST | No | No | Yes | Dense + graph context |

**Config B is the reference config** — all significance tests report deltas relative to B.

---

## Step 1: Validate Datasets

Verify all four frozen query sets are present and intact (no network required):

```bash
make datasets
# or: python bench/fetch_datasets.py --mode validate
```

Expected output: `bench/datasets_manifest.lock.json` with SHA256 checksums and metadata
for all four query sets (custom, cs1qa, mbpp, staqc).

To refresh the publicly sourced sets from upstream GitHub (optional):

```bash
python bench/fetch_datasets.py --mode refresh --sources cs1qa mbpp
```

Note: `custom` and `staqc` are frozen and cannot be refreshed. `staqc` refresh is
disabled intentionally due to upstream format inconsistency.

---

## Step 2: Build Chunk Manifest

Parse the corpus into retrievable chunks (AST-based for code, fixed-size for text).

> **Note:** `data/corpus/` ships pre-populated with 100 documents. No ingest step is needed — run the command below directly.

```bash
make chunks
# or: python bench/build_chunks.py
```

Outputs written to `data/corpus_meta/`:
- `chunk_manifest.json` — all chunks with content, type tag (`fixed-v1`, `ast-v2`, `text-v1`), and concept tags
- `chunk_stats_report.json` — summary counts and size distribution
- `parser_failure_report.json` — any documents where AST parsing failed (falls back to text chunking)

Chunk parameters (hardcoded):
- Chunk size: **450 characters**
- Code chunks (`ast-v2`): extracted from fenced Python code blocks, split at function/class/statement boundaries using Python's built-in `ast` module
- Text fallback (`text-v1`): fixed-size word-boundary splits when no parseable code blocks are found
- Fixed chunks (`fixed-v1`): used by configs A–C regardless of content type

---

## Step 3: Build Retrieval Indexes

Build the vector (dense) and BM25 keyword indexes from the frozen corpus snapshot:

```bash
make index
# or: python -m src.indexing.build_indexes --snapshot-id 1538fae68752ebed
```

The snapshot ID `1538fae68752ebed` identifies the exact frozen version of the corpus used
in the reported results. Do not change this unless you are building a new corpus version.

Outputs written to `indexes/<index_id>/`:
- `vector_index.json` — semantic embeddings (all-MiniLM-L6-v2, 384-dim, L2-normalized)
- `bm25_index.json` — BM25 parameters (k1=1.2, b=0.75), token index, document frequencies
- `manifest.json` — index metadata (snapshot_id, embedder version, creation timestamp)
- `build_log.json` — build status and file paths

**Embedder fallback:** If `sentence-transformers` fails to load (e.g., missing PyTorch),
the indexer falls back to a deterministic 64-dim hash embedding. Results will differ from
the reported numbers in this case. Ensure `sentence-transformers` and `torch` install
correctly from `requirements.lock`.

Expected runtime: ~5 minutes on CPU with sentence-transformers loaded.

---

## Step 4: Build Silver Labels

Generate deterministic relevance labels for all query-chunk pairs:

**Primary benchmark (custom dataset only — recommended for initial runs):**

```bash
make labels
# or: python bench/build_silver_labels.py --benchmark bench/benchmarks/obj1_primary.yaml
```

**Full benchmark (all four datasets):**

```bash
python bench/build_silver_labels.py --benchmark bench/benchmarks/obj1_full.yaml
```

Outputs written to `bench/labels/`:
- `silver_labels_custom.csv`
- `silver_labels_cs1qa.csv` (full only)
- `silver_labels_mbpp.csv` (full only)
- `silver_labels_staqc.csv` (full only)

**Label schema:** `query_id, query, chunk_id, doc_id, relevance, silver_score, label_source, notes`

**Relevance scale:**
- `2` — Highly relevant (top 2 scoring chunks per query)
- `1` — Relevant (next 4 scoring chunks per query)
- `0` — Not relevant (below threshold)

**Scoring signals (hardcoded weights):**
- Lexical F1 (text overlap): 55%
- Title overlap: 25%
- Topic match (concept tags): 35%
- Category bonuses, chunk length/type bonuses applied on top

Minimum score to receive any label: **0.20**. Labels are fully deterministic — identical
results on every run given the same chunk manifest.

---

## Step 5: Run Objective 1 — Retrieval Benchmark

### Primary benchmark (recommended — faster, high-signal)

Runs 120 custom queries across all 6 configs (A–F). Dry-run mode skips LLM generation
and evaluates retrieval only:

```bash
make obj1-primary
# or: python bench/run_obj1.py --benchmark bench/benchmarks/obj1_primary.yaml
```

Total eval rows: 120 queries × 6 configs = 720 rows.

### Full benchmark (complete — all datasets)

Runs 1,557 queries across configs B–F (Config A is excluded from the full run spec):

```bash
make obj1-full
# or: python bench/run_obj1.py --benchmark bench/benchmarks/obj1_full.yaml
```

Total eval rows: 1,557 queries × 5 configs = 7,785 rows. Allow 20–60 minutes depending
on hardware and whether generation is enabled.

### Smoke test (quick verification)

5 queries per dataset, primary spec only:

```bash
make obj1-smoke
```

### CLI flags for run_obj1.py

| Flag | Default | Description |
|------|---------|-------------|
| `--benchmark` | `obj1_primary.yaml` | YAML benchmark spec |
| `--configs` | (from spec) | Override config list, e.g. `B,D,E` |
| `--provider` | (from config.yaml) | LLM provider: `ollama`, `gemini`, `mock` |
| `--model` | (from config.yaml) | Model name, e.g. `mistral` |
| `--delay` | 0 | Seconds between requests (rate limiting) |
| `--dry-run` | `retrieval` | `none` (full gen), `retrieval` (skip gen), `graph` |
| `--sample-per-dataset` | 0 (all) | Limit queries per dataset for quick tests |
| `--out-dir` | `bench/runs/` | Output directory |

### Outputs

Written to `bench/runs/`:
- `run_<timestamp>_<benchmark>.jsonl` — one row per query+config with retrieval debug info and (if enabled) generated artifact
- `run_<timestamp>_<benchmark>.manifest.json` — run metadata (configs, query counts, spec SHA256, provider, model)

---

## Step 6: Score Objective 1

Score the latest run against silver labels, compute metrics, and run significance tests:

```bash
make obj1-score
# or: python bench/score_obj1.py --reference-config B
```

### CLI flags for score_obj1.py

| Flag | Default | Description |
|------|---------|-------------|
| `--run-file` | (latest in bench/runs/) | Specific run JSONL to score |
| `--runs-dir` | `bench/runs/` | Directory to find latest run |
| `--labels-dir` | `bench/labels/` | Directory with silver label CSVs |
| `--reference-config` | `B` | Config used as significance test baseline |
| `--out-dir` | `bench/results/obj1_latest/` | Output directory |
| `--seed` | `13` | Random seed for bootstrap/randomization tests |

### Metrics

| Metric | Description |
|--------|-------------|
| MRR@10 | Mean Reciprocal Rank at cutoff 10 |
| nDCG@10 | Normalized Discounted Cumulative Gain at cutoff 10 |
| P@5 | Precision at 5 |
| Hit@5 | Binary: was any relevant doc in top 5? |
| retrieval_latency_ms | Wall-clock retrieval time per query |

### Outputs

Written to `bench/results/obj1_latest/`:
- `per_query.csv` — per-query metric breakdown by dataset and config
- `summary.csv` — aggregated means with 95% bootstrap CI (2000 iterations, seed=13)
- `significance_vs_reference.csv` — delta metrics vs. Config B with p-values from paired randomization test (5000 iterations, seed=13)

---

## Step 6b: Validate Silver Label Quality (Recommended)

Silver labels are generated using a lexical scoring function (keyword overlap, title match,
concept tags). To address the circularity concern — where keyword-based labels could
systematically favour keyword-based retrievers — two independent validation checks are
provided. Both are optional but strongly recommended before reporting results.

### Option A: LLM-as-Judge cross-validation (primary)

Uses an external LLM (ChatGPT or Gemini) to independently score chunk relevance for each
query-chunk pair. Agreement with silver labels is measured via Cohen's κ.

**κ ≥ 0.60** = substantial independent validation, cite as methodological support.
**κ 0.40–0.60** = moderate agreement, acknowledge as a limitation.
**κ < 0.40** = weak alignment, consider revising labels.

#### Step 1 — Export judging batches

```bash
python bench/export_for_llm_judge.py
```

Outputs written to `bench/llm_judge/input/`:
- `judge_input_batch_01.csv` through `judge_input_batch_09.csv` (80 rows each)
- `../PROMPT.txt` — instructions to paste into ChatGPT / Gemini

Each CSV has columns: `row_id, query_id, query_text, chunk_id, chunk_text, silver_label`
(chunk text is truncated to 300 characters to fit within chat context windows).

#### Step 2 — Run each batch through ChatGPT or Gemini

For each batch file:
1. Open ChatGPT (GPT-4 or above) or Gemini 1.5 Pro / 2.0 Flash
2. Upload `judge_input_batch_XX.csv` as a file attachment
3. Paste the full contents of `PROMPT.txt` as your message
4. The model returns only two-column CSV rows: `row_id,llm_score`
5. Copy the response and save it as `judge_scores_batch_XX.csv` in `bench/llm_judge/scores/`

The prompt instructs the model to return **only** bare CSV rows with no headers or
explanation. GPT-4 and Gemini 1.5+ follow this reliably. If stray text appears,
`import_llm_scores.py` skips non-numeric lines automatically.

#### Step 3 — Compute Cohen's κ

```bash
python bench/import_llm_scores.py
```

Reads all `judge_scores_batch_*.csv` from `bench/llm_judge/scores/`, merges with silver
labels by `row_id`, and prints agreement statistics.

Output written to `bench/results/llm_judge_summary.json`:
- `cohens_kappa` — primary agreement metric
- `pct_exact_agreement` — fraction of pairs with identical scores
- `pct_within_one` — fraction of pairs within 1 score point
- `score_distribution` — label distribution for both raters
- `disagreement_breakdown` — count of each silver→llm discrepancy pattern

**CLI flags for import_llm_scores.py:**

| Flag | Default | Description |
|------|---------|-------------|
| `--scores-dir` | `bench/llm_judge/scores/` | Directory with judge score CSVs |
| `--out` | `bench/results/llm_judge_summary.json` | Summary output path |

**Dependencies:** `scikit-learn` (for `cohen_kappa_score`). If not installed, κ is
computed via a built-in fallback — no additional install required.

---

### Option B: Semantic cross-validation (supplementary)

Computes Spearman ρ between silver label rankings and cosine similarity rankings from
`all-MiniLM-L6-v2`. This measures whether lexical labels are broadly aligned with an
independent embedding signal. Use as a supplementary check; it does not replace Option A.

**ρ ≥ 0.60** = acceptable alignment, partially defuses circularity.

Requires a completed Obj1 run file (Step 5):

```bash
python bench/validate_silver_labels.py
# or: python bench/validate_silver_labels.py --run-file bench/runs/run_<timestamp>.jsonl
```

Output written to `bench/results/silver_validation_summary.json` and
`bench/results/silver_validation_per_query.csv`.

**Dependencies:**

```bash
pip install sentence-transformers scipy
```

---

## Step 7: Objective 2 — RAGAS Evaluation

Objective 2 evaluates generation quality using the RAGAS framework. It is **not** part of
`make reproduce` and must be run separately.

**Default provider: Ollama (local, no API key needed).** Gemini is available as an
optional fallback via `--provider gemini --api-key YOUR_KEY`.

Install the extra dependencies (not in requirements.lock):

```bash
pip install "ragas==0.4.3" langchain-ollama langchain-core datasets
```

### Sub-step A: Build golden reference answers

Golden answers are LLM-generated reference responses used as RAGAS ground truth.
Run once against an existing Obj1 run file:

```bash
python bench/run_obj2.py build-golden \
  --queries bench/queries_custom.jsonl \
  --run-file bench/runs/run_<timestamp>_obj1_primary.jsonl \
  --out bench/results/golden_custom.jsonl
```

Replace `<timestamp>` with the actual filename from `bench/runs/`. The command uses
Ollama + `mistral` by default (reads `OLLAMA_BASE_URL` and `OLLAMA_MODEL` from `.env`).
No rate-limiting delay is needed for local Ollama (default `--delay 0`).

### Sub-step B: Run RAGAS evaluation

```bash
python bench/run_obj2.py evaluate \
  --run-file bench/runs/run_<timestamp>_obj1_primary.jsonl \
  --golden bench/results/golden_custom.jsonl \
  --out bench/results/obj2_ragas.csv \
  --configs A,B,D,E,F
```

Default configs evaluated: A, B, D, E, F (C is omitted as it differs only in BM25 fusion,
not generation).

**Output:** `bench/results/obj2_ragas.csv` with RAGAS metrics: faithfulness, answer
relevancy, context precision, context recall per config.

Or use the Makefile shortcut (uses `.env` defaults):

```bash
make obj2
```

### Optional: Gemini provider

If you want to use Gemini instead of Ollama for golden answer generation or evaluation:

```bash
python bench/run_obj2.py build-golden \
  --provider gemini \
  --api-key $GEMINI_API_KEY \
  --delay 7.0 \
  --queries bench/queries_custom.jsonl \
  --run-file bench/runs/run_<timestamp>_obj1_primary.jsonl \
  --out bench/results/golden_custom.jsonl
```

The `--delay 7.0` is recommended for the Gemini free tier to avoid rate limit errors.

---

## Makefile Reference

| Target | Description |
|--------|-------------|
| `make help` | List all targets |
| `make setup` | Install dependencies from requirements.lock |
| `make datasets` | Validate shipped query files |
| `make chunks` | Build chunk manifest from corpus |
| `make index` | Build vector + BM25 indexes (snapshot 1538fae68752ebed) |
| `make labels` | Build silver labels (primary benchmark / custom dataset) |
| `make reproduce` | Full primary run: datasets → chunks → index → labels → obj1-primary → obj1-score |
| `make obj1-primary` | Run Obj1 with primary benchmark spec (all configs, custom queries) |
| `make obj1-full` | Run Obj1 with full benchmark spec (all datasets, configs B–F) |
| `make obj1-score` | Score the latest run against silver labels |
| `make obj1-smoke` | Quick 5-query smoke test + score |
| `make obj2` | Run Objective 2 RAGAS evaluation |
| `make test` | Syntax-check all bench scripts |

**Label validation scripts (no Makefile target — run directly):**

| Script | Description |
|--------|-------------|
| `python bench/export_for_llm_judge.py` | Export batched CSVs for manual LLM judging |
| `python bench/import_llm_scores.py` | Import LLM scores and compute Cohen's κ |
| `python bench/validate_silver_labels.py` | Semantic cross-validation (Spearman ρ) |

---

## Key Parameter Reference

These values are hardcoded in the pipeline and match the reported experimental setup:

| Parameter | Value | Location |
|-----------|-------|----------|
| Corpus snapshot ID | `1538fae68752ebed` | Makefile, `config.yaml` |
| Chunk size | 450 characters | `bench/build_chunks.py` |
| Retrieval top-k | 5 | `config.yaml` |
| RRF fusion k | 5 | `config.yaml` |
| Rerank weights | retrieval=0.70, graph=0.30 | `config.yaml` |
| BM25 k1 | 1.2 | `src/indexing/build_indexes.py` |
| BM25 b | 0.75 | `src/indexing/build_indexes.py` |
| Embedding model | `all-MiniLM-L6-v2` (384-dim) | `src/indexing/build_indexes.py` |
| Embedding fallback | Hash-based (64-dim) | `src/indexing/build_indexes.py` |
| Silver label high-k | 2 (relevance = 2) | `bench/build_silver_labels.py` |
| Silver label mid-k | 4 (relevance = 1) | `bench/build_silver_labels.py` |
| Silver label min-score | 0.20 | `bench/build_silver_labels.py` |
| Bootstrap CI iterations | 2000 | `bench/score_obj1.py` |
| Significance test iterations | 5000 | `bench/score_obj1.py` |
| Random seed | 13 | `bench/score_obj1.py` |
| Default LLM | mistral (Ollama) | `config.yaml`, `.env` |
| Ollama base URL | `http://localhost:11434` | `.env` |
| Obj2 LLM delay | 0 s (Ollama default); 7 s recommended for Gemini free tier | `bench/run_obj2.py` |

---

## Dataset and Corpus Provenance

### Query Datasets

| Dataset | Origin | License | Notes |
|---------|--------|---------|-------|
| `custom` | Internal development set | N/A | Frozen; not refreshable |
| `cs1qa` | [cyoon47/CS1QA](https://github.com/cyoon47/CS1QA) on GitHub | Research use | Refreshable; filtered to 437 questions |
| `mbpp` | [google-research/mbpp](https://github.com/google-research/mbpp) on GitHub | CC BY 4.0 | Refreshable; filtered to 500 problems |
| `staqc` | StaQC extract | Research use | Frozen; upstream refresh disabled |

### Corpus Documents

The corpus in `data/corpus/` was assembled from multiple public sources:

| Source | License | Count | Redistribution |
|--------|---------|-------|----------------|
| Python official docs (python.org) | PSF Documentation License | ~20 docs | Open with attribution |
| *Automate the Boring Stuff with Python* 2e | CC BY-NC-SA 3.0 | ~30 docs | Non-commercial only |
| *Think Python* 2e | CC BY-NC 3.0 | ~20 docs | Non-commercial only |
| Python PEPs | Public domain | ~5 docs | Freely redistributable |
| AI-generated reference pages | N/A | ~26 docs | Original to this project |

**Construction process:** Documents were manually scraped from their respective public URLs
using `scripts/scrape_corpus.py`, then selectively edited using an LLM to embed
representative Python code examples, concept explanations, and illustrative snippets
appropriate for a CS1 tutoring context. The AI-editing step augmented chapters that were
primarily prose with inline code demonstrations tied to the queried concept tags.

**Redistribution restriction:** Because a significant portion of the corpus derives from
CC BY-NC-SA and CC BY-NC licensed works, **the full `data/corpus/` directory cannot
be redistributed in derivative projects or used in commercial applications without explicit
permission from the original authors** (Al Sweigart for ATBS; Allen Downey for Think Python).

**To rebuild the corpus from scratch:**

```bash
pip install requests beautifulsoup4
python scripts/scrape_corpus.py --out-dir data/corpus/ --sources all
```

Available source targets: `python_docs`, `atbs`, `think_python`, or `all`.
The scraper applies rate limiting (1.2 s per request). The AI-edited enhancements
are not reproducible from the scraper alone — those edits exist only in `data/corpus/`.
