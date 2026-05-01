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

Build the vector (dense) and BM25 keyword indexes from the chunk manifest:

```bash
make index
# or: python -m src.indexing.build_indexes
```

The index ID is derived automatically from the chunker and embedder versions — no manual ID needed.

Outputs written to `indexes/<index_id>/`:
- `vector_index.json` — semantic embeddings (all-MiniLM-L6-v2, 384-dim, L2-normalized)
- `bm25_index.json` — BM25 parameters (k1=1.2, b=0.75), token index, document frequencies
- `manifest.json` — index metadata (chunker version, embedder version, creation timestamp)
- `build_log.json` — build status and file paths

**Embedder fallback:** If `sentence-transformers` fails to load (e.g., missing PyTorch),
the indexer falls back to a deterministic 64-dim hash embedding. Results will differ from
the reported numbers in this case. Ensure `sentence-transformers` and `torch` install
correctly from `requirements.lock`.

Expected runtime: ~5 minutes on CPU with sentence-transformers loaded.

---

## Step 4: Build Candidate Labels (Silver Labels)

Generate lexical candidate labels used as input to the qrels pipeline (Step 6b).
These are **not** the final ground-truth labels — they serve as the silver-positive
source in the judging pool. Final ground truth comes from the LLM judge in Step 6b.

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

**Scoring signals (hardcoded weights):**
- Lexical F1 (text overlap): 55%
- Title overlap: 25%
- Topic match (concept tags): 35%
- Category bonuses, chunk length/type bonuses applied on top

Minimum score threshold: **0.55** (used in Step 6b pool building). Labels are fully
deterministic — identical results on every run given the same chunk manifest.

> **Note:** `make reproduce` uses these silver labels directly for a quick initial
> scoring pass. For the final paper results, run Step 6b to build LLM-validated qrels.

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

Score the latest run against relevance labels, compute metrics, and run significance tests.

**Quick pass against silver labels** (before running Step 6b):

```bash
make obj1-score
# or: python bench/score_obj1.py --reference-config B
```

**Final paper results against LLM-validated qrels** (after Step 6b):

```bash
python bench/score_obj1.py --reference-config B --labels-dir bench/labels
```

`score_obj1.py` reads all `*.csv` files in `--labels-dir` matching `silver_labels_*` or
`qrels_*`. After running Step 6b, both will be present; qrels take precedence for any
query_id that appears in both (last-write-wins in the labels dict).

### CLI flags for score_obj1.py

| Flag | Default | Description |
|------|---------|-------------|
| `--run-file` | (latest in bench/runs/) | Specific run JSONL to score |
| `--runs-dir` | `bench/runs/` | Directory to find latest run |
| `--labels-dir` | `bench/labels/` | Directory with label CSVs (silver or qrels) |
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

## Step 6b: Build LLM-Validated Qrels (Recommended)

The lexical silver labels are used only as a candidate nominator. Final ground-truth
relevance labels come from a blind LLM judge scoring a four-source pool that includes
positives, hard negatives, retrieved chunks from every config, and random negatives.
This eliminates circularity: dense and semantic retrievers get credit for chunks the
lexical scorer would have missed.

**Full pipeline:**
```
build_judge_pool.py → run_llm_judge.py → build_qrels.py → score_obj1.py
```

---

### Step 1 — Build the judging pool

```bash
python bench/build_judge_pool.py --benchmark bench/benchmarks/obj1_full.yaml
```

Reads the latest run file from `bench/runs/` automatically. Outputs
`bench/judge_pool/pool.csv` with columns:
`query_id, query_text, dataset, chunk_id, doc_id, chunk_text, source, silver_score`

Pool sources per query:
- `silver_positive` — chunks rated ≥ 0.55 by lexical scorer
- `hard_negative` — top-3 topic-matching chunks rejected by lexical scorer
- `retrieved` — all chunks retrieved by any config in the run file
- `random_negative` — 2 randomly sampled chunks with no topic connection

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--benchmark` | `bench/benchmarks/obj1_full.yaml` | Benchmark spec |
| `--run-file` | latest in `bench/runs/` | Run JSONL to extract retrieved chunks |
| `--min-score` | `0.55` | Lexical score threshold for silver positives |
| `--n-hard-neg` | `3` | Hard negatives per query |
| `--n-random-neg` | `2` | Random negatives per query |
| `--seed` | `42` | Random seed for reproducibility |

---

### Step 2 — Score the pool with a local LLM

**Recommended models** (pull one first):

| Model | Pull command | VRAM needed | Notes |
|-------|-------------|-------------|-------|
| `qwen3.5:9b`  | `ollama pull qwen3.5:9b`  | ~7 GB  | **Recommended** — fits on 16GB GPU, fast |
| `qwen3.6:27b` | `ollama pull qwen3.6:27b` | ~17 GB | Latest Qwen; requires 20GB+ VRAM |
| `qwen3.5:27b` | `ollama pull qwen3.5:27b` | ~17 GB | Strong; requires 20GB+ VRAM |
| `llama3.3:70b`| `ollama pull llama3.3:70b`| ~43 GB | Highest accuracy; requires 48GB+ VRAM |

```bash
ollama pull qwen3.5:9b
python bench/run_llm_judge.py --model qwen3.5:9b --pool-file bench/judge_pool/pool.csv
```

Shows live progress with ETA. If interrupted, resume without re-scoring completed batches:

```bash
python bench/run_llm_judge.py --model qwen3.5:9b --pool-file bench/judge_pool/pool.csv --resume
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `qwen3.5:9b` | Ollama model tag |
| `--pool-file` | *(empty)* | Pool CSV from build_judge_pool.py (use this for qrels) |
| `--batch-size` | `30` | Pairs per Ollama call |
| `--timeout` | `600` | Seconds per call before timeout |
| `--resume` | off | Skip batches whose output file already exists |

---

### Step 3 — Build qrels from LLM scores

```bash
python bench/build_qrels.py
```

Reads `bench/judge_pool/pool.csv` and all `judge_scores_batch_*.csv` files.
Outputs one file per dataset in `bench/labels/`:
- `bench/labels/qrels_custom.csv`
- `bench/labels/qrels_cs1qa.csv`
- `bench/labels/qrels_mbpp.csv`
- `bench/labels/qrels_staqc.csv`

Schema matches `silver_labels_*.csv` exactly — `score_obj1.py` can read them directly.
Only pairs with relevance ≥ 1 are written (zeros are implicit).

---

### Step 4 — Re-score Objective 1 against qrels

```bash
python bench/score_obj1.py --reference-config B --labels-dir bench/labels
```

`score_obj1.py` will now load the `qrels_*.csv` files alongside any remaining
`silver_labels_*.csv` files. To use qrels exclusively, either remove or rename
the old silver label files, or point `--labels-dir` to a directory containing
only the qrels.

---

### Alternative: Manual judging via ChatGPT or Gemini

If you do not have Ollama available, export the pool as batched CSVs for manual scoring:

```bash
python bench/export_for_llm_judge.py --batch-size 200
```

Then follow the manual workflow in `bench/llm_judge/input/PROMPT.txt`.
After collecting scores, run `build_qrels.py` as above.

---

### Diagnostic: Cohen's κ against silver labels

To measure how much the LLM judge disagreed with the old silver labels (useful for
reporting in the manuscript limitations section):

```bash
python bench/import_llm_scores.py
```

Output: `bench/results/llm_judge_summary.json` with Cohen's κ, exact agreement,
and disagreement breakdown. This is diagnostic only — the qrels are the ground truth.

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
| `make index` | Build vector + BM25 indexes from chunk manifest |
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
| `python bench/build_judge_pool.py` | Build 4-source judging pool (positives + hard/random negatives + retrieved) |
| `python bench/run_llm_judge.py --pool-file bench/judge_pool/pool.csv` | Score pool with local Ollama LLM |
| `python bench/build_qrels.py` | Merge LLM scores → per-dataset qrels CSVs (ground truth for scoring) |
| `python bench/import_llm_scores.py` | Diagnostic: Cohen's κ between LLM scores and silver labels |
| `python bench/export_for_llm_judge.py` | Export batched CSVs for manual judging (ChatGPT / Gemini fallback) |
| `python bench/validate_silver_labels.py` | Supplementary: semantic cross-validation (Spearman ρ) |

---

## Key Parameter Reference

These values are hardcoded in the pipeline and match the reported experimental setup:

| Parameter | Value | Location |
|-----------|-------|----------|
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
| Silver label min-score | 0.55 (candidate pool threshold) | `bench/build_judge_pool.py` |
| Judge pool hard negatives | 3 per query | `bench/build_judge_pool.py` |
| Judge pool random negatives | 2 per query | `bench/build_judge_pool.py` |
| Judge pool random seed | 42 | `bench/build_judge_pool.py` |
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
