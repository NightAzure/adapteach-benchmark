# Reproducing the AdapTeach Benchmark

Complete step-by-step guide for reproducing all benchmark results reported in the thesis.
Covers Objective 1 (retrieval evaluation, configs A–F) and Objective 2 (RAGAS generation quality).

---

## Prerequisites

### Software

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Tested on 3.10 and 3.11 |
| Ollama | Any recent | Required for Obj 1 LLM judge (Step 5) |
| Gemini API key | — | Required for Obj 2 (generation + RAGAS evaluation) |

Install Ollama from https://ollama.com, then pull the judge model used in Step 5:

```bash
ollama pull qwen3.5:9b
```

Confirm Ollama is serving at `http://localhost:11434` (default):

```bash
curl http://localhost:11434/api/tags
```

### Hardware

- **RAM:** 8 GB minimum; 16 GB recommended (sentence-transformers loads a 90 MB model)
- **Disk:** ~500 MB for indexes + run outputs
- **GPU:** Not required for Obj 1; Ollama runs on CPU. Gemini is remote for Obj 2.

### Python Dependencies

Core dependencies (Objective 1):

```bash
pip install -r requirements.lock
```

Additional dependencies for Objective 2 (RAGAS evaluation) — **not in requirements.lock**:

```bash
pip install "ragas==0.4.3" langchain-google-genai langchain-core datasets sentence-transformers
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
OLLAMA_MODEL=qwen3.5:9b
GEMINI_API_KEY=your_key_here
```

The `GEMINI_API_KEY` is used by Obj 2 commands. Alternatively, pass it via `--api-key` on the command line.

To override any `config.yaml` value without editing the file, use the `ADAPTEACH__` prefix
with double-underscores as separators:

```bash
export ADAPTEACH__LLM__PROVIDER=gemini
export ADAPTEACH__RETRIEVAL__K=5
```

Config precedence (highest to lowest): shell env vars > `ADAPTEACH__` overrides > `config.yaml`.

---

## Pipeline Configurations

Six ablation configs are evaluated. Each is defined in `configs/<letter>.yaml`.

| Config | Retrieval | Chunking | Hybrid (BM25) | Graph Expansion | Graph Context | Purpose |
|--------|-----------|----------|---------------|-----------------|---------------|---------|
| **A** | Disabled | Fixed | No | No | No | No-retrieval **negative control** — context metrics (precision/recall) are N/A by design |
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

## Step 4: Run Objective 1 — Retrieval Benchmark

Runs retrieval only (`dry_run: retrieval`); no LLM generation. Config A is included in the primary run but is functionally a no-retrieval baseline (empty retrieved sets).

### Primary benchmark (recommended — faster, high-signal)

Runs 120 custom queries across all 6 configs (A–F):

```bash
make obj1-primary
# or: python bench/run_obj1.py --benchmark bench/benchmarks/obj1_primary.yaml
```

Total eval rows: 120 queries × 6 configs = 720 rows.

### Full benchmark (complete — all datasets)

Runs 1,557 queries across configs B–F (Config A excluded from full spec; no-retrieval baseline is evaluated on the custom set only):

```bash
make obj1-full
# or: python bench/run_obj1.py --benchmark bench/benchmarks/obj1_full.yaml
```

Total eval rows: 1,557 queries × 5 configs = 7,785 rows. Allow 20–60 minutes depending on hardware.

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
| `--model` | (from config.yaml) | Model name, e.g. `gemini-2.5-flash` |
| `--api-key` | (from `.env`) | API key for Gemini |
| `--delay` | 0 | Seconds between requests (rate limiting) |
| `--dry-run` | `retrieval` | `none` (full gen), `retrieval` (skip gen), `graph` |
| `--sample-per-dataset` | 0 (all) | Limit queries per dataset for quick tests |
| `--out-dir` | `bench/runs/` | Output directory |

### Outputs

Written to `bench/runs/`:
- `run_<timestamp>_<benchmark>.jsonl` — one row per query+config with retrieval debug info and (if enabled) generated artifact
- `run_<timestamp>_<benchmark>.manifest.json` — run metadata (configs, query counts, spec SHA256, provider, model)

---

## Step 5: Score Objective 1 (LLM-Validated Ground Truth)

The scoring pipeline has one path: build a mixed judging pool, have a blind LLM judge rate
query-chunk pairs from all retrieval methods, build qrels, then score against those qrels.
There is no standalone lexical-label scoring fallback.

**Pipeline:**
```
build_judge_pool.py → run_llm_judge.py → build_qrels.py → score_obj1.py
```

`score_obj1.py` requires `qrels_*.csv`. If qrels are missing, run the judge pipeline below.

---

### 1. Build the judging pool

```bash
python bench/build_judge_pool.py --benchmark bench/benchmarks/obj1_full.yaml
```

Reads the latest run file from `bench/runs/` automatically. Outputs
`bench/judge_pool/pool.csv`.

Pool sources per query:
- `lexical_positive` — chunks rated ≥ 0.55 by the lexical candidate scorer
- `hard_negative` — top-3 topic-matching chunks rejected by lexical scorer
- `retrieved` — all chunks retrieved by any config in the run file
- `random_negative` — 2 randomly sampled chunks with no topic connection

---

### 2. Score the pool with a local LLM

```bash
python bench/run_llm_judge.py --model qwen3.5:9b --pool-file bench/judge_pool/pool.csv
```

Shows live progress with ETA. If interrupted, resume without re-scoring completed batches:

```bash
python bench/run_llm_judge.py --model qwen3.5:9b --pool-file bench/judge_pool/pool.csv --resume
```

Model requirements (choose one):

| Model | VRAM needed | Notes |
|-------|-------------|-------|
| `qwen3.5:9b` | ~7 GB | **Recommended** — fits on 16 GB GPU, fast |
| `qwen3.5:27b` | ~17 GB | Higher accuracy; requires 20 GB+ VRAM |
| `qwen3.6:27b` | ~17 GB | Latest Qwen; requires 20 GB+ VRAM |

---

### 3. Build qrels from LLM scores

```bash
python bench/build_qrels.py
```

Reads `bench/judge_pool/pool.csv` and all `judge_scores_batch_*.csv` files.
Outputs one file per dataset in `bench/labels/`:
- `bench/labels/qrels_custom.csv`
- `bench/labels/qrels_cs1qa.csv`
- `bench/labels/qrels_mbpp.csv`
- `bench/labels/qrels_staqc.csv`

Only pairs with relevance ≥ 1 are written (zeros are implicit negatives).

---

### 4. Score against qrels

```bash
python bench/score_obj1.py --reference-config B --labels-dir bench/labels
```

`score_obj1.py` reads only `qrels_*.csv` from `--labels-dir`.

Metrics reported:

| Metric | Description |
|--------|-------------|
| MRR@10 | Mean Reciprocal Rank at cutoff 10 |
| nDCG@10 | Normalized Discounted Cumulative Gain at cutoff 10 |
| P@5 | Precision at 5 |
| Hit@5 | Binary: was any relevant doc in top 5? |
| retrieval_latency_ms | Wall-clock retrieval time per query |

Outputs written to `bench/results/obj1_latest/`:
- `per_query.csv` — per-query breakdown by dataset and config
- `summary.csv` — aggregated means with 95% bootstrap CI (2000 iterations, seed=13)
- `significance_vs_reference.csv` — delta metrics vs. Config B with p-values (5000 iterations, seed=13)

---

## Step 6: Objective 2 — RAGAS Evaluation

Objective 2 evaluates generation quality (faithfulness, answer relevancy, context precision, context recall) using the RAGAS framework across all six configs including the no-retrieval baseline (Config A).

**Default provider: Gemini 2.5 Flash.** Set your API key once and all three sub-steps use it:

```bash
export GEMINI_API_KEY=your_key_here
# or add GEMINI_API_KEY=your_key_here to .env
```

> **Ollama alternative:** Replace `--provider gemini --api-key $GEMINI_API_KEY` with `--provider ollama --ollama-model qwen3.5:9b` in all commands below.

---

### Sub-step A: Run generation across all configs (A–F)

The primary benchmark spec runs in retrieval-only mode by default (no LLM answers). For RAGAS, you need a separate generation run with all six configs including Config A:

```bash
python bench/run_obj1.py \
  --benchmark bench/benchmarks/obj1_primary.yaml \
  --dry-run none \
  --provider gemini \
  --api-key $GEMINI_API_KEY \
  --out-dir bench/runs
```

This produces `bench/runs/run_<timestamp>_obj1_primary.jsonl` with 720 rows (120 queries × 6 configs A–F) including generated answers. Note the exact filename — you will pass it to Sub-steps B and C.

Estimated time: ~30–60 minutes on the Gemini free tier (10 RPM limit). Use `--delay 6` if you hit rate limit errors.

---

### Sub-step B: Build golden reference answers

Golden answers are LLM-generated reference responses used as RAGAS ground truth.
Run once — resumes automatically if interrupted:

```bash
python bench/run_obj2.py build-golden \
  --queries bench/queries_custom.jsonl \
  --run-file bench/runs/run_<timestamp>_obj1_primary.jsonl \
  --out bench/results/golden_custom.jsonl \
  --provider gemini \
  --api-key $GEMINI_API_KEY \
  --delay 7
```

Replace `<timestamp>` with the actual filename from Sub-step A.

---

### Sub-step C: Evaluate with RAGAS

```bash
python bench/run_obj2.py evaluate \
  --run-file bench/runs/run_<timestamp>_obj1_primary.jsonl \
  --golden bench/results/golden_custom.jsonl \
  --out bench/results/obj2_ragas.csv \
  --provider gemini \
  --api-key $GEMINI_API_KEY \
  --configs A,B,C,D,E,F \
  --gemini-rpm 60
```

RAGAS evaluates only queries present in the golden file (~120 samples). Each config runs sequentially; progress is checkpointed to the CSV after each config — if interrupted, re-running skips already-completed configs automatically.

**Output:** `bench/results/obj2_ragas.csv`

| Config | faithfulness | answer_relevancy | context_precision | context_recall |
|--------|-------------|-----------------|-------------------|----------------|
| A | 0.0 (N/A — no retrieval) | ✓ | N/A | N/A |
| B–F | ✓ | ✓ | ✓ | ✓ |

Config A reports only `answer_relevancy` (no retrieved contexts to evaluate).

**Rate limit note:** `--gemini-rpm 60` is appropriate for paid tier (flash-lite supports ~1,000 RPM). RAGAS's own sequential processing per sample is the real throughput ceiling — setting higher won't speed things up. On free tier (30 RPM for flash-lite), use `--gemini-rpm 25` instead.

To reduce load or speed up a partial re-run, limit to specific configs:

```bash
python bench/run_obj2.py evaluate \
  --run-file bench/runs/run_<timestamp>_obj1_primary.jsonl \
  --golden bench/results/golden_custom.jsonl \
  --out bench/results/obj2_ragas.csv \
  --provider gemini \
  --api-key $GEMINI_API_KEY \
  --configs D,E,F
```

---

## Makefile Reference

| Target | Description |
|--------|-------------|
| `make help` | List all targets |
| `make setup` | Install dependencies from requirements.lock |
| `make datasets` | Validate shipped query files |
| `make chunks` | Build chunk manifest from corpus |
| `make index` | Build vector + BM25 indexes from chunk manifest |
| `make reproduce` | Primary run: datasets → chunks → index → obj1-primary → judge-pool → llm-judge → qrels → obj1-score |
| `make obj1-primary` | Run Obj1 with primary benchmark spec (all configs, custom queries) |
| `make obj1-full` | Run Obj1 with full benchmark spec (all datasets, configs B–F) |
| `make judge-pool` | Build LLM judging pool from latest Obj1 run |
| `make llm-judge` | Score judging pool with local Ollama |
| `make qrels` | Merge LLM scores into qrels |
| `make obj1-score` | Score the latest run against qrels |
| `make obj1-smoke` | Quick 5-query retrieval-only smoke test |
| `make obj2` | Run Objective 2 RAGAS evaluation |
| `make test` | Syntax-check all bench scripts |

**Qrels pipeline scripts:**

| Script | Description |
|--------|-------------|
| `python bench/build_judge_pool.py` | Build 4-source judging pool (lexical positives + hard/random negatives + retrieved) |
| `python bench/run_llm_judge.py --pool-file bench/judge_pool/pool.csv` | Score pool with local Ollama LLM |
| `python bench/build_qrels.py` | Merge LLM scores → per-dataset qrels CSVs (ground truth for scoring) |

---

## Key Parameter Reference

These values are hardcoded in the pipeline and match the reported experimental setup:

| Parameter | Value | Location |
|-----------|-------|----------|
| Chunk size | 450 characters | `bench/build_chunks.py` |
| Retrieval top-k | 5 | `config.yaml` |
| RRF smoothing constant | 60 | `src/retrieval/engine.py` (hardcoded `rrf_k`) |
| Rerank weights | retrieval=0.70, graph=0.30 | `config.yaml` |
| BM25 k1 | 1.2 | `src/indexing/build_indexes.py` |
| BM25 b | 0.75 | `src/indexing/build_indexes.py` |
| Embedding model | `all-MiniLM-L6-v2` (384-dim) | `src/indexing/build_indexes.py` |
| Embedding fallback | Hash-based (64-dim) | `src/indexing/build_indexes.py` |
| Lexical candidate threshold | 0.55 | `bench/build_judge_pool.py` |
| Judge pool hard negatives | 3 per query | `bench/build_judge_pool.py` |
| Judge pool random negatives | 2 per query | `bench/build_judge_pool.py` |
| Judge pool random seed | 42 | `bench/build_judge_pool.py` |
| Bootstrap CI iterations | 2000 | `bench/score_obj1.py` |
| Significance test iterations | 5000 | `bench/score_obj1.py` |
| Random seed | 13 | `bench/score_obj1.py` |
| Obj1 judge model | `qwen3.5:9b` (Ollama) | `bench/run_llm_judge.py` |
| Obj2 eval model | `gemini-2.5-flash-lite` (RAGAS judge) | `bench/ragas_eval.py` |
| Obj2 golden model | `gemini-2.5-flash` (golden generation) | `bench/ragas_eval.py` |
| Obj2 RAGAS batch size | 4 samples per batch | `bench/ragas_eval.py` |
| Obj2 max contexts per sample | 5 | `bench/ragas_eval.py` (matches retrieval k=5) |
| Obj2 default Gemini RPM | 60 (paid tier); use 25 on free tier | `bench/run_obj2.py` |
| Ollama base URL | `http://localhost:11434` | `.env` |

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
