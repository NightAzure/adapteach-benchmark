# AdapTeach Benchmark (Clean Reproducible Layout)

Retrieval-augmented generation benchmark for adaptive CS1 tutoring artifacts.
Evaluates six pipeline configurations (A–F) across retrieval strategy, chunking method,
and graph-based context augmentation. Developed as part of a thesis project submitted to
Mapua Malayan Colleges Mindanao, CSAI 2026.

---

## Repository Layout

```
adapteach-benchmark/
├── bench/                  # Benchmark scripts, query sets, labels, runs, results
│   ├── benchmarks/         # YAML benchmark specs (obj1_primary, obj1_full)
│   ├── labels/             # Generated LLM-validated qrels
│   ├── runs/               # Generated run JSONL files (git-ignored)
│   └── results/            # Generated score CSVs (git-ignored)
├── configs/                # Pipeline configs A–F + defaults.yaml
├── data/
│   ├── corpus/             # 100 corpus documents (JSON)
│   ├── corpus_meta/        # Chunk manifest, corpus manifest, stats
│   └── snapshots/          # Frozen chunk snapshots for index building
├── scripts/                # Utility scripts (corpus scraper)
├── src/                    # Core pipeline source code
│   ├── generation/         # LLM provider interface (Ollama, Gemini, Mock)
│   ├── graphs/             # CKG and CPG graph modules
│   ├── indexing/           # Vector + BM25 index builder
│   ├── pipelines/          # Pipeline orchestrator
│   └── retrieval/          # Retrieval engine (dense, BM25, RRF)
├── config.yaml             # Runtime configuration
├── .env                    # Local environment variables (not committed)
├── Makefile                # Convenience targets
└── requirements.lock       # Pinned Python dependencies
```

---

## Quick Start (Smoke Test)

Run 5 queries per dataset end-to-end to verify the setup works before committing to a full run.

```bash
# 1. Install dependencies
pip install -r requirements.lock

# 2. Copy and fill in environment variables
cp .env.example .env        # then edit .env with your Ollama URL and model

# 3. Validate shipped query files (offline, no network needed)
make datasets

# 4. Build chunk manifest and retrieval indexes
make chunks
make index

# 5. Smoke test: 5 queries per dataset (retrieval only)
make obj1-smoke
```

For the full benchmark, see [bench/REPRODUCE_BENCHMARK.md](bench/REPRODUCE_BENCHMARK.md).

---

## Corpus and Dataset Provenance

### Query Sets

Four frozen query sets ship with the repository (no download required):

| File | Queries | Source | Refreshable |
|------|---------|--------|-------------|
| `bench/queries_custom.jsonl` | 120 | Internal dev set | No |
| `bench/queries_cs1qa.jsonl` | 437 | CS1QA (GitHub) | Yes |
| `bench/queries_mbpp.jsonl` | 500 | MBPP (Google Research) | Yes |
| `bench/queries_staqc.jsonl` | 500 | StaQC extract | No (upstream frozen) |

To refresh the refreshable sets from upstream:

```bash
python bench/fetch_datasets.py --mode refresh --sources cs1qa mbpp
```

### Corpus Documents

The 100 documents in `data/corpus/` were assembled from multiple public sources:

| Source | License | Notes |
|--------|---------|-------|
| Python official documentation (python.org) | PSF Documentation License | Open; freely redistributable with attribution |
| *Automate the Boring Stuff with Python*, 2nd ed. | CC BY-NC-SA 3.0 | **Non-commercial only** |
| *Think Python*, 2nd ed. | CC BY-NC 3.0 | **Non-commercial only** |
| Python PEPs (peps.python.org) | Public domain | Freely redistributable |
| AI-generated reference pages | N/A | Original to this project |

**Important:** A significant portion of the corpus was manually scraped from the sources
above, then AI-edited to embed representative Python code examples and explanations
suitable for a CS1 tutoring context. Because the CC BY-NC-SA and CC BY-NC licensed
content cannot be redistributed in modified form for commercial purposes, **the full
corpus cannot be included in derivative works or redistributed without verifying
compliance with each source license.**

If you need to rebuild the corpus from scratch (e.g., for a derivative project):

```bash
pip install requests beautifulsoup4
python scripts/scrape_corpus.py --out-dir data/corpus/ --sources all
```

The scraper fetches from the same public URLs with rate limiting (1.2 s between requests).
Post-scraping, the AI-edited enhancements to individual documents are not reproducible
from the scraper alone — those edits exist only in `data/corpus/`.

---

## Full Reproduction Guide

See **[bench/REPRODUCE_BENCHMARK.md](bench/REPRODUCE_BENCHMARK.md)** for:
- Complete system prerequisites (Python, Ollama, hardware)
- All pipeline configs A–F explained
- Objective 1 (retrieval) and Objective 2 (RAGAS) full run instructions
- Key parameter reference table
- Troubleshooting tips
