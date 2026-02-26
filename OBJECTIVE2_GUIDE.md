# Objective 2 — Comparative Component Analysis with RAGAS

> **Thesis Objective:** Conduct a comparative component analysis on retrieval precision and
> context coherence across a baseline LLM configuration and multiple RAG pipeline baselines,
> using the RAGAS framework.

---

## 1. What RAGAS Actually Measures

RAGAS (Retrieval-Augmented Generation Assessment) evaluates a RAG pipeline on four axes.
Each metric answers a different question about your system:

| Metric | Question it answers | Needs ground truth? |
|---|---|---|
| **Faithfulness** | Is the generated answer grounded in the retrieved context? (hallucination proxy) | No |
| **Answer Relevancy** | Does the answer actually address the question? | No |
| **Context Precision** | Are the top-k retrieved chunks actually relevant? (signal-to-noise) | Yes |
| **Context Recall** | Does the retrieved context cover the ground truth answer? | Yes |

### Why each matters for your five configs

| Config | Expected pattern |
|---|---|
| **A — Baseline LLM** | Low Faithfulness (no context to ground answers in), decent Answer Relevancy |
| **B — Dense Retrieval** | Faithfulness improves, Context Precision moderate |
| **C — Hybrid (BM25+Dense)** | Context Precision improves over B (lexical matching helps exact terms) |
| **D — AST Chunking** | Context Recall improves (AST chunks preserve code structure, less info loss) |
| **E — Full GraphRAG** | Highest Context Precision + Recall (graph expansion finds cross-concept chunks) |

This is the story your Objective 2 results should tell.

---

## 2. Prerequisites

```bash
pip install "ragas==0.4.3"
pip install langchain-google-genai langchain-core
pip install datasets
pip install google-generativeai   # for --build-golden step
pip install sentence-transformers  # already installed
```

> **Why 0.4.3?** RAGAS had major breaking changes between 0.1, 0.2, and 0.4.
> The eval script (`bench/ragas_eval.py`) uses the 0.4.x API exactly:
> `from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ...`
> and `SingleTurnSample` with fields `user_input`, `response`,
> `retrieved_contexts`, `reference` (not the old `ground_truths` list from 0.1).

Set your API key (same one used by the pipeline):
```bash
# in backend/.env or benchmark/.env
ADAPTEACH_LLM_GEMINI_API_KEY=your-key-here
```

---

## 3. Step-by-Step Execution

### Step 1 — Grow the corpus (optional but recommended)

60 docs is borderline for GraphRAG. Run the scraper to expand to ~200 docs:

```bash
cd benchmark
py scripts/scrape_corpus.py --out-dir data/corpus_raw/scraped/ --sources all
```

Then rebuild chunks and index:

```bash
py -m src.chunking.build_chunks --chunker ast --chunk-size 450 --overlap 80
py -m src.indexing.build_indexes --snapshot-id <new-id> --embedder-version minilm-v2
# Copy the printed index_id into ALL configs A-E under index_hash:
```

### Step 2 — Create the golden test set

RAGAS needs `(question, ground_truth_answer)` pairs. A script generates them via Gemini
using your own corpus as the source of truth:

```bash
py bench/ragas_eval.py --build-golden \
  --queries bench/queries_test.jsonl \
  --out bench/golden_test.jsonl \
  --api-key $ADAPTEACH_LLM_GEMINI_API_KEY
```

This queries Gemini with each question + the top-5 retrieved corpus chunks and asks it to
write a concise ground-truth answer. Review the output file and edit any answers that look
wrong before continuing.

### Step 3 — Run the pipeline for all configs

```bash
py bench/run_benchmark_suite.py \
  --query-set bench/queries_test.jsonl \
  --configs A B C D E \
  --artifact-type mutation \
  --out-dir bench/runs/
```

Takes ~15–30 min for 30 queries × 5 configs. The output is
`bench/runs/run_<timestamp>.jsonl`.

### Step 4 — Run RAGAS evaluation

```bash
py bench/ragas_eval.py \
  --run-file bench/runs/run_<timestamp>.jsonl \
  --golden bench/golden_test.jsonl \
  --api-key $ADAPTEACH_LLM_GEMINI_API_KEY \
  --out bench/results/ragas_results.csv
```

Produces one row per config with all four RAGAS metrics + latency.

### Step 5 — Run ablations

```bash
py bench/run_ablations.py --query-set bench/queries_test.jsonl
```

Isolates each component's contribution:
- Chunking effect: B (fixed) vs D (AST)
- Retrieval effect: B (dense) vs C (hybrid) vs E (hybrid+graph)
- Graph effect: C (no-graph) vs E (graph)

---

## 4. Understanding the Output

`ragas_results.csv` columns:

| Column | Range | Better is |
|---|---|---|
| `faithfulness` | 0–1 | Higher |
| `answer_relevancy` | 0–1 | Higher |
| `context_precision` | 0–1 | Higher |
| `context_recall` | 0–1 | Higher |
| `retrieval_latency_ms_avg` | ms | Lower |
| `total_tokens_avg` | tokens | Lower |

### Reporting in your thesis

Present a table like this for Objective 2:

| Config | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall |
|---|---|---|---|---|
| A — Baseline LLM | ... | ... | n/a | n/a |
| B — Dense Retrieval | ... | ... | ... | ... |
| C — Hybrid | ... | ... | ... | ... |
| D — AST Chunking | ... | ... | ... | ... |
| E — Full GraphRAG | ... | ... | ... | ... |

Then in the ablation section, show the delta each component adds:
- Δ(C − B): contribution of hybrid retrieval
- Δ(D − B): contribution of AST chunking
- Δ(E − C): contribution of graph expansion

---

## 5. Troubleshooting

**`ragas` returns NaN for context_precision / context_recall**
→ Your golden test set is missing `ground_truth` values. Check `golden_test.jsonl`.

**Config A scores are all 0 for context metrics**
→ This is expected — Config A has no retrieval, so `contexts` is empty. Skip those metrics
for Config A in your results table.

**Very slow evaluation**
→ RAGAS calls Gemini once per sample per metric. For 30 queries × 4 metrics = 120 API
calls. Use `--configs B C D E` to skip Config A if you only need retrieval metrics.

**Faithfulness is suspiciously high for Config A**
→ RAGAS faithfulness with empty context defaults to 1.0 in some versions. Use the proxy
metric from `compute_metrics.py` for Config A instead.

---

## 6. Corpus Quality Checklist

Before running, verify your corpus covers all query topics:

```bash
py -c "
import json, glob, collections
tags = collections.Counter()
for f in glob.glob('data/corpus_raw/**/*.json', recursive=True):
    doc = json.load(open(f))
    tags.update(doc.get('concept_tags', []))
for tag, n in tags.most_common(30):
    print(f'{n:3d}  {tag}')
"
```

You want **≥ 5 documents per concept** that appears in your query set. Concepts with fewer
than 3 docs will produce weak retrieval results that hurt all configs equally.
