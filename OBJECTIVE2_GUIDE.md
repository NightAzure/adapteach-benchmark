# Objective 2 — Generation Quality Evaluation with RAGAS

> **Thesis Objective:** Evaluate the generation quality of educational artifacts produced by
> the highest-performing retrieval configurations, assessed via RAGAS metrics (faithfulness,
> answer relevancy, context precision, and context recall) against a reference answer set.

---

## Status

- **Objective 1 (retrieval benchmarks) — DONE.** Results in `bench/results/obj1_all_datasets.csv`.
- **Objective 2 (RAGAS generation quality) — PENDING.** Run this on AWS.

---

## 1. What RAGAS Measures

| Metric | Question it answers | Needs reference answer? |
|---|---|---|
| **Faithfulness** | Is the generated answer grounded in the retrieved context? (hallucination proxy) | No |
| **Answer Relevancy** | Does the answer actually address the question? | No |
| **Context Precision** | Are the top-k retrieved chunks actually relevant? | Yes |
| **Context Recall** | Does the retrieved context cover the reference answer? | Yes |

### Expected pattern (based on Obj 1 retrieval findings)

| Config | Expected RAGAS behaviour |
|---|---|
| **A — Baseline LLM** | Low Faithfulness (no context), skip Context metrics (no retrieval) |
| **B — Dense Retrieval** | Moderate all-round; strongest on conceptual queries |
| **C — Hybrid (BM25+Dense)** | Similar to B, slight improvement on exact-term queries |
| **D — AST Chunking + Dense** | Higher Context Precision on code-adjacent queries; may match or beat B overall |
| **E — Full GraphRAG** | Expected to underperform B and D (consistent with Obj 1 retrieval findings) |
| **F — D + Graph Context Injection** | Same retrieval as D; may improve Faithfulness if graph context enriches LLM prompt |

> Note: Obj 1 showed Config E (Full GraphRAG) is the weakest retrieval config on all 4
> datasets. Obj 2 tests whether generation quality follows the same pattern.

---

## 2. Prerequisites (on AWS)

```bash
pip install "ragas==0.4.3"
pip install langchain-google-genai langchain-core
pip install datasets sentence-transformers
pip install google-generativeai
```

> **Why ragas 0.4.3?** The eval script uses the 0.4.x API:
> `from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ...`
> and `SingleTurnSample` with fields `user_input`, `response`, `retrieved_contexts`, `reference`.
> Do NOT install a different version.

Set your API key:
```bash
export GEMINI_API_KEY=AIzaSyAvRtEtO8qgvhiBIB5WkQ-KdYXtVVvou1U
```

---

## 3. Configs to Run

Run **B, D, E, F** (skip A for RAGAS — no retrieval context makes most metrics meaningless).
C is optional; include it if you want the full ablation story.

**Recommended minimum:** `--configs B D E F`

---

## 4. Step-by-Step Execution

### Step 1 — Full pipeline run (no dry-run)

```bash
cd benchmark

python bench/run_benchmark_suite.py \
  --query-set bench/queries_custom.jsonl \
  --configs B D E F \
  --out-dir bench/runs/ \
  --tag custom_full
```

- Uses `bench/queries_custom.jsonl` (120 curriculum-aligned queries)
- This calls the LLM (Gemini Flash) to generate actual artifact answers — takes ~30–60 min
- Output: `bench/runs/run_<TIMESTAMP>_custom_full.jsonl`

### Step 2 — Build the reference answer set (golden)

```bash
RUNFILE=$(ls -t bench/runs/run_*_custom_full.jsonl | head -1)

python bench/ragas_eval.py \
  --build-golden \
  --queries bench/queries_custom.jsonl \
  --run-file "$RUNFILE" \
  --out bench/golden_custom.jsonl \
  --api-key $GEMINI_API_KEY
```

This calls Gemini to generate a reference answer for each query using the retrieved corpus
chunks. Review `bench/golden_custom.jsonl` — if any reference answers look wrong or hallucinated,
edit them before Step 3.

### Step 3 — RAGAS evaluation

```bash
python bench/ragas_eval.py \
  --eval \
  --run-file "$RUNFILE" \
  --golden bench/golden_custom.jsonl \
  --out bench/results/ragas_custom.csv \
  --api-key $GEMINI_API_KEY
```

Output: `bench/results/ragas_custom.csv` — one row per config with all four RAGAS metrics.

---

## 5. Output Format

`ragas_custom.csv` columns:

| Column | Range | Better is |
|---|---|---|
| `config` | A–F | — |
| `faithfulness` | 0–1 | Higher |
| `answer_relevancy` | 0–1 | Higher |
| `context_precision` | 0–1 | Higher |
| `context_recall` | 0–1 | Higher |
| `retrieval_latency_ms_avg` | ms | Lower |

### Results table for thesis (Objective 2)

| Config | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall |
|---|---|---|---|---|
| B — Dense | ... | ... | ... | ... |
| C — Hybrid | ... | ... | ... | ... |
| D — AST+Dense | ... | ... | ... | ... |
| E — Full GraphRAG | ... | ... | ... | ... |
| F — D+Graph Context | ... | ... | ... | ... |

**Ablation deltas to report:**
- Δ(D − B): contribution of AST chunking on generation quality
- Δ(E − D): contribution of full graph integration (expected negative or near-zero)
- Δ(F − D): contribution of graph context injection at generation stage

---

## 6. Troubleshooting

**`ragas` returns NaN for context_precision / context_recall**
→ Reference answer missing in `golden_custom.jsonl`. Check that every query_id has a `reference` field.

**Config A context metrics are 0 or NaN**
→ Expected — Config A has no retrieval, so contexts is empty. Omit Config A from the RAGAS results table.

**Faithfulness is suspiciously 1.0 for all configs**
→ Known ragas bug with empty context. Ensure your run file has non-empty `context` fields in the response.

**Very slow (>2 hrs)**
→ RAGAS calls Gemini once per sample per metric. 120 queries × 4 metrics = 480 API calls.
→ Use `--configs D F` first to get the most important comparison. Add B E C after.

**`run_benchmark_suite.py` errors on LLM call**
→ Check `GEMINI_API_KEY` is exported. Check `configs/config_B.yaml` (and D, E, F) have `provider: gemini`.

**`bench/ragas_eval.py` KeyError on response**
→ Context and answer are nested under `row["response"]` — the script reads `row["response"]["context"]` and `row["response"]["answer"]`. If you see KeyError, the run file format may have changed; check one row manually with `head -1 bench/runs/run_*.jsonl | python -m json.tool`.
