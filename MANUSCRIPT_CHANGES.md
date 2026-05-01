# Manuscript Changes — Evaluation Label Methodology

Changes made to address the reviewer concern about circularity in silver label evaluation.
Edit your manuscript at the sections noted below.

---

## What the concern is (for your reference)

Silver labels are generated using a lexical scoring function (keyword overlap, title match,
concept tags). Evaluating retrievers against these labels could systematically favour
keyword-based retrievers (BM25) over semantic/dense retrievers — not because BM25 is
better, but because the labels reward lexical overlap. A reviewer would flag this as
circular evaluation.

---

## How it is resolved — LLM-validated qrels

The pipeline has been updated to eliminate the circularity. The lexical scorer is no longer
the ground truth — it is used only as a *candidate nominator*. Final relevance labels come
from a blind LLM judge scoring a mixed pool that includes positives, hard negatives,
retrieved chunks from every config, and random negatives.

**New evaluation flow:**

```
build_judge_pool.py  →  run_llm_judge.py  →  build_qrels.py  →  score_obj1.py
  (pool 4 sources)       (LLM scores 0/1/2)   (qrels per dataset)  (MRR, nDCG, etc.)
```

---

## Changes to report in the manuscript

### 1. Methodology section — Relevance Label Construction

**Replace or rewrite the silver label section as:**

> We construct relevance labels using an LLM-as-judge procedure to avoid lexical circularity.
> A candidate judging pool is built from four sources for each query: (1) chunks rated
> relevant by the lexical scorer (silver positives, threshold ≥ 0.55); (2) topic-matching
> chunks rejected by the lexical scorer (hard negatives); (3) all chunks retrieved by any
> pipeline configuration in the benchmark run; and (4) randomly sampled chunks with no
> topic connection (random negatives). This pooling strategy ensures that dense and
> semantic retrievers (Configs D–F) can receive credit for chunks the lexical scorer would
> have missed.
>
> The resulting pool of [N] (query, chunk) pairs is scored blindly by
> [model name, e.g. Qwen3.5-9B / Qwen3.6-27B] on a three-point scale:
> 0 = not relevant, 1 = partially relevant, 2 = directly relevant. The LLM receives only
> the query text and chunk text — no retrieval scores or system identifiers. Pairs scored 0
> are treated as implicit negatives (not written to the qrels file). The resulting qrels
> are used as ground truth for all Objective 1 metrics (MRR@10, nDCG@10, P@5, Hit@5).

**Then note pool composition:**

> The judging pool contained [N] unique (query, chunk) pairs: [X] silver positives,
> [Y] hard negatives, [Z] retrieved-only pairs, and [W] random negatives. After LLM
> scoring, [M] pairs received relevance ≥ 1 and form the final qrels.

---

### 2. Evaluation setup / Experimental design section

**Add:**

> Relevance labels are LLM-validated qrels built via a four-source pooling strategy
> (see §Methodology). All evaluation scripts and the judging pool are included in the
> benchmark repository. The lexical silver labeler (`bench/build_silver_labels.py`) is
> retained as an upstream candidate nominator but is no longer the source of ground-truth
> labels. Final qrels are produced by `bench/build_qrels.py` from blind LLM scores.

---

### 3. Limitations section

**Add or expand:**

> Relevance labels are produced by an LLM judge ([model name]) rather than human
> annotators. While LLM-as-judge approaches have been shown to correlate well with
> human judgements at the query–chunk level (Faggioli et al., 2023), model consistency
> is not guaranteed across all query types. The judging pool uses a four-source pooling
> strategy — including random negatives and retrieved chunks from all pipeline
> configurations — to minimise systematic bias. Human annotation over a subset of
> queries would provide stronger ground truth and is left as future work.

**If you keep the semantic cross-validation (Spearman ρ) as supplementary:**

> As a supplementary check, we compute Spearman ρ between the LLM-judged relevance
> rankings and cosine similarity rankings from all-MiniLM-L6-v2 over the top-10
> retrieved chunks per query (mean ρ = [fill in]).

---

### 4. Results table for label quality (if applicable)

| Pool source | Pairs | After LLM scoring (rel ≥ 1) |
|-------------|-------|------------------------------|
| Silver positive | [fill in] | [fill in] |
| Hard negative | [fill in] | [fill in] |
| Retrieved (any config) | [fill in] | [fill in] |
| Random negative | [fill in] | [fill in] |
| **Total** | **[fill in]** | **[fill in]** |

---

## Files added or updated in the repository

| File | Purpose |
|------|---------|
| `bench/build_judge_pool.py` | Builds deduplicated judging pool (4 sources) |
| `bench/run_llm_judge.py` | Automated LLM judging via Ollama; supports `--pool-file` |
| `bench/build_qrels.py` | Merges LLM scores → per-dataset qrels CSVs |
| `bench/import_llm_scores.py` | Computes Cohen's κ (used for diagnostics, not ground truth) |
| `bench/export_for_llm_judge.py` | Manual judging export for ChatGPT/Gemini (fallback) |
| `bench/validate_silver_labels.py` | Semantic cross-validation via Spearman ρ (supplementary) |
| `bench/REPRODUCE_BENCHMARK.md` | Updated with full qrels pipeline instructions |

---

## Reference to cite for LLM-as-judge

> Faggioli, G., et al. (2023). Perspectives on large language models for relevance
> judgment. *Proceedings of SIGIR 2023*.

## Reference to cite for Cohen's κ (if still reporting it as diagnostic)

> Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for
> categorical data. *Biometrics*, 33(1), 159–174.

Kappa interpretation thresholds (Landis & Koch):
- κ < 0.20: slight
- κ 0.21–0.40: fair
- κ 0.41–0.60: moderate
- κ 0.61–0.80: substantial
- κ > 0.80: almost perfect
