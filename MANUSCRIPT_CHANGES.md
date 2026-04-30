# Manuscript Changes — Silver Label Validation

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

## Changes to report in the manuscript

### 1. Methodology section — Silver Label Construction

**Add after the description of the scoring function:**

> To validate that the lexically-constructed labels are not systematically misaligned
> with semantic relevance, we cross-validate them against two independent signals.
> First, an LLM-as-judge procedure: for each query-chunk pair in the custom dataset,
> we prompt [GPT-4 / Gemini 1.5 Pro — fill in whichever you used] to independently
> score chunk relevance on a 0–2 scale (0 = not relevant, 1 = partially relevant,
> 2 = directly relevant). We measure Cohen's κ between the LLM scores and our silver
> labels across all [N] pairs. Second, as a supplementary check, we compute Spearman ρ
> between the silver label ranking and cosine similarity rankings from all-MiniLM-L6-v2
> over the top-10 retrieved chunks per query.

**Then add the results inline or in a table:**

> Cohen's κ = [fill in value] between LLM judge and silver labels, indicating
> [substantial / moderate] agreement (Landis & Koch, 1977). Mean Spearman ρ =
> [fill in value] across [N] queries. These results suggest that the lexical labels
> are broadly consistent with independent semantic and model-based relevance judgements,
> partially mitigating the circularity concern inherent in automated label generation.

**If κ < 0.60:** Reframe as a limitation rather than a validation — see Limitations section below.

---

### 2. Limitations section

**Add or expand:**

> Our silver labels are constructed using a weighted lexical scoring function combining
> token overlap, title similarity, and concept tag matching. This introduces a potential
> bias toward keyword-based retrieval configurations (e.g., Config C with BM25 fusion)
> relative to purely dense configurations. To partially address this, we validate labels
> against an independent LLM judge (Cohen's κ = [value]) and semantic similarity
> rankings (mean Spearman ρ = [value]). While these checks indicate reasonable
> alignment, human annotation over a subset of queries would provide stronger ground
> truth and is left as future work.

---

### 3. Evaluation setup / Experimental design section

**Add a sentence noting the validation scripts are available in the reproducibility package:**

> Label validation scripts and per-query agreement statistics are included in the
> benchmark repository (`bench/export_for_llm_judge.py`, `bench/import_llm_scores.py`,
> `bench/validate_silver_labels.py`).

---

### 4. If you have a Results table for silver label quality

Add a small table or inline values:

| Validation method | Metric | Value |
|-------------------|--------|-------|
| LLM-as-judge (GPT-4 / Gemini) | Cohen's κ | [fill in] |
| Semantic cross-validation | Mean Spearman ρ | [fill in] |
| Semantic cross-validation | Queries with ρ > 0.60 | [fill in %] |

---

## Files added to the repository

| File | Purpose |
|------|---------|
| `bench/export_for_llm_judge.py` | Exports batched CSVs for manual LLM judging |
| `bench/import_llm_scores.py` | Imports LLM scores, computes Cohen's κ |
| `bench/validate_silver_labels.py` | Semantic cross-validation via Spearman ρ |
| `bench/REPRODUCE_BENCHMARK.md` | Updated with Step 6b validation instructions |

---

## Reference to cite for Cohen's κ interpretation

> Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for
> categorical data. *Biometrics*, 33(1), 159–174.

Kappa interpretation thresholds (Landis & Koch):
- κ < 0.20: slight
- κ 0.21–0.40: fair
- κ 0.41–0.60: moderate
- κ 0.61–0.80: substantial
- κ > 0.80: almost perfect
