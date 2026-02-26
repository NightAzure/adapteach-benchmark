# Objective 1 Guide (Friendly, Non-Technical)

This guide explains what is still needed for **Objective 1** and which parts require researcher input.

Objective 1 from your framework:

- "Iteratively design a structure-aware retrieval framework and compare component variants"
- Variants: fixed vs AST chunking, dense vs BM25 vs hybrid, graph integration/reranking

---

## Short Answer

Detailed execution runbook: `docs/OBJECTIVE1_RUNBOOK.md`

Yes, the remaining blockers are mostly researcher-side inputs, especially because the corpus is not finalized yet.

The 3 key researcher inputs are:

1. Final benchmark queries (`bench/queries_dev.jsonl`, `bench/queries_test.jsonl`)
2. Relevance labels (`bench/labels/labels.csv`)
3. Inter-rater agreement process (two raters + kappa check)

---

## Why Corpus Comes First

For Objective 1, retrieval is being compared. Retrieval quality depends on what is in the corpus.

If corpus is incomplete:

- rankings are unstable
- metrics are not trustworthy
- ablation conclusions can be misleading

So the order should be:

1. finalize corpus
2. freeze corpus snapshot/index
3. run benchmark + labeling
4. compare A-E and ablations

---

## Where This Appears in Your Research Docs

In `AdapTeach_Guided_Experiment_Guide.md`:

- Section `3.2 Specific objectives`: Objective 1 definition
- Section `7 Data Sources and Corpus Construction`: corpus and governance requirements
- Section `8 System Configurations for Benchmarking (A-E)`: the exact variants to compare
- Section `4.1 Two phases`: developmental + technical benchmarking flow

So the "3 inputs" are the practical evidence-collection part for Objective 1 benchmarking.

---

## What Each Input Means (Plain Language)

## 1) Benchmark queries

What it is:

- A list of representative questions/tasks you will test the system on.

Why it matters:

- If query quality is poor, retrieval comparison is not meaningful.

Your files:

- `bench/queries_dev.jsonl` (for tuning/iteration)
- `bench/queries_test.jsonl` (held-out final check)

## 2) Relevance labels

What it is:

- Human judgment of whether retrieved chunks are relevant to each query.

Why it matters:

- Metrics like `MRR@10`, `nDCG@10`, and context precision need gold labels.

Your file:

- `bench/labels/labels.csv`

## 3) Inter-rater agreement

What it is:

- Two raters label the same samples; agreement is measured (kappa).

Why it matters:

- Shows labels are reliable, not arbitrary.

Your command:

- `py bench/compute_kappa.py --labels-csv bench/labels/labels.csv`

---

## Practical "Do This Now" Checklist

## Step A: Finish corpus readiness

- Use `docs/CORPUS_GATHERING_SOP_FOR_NON_TECH.md`
- Ensure licensing/provenance/tagging are complete
- Build clean corpus, chunks, indexes, graphs

## Step B: Lock benchmark query sets

- Review `bench/queries_dev.jsonl` and `bench/queries_test.jsonl`
- Remove duplicates/ambiguous queries
- Ensure coverage of loops/conditionals/functions + artifact types

## Step C: Run benchmark once to generate retrieval outputs

- `py bench/run_benchmark_suite.py --query-set bench/queries_dev.jsonl --configs A,B,C,D,E`

## Step D: Label relevance

- Copy template to `bench/labels/labels.csv`
- Fill `chunk_id` + `relevance` using `bench/labels/rubric.md`

## Step E: Check agreement and freeze

- `py bench/compute_kappa.py --labels-csv bench/labels/labels.csv`
- `py bench/freeze_labels.py --labels-csv bench/labels/labels.csv --queries-dev bench/queries_dev.jsonl --queries-test bench/queries_test.jsonl`

## Step F: Compute results for Objective 1

- `py bench/compute_metrics.py --labels-csv bench/labels/labels.csv`
- `py bench/run_ablations.py --query-set bench/queries_dev.jsonl`

---

## Definition of "Objective 1 Ready"

You are ready to claim Objective 1 evidence when all are true:

- corpus snapshot is frozen
- query sets are frozen
- labels are completed and agreement checked
- metrics CSV exists
- ablation CSV exists
- results are reported per variant (fixed/AST, dense/hybrid/graph)

Until those are complete, Objective 1 is implementation-ready but evidence-incomplete.
