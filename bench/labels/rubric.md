# LLM Judge Relevance Rubric

The Objective 1 qrels pipeline asks the LLM judge to score each query-chunk pair as:

- `2` = directly relevant
- `1` = partially relevant
- `0` = not relevant

Judge only whether the chunk supports the query intent. Do not reward writing style,
retrieval rank, source configuration, or whether the chunk was selected by lexical,
retrieved, hard-negative, or random pool nomination.
