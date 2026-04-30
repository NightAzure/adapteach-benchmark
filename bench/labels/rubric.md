# Relevance Labeling Rubric

Label each retrieved chunk for a query using:

- `3` = highly relevant, directly supports answer
- `2` = relevant, useful support
- `1` = partially relevant, useful but incomplete
- `0` = not relevant

Guidelines:

- judge only relevance to query intent
- do not reward stylistic quality
- prefer conservative labels when uncertain
- label independently before discussion

Inter-rater:

- Use two raters (`rater_a`, `rater_b`)
- compute Cohen's kappa from overlapping labels
