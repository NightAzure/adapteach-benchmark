# Labels

Objective 1 uses LLM-validated qrels as the only scoring labels.

Canonical path:

```text
bench/build_judge_pool.py -> bench/run_llm_judge.py -> bench/build_qrels.py -> bench/score_obj1.py
```

Generated files in this directory:

- `qrels_custom.csv`
- `qrels_cs1qa.csv`
- `qrels_mbpp.csv`
- `qrels_staqc.csv`

The lexical scorer is used only inside `bench/build_judge_pool.py` to nominate candidate
query-chunk pairs for judging. It does not write standalone labels and is not a scoring fallback.
