PYTHON ?= python

.PHONY: help setup chunks index datasets reproduce \
        obj1-primary obj1-full judge-pool llm-judge qrels obj1-score obj1-smoke obj2 test

help:
	@echo "AdapTeach Benchmark — available targets:"
	@echo ""
	@echo "  setup          Install dependencies from requirements.lock"
	@echo "  datasets       Validate shipped query files"
	@echo "  chunks         Build chunk_manifest.json from corpus/ (AST + text fallback)"
	@echo "  index          Build vector + BM25 indexes from chunk manifest"
	@echo "  reproduce      Primary benchmark: datasets → chunks → index → obj1-primary → judge-pool → llm-judge → qrels → obj1-score"
	@echo ""
	@echo "  obj1-primary   Run Objective 1 (primary query set)"
	@echo "  obj1-full      Run Objective 1 (full query set)"
	@echo "  judge-pool     Build LLM judging pool from latest Obj1 run"
	@echo "  llm-judge      Score judging pool with local Ollama"
	@echo "  qrels          Build qrels from LLM judge scores"
	@echo "  obj1-score     Score Objective 1 results against qrels, reference config B"
	@echo "  obj1-smoke     Quick retrieval-only smoke test: 5 samples per dataset"
	@echo "  obj2           Run Objective 2 RAGAS evaluation"
	@echo ""
	@echo "  test           Syntax-check all bench scripts"
	@echo ""
	@echo "See bench/REPRODUCE_BENCHMARK.md for the full guide."

setup:
	$(PYTHON) -m pip install -r requirements.lock

datasets:
	$(PYTHON) bench/fetch_datasets.py --mode validate

chunks:
	$(PYTHON) bench/build_chunks.py

index:
	$(PYTHON) -m src.indexing.build_indexes

reproduce: datasets chunks index obj1-primary judge-pool llm-judge qrels obj1-score

obj1-primary:
	$(PYTHON) bench/run_obj1.py --benchmark bench/benchmarks/obj1_primary.yaml

obj1-full:
	$(PYTHON) bench/run_obj1.py --benchmark bench/benchmarks/obj1_full.yaml

judge-pool:
	$(PYTHON) bench/build_judge_pool.py --benchmark bench/benchmarks/obj1_primary.yaml

llm-judge:
	$(PYTHON) bench/run_llm_judge.py --pool-file bench/judge_pool/pool.csv

qrels:
	$(PYTHON) bench/build_qrels.py

obj1-score:
	$(PYTHON) bench/score_obj1.py --reference-config B

obj1-smoke:
	$(PYTHON) bench/run_obj1.py --benchmark bench/benchmarks/obj1_primary.yaml --sample-per-dataset 5

obj2:
	$(PYTHON) bench/run_obj2.py

test:
	$(PYTHON) -m py_compile bench/common.py bench/fetch_datasets.py bench/build_chunks.py bench/lexical_candidates.py bench/build_judge_pool.py bench/run_llm_judge.py bench/build_qrels.py bench/run_obj1.py bench/score_obj1.py bench/run_obj2.py
