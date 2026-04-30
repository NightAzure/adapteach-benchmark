PYTHON ?= python

.PHONY: help setup chunks index datasets labels reproduce \
        obj1-primary obj1-full obj1-score obj1-smoke obj2 test

help:
	@echo "AdapTeach Benchmark — available targets:"
	@echo ""
	@echo "  setup          Install dependencies from requirements.lock"
	@echo "  datasets       Validate shipped query files"
	@echo "  chunks         Build chunk_manifest.json from corpus/ (AST + text fallback)"
	@echo "  index          Build vector + BM25 indexes from chunk manifest"
	@echo "  labels         Build deterministic silver labels (primary benchmark)"
	@echo "  reproduce      Full primary benchmark run: datasets → chunks → index → labels → obj1-primary → obj1-score"
	@echo ""
	@echo "  obj1-primary   Run Objective 1 (primary query set)"
	@echo "  obj1-full      Run Objective 1 (full query set)"
	@echo "  obj1-score     Score Objective 1 results against config B"
	@echo "  obj1-smoke     Quick smoke test: 5 samples per dataset + score"
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

labels:
	$(PYTHON) bench/build_silver_labels.py --benchmark bench/benchmarks/obj1_primary.yaml

reproduce: datasets chunks index labels obj1-primary obj1-score

obj1-primary:
	$(PYTHON) bench/run_obj1.py --benchmark bench/benchmarks/obj1_primary.yaml

obj1-full:
	$(PYTHON) bench/run_obj1.py --benchmark bench/benchmarks/obj1_full.yaml

obj1-score:
	$(PYTHON) bench/score_obj1.py --reference-config B

obj1-smoke:
	$(PYTHON) bench/run_obj1.py --benchmark bench/benchmarks/obj1_primary.yaml --sample-per-dataset 5
	$(PYTHON) bench/score_obj1.py --reference-config B

obj2:
	$(PYTHON) bench/run_obj2.py

test:
	$(PYTHON) -m py_compile bench/common.py bench/fetch_datasets.py bench/build_chunks.py bench/build_silver_labels.py bench/run_obj1.py bench/score_obj1.py bench/run_obj2.py
