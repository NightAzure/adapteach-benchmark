PYTHON ?= py

.PHONY: setup corpus snapshot chunk-fixed chunk-ast graphs index bench bench-prepare bench-run bench-metrics bench-ablations bench-kappa test check-lock

setup:
	$(PYTHON) -m pip install -r requirements.lock

corpus:
	$(PYTHON) -m src.indexing.corpus_pipeline

snapshot:
ifdef EXCLUDE_AI
	$(PYTHON) -m src.indexing.snapshot_tool create --exclude-ai
else
	$(PYTHON) -m src.indexing.snapshot_tool create
endif

index:
ifndef SNAPSHOT_ID
	$(error SNAPSHOT_ID is required. Example: make index SNAPSHOT_ID=abc123 CHUNKER=ast)
endif
ifndef CHUNKER
	$(eval CHUNKER=fixed)
endif
	$(PYTHON) -m src.chunking.build_chunks --chunker $(CHUNKER)
	$(PYTHON) -m src.indexing.build_indexes --snapshot-id $(SNAPSHOT_ID)

chunk-fixed:
	$(PYTHON) -m src.chunking.build_chunks --chunker fixed

chunk-ast:
	$(PYTHON) -m src.chunking.build_chunks --chunker ast

graphs:
	$(PYTHON) -m src.graphs.build_graphs

bench:
	$(PYTHON) bench/run_benchmark_suite.py --query-set bench/queries_dev.jsonl --configs A,B,C,D,E --dry-run retrieval
	$(PYTHON) bench/compute_metrics.py
	$(PYTHON) bench/run_ablations.py --query-set bench/queries_dev.jsonl --dry-run retrieval

bench-prepare:
	$(PYTHON) bench/prepare_benchmark.py

bench-run:
	$(PYTHON) bench/run_benchmark_suite.py --query-set bench/queries_dev.jsonl --configs A,B,C,D,E --dry-run retrieval

bench-metrics:
	$(PYTHON) bench/compute_metrics.py

bench-ablations:
	$(PYTHON) bench/run_ablations.py --query-set bench/queries_dev.jsonl --dry-run retrieval

bench-kappa:
	$(PYTHON) bench/compute_kappa.py --labels-csv bench/labels/labels.csv

test:
	$(PYTHON) -m pytest -q -p no:cacheprovider tests

check-lock:
	$(PYTHON) scripts/check_lockfile.py
