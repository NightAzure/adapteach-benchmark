"""
RAGAS evaluation for AdapTeach — Objective 2 comparative component analysis.

Uses RAGAS 0.4.x API (tested against ragas==0.4.3).
Primary LLM provider: Ollama (local). Gemini available via --provider gemini.

Modes:
  --build-golden   Generate ground-truth answers for your query set (run once before --eval).
  --eval           Run RAGAS metrics across all configs in a benchmark run JSONL.

Usage (Ollama — default):
  # Step 1 — generate ground truth
  py bench/ragas_eval.py --build-golden \\
      --queries bench/queries_test.jsonl \\
      --run-file bench/runs/run_<ts>.jsonl \\
      --out bench/golden_test.jsonl

  # Step 2 — evaluate
  py bench/ragas_eval.py --eval \\
      --run-file bench/runs/run_<ts>.jsonl \\
      --golden bench/golden_test.jsonl \\
      --out bench/results/ragas_results.csv

Usage (Gemini fallback):
  py bench/ragas_eval.py --build-golden --provider gemini --api-key YOUR_KEY ...
  py bench/ragas_eval.py --eval --provider gemini --api-key YOUR_KEY ...

Install (Ollama):
  pip install "ragas==0.4.3" langchain-ollama langchain-core datasets

Install (Gemini):
  pip install "ragas==0.4.3" langchain-google-genai langchain-core datasets google-genai
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Lazy imports — only pulled in when actually running RAGAS so that
# --build-golden can work with just google-generativeai installed.
# ---------------------------------------------------------------------------

def _require_ragas() -> None:
    try:
        import ragas  # noqa: F401
    except ImportError:
        raise SystemExit(
            "RAGAS not installed. Run:\n"
            "  pip install \"ragas==0.4.3\" langchain-ollama langchain-core datasets\n"
            "  (or langchain-google-genai instead of langchain-ollama for --provider gemini)"
        )


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_golden(path: Path) -> dict[str, str]:
    """Returns {query_id: ground_truth_answer}."""
    rows = _load_jsonl(path)
    return {r["id"]: r["ground_truth"] for r in rows if "ground_truth" in r}


def _contexts_from_record(record: dict) -> list[str]:
    """Extract retrieved context strings from a run record.

    Handles both flat format (context at top level) and our nested format
    where context lives under record["response"]["context"].
    """
    # Nested format: response.context
    response = record.get("response") or {}
    context_items = response.get("context") or record.get("context") or []
    texts = []
    for item in context_items:
        text = item.get("text") or item.get("content") or ""
        if text:
            texts.append(text.strip())
    return texts


# ---------------------------------------------------------------------------
# Build golden dataset
# ---------------------------------------------------------------------------

def build_golden(
    queries_path: Path,
    run_file: Path,
    out_path: Path,
    api_key: str = "",
    delay: float = 7.0,
    provider: str = "ollama",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "mistral",
) -> None:
    """
    Generate ground-truth answers for each query using the configured LLM +
    retrieved context from Config E (best config). Saves JSONL with {id, query, ground_truth}.
    """
    import urllib.request

    if provider == "ollama":
        def _generate(prompt: str) -> str:
            import json as _json
            url = ollama_url.rstrip("/") + "/api/generate"
            payload = {"model": ollama_model, "prompt": prompt, "stream": False}
            req = urllib.request.Request(
                url,
                data=_json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                return str(_json.loads(resp.read().decode("utf-8")).get("response", "")).strip()
    else:
        try:
            from google import genai
        except ImportError:
            raise SystemExit("google-genai not installed. Run:\n  pip install google-genai")
        client = genai.Client(api_key=api_key)

        def _generate(prompt: str) -> str:
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return response.text.strip()

    queries = {r["id"]: r["query"] for r in _load_jsonl(queries_path)}
    run_rows = _load_jsonl(run_file)

    # Use best available context by priority: E > F > D > C > B
    preferred_configs = ["E", "F", "D", "C", "B"]
    best_context_by_qid: dict[str, list[str]] = {}
    best_config_by_qid: dict[str, str] = {}
    for row in run_rows:
        qid = row.get("query_id") or row.get("id", "")
        config = row.get("config", "")
        if config not in preferred_configs:
            continue
        ctx = _contexts_from_record(row)
        if qid not in best_context_by_qid:
            best_context_by_qid[qid] = ctx
            best_config_by_qid[qid] = config
        elif preferred_configs.index(config) < preferred_configs.index(best_config_by_qid[qid]):
            best_context_by_qid[qid] = ctx
            best_config_by_qid[qid] = config

    existing: dict[str, str] = {}
    if out_path.exists():
        existing = _load_golden(out_path)
        print(f"Resuming — {len(existing)} ground-truth entries already exist.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with out_path.open("a", encoding="utf-8") as f:
        for qid, query in queries.items():
            if qid in existing:
                print(f"  [skip] {qid} already has ground truth")
                continue

            context_texts = best_context_by_qid.get(qid, [])
            if not context_texts:
                print(f"  [skip] {qid}: no context in run file (not in sample)")
                continue
            context_block = "\n\n---\n\n".join(context_texts[:5])

            prompt = textwrap.dedent(f"""
                You are an expert Python programming instructor.
                Based ONLY on the context below, write a concise, accurate ground-truth
                answer to the question. Limit to 2-3 sentences. Be factual and specific.

                Question: {query}

                Context:
                {context_block}

                Ground-truth answer:
            """).strip()

            try:
                ground_truth = _generate(prompt)
                print(f"  [ok] {qid}: {ground_truth[:80]}...")
            except Exception as e:
                print(f"  [error] {qid}: {e}")
                ground_truth = ""

            if ground_truth:
                row = {"id": qid, "query": query, "ground_truth": ground_truth}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                written += 1

            if delay > 0:
                time.sleep(delay)

    print(f"\nDone. Wrote {written} new ground-truth entries to {out_path}")


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def run_eval(
    run_file: Path,
    golden_path: Path,
    out_path: Path,
    api_key: str = "",
    configs: list[str] | None = None,
    provider: str = "ollama",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "mistral",
    timeout: int = 600,
) -> None:
    _require_ragas()

    import warnings
    from ragas.llms import LangchainLLMWrapper
    from ragas import SingleTurnSample, EvaluationDataset, evaluate
    # Use OLD-style metric instances — the new ragas.metrics.collections classes
    # inherit from SimpleBaseMetric, which is NOT a subclass of the Metric ABC
    # that evaluate() checks. Only old-style metrics work with evaluate().
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    class _STEmbeddings:
        """Minimal sentence-transformers adapter for old-style RAGAS metrics."""
        def __init__(self, model_name: str):
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)

        def embed_query(self, text: str) -> list:
            return self._model.encode(text, convert_to_numpy=True).tolist()

        def embed_documents(self, texts: list) -> list:
            return self._model.encode(texts, convert_to_numpy=True).tolist()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError:
                raise SystemExit(
                    "langchain-ollama not installed. Run:\n  pip install langchain-ollama"
                )
            # Disable Qwen3 thinking blocks — RAGAS's output parser can't handle
            # <think>...</think> preambles. Safe to pass for all models; non-thinking
            # models ignore it.
            evaluator_llm = LangchainLLMWrapper(
                ChatOllama(model=ollama_model, base_url=ollama_url, temperature=0.0,
                           request_timeout=timeout, model_kwargs={"think": False})
            )
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            evaluator_llm = LangchainLLMWrapper(
                ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.0,
                    google_api_key=api_key,
                )
            )
        evaluator_embeddings = _STEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

    # Attach LLM/embeddings to the singleton metric instances
    faithfulness.llm = evaluator_llm
    answer_relevancy.llm = evaluator_llm
    answer_relevancy.embeddings = evaluator_embeddings
    context_precision.llm = evaluator_llm
    context_recall.llm = evaluator_llm

    all_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    # Config A: no retrieval context — only answer_relevancy is meaningful.
    # faithfulness is undefined (nothing to ground against); context metrics are N/A.
    # We report faithfulness=0.000 by convention and context metrics as None.
    config_a_metrics = [answer_relevancy]

    golden = _load_golden(golden_path)
    run_rows = _load_jsonl(run_file)

    # Pre-filter to only queries that have a golden entry — everything else
    # is silently skipped later anyway, so evaluating it wastes time.
    golden_ids = set(golden.keys())
    run_rows_filtered = [r for r in run_rows if (r.get("query_id") or r.get("id", "")) in golden_ids]
    dropped = len(run_rows) - len(run_rows_filtered)
    print(f"Golden set: {len(golden_ids)} queries. "
          f"Run file: {len(run_rows)} rows → {len(run_rows_filtered)} rows match golden "
          f"({dropped} without golden entry skipped before evaluation).")

    # Group rows by config
    by_config: dict[str, list[dict]] = defaultdict(list)
    for row in run_rows_filtered:
        cfg = row.get("config", "?")
        if configs and cfg not in configs:
            continue
        by_config[cfg].append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for cfg in sorted(by_config.keys()):
        rows = by_config[cfg]
        is_config_a = cfg == "A"
        active_metrics = config_a_metrics if is_config_a else all_metrics
        print(f"\n── Config {cfg} ({len(rows)} samples){' [answer_relevancy only — no retrieval]' if is_config_a else ''} ──")

        samples = []
        for row in rows:
            qid = row.get("query_id") or row.get("id", "")
            query = row.get("query", "")
            resp_obj = row.get("response") or {}
            response = (
                resp_obj.get("answer")
                or row.get("answer")
                or ""
            )
            if not isinstance(response, str):
                response = ""
            contexts = [] if is_config_a else _contexts_from_record(row)
            reference = golden[qid]  # guaranteed present after pre-filter

            sample = SingleTurnSample(
                user_input=query,
                response=response,
                retrieved_contexts=contexts,
                reference=reference,
            )
            samples.append(sample)

        if not samples:
            print(f"  No samples for config {cfg}, skipping.")
            continue

        try:
            dataset = EvaluationDataset(samples=samples)
            result = evaluate(
                dataset=dataset,
                metrics=active_metrics,
                show_progress=True,
                raise_exceptions=False,
                batch_size=4,
            )
            df = result.to_pandas()

            means: dict[str, float | None] = {}
            for m in metric_names:
                if is_config_a and m == "faithfulness":
                    means[m] = 0.0  # undefined by convention: no context to ground against
                elif is_config_a and m in ("context_precision", "context_recall"):
                    means[m] = None  # N/A: no context was provided
                elif m in df.columns:
                    means[m] = round(float(df[m].mean(skipna=True)), 4)
                else:
                    means[m] = None

            print(f"  Results: {means}")
            all_results.append({"config": cfg, **means})

        except Exception as e:
            import traceback
            print(f"  RAGAS evaluation failed for config {cfg}: {e}")
            traceback.print_exc()
            all_results.append({"config": cfg, **{m: None for m in metric_names}})

    # Write CSV
    if all_results:
        fieldnames = ["config"] + metric_names
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\nResults saved to {out_path}")
        print("\nSummary:")
        print(f"{'Config':<10} {'Faithfulness':<14} {'Ans.Relevancy':<16} {'Ctx.Precision':<16} {'Ctx.Recall'}")
        print("-" * 65)
        for r in all_results:
            def _fmt(v): return f"{v:.4f}" if v is not None else "  N/A "
            print(f"{r['config']:<10} {_fmt(r.get('faithfulness')):<14} {_fmt(r.get('answer_relevancy')):<16} {_fmt(r.get('context_precision')):<16} {_fmt(r.get('context_recall'))}")
    else:
        print("No results to write.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import sys as _sys
    _root = Path(__file__).resolve().parents[1]
    _sys.path.insert(0, str(_root))
    from src.utils.config import load_dotenv
    load_dotenv(_root / ".env")

    parser = argparse.ArgumentParser(
        description="RAGAS evaluation for AdapTeach Objective 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-golden", action="store_true", help="Generate ground-truth answers using the configured LLM")
    mode.add_argument("--eval", action="store_true", help="Run RAGAS evaluation on a benchmark run file")

    parser.add_argument("--run-file", type=Path, required=True, help="Path to benchmark run JSONL")
    parser.add_argument("--queries", type=Path, help="Queries JSONL (required for --build-golden)")
    parser.add_argument("--golden", type=Path, help="Golden test set JSONL (required for --eval)")
    parser.add_argument("--out", type=Path, required=True, help="Output path (.jsonl for golden, .csv for eval)")
    parser.add_argument("--provider", default="ollama",
                        choices=["ollama", "gemini"],
                        help="LLM provider (default: ollama). Use 'gemini' with --api-key.")
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                        help="Ollama base URL (default: http://localhost:11434)")
    parser.add_argument("--ollama-model", default=os.environ.get("OLLAMA_MODEL", "mistral"),
                        help="Ollama model name (default: mistral)")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", os.environ.get("ADAPTEACH_LLM_GEMINI_API_KEY", "")),
                        help="Google AI Studio API key (only required when --provider gemini)")
    parser.add_argument("--configs", nargs="+", default=None, help="Configs to evaluate (default: all)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to sleep between LLM calls in --build-golden (default 0). Set to 7 when using Gemini free tier.")

    args = parser.parse_args()

    if args.provider == "gemini" and not args.api_key:
        raise SystemExit("--api-key (or GEMINI_API_KEY env var) is required when --provider gemini")

    if args.build_golden:
        if not args.queries:
            raise SystemExit("--queries is required for --build-golden")
        build_golden(
            queries_path=args.queries,
            run_file=args.run_file,
            out_path=args.out,
            api_key=args.api_key,
            delay=args.delay,
            provider=args.provider,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
        )
    else:
        if not args.golden:
            raise SystemExit("--golden is required for --eval")
        run_eval(
            run_file=args.run_file,
            golden_path=args.golden,
            out_path=args.out,
            api_key=args.api_key,
            configs=args.configs,
            provider=args.provider,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
        )


if __name__ == "__main__":
    main()
