"""
RAGAS evaluation for AdapTeach — Objective 2 comparative component analysis.

Uses RAGAS 0.4.x API (tested against ragas==0.4.3).

Modes:
  --build-golden   Use Gemini to generate ground-truth answers for your query set.
                   Run this once before --eval.
  --eval           Run RAGAS metrics across all configs in a benchmark run JSONL.

Usage:
  # Step 1 — generate ground truth
  py bench/ragas_eval.py --build-golden \\
      --queries bench/queries_test.jsonl \\
      --run-file bench/runs/run_<ts>.jsonl \\
      --out bench/golden_test.jsonl \\
      --api-key YOUR_GOOGLE_AI_STUDIO_KEY

  # Step 2 — evaluate
  py bench/ragas_eval.py --eval \\
      --run-file bench/runs/run_<ts>.jsonl \\
      --golden bench/golden_test.jsonl \\
      --out bench/results/ragas_results.csv \\
      --api-key YOUR_GOOGLE_AI_STUDIO_KEY

Install:
  pip install "ragas==0.4.3" langchain-google-genai langchain-core datasets
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
            "  pip install \"ragas==0.4.3\" langchain-google-genai langchain-core datasets"
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
    api_key: str,
    delay: float = 7.0,
) -> None:
    """
    Generate ground-truth answers for each query using Gemini + the retrieved
    context from Config E (best config). Saves a JSONL with {id, query, ground_truth}.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise SystemExit(
            "google-generativeai not installed. Run:\n"
            "  pip install google-generativeai"
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    queries = {r["id"]: r["query"] for r in _load_jsonl(queries_path)}
    run_rows = _load_jsonl(run_file)

    # Use Config E context if available; fall back to C, D, B in order
    preferred_configs = ["E", "D", "C", "B"]
    best_context_by_qid: dict[str, list[str]] = {}
    for row in run_rows:
        qid = row.get("query_id") or row.get("id", "")
        config = row.get("config", "")
        ctx = _contexts_from_record(row)
        if qid not in best_context_by_qid and config in preferred_configs:
            best_context_by_qid[qid] = ctx
        elif qid in best_context_by_qid:
            current_cfg = next(
                (r.get("config", "") for r in run_rows if (r.get("query_id") or r.get("id")) == qid),
                "Z",
            )
            if preferred_configs.index(config) < preferred_configs.index(current_cfg):
                best_context_by_qid[qid] = ctx

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
            context_block = "\n\n---\n\n".join(context_texts[:5]) if context_texts else "(no context available)"

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
                response = model.generate_content(prompt)
                ground_truth = response.text.strip()
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
    api_key: str,
    configs: list[str] | None = None,
) -> None:
    _require_ragas()

    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas import SingleTurnSample, EvaluationDataset, evaluate
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )

    os.environ.setdefault("GOOGLE_API_KEY", api_key)

    # LLM judge + embeddings (Gemini)
    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            google_api_key=api_key,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
    )

    # Metrics — attach LLM/embeddings at construction per RAGAS 0.4 API
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    golden = _load_golden(golden_path)
    run_rows = _load_jsonl(run_file)

    # Group rows by config
    by_config: dict[str, list[dict]] = defaultdict(list)
    for row in run_rows:
        cfg = row.get("config", "?")
        if configs and cfg not in configs:
            continue
        by_config[cfg].append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for cfg in sorted(by_config.keys()):
        rows = by_config[cfg]
        print(f"\n── Config {cfg} ({len(rows)} samples) ──")

        samples = []
        for row in rows:
            qid = row.get("query_id") or row.get("id", "")
            query = row.get("query", "")
            # Answer is nested under response.answer in our run format
            resp_obj = row.get("response") or {}
            response = (
                resp_obj.get("answer")
                or row.get("answer")
                or ""
            )
            if not isinstance(response, str):
                response = ""
            contexts = _contexts_from_record(row)
            reference = golden.get(qid, "")

            # Config A has no retrieval — ContextPrecision/Recall will be NaN
            sample = SingleTurnSample(
                user_input=query,
                response=response,
                retrieved_contexts=contexts,
                reference=reference if reference else None,
            )
            samples.append(sample)

        if not samples:
            print(f"  No samples for config {cfg}, skipping.")
            continue

        dataset = EvaluationDataset(samples=samples)

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                show_progress=True,
                raise_exceptions=False,
            )
            df = result.to_pandas()

            means = {}
            for m in metric_names:
                if m in df.columns:
                    means[m] = round(float(df[m].mean(skipna=True)), 4)
                else:
                    means[m] = None

            print(f"  Results: {means}")
            all_results.append({"config": cfg, **means})

        except Exception as e:
            print(f"  RAGAS evaluation failed for config {cfg}: {e}")
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
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation for AdapTeach Objective 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-golden", action="store_true", help="Generate ground-truth answers via Gemini")
    mode.add_argument("--eval", action="store_true", help="Run RAGAS evaluation on a benchmark run file")

    parser.add_argument("--run-file", type=Path, required=True, help="Path to benchmark run JSONL")
    parser.add_argument("--queries", type=Path, help="Queries JSONL (required for --build-golden)")
    parser.add_argument("--golden", type=Path, help="Golden test set JSONL (required for --eval)")
    parser.add_argument("--out", type=Path, required=True, help="Output path (.jsonl for golden, .csv for eval)")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", os.environ.get("ADAPTEACH_LLM_GEMINI_API_KEY", "")), help="Google AI Studio API key")
    parser.add_argument("--configs", nargs="+", default=None, help="Configs to evaluate (default: all)")
    parser.add_argument("--delay", type=float, default=7.0,
                        help="Seconds to sleep between Gemini calls in --build-golden (default 7 for free tier 10 RPM). Set 0 to disable.")

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("API key required. Pass --api-key or set ADAPTEACH_LLM_GEMINI_API_KEY.")

    if args.build_golden:
        if not args.queries:
            raise SystemExit("--queries is required for --build-golden")
        build_golden(
            queries_path=args.queries,
            run_file=args.run_file,
            out_path=args.out,
            api_key=args.api_key,
            delay=args.delay,
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
        )


if __name__ == "__main__":
    main()
