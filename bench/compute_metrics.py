import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


# ── Dataset detection ─────────────────────────────────────────────────────────
# Maps query_id prefix → dataset name. Used when --query-files is not given.
_ID_PREFIX_MAP: list[tuple[str, str]] = [
    ("obj1-dev-",  "custom"),
    ("obj1-test-", "custom"),
    ("q-",         "custom"),
    ("d-",         "custom"),
    ("cs1qa-",     "cs1qa"),
    ("mbpp-",      "mbpp"),
    ("staqc-",     "staqc"),
    ("conala-",    "conala"),
]


def _dataset_from_id(qid: str) -> str:
    for prefix, name in _ID_PREFIX_MAP:
        if qid.startswith(prefix):
            return name
    return "unknown"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _latest_run(out_dir: Path) -> Path:
    files = sorted(out_dir.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("No benchmark run files found.")
    return files[0]


def _read_labels(path: Path) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        qid = row.get("query_id", "")
        chunk_id = row.get("chunk_id", "")
        rel = row.get("relevance", "")
        if not qid or not chunk_id or rel == "":
            continue
        try:
            rel_v = int(rel)
        except ValueError:
            continue
        if qid not in out:
            out[qid] = {}
        out[qid][chunk_id] = max(rel_v, out[qid].get(chunk_id, rel_v))
    return out


def _build_qid_to_dataset(query_files_arg: str) -> dict[str, str]:
    """Parse --query-files 'name:path,name:path,...' into a qid→dataset map."""
    qid_map: dict[str, str] = {}
    for entry in query_files_arg.split(","):
        entry = entry.strip()
        if ":" not in entry:
            continue
        name, path_str = entry.split(":", 1)
        path = Path(path_str.strip())
        if not path.exists():
            print(f"  [warn] query file not found: {path}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid_map[row["id"]] = name.strip()
    return qid_map


def _score_mrr(retrieved_chunk_ids: list[str], labels_for_query: dict[str, int]) -> float:
    if not retrieved_chunk_ids or not labels_for_query:
        return 0.0
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if labels_for_query.get(cid, 0) > 0:
            return 1.0 / rank
    return 0.0


def _score_ndcg_at_k(retrieved_chunk_ids: list[str], labels_for_query: dict[str, int], k: int = 10) -> float:
    if not retrieved_chunk_ids or not labels_for_query:
        return 0.0
    ranked = retrieved_chunk_ids[:k]
    dcg = 0.0
    for i, cid in enumerate(ranked, start=1):
        rel = max(0, labels_for_query.get(cid, 0))
        if rel > 0:
            dcg += rel / (1.0 if i == 1 else math.log2(i + 1))
    ideal_rels = sorted((max(0, v) for v in labels_for_query.values()), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        if rel > 0:
            idcg += rel / (1.0 if i == 1 else math.log2(i + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _score_context_precision(context: list[dict[str, Any]]) -> float:
    if not context:
        return 0.0
    relevant = 0
    for c in context:
        ev = c.get("evidence", [])
        if ev or c.get("score", 0) > 0:
            relevant += 1
    return relevant / len(context)


def _score_faithfulness(answer: str, context: list[dict[str, Any]]) -> float:
    if not answer or not context:
        return 0.0
    support_ids = [c.get("chunk_id", "") for c in context[:3]]
    matches = sum(1 for cid in support_ids if cid and cid in answer)
    return matches / max(1, len(support_ids))


def _score_answer_relevance(answer: str, query: str) -> float:
    q_tokens = {t.lower() for t in query.split() if len(t) > 2}
    a_tokens = {t.lower() for t in answer.split() if len(t) > 2}
    if not q_tokens:
        return 0.0
    return len(q_tokens & a_tokens) / len(q_tokens)


def _accumulate(accum: dict, it: dict[str, Any], labels: dict[str, dict[str, int]]) -> None:
    """Add one run row's metrics into the accumulator dict."""
    record = it["record"]
    response = it["response"]
    context = response.get("context", [])
    qid = it.get("query_id", "")
    answer = response.get("answer", "") or ""
    query = it.get("query", "")
    retrieved_ids = [c.get("chunk_id", "") for c in context if c.get("chunk_id")]

    accum["n"] += 1
    if qid in labels and labels[qid]:
        accum["labeled"] += 1
        accum["mrr"]  += _score_mrr(retrieved_ids, labels[qid])
        accum["ndcg"] += _score_ndcg_at_k(retrieved_ids, labels[qid], k=10)
        rel_hits = sum(1 for cid in retrieved_ids if labels[qid].get(cid, 0) > 0)
        accum["ctx_prec"] += (rel_hits / len(retrieved_ids)) if retrieved_ids else 0.0
    else:
        if context:
            accum["mrr"] += 1.0
        accum["ctx_prec"] += _score_context_precision(context)

    accum["faithfulness"]     += _score_faithfulness(answer, context)
    accum["answer_relevance"] += _score_answer_relevance(answer, query)

    validation = record.get("stage_outputs", {}).get("validation")
    if isinstance(validation, dict):
        accum["parsability"]    += 1.0 if validation.get("deterministic", {}).get("parsability", {}).get("valid", False) else 0.0
        accum["validator_pass"] += 1.0 if validation.get("valid", False) else 0.0

    accum["fallback"]          += 1.0 if response["status"] == "ok_with_fallback" else 0.0
    accum["ret_lat"]           += float(record["stage_timings_ms"].get("retrieval", 0.0))
    accum["gen_lat"]           += float(record["stage_timings_ms"].get("generation", 0.0))
    accum["tokens"]            += (
        int(record.get("token_counts", {}).get("prompt", 0))
        + int(record.get("token_counts", {}).get("response", 0))
    )


def _empty_accum() -> dict:
    return {k: 0 for k in (
        "n", "labeled", "mrr", "ndcg", "ctx_prec",
        "faithfulness", "answer_relevance", "parsability",
        "validator_pass", "fallback", "ret_lat", "gen_lat", "tokens",
    )}


def _to_row(dataset: str, cfg: str, accum: dict) -> dict:
    n = accum["n"]
    if n == 0:
        return {}
    return {
        "dataset": dataset,
        "config": cfg,
        "n": n,
        "labeled_rows": accum["labeled"],
        "MRR@10":            round(accum["mrr"]  / n, 6),
        "nDCG@10":           round(accum["ndcg"] / n, 6),
        "context_precision": round(accum["ctx_prec"] / n, 6),
        "faithfulness_proxy":    round(accum["faithfulness"]     / n, 6),
        "answer_relevance_proxy":round(accum["answer_relevance"] / n, 6),
        "parsability_rate":      round(accum["parsability"]      / n, 6),
        "validator_pass_rate":   round(accum["validator_pass"]   / n, 6),
        "fallback_rate":         round(accum["fallback"]         / n, 6),
        "retrieval_latency_ms_avg":  round(accum["ret_lat"] / n, 3),
        "generation_latency_ms_avg": round(accum["gen_lat"] / n, 3),
        "total_tokens": accum["tokens"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute benchmark metrics from run logs")
    p.add_argument("--run-file",    default="")
    p.add_argument("--runs-dir",    default="bench/runs")
    p.add_argument("--labels-csv",  default="bench/labels/labels.csv")
    p.add_argument("--out-csv",     default="bench/runs/metrics_latest.csv")
    p.add_argument(
        "--query-files",
        default="",
        help=(
            "Comma-separated 'name:path' pairs to assign each query_id a dataset label. "
            "Example: custom:bench/queries_custom.jsonl,cs1qa:bench/queries_cs1qa.jsonl. "
            "If omitted, dataset is inferred from query_id prefix."
        ),
    )
    p.add_argument(
        "--no-combined",
        action="store_true",
        help="Skip the 'combined' row that aggregates all datasets.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    run_file = Path(args.run_file) if args.run_file else _latest_run(runs_dir)
    rows = _read_jsonl(run_file)
    labels = _read_labels(Path(args.labels_csv))

    # Build qid → dataset mapping
    if args.query_files:
        qid_to_dataset = _build_qid_to_dataset(args.query_files)
    else:
        qid_to_dataset = {}  # fall back to prefix detection

    def get_dataset(qid: str) -> str:
        if qid_to_dataset:
            return qid_to_dataset.get(qid, "unknown")
        return _dataset_from_id(qid)

    # Accumulate per (dataset, config)
    # key: (dataset, config) → accum dict
    by_ds_cfg: dict[tuple[str, str], dict] = {}
    by_cfg_combined: dict[str, dict] = {}  # for the "combined" row

    for row in rows:
        cfg = row["config"]
        qid = row.get("query_id", "")
        ds  = get_dataset(qid)
        key = (ds, cfg)
        if key not in by_ds_cfg:
            by_ds_cfg[key] = _empty_accum()
        _accumulate(by_ds_cfg[key], row, labels)

        if cfg not in by_cfg_combined:
            by_cfg_combined[cfg] = _empty_accum()
        _accumulate(by_cfg_combined[cfg], row, labels)

    # Build output rows: sorted by dataset then config
    out_rows: list[dict] = []

    # Determine dataset order: custom first, then alphabetical externals, unknown last
    ds_order = ["custom", "cs1qa", "mbpp", "staqc", "conala", "unknown"]
    seen_datasets = sorted(
        {ds for ds, _ in by_ds_cfg.keys()},
        key=lambda d: (ds_order.index(d) if d in ds_order else 99, d),
    )

    for ds in seen_datasets:
        for cfg in sorted({cfg for d, cfg in by_ds_cfg if d == ds}):
            row = _to_row(ds, cfg, by_ds_cfg[(ds, cfg)])
            if row:
                out_rows.append(row)

    # Combined row across all datasets
    if not args.no_combined:
        for cfg in sorted(by_cfg_combined.keys()):
            row = _to_row("combined", cfg, by_cfg_combined[cfg])
            if row:
                out_rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else ["dataset", "config", "n"]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    datasets_found = seen_datasets + (["combined"] if not args.no_combined else [])
    print(json.dumps({
        "run_file":      str(run_file.as_posix()),
        "metrics_csv":   str(out_csv.as_posix()),
        "datasets":      datasets_found,
        "configs":       sorted(by_cfg_combined.keys()),
        "total_rows":    len(out_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
