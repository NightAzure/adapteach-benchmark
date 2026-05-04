from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8', newline='') as f:
        buf = ''
        for raw_line in f:
            buf += raw_line
            try:
                obj = json.loads(buf.strip())
                rows.append(obj)
                buf = ''
            except json.JSONDecodeError:
                pass  # line was split by embedded newline — accumulate and retry
    if buf.strip():
        rows.append(json.loads(buf.strip()))
    return rows


def load_labels_from_dir(labels_dir: Path) -> dict[str, dict[str, int]]:
    qrels_files = sorted(labels_dir.glob('qrels_*.csv'))
    if not qrels_files:
        raise SystemExit(
            f'No qrels_*.csv files found in {labels_dir}.\n'
            'Run the single Obj1 label path first: build_judge_pool.py -> run_llm_judge.py -> build_qrels.py.'
        )
    print(f'Scoring against LLM-validated qrels ({len(qrels_files)} file(s)).')
    labels: dict[str, dict[str, int]] = defaultdict(dict)
    for csv_path in qrels_files:
        with csv_path.open('r', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                qid = row['query_id']
                target_id = row.get('doc_id') or row.get('chunk_id')
                if not target_id:
                    continue
                rel = int(row['relevance'])
                labels[qid][target_id] = max(rel, labels[qid].get(target_id, 0))
    return labels


def latest_run(runs_dir: Path) -> Path:
    files = sorted(runs_dir.glob('run_*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f'No run files found in {runs_dir}')
    return files[0]


def score_mrr(retrieved: list[str], labels: dict[str, int], k: int = 10) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if rank > k:
            break
        if labels.get(cid, 0) > 0:
            return 1.0 / rank
    return 0.0


def score_hit_at_k(retrieved: list[str], labels: dict[str, int], k: int = 5) -> float:
    return 1.0 if any(labels.get(cid, 0) > 0 for cid in retrieved[:k]) else 0.0


def score_precision_at_k(retrieved: list[str], labels: dict[str, int], k: int = 5) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for cid in top if labels.get(cid, 0) > 0)
    return hits / len(top)


def score_ndcg_at_k(retrieved: list[str], labels: dict[str, int], k: int = 10) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    dcg = 0.0
    for rank, cid in enumerate(top, start=1):
        rel = max(0, labels.get(cid, 0))
        if rel:
            dcg += rel / (1.0 if rank == 1 else math.log2(rank + 1))
    ideal = sorted((rel for rel in labels.values() if rel > 0), reverse=True)[:k]
    if not ideal:
        return 0.0
    idcg = 0.0
    for rank, rel in enumerate(ideal, start=1):
        idcg += rel / (1.0 if rank == 1 else math.log2(rank + 1))
    return dcg / idcg if idcg else 0.0


def bootstrap_ci(values: list[float], iterations: int = 2000, seed: int = 13) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rnd = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(iterations):
        draw = [values[rnd.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    samples.sort()
    lo = samples[int(0.025 * (iterations - 1))]
    hi = samples[int(0.975 * (iterations - 1))]
    return (lo, hi)


def paired_randomization_p(a: list[float], b: list[float], iterations: int = 5000, seed: int = 13) -> float:
    if len(a) != len(b) or not a:
        return 1.0
    diffs = [x - y for x, y in zip(a, b)]
    observed = abs(sum(diffs) / len(diffs))
    rnd = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        flipped = [d if rnd.random() < 0.5 else -d for d in diffs]
        test_stat = abs(sum(flipped) / len(flipped))
        if test_stat >= observed:
            extreme += 1
    return (extreme + 1) / (iterations + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Strict Obj1 retrieval scoring with confidence intervals and paired significance tests.')
    parser.add_argument('--run-file', default='')
    parser.add_argument('--runs-dir', default='bench/runs')
    parser.add_argument('--labels-dir', default='bench/labels')
    parser.add_argument('--reference-config', default='B')
    parser.add_argument('--out-dir', default='bench/results/obj1_latest')
    parser.add_argument('--seed', type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_path = ROOT / args.run_file if args.run_file else latest_run(ROOT / args.runs_dir)
    rows = read_jsonl(run_path)
    labels = load_labels_from_dir(ROOT / args.labels_dir)
    if not labels:
        raise SystemExit('Qrels files were found, but no relevance rows were loaded.')

    per_query: list[dict[str, Any]] = []
    by_dataset_config: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        qid = row['query_id']
        if qid not in labels:
            continue
        dataset = row.get('dataset', 'unknown')
        config = row['config']
        context = row.get('response', {}).get('context', [])
        seen_ids: set[str] = set()
        retrieved = []
        for item in context:
            target_id = item.get('doc_id') or item.get('provenance', {}).get('doc_id') or item.get('chunk_id', '')
            if not target_id or target_id in seen_ids:
                continue
            seen_ids.add(target_id)
            retrieved.append(target_id)
        record = row.get('record', {})
        latency = float(record.get('stage_timings_ms', {}).get('retrieval', 0.0))
        metric_row = {
            'dataset': dataset,
            'config': config,
            'query_id': qid,
            'MRR@10': score_mrr(retrieved, labels[qid]),
            'nDCG@10': score_ndcg_at_k(retrieved, labels[qid], 10),
            'P@5': score_precision_at_k(retrieved, labels[qid], 5),
            'Hit@5': score_hit_at_k(retrieved, labels[qid], 5),
            'retrieval_latency_ms': latency,
        }
        per_query.append(metric_row)
        by_dataset_config[(dataset, config)].append(metric_row)

    if not by_dataset_config:
        raise SystemExit('No run rows matched the available qrels. Check that query IDs and dataset outputs align.')

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = out_dir / 'per_query.csv'
    with per_query_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['dataset', 'config', 'query_id', 'MRR@10', 'nDCG@10', 'P@5', 'Hit@5', 'retrieval_latency_ms'])
        writer.writeheader()
        writer.writerows(per_query)

    summary_rows: list[dict[str, Any]] = []
    grouped_for_significance: dict[tuple[str, str], dict[str, list[float]]] = {}
    for key, items in sorted(by_dataset_config.items()):
        dataset, config = key
        ndcg_values = [float(item['nDCG@10']) for item in items]
        mrr_values = [float(item['MRR@10']) for item in items]
        p5_values = [float(item['P@5']) for item in items]
        hit_values = [float(item['Hit@5']) for item in items]
        lat_values = [float(item['retrieval_latency_ms']) for item in items]
        ndcg_ci = bootstrap_ci(ndcg_values, seed=args.seed)
        mrr_ci = bootstrap_ci(mrr_values, seed=args.seed)
        summary_rows.append({
            'dataset': dataset,
            'config': config,
            'n': len(items),
            'MRR@10_mean': round(mean(mrr_values), 6),
            'MRR@10_ci_low': round(mrr_ci[0], 6),
            'MRR@10_ci_high': round(mrr_ci[1], 6),
            'nDCG@10_mean': round(mean(ndcg_values), 6),
            'nDCG@10_ci_low': round(ndcg_ci[0], 6),
            'nDCG@10_ci_high': round(ndcg_ci[1], 6),
            'P@5_mean': round(mean(p5_values), 6),
            'Hit@5_mean': round(mean(hit_values), 6),
            'retrieval_latency_ms_mean': round(mean(lat_values), 3),
        })
        grouped_for_significance[(dataset, config)] = {'nDCG@10': ndcg_values, 'MRR@10': mrr_values}

    summary_path = out_dir / 'summary.csv'
    with summary_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    sig_rows: list[dict[str, Any]] = []
    datasets = sorted({dataset for dataset, _ in grouped_for_significance.keys()})
    for dataset in datasets:
        ref_key = (dataset, args.reference_config)
        if ref_key not in grouped_for_significance:
            continue
        ref_ndcg = grouped_for_significance[ref_key]['nDCG@10']
        ref_mrr = grouped_for_significance[ref_key]['MRR@10']
        for (ds, config), metric_map in sorted(grouped_for_significance.items()):
            if ds != dataset or config == args.reference_config:
                continue
            ndcg = metric_map['nDCG@10']
            mrr = metric_map['MRR@10']
            if len(ndcg) != len(ref_ndcg):
                continue
            ndcg_diff = [a - b for a, b in zip(ndcg, ref_ndcg)]
            mrr_diff = [a - b for a, b in zip(mrr, ref_mrr)]
            ndcg_ci = bootstrap_ci(ndcg_diff, seed=args.seed)
            mrr_ci = bootstrap_ci(mrr_diff, seed=args.seed)
            sig_rows.append({
                'dataset': dataset,
                'reference_config': args.reference_config,
                'compare_config': config,
                'delta_nDCG@10_mean': round(mean(ndcg_diff), 6),
                'delta_nDCG@10_ci_low': round(ndcg_ci[0], 6),
                'delta_nDCG@10_ci_high': round(ndcg_ci[1], 6),
                'delta_nDCG@10_p': round(paired_randomization_p(ndcg, ref_ndcg, seed=args.seed), 6),
                'delta_MRR@10_mean': round(mean(mrr_diff), 6),
                'delta_MRR@10_ci_low': round(mrr_ci[0], 6),
                'delta_MRR@10_ci_high': round(mrr_ci[1], 6),
                'delta_MRR@10_p': round(paired_randomization_p(mrr, ref_mrr, seed=args.seed), 6),
            })

    if sig_rows:
        sig_path = out_dir / 'significance_vs_reference.csv'
        with sig_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sig_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sig_rows)
    else:
        sig_path = None

    print(json.dumps({
        'run_file': run_path.as_posix(),
        'summary_csv': summary_path.as_posix(),
        'per_query_csv': per_query_path.as_posix(),
        'significance_csv': sig_path.as_posix() if sig_path else '',
        'reference_config': args.reference_config,
    }, indent=2))


if __name__ == '__main__':
    main()
