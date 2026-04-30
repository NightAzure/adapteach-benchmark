from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.common import load_benchmark_spec, read_jsonl, sha256_file
from src.pipelines.runner import run_pipeline
from src.utils.config import load_app_config, load_dotenv, load_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Objective 1 retrieval benchmark from a single benchmark spec.')
    parser.add_argument('--benchmark', default='bench/benchmarks/obj1_primary.yaml')
    parser.add_argument('--out-dir', default='bench/runs')
    parser.add_argument('--sample-per-dataset', type=int, default=0)
    parser.add_argument('--configs', default='', help='Comma-separated override, e.g. B,C,D,E,F')
    parser.add_argument('--dry-run', default='', choices=['', 'none', 'retrieval', 'graph'])
    parser.add_argument('--provider', default='', choices=['', 'mock', 'gemini', 'ollama'])
    parser.add_argument('--model', default='')
    parser.add_argument('--delay', type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    spec_path = ROOT / args.benchmark
    spec = load_benchmark_spec(spec_path)
    dry_run = args.dry_run or spec.get('dry_run', 'retrieval')
    artifact_type = spec.get('artifact_type', 'flashcard')
    configs = [c.strip() for c in (args.configs or ','.join(spec.get('configs', []))).split(',') if c.strip()]
    if not configs:
        raise SystemExit('No configs provided in benchmark spec or --configs.')

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    bench_name = spec.get('name', 'obj1')
    run_path = out_dir / f'run_{timestamp}_{bench_name}.jsonl'
    manifest_path = out_dir / f'run_{timestamp}_{bench_name}.manifest.json'

    app_cfg = load_app_config()
    if args.provider:
        app_cfg['llm']['provider'] = args.provider
    if args.model:
        app_cfg['llm']['model'] = args.model

    query_total = 0
    query_counts: dict[str, int] = {}
    with run_path.open('w', encoding='utf-8') as handle:
        for dataset in spec.get('datasets', []):
            dataset_name = dataset['name']
            query_file = ROOT / dataset['query_file']
            query_rows = read_jsonl(query_file)
            if args.sample_per_dataset > 0:
                query_rows = query_rows[: args.sample_per_dataset]
            query_counts[dataset_name] = len(query_rows)
            query_total += len(query_rows)
            for cfg_name in configs:
                pipeline_cfg = load_pipeline_config(cfg_name)
                for row in query_rows:
                    request = {
                        'query': row['query'],
                        'artifact_type': artifact_type,
                        'query_id': row['id'],
                        'category': row.get('category', ''),
                        'topic': row.get('topic', ''),
                    }
                    response, record = run_pipeline(request=request, app_config=app_cfg, pipeline_config=pipeline_cfg, dry_run=dry_run)
                    handle.write(json.dumps({
                        'timestamp': timestamp,
                        'benchmark': bench_name,
                        'dataset': dataset_name,
                        'query_id': row['id'],
                        'query': row['query'],
                        'config': cfg_name,
                        'artifact_type': artifact_type,
                        'dry_run': dry_run,
                        'response': response,
                        'record': record,
                    }, ensure_ascii=False) + '\n')
                    if args.delay > 0 and dry_run == 'none':
                        time.sleep(args.delay)

    manifest = {
        'benchmark': bench_name,
        'benchmark_spec': spec_path.as_posix(),
        'benchmark_spec_sha256': sha256_file(spec_path),
        'timestamp': timestamp,
        'run_file': run_path.as_posix(),
        'dry_run': dry_run,
        'artifact_type': artifact_type,
        'configs': configs,
        'datasets': query_counts,
        'query_total': query_total,
        'provider': app_cfg['llm']['provider'],
        'model': app_cfg['llm']['model'],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
