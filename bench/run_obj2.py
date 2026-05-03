from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import ragas_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Objective 2 automated evaluation.')
    sub = parser.add_subparsers(dest='command', required=True)

    build = sub.add_parser('build-golden')
    build.add_argument('--queries', required=True)
    build.add_argument('--run-file', required=True)
    build.add_argument('--out', required=True)
    build.add_argument('--provider', default='ollama', choices=['ollama', 'gemini'])
    build.add_argument('--ollama-url', default='http://localhost:11434')
    build.add_argument('--ollama-model', default='mistral')
    build.add_argument('--api-key', default='')
    build.add_argument('--delay', type=float, default=0.0)

    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--run-file', required=True)
    evaluate.add_argument('--golden', required=True)
    evaluate.add_argument('--out', required=True)
    evaluate.add_argument('--provider', default='ollama', choices=['ollama', 'gemini'])
    evaluate.add_argument('--ollama-url', default='http://localhost:11434')
    evaluate.add_argument('--ollama-model', default='mistral')
    evaluate.add_argument('--api-key', default='')
    evaluate.add_argument('--configs', default='A,B,D,E,F')
    evaluate.add_argument('--timeout', type=int, default=600)
    evaluate.add_argument('--gemini-rpm', type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == 'build-golden':
        ragas_eval.build_golden(
            queries_path=ROOT / args.queries,
            run_file=ROOT / args.run_file,
            out_path=ROOT / args.out,
            api_key=args.api_key,
            delay=args.delay,
            provider=args.provider,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
        )
        return
    configs = [c.strip() for c in args.configs.split(',') if c.strip()]
    ragas_eval.run_eval(
        run_file=ROOT / args.run_file,
        golden_path=ROOT / args.golden,
        out_path=ROOT / args.out,
        api_key=args.api_key,
        configs=configs,
        provider=args.provider,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        timeout=args.timeout,
        gemini_rpm=args.gemini_rpm,
    )


if __name__ == '__main__':
    main()
