from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import ragas_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Clean wrapper for Objective 2 automated evaluation.')
    sub = parser.add_subparsers(dest='command', required=True)

    build = sub.add_parser('build-golden', help='Generate frozen ground-truth answers for a query set.')
    build.add_argument('--queries', required=True)
    build.add_argument('--run-file', required=True)
    build.add_argument('--out', required=True)
    build.add_argument('--provider', default='ollama', choices=['ollama', 'gemini'],
                       help='LLM provider (default: ollama)')
    build.add_argument('--ollama-url', default='http://localhost:11434')
    build.add_argument('--ollama-model', default='mistral')
    build.add_argument('--api-key', default='', help='Gemini API key (only required when --provider gemini)')
    build.add_argument('--delay', type=float, default=0.0,
                       help='Seconds between LLM calls (default 0; set ~7 when using Gemini free tier)')

    evaluate = sub.add_parser('evaluate', help='Run RAGAS metrics over an existing run file.')
    evaluate.add_argument('--run-file', required=True)
    evaluate.add_argument('--golden', required=True)
    evaluate.add_argument('--out', required=True)
    evaluate.add_argument('--provider', default='ollama', choices=['ollama', 'gemini'],
                          help='LLM provider (default: ollama)')
    evaluate.add_argument('--ollama-url', default='http://localhost:11434')
    evaluate.add_argument('--ollama-model', default='mistral')
    evaluate.add_argument('--api-key', default='', help='Gemini API key (only required when --provider gemini)')
    evaluate.add_argument('--configs', default='A,B,D,E,F', help='Comma-separated configs to evaluate.')
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
    )


if __name__ == '__main__':
    main()
