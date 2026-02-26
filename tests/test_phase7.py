import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.providers import generate_with_provider


def _run_py(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=True,
    )


def test_provider_gemini_missing_key_falls_back_to_mock() -> None:
    app = {
        "llm": {
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "prompt_template_version": "v1",
            "gemini_api_key_env": "TEST_GEMINI_KEY",
            "ollama_base_url": "http://localhost:11434",
        }
    }
    os.environ.pop("TEST_GEMINI_KEY", None)
    answer, debug = generate_with_provider("Explain loops", [], "A", app)
    assert debug["provider_used"] == "mock_fallback"
    assert "missing_env:TEST_GEMINI_KEY" in debug.get("fallback_reason", "")
    assert answer.startswith("[A]")


def test_phase7_benchmark_scripts_smoke() -> None:
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="phase7-") as tmp:
        tmp_dir = Path(tmp)
        dev = tmp_dir / "queries_dev.jsonl"
        test = tmp_dir / "queries_test.jsonl"
        labels_template = tmp_dir / "labels_template.csv"
        labels = tmp_dir / "labels.csv"
        runs = tmp_dir / "runs"
        metrics = tmp_dir / "metrics.csv"
        ablations = tmp_dir / "ablations.csv"
        freeze = tmp_dir / "freeze_manifest.json"

        _run_py(
            [
                "bench/prepare_benchmark.py",
                "--dev-count",
                "2",
                "--test-count",
                "2",
                "--dev-out",
                str(dev),
                "--test-out",
                str(test),
                "--labels-template",
                str(labels_template),
            ],
            cwd=repo,
        )
        assert dev.exists()
        assert test.exists()
        assert labels_template.exists()

        rows = list(csv.DictReader(labels_template.open("r", encoding="utf-8")))
        rows[0]["chunk_id"] = "chunk-1"
        rows[0]["relevance"] = "3"
        rows[1]["chunk_id"] = "chunk-1"
        rows[1]["relevance"] = "2"
        with labels.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        kappa = _run_py(
            ["bench/compute_kappa.py", "--labels-csv", str(labels)],
            cwd=repo,
        )
        parsed = json.loads(kappa.stdout)
        assert "cohen_kappa" in parsed

        _run_py(
            [
                "bench/freeze_labels.py",
                "--labels-csv",
                str(labels),
                "--queries-dev",
                str(dev),
                "--queries-test",
                str(test),
                "--out",
                str(freeze),
            ],
            cwd=repo,
        )
        assert freeze.exists()

        _run_py(
            [
                "bench/run_benchmark_suite.py",
                "--query-set",
                str(dev),
                "--configs",
                "A",
                "--out-dir",
                str(runs),
            ],
            cwd=repo,
        )
        run_files = list(runs.glob("run_*.jsonl"))
        assert run_files

        _run_py(
            [
                "bench/compute_metrics.py",
                "--run-file",
                str(run_files[0]),
                "--labels-csv",
                str(labels),
                "--out-csv",
                str(metrics),
            ],
            cwd=repo,
        )
        assert metrics.exists()

        _run_py(
            [
                "bench/run_ablations.py",
                "--query-set",
                str(dev),
                "--out-csv",
                str(ablations),
            ],
            cwd=repo,
        )
        assert ablations.exists()
