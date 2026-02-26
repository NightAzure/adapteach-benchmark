import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.build_chunks import build_chunks
from src.graphs.build_graphs import main as _graphs_main  # smoke import
from src.graphs.ckg import build_ckg
from src.graphs.cpg import build_cpg
from src.indexing.build_indexes import build_indexes
from src.indexing.corpus_pipeline import build_corpus, CorpusPaths
from src.indexing.snapshot_tool import create_snapshot
from src.pipelines.runner import run_pipeline
from src.retrieval.engine import _rrf_fuse
from src.utils.config import load_app_config, load_pipeline_config
def _prepare_assets() -> str:
    build_corpus(CorpusPaths())
    build_chunks(chunker="ast")
    snap = create_snapshot()
    build_indexes(snapshot_id=snap["snapshot_id"])
    build_ckg()
    build_cpg()
    return snap["snapshot_id"]


def _app_config_with_temp_logs() -> dict:
    cfg = load_app_config()
    cfg["runtime"]["log_dir"] = tempfile.mkdtemp(prefix="phase5-logs-")
    return cfg


def test_config_a_baseline_has_safety_prompt_logging() -> None:
    _prepare_assets()
    app = _app_config_with_temp_logs()
    pipeline = load_pipeline_config("A")
    _, record = run_pipeline({"query": "Explain loops safely"}, app, pipeline)
    gen_debug = record["stage_outputs"]["generation"]["debug"]
    assert "prompt_version" in gen_debug


def test_config_b_dense_topk_saved_in_logs() -> None:
    _prepare_assets()
    app = _app_config_with_temp_logs()
    pipeline = load_pipeline_config("B")
    _, record = run_pipeline({"query": "for loop range"}, app, pipeline)
    retrieval_debug = record["stage_outputs"]["retrieval"]["debug"]
    assert retrieval_debug["retrieval_mode"] == "dense"
    assert retrieval_debug["dense_topk"]


def test_config_c_fusion_debug_available() -> None:
    _prepare_assets()
    app = _app_config_with_temp_logs()
    pipeline = load_pipeline_config("C")
    _, record = run_pipeline({"query": "function return value"}, app, pipeline)
    retrieval_debug = record["stage_outputs"]["retrieval"]["debug"]
    assert retrieval_debug["retrieval_mode"] == "hybrid"
    assert "fusion_debug" in retrieval_debug
    assert retrieval_debug["fusion_debug"]["rows"]


def test_rrf_fuse_preserves_topic_overlap_signal() -> None:
    dense_rows = [
        {"chunk_id": "c1", "rank": 1, "score": 0.9, "topic_overlap": 1.0},
        {"chunk_id": "c2", "rank": 2, "score": 0.8, "topic_overlap": 0.0},
    ]
    bm25_rows = [
        {"chunk_id": "c2", "rank": 1, "score": 4.2, "topic_overlap": 1.0},
        {"chunk_id": "c1", "rank": 2, "score": 3.0, "topic_overlap": 0.0},
    ]
    fused, debug = _rrf_fuse(dense_rows, bm25_rows, k=2)
    by_chunk = {r["chunk_id"]: r for r in fused}
    assert by_chunk["c1"]["topic_overlap"] == 1.0
    assert by_chunk["c2"]["topic_overlap"] == 1.0
    debug_rows = {r["chunk_id"]: r for r in debug["rows"]}
    assert debug_rows["c1"]["topic_overlap"] == 1.0
    assert debug_rows["c2"]["topic_overlap"] == 1.0


def test_config_d_parser_fallback_logged() -> None:
    _prepare_assets()
    app = _app_config_with_temp_logs()
    pipeline = load_pipeline_config("D")
    _, record = run_pipeline({"query": "if else statement"}, app, pipeline)
    retrieval_debug = record["stage_outputs"]["retrieval"]["debug"]
    assert "parser_fallback" in retrieval_debug
    assert "chunker_counts" in retrieval_debug["parser_fallback"]


def test_config_e_graph_evidence_and_rerank_debug() -> None:
    _prepare_assets()
    app = _app_config_with_temp_logs()
    pipeline = load_pipeline_config("E")
    _, record = run_pipeline({"query": "explain loops and functions relationship"}, app, pipeline)
    graph_debug = record["stage_outputs"]["graph_expansion"]["debug"]
    assert graph_debug["graph_used"] is True
    assert "evidence_paths" in graph_debug
    assert "rerank" in graph_debug
