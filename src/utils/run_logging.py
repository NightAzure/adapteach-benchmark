import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunLogger:
    log_dir: Path
    app_config: dict[str, Any]
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stage_outputs: dict[str, Any] = field(default_factory=dict)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    stage_events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "runs.jsonl"

    def stage_start(self, stage: str) -> float:
        return time.perf_counter()

    def stage_end(self, stage: str, started_at: float, output: Any) -> None:
        latency_ms = round((time.perf_counter() - started_at) * 1000, 3)
        self.stage_outputs[stage] = output
        self.stage_timings_ms[stage] = latency_ms
        self.stage_events.append(
            {
                "trace_id": self.trace_id,
                "timestamp": utc_now_iso(),
                "stage": stage,
                "latency_ms": latency_ms,
                "output": output,
            }
        )

    def write_final_record(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        token_counts: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        llm = self.app_config["llm"]
        retrieval = self.app_config["retrieval"]
        record = {
            "trace_id": self.trace_id,
            "timestamp": utc_now_iso(),
            "request": request,
            "response": response,
            "model_provider": llm["provider"],
            "model_name": llm["model"],
            "model_version_tag": llm["version_tag"],
            "prompt_template_version": llm["prompt_template_version"],
            "retrieval_config": {
                "chunking": retrieval["chunking"],
                "k": retrieval["k"],
                "fusion": retrieval["fusion"],
                "rerank_weights": retrieval["rerank_weights"],
            },
            "corpus_snapshot": retrieval["corpus_snapshot"],
            "index_hash": retrieval["index_hash"],
            "stage_timings_ms": self.stage_timings_ms,
            "stage_outputs": self.stage_outputs,
            "stage_events": self.stage_events,
            "token_counts": token_counts or {"prompt": 0, "response": 0},
        }
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return record
