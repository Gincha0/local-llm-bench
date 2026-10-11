from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from local_llm_bench.config import load_config
from local_llm_bench.environment import Environment
from local_llm_bench.report import Run, Summary, describe_failure, load_runs, markdown, summarize
from local_llm_bench.results import Measurement

CONFIG = load_config(Path(__file__).parent.parent / "config" / "bench.yaml")
GIB = 1024**3


def measurement(**fields: Any) -> Measurement:
    values: dict[str, Any] = {
        "tier_id": "T1",
        "model_id": "qwen3-0.6b-q4",
        "regime_id": "short",
        "rep": 1,
        "warmup": False,
        "think": False,
        "started_at": datetime(2026, 10, 10, tzinfo=UTC),
    }
    return Measurement(**(values | fields))


def environment(tier_id: str = "T1", started_at: datetime | None = None) -> Environment:
    return Environment(
        tier_id=tier_id,
        started_at=started_at or datetime(2026, 10, 10, tzinfo=UTC),
        system="Linux 7.2.8",
        arch="x86_64",
        cpu="test CPU",
        cpu_count=8,
        memory_bytes=32 * GIB,
        gpu=None,
        ollama_version="0.34.4",
        python_version="3.13.15",
        git_commit="e83569f8b32f9bd519a3fbeb4a09788d3398fd76",
        git_dirty=False,
    )


def run_of(*measurements: Measurement) -> Run:
    return Run(Path("results/T1-test.jsonl"), environment(), list(measurements))


def row(run: Run, model_id: str = "qwen3-0.6b-q4", regime_id: str = "short") -> Summary:
    rows = summarize(CONFIG, run)
    return next(r for r in rows if r.model_id == model_id and r.regime_id == regime_id)


def test_summary_is_the_median_and_leaves_out_the_warmup() -> None:
    run = run_of(
        measurement(warmup=True, rep=0, ttft_ns=2_000_000_000),  # includes model load
        measurement(ttft_ns=10_000_000),
        measurement(ttft_ns=30_000_000),
        measurement(ttft_ns=20_000_000),
    )
    summary = row(run)
    assert summary.measured == 3
    assert summary.ttft_ms == pytest.approx(20.0)


def test_failed_load_has_a_reason_and_no_numbers() -> None:
    error = 'OllamaError: HTTP 500: {"error":"... ggml_aligned_malloc: insufficient memory ..."}'
    summary = row(run_of(measurement(warmup=True, rep=0, error=error)))
    assert summary.measured == 0
    assert summary.gen_tps is None
    assert summary.failure == "not enough memory to load"


def test_pair_that_never_ran_is_still_listed() -> None:
    summary = row(run_of(measurement()), regime_id="long_context")
    assert summary.measured == 0
    assert summary.failure == "not run"


def test_unknown_error_is_kept_but_shortened() -> None:
    reason = describe_failure("SomeNewError: " + "x" * 100)
    assert reason.startswith("SomeNewError: ")
    assert len(reason) == 60


def test_split_with_system_ram_shows_in_the_table() -> None:
    run = run_of(measurement(loaded_bytes=10 * GIB, loaded_vram_bytes=8 * GIB))
    summary = row(run)
    assert summary.gpu_share == pytest.approx(0.8)
    table = markdown(CONFIG, {"T1": run}, summarize(CONFIG, run))
    assert "| qwen3-0.6b-q4 | short | 1/10 | – | – | – | 10.00 | 80% |  |" in table


def test_only_the_newest_run_of_a_tier_counts(tmp_path: Path) -> None:
    for name, day in (("T1-older", 9), ("T1-newer", 10)):
        env = environment(started_at=datetime(2026, 10, day, tzinfo=UTC))
        (tmp_path / f"{name}.env.json").write_text(env.model_dump_json(), encoding="utf-8")
        (tmp_path / f"{name}.jsonl").write_text(measurement().model_dump_json() + "\n")
    runs = load_runs(tmp_path)
    assert list(runs) == ["T1"]
    assert runs["T1"].path.name == "T1-newer.jsonl"
