import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from local_llm_bench.cli import probe, progress
from local_llm_bench.config import load_config
from local_llm_bench.results import Measurement

CONFIG = load_config(Path(__file__).parent.parent / "config" / "bench.yaml")


def test_probe_prints_timings_rates_and_answer() -> None:
    lines = [
        {"response": "A switch "},
        {"response": "forwards frames."},
        {
            "done": True,
            "response": "",
            "load_duration": 5_000_000,
            "prompt_eval_count": 20,
            "prompt_eval_cached_count": 0,
            "prompt_eval_duration": 10_000_000,
            "eval_count": 100,
            "eval_duration": 2_000_000_000,
        },
    ]
    body = b"".join(json.dumps(line).encode() + b"\n" for line in lines)
    loaded = {
        "model": CONFIG.models[0].ollama_tag,
        "size": 4_000_000_000,
        "size_vram": 3_000_000_000,
    }

    def ollama(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [loaded]})
        return httpx.Response(200, content=body)

    client = httpx.Client(transport=httpx.MockTransport(ollama), base_url="http://ollama.test")
    out = probe(CONFIG.run, CONFIG.tier("T1"), CONFIG.models[0], CONFIG.regime("short"), client)
    assert "model load    5.0 ms" in out
    assert "20 tokens (0 cached), 2000.0 tok/s" in out
    assert "100 tokens, 50.0 tok/s" in out
    assert "memory        3.7 GiB, 75% GPU" in out
    assert out.endswith("A switch forwards frames.")


def test_progress_line_shows_errors_instead_of_numbers() -> None:
    failed = Measurement(
        tier_id="T4",
        model_id="qwen3-8b-q8",
        regime_id="short",
        rep=0,
        warmup=True,
        think=False,
        started_at=datetime(2026, 10, 5, tzinfo=UTC),
        error="OllamaError: model requires more system memory",
    )
    line = progress(failed)
    assert "warmup" in line and "ERROR OllamaError" in line and "ttft" not in line
