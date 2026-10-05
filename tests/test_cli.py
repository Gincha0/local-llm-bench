import json
from pathlib import Path

import httpx

from local_llm_bench.cli import probe
from local_llm_bench.config import load_config

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
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)),
        base_url="http://ollama.test",
    )
    out = probe(CONFIG.run, CONFIG.tier("T1"), CONFIG.models[0], CONFIG.regime("short"), client)
    assert "model load    5.0 ms" in out
    assert "20 tokens (0 cached), 2000.0 tok/s" in out
    assert "100 tokens, 50.0 tok/s" in out
    assert out.endswith("A switch forwards frames.")
