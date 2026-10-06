import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_llm_bench.config import BenchConfig, load_config
from local_llm_bench.measure import StreamResult
from local_llm_bench.results import Measurement
from local_llm_bench.runner import build_payload, run_tier, to_measurement

CONFIG = load_config(Path(__file__).parent.parent / "config" / "bench.yaml")


def payload_for(tier_id: str, regime_id: str) -> dict[str, object]:
    model = CONFIG.models[0]
    regime = next(r for r in CONFIG.regimes if r.id == regime_id)
    return build_payload(model, regime, CONFIG.run, CONFIG.tier(tier_id), prompt="Hi")


def test_cpu_tier_keeps_every_layer_off_the_gpu() -> None:
    assert payload_for("T2", "short")["options"] == {
        "temperature": CONFIG.run.temperature,
        "seed": CONFIG.run.seed,
        "num_ctx": 2048,
        "num_predict": CONFIG.run.max_tokens,
        "num_gpu": 0,
    }


def test_gpu_tier_leaves_offload_to_ollama() -> None:
    options = payload_for("T1", "short")["options"]
    assert isinstance(options, dict)
    assert "num_gpu" not in options


def test_only_structured_regime_asks_for_json() -> None:
    assert payload_for("T1", "structured")["format"] == "json"
    assert "format" not in payload_for("T1", "short")


def test_payload_streams_and_states_thinking_explicitly() -> None:
    payload = payload_for("T1", "short")
    assert payload["stream"] is True
    assert payload["think"] is CONFIG.run.think
    assert payload["model"] == CONFIG.models[0].ollama_tag


def test_ollama_counters_map_to_measurement_fields() -> None:
    result = StreamResult(
        ttft_ns=10,
        first_answer_ns=20,
        total_ns=30,
        answer="Hi",
        final={
            "load_duration": 1,
            "prompt_eval_count": 2,
            "prompt_eval_cached_count": 0,
            "prompt_eval_duration": 3,
            "eval_count": 100,
            "eval_duration": 2_000_000_000,
        },
    )
    base = Measurement(
        tier_id="T1",
        model_id="qwen3-4b-q4",
        regime_id="short",
        rep=0,
        warmup=False,
        think=False,
        started_at=datetime(2026, 10, 4, tzinfo=UTC),
    )
    m = to_measurement(base, result)
    assert m.model_id == "qwen3-4b-q4"
    assert (m.ttft_ns, m.first_answer_ns, m.total_ns) == (10, 20, 30)
    assert (m.load_ns, m.prompt_tokens, m.prompt_eval_ns) == (1, 2, 3)
    assert m.prompt_cached_tokens == 0
    assert m.answer == "Hi"
    assert m.gen_tps == pytest.approx(50.0)


def ndjson(*lines: dict[str, Any]) -> bytes:
    return b"".join(json.dumps(line).encode() + b"\n" for line in lines)


OK_STREAM = ndjson({"response": "Hi"}, {"done": True, "eval_count": 1, "eval_duration": 1})
OUT_OF_MEMORY = b'{"error": "model requires more system memory than is available"}'


class FakeOllama:
    """Answers every generate request; models listed in `too_big` fail like an OOM load."""

    def __init__(self, too_big: frozenset[str] = frozenset()) -> None:
        self.too_big = too_big
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if "prompt" not in body:  # unload request
            return httpx.Response(200, json={"done": True})
        if body["model"] in self.too_big:
            return httpx.Response(500, content=OUT_OF_MEMORY)
        return httpx.Response(200, content=OK_STREAM)

    def prompts(self) -> list[str]:
        return [b["prompt"] for b in self.bodies if "prompt" in b]


def small_run(models: int = 1) -> BenchConfig:
    """First `models` models, short regime only, 1 warmup + 2 measured requests."""
    run = CONFIG.run.model_copy(update={"warmup": 1, "repetitions": 2})
    return CONFIG.model_copy(
        update={"models": CONFIG.models[:models], "regimes": [CONFIG.regime("short")], "run": run}
    )


def run_against(fake: FakeOllama, config: BenchConfig) -> list[Measurement]:
    client = httpx.Client(transport=httpx.MockTransport(fake), base_url="http://ollama.test")
    return list(run_tier(config, CONFIG.tier("T1"), client))


def test_warmup_comes_first_and_is_flagged() -> None:
    results = run_against(FakeOllama(), small_run())
    assert [(m.rep, m.warmup) for m in results] == [(0, True), (1, False), (2, False)]


def test_every_request_starts_with_its_own_line() -> None:
    fake = FakeOllama()
    run_against(fake, small_run())
    first_lines = [p.splitlines()[0] for p in fake.prompts()]
    assert len(first_lines) == len(set(first_lines)) == 3


def test_model_is_unloaded_after_its_regimes() -> None:
    fake = FakeOllama()
    config = small_run()
    run_against(fake, config)
    assert fake.bodies[-1] == {"model": config.models[0].ollama_tag, "keep_alive": 0}


def test_model_that_does_not_fit_is_a_result_not_a_crash() -> None:
    config = small_run(models=2)
    fake = FakeOllama(too_big=frozenset({config.models[0].ollama_tag}))
    results = run_against(fake, config)
    failed, *rest = results
    assert failed.error is not None and "more system memory" in failed.error
    assert [m.model_id for m in rest] == [config.models[1].id] * 3  # next model still runs
