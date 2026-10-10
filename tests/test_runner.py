import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_llm_bench.config import BenchConfig, load_config
from local_llm_bench.measure import StreamResult
from local_llm_bench.results import Measurement
from local_llm_bench.runner import add_memory, build_payload, run_tier, to_measurement

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
LOADED = {"size": 4_000_000_000, "size_vram": 3_000_000_000, "context_length": 2048}


class FakeOllama:
    """Answers every generate request; models listed in `too_big` fail like an OOM load.

    /api/ps lists every model that answered and was not unloaded since, sized as LOADED.
    """

    def __init__(self, too_big: frozenset[str] = frozenset(), ps_status: int = 200) -> None:
        self.too_big = too_big
        self.ps_status = ps_status
        self.loaded: set[str] = set()
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            models = [{"model": tag, **LOADED} for tag in sorted(self.loaded)]
            return httpx.Response(self.ps_status, json={"models": models})
        body = json.loads(request.content)
        self.bodies.append(body)
        if "prompt" not in body:  # unload request
            self.loaded.discard(body["model"])
            return httpx.Response(200, json={"done": True})
        if body["model"] in self.too_big:
            return httpx.Response(500, content=OUT_OF_MEMORY)
        self.loaded.add(body["model"])
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


def test_memory_is_read_after_every_request() -> None:
    results = run_against(FakeOllama(), small_run())
    assert [(m.loaded_bytes, m.loaded_vram_bytes, m.loaded_context) for m in results] == [
        (4_000_000_000, 3_000_000_000, 2048)
    ] * 3


def test_failed_memory_read_does_not_stop_the_run() -> None:
    results = run_against(FakeOllama(ps_status=500), small_run())
    assert len(results) == 3
    assert all(m.error is None and m.loaded_bytes is None for m in results)


def test_only_the_measured_models_memory_counts() -> None:
    fake = FakeOllama()
    fake.loaded.add("some-other-model:latest")
    client = httpx.Client(transport=httpx.MockTransport(fake), base_url="http://ollama.test")
    base = Measurement(
        tier_id="T1",
        model_id="qwen3-4b-q4",
        regime_id="short",
        rep=1,
        warmup=False,
        think=False,
        started_at=datetime(2026, 10, 6, tzinfo=UTC),
    )
    assert add_memory(base, client, CONFIG.models[0]).loaded_bytes is None


class CrashingOllama(FakeOllama):
    """FakeOllama that dies on the first request for `crash_model`, as when the kernel kills
    it for running out of memory, then refuses connections until systemd restarts it after
    `refusals` refused requests (or never, if `refusals` is None)."""

    def __init__(self, crash_model: str, refusals: int | None) -> None:
        super().__init__()
        self.crash_model = crash_model
        self.refusals = refusals
        self.crashed = False
        self.down = False
        self.refused = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            if self.refusals is not None and self.refused >= self.refusals:
                self.down = False
            else:
                self.refused += 1
                raise httpx.ConnectError("[Errno 111] Connection refused", request=request)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.34.4"})
        body = json.loads(request.content) if request.content else {}
        if not self.crashed and "prompt" in body and body["model"] == self.crash_model:
            self.crashed = self.down = True
            raise httpx.RemoteProtocolError("Server disconnected", request=request)
        return super().__call__(request)


def two_regimes(models: int = 1) -> BenchConfig:
    config = small_run(models)
    return config.model_copy(
        update={"regimes": [CONFIG.regime("short"), CONFIG.regime("structured")]}
    )


def test_run_waits_for_a_crashed_server_and_carries_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_llm_bench.runner.time.sleep", lambda _: None)
    config = two_regimes()
    results = run_against(CrashingOllama(config.models[0].ollama_tag, refusals=3), config)
    crashed, *after = results
    assert crashed.error is not None and "RemoteProtocolError" in crashed.error
    assert [(m.regime_id, m.error) for m in after] == [("structured", None)] * 3


def test_server_that_stays_down_ends_in_errors_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_llm_bench.runner.time.sleep", lambda _: None)
    config = two_regimes(models=2)
    results = run_against(CrashingOllama(config.models[0].ollama_tag, refusals=None), config)
    assert len(results) == 4  # one failed request per model and regime, then the run ends
    assert all(m.error is not None for m in results)
