from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_llm_bench.config import load_config
from local_llm_bench.measure import StreamResult
from local_llm_bench.results import Measurement
from local_llm_bench.runner import build_payload, to_measurement

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
