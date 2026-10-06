from datetime import UTC, datetime

import pytest

from local_llm_bench.results import Measurement


def test_gen_tps_converts_nanoseconds_to_seconds() -> None:
    # The model generated 100 tokens in 2 second.
    m = Measurement(
        tier_id="T1",
        model_id="qwen3-4b-q4",
        regime_id="short",
        rep=0,
        warmup=False,
        think=False,
        started_at=datetime(2026, 9, 29, tzinfo=UTC),
        gen_tokens=100,
        gen_ns=2_000_000_000,
    )
    assert m.gen_tps == pytest.approx(50.0)


def test_prompt_tps_converts_nanoseconds_to_seconds() -> None:
    # the model read 3000 tokens in 0.5 second.
    m = Measurement(
        tier_id="T1",
        model_id="qwen3-4b-q4",
        regime_id="short",
        rep=0,
        warmup=False,
        think=False,
        started_at=datetime(2026, 9, 29, tzinfo=UTC),
        prompt_tokens=3000,
        prompt_eval_ns=500_000_000,
    )
    assert m.prompt_tps == pytest.approx(6000.0)


def test_prompt_tps_counts_only_tokens_actually_evaluated() -> None:
    # 3000 tokens in, 2000 of them reused from the cache: 1000 evaluated in 0.5 s.
    m = Measurement(
        tier_id="T1",
        model_id="qwen3-4b-q4",
        regime_id="long_context",
        rep=1,
        warmup=False,
        think=False,
        started_at=datetime(2026, 10, 5, tzinfo=UTC),
        prompt_tokens=3000,
        prompt_cached_tokens=2000,
        prompt_eval_ns=500_000_000,
    )
    assert m.prompt_tps == pytest.approx(2000.0)
