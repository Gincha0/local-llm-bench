import itertools
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from local_llm_bench.measure import OllamaError, stream_generate

PAYLOAD: dict[str, Any] = {"model": "qwen3:4b-q4_K_M", "prompt": "Hi", "stream": True}

FINAL_LINE: dict[str, Any] = {
    "done": True,
    "done_reason": "stop",
    "response": "",
    "load_duration": 100,
    "prompt_eval_count": 12,
    "prompt_eval_duration": 200,
    "eval_count": 3,
    "eval_duration": 300,
}


def ndjson(*lines: dict[str, Any]) -> bytes:
    """Ollama's stream format: one JSON object per line."""
    return b"".join(json.dumps(line).encode() + b"\n" for line in lines)


def fake_ollama(body: bytes, status: int = 200) -> httpx.Client:
    """A client whose every request gets this canned response. No server involved."""

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body)

    return httpx.Client(transport=httpx.MockTransport(respond), base_url="http://ollama.test")


def ticking_clock(step_ns: int = 1_000) -> Callable[[], int]:
    """Fake clock: each reading is step_ns later than the one before, starting at 0."""
    ticks = itertools.count(start=0, step=step_ns)
    return lambda: next(ticks)


def test_thinking_counts_for_ttft_but_not_for_first_answer() -> None:
    body = ndjson(
        {"thinking": "Hmm"},  # line 1 at 1000 ns
        {"thinking": " ok"},
        {"response": "Hi"},  # line 3 at 3000 ns
        {"response": "!"},
        FINAL_LINE,  # line 5 at 5000 ns
    )
    result = stream_generate(fake_ollama(body), PAYLOAD, clock=ticking_clock())
    assert result.ttft_ns == 1_000
    assert result.first_answer_ns == 3_000
    assert result.total_ns == 5_000


def test_without_thinking_ttft_equals_first_answer() -> None:
    body = ndjson({"response": "Hi"}, {"response": "!"}, FINAL_LINE)
    result = stream_generate(fake_ollama(body), PAYLOAD, clock=ticking_clock())
    assert result.ttft_ns == result.first_answer_ns == 1_000


def test_empty_response_is_not_a_first_token() -> None:
    body = ndjson({"response": ""}, {"response": "Hi"}, FINAL_LINE)
    result = stream_generate(fake_ollama(body), PAYLOAD, clock=ticking_clock())
    assert result.ttft_ns == 2_000


def test_answer_is_joined_and_final_line_kept() -> None:
    body = ndjson({"thinking": "Hmm"}, {"response": "Hi"}, {"response": "!"}, FINAL_LINE)
    result = stream_generate(fake_ollama(body), PAYLOAD, clock=ticking_clock())
    assert result.answer == "Hi!"
    assert result.final["eval_count"] == 3


def test_blank_lines_are_skipped() -> None:
    body = b"\n" + ndjson({"response": "Hi"}) + b"\n" + ndjson(FINAL_LINE)
    result = stream_generate(fake_ollama(body), PAYLOAD, clock=ticking_clock())
    assert result.answer == "Hi"


def test_http_error_carries_ollamas_message() -> None:
    # What a model that does not fit looks like: this is how the cliff gets recorded.
    body = json.dumps({"error": "model requires more system memory than is available"})
    with pytest.raises(OllamaError, match="more system memory"):
        stream_generate(fake_ollama(body.encode(), status=500), PAYLOAD)


def test_error_line_mid_stream_raises() -> None:
    body = ndjson({"response": "Hi"}, {"error": "model runner has unexpectedly stopped"})
    with pytest.raises(OllamaError, match="unexpectedly stopped"):
        stream_generate(fake_ollama(body), PAYLOAD)


def test_stream_without_final_line_raises() -> None:
    body = ndjson({"response": "Hi"})
    with pytest.raises(OllamaError, match="done"):
        stream_generate(fake_ollama(body), PAYLOAD)


def test_payload_is_posted_to_generate() -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=ndjson(FINAL_LINE))

    client = httpx.Client(transport=httpx.MockTransport(respond), base_url="http://ollama.test")
    stream_generate(client, PAYLOAD)
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/api/generate"
    assert json.loads(seen[0].content) == PAYLOAD
