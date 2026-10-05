"""Time one streamed generate request against Ollama.

Ollama streams one JSON object per line. Each line is timed on the client with a steady
clock as it arrives:

- ttft_ns: first token of any kind, thinking included. Compares engines, not how long a
  model chooses to think.
- first_answer_ns: first token of the answer itself. How long a user waits for it.
- total_ns: until the final line (done = true), which carries Ollama's own counters.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx


class OllamaError(RuntimeError):
    """Ollama refused or failed the request: out of memory, unknown model, crashed runner."""


@dataclass(frozen=True)
class StreamResult:
    ttft_ns: int | None
    first_answer_ns: int | None
    total_ns: int
    answer: str
    final: dict[str, Any]


def stream_generate(
    client: httpx.Client,
    payload: dict[str, Any],
    clock: Callable[[], int] = time.perf_counter_ns,
) -> StreamResult:
    """Send `payload` to /api/generate as a stream and time it.

    Durations count from just before the request is sent, one clock reading per line.
    Raises OllamaError if Ollama reports an error, as an HTTP status or as an
    {"error": ...} line, or if the stream ends without a final line.
    """
    ttft_ns: int | None = None
    first_answer_ns: int | None = None
    answer_parts: list[str] = []
    start = clock()
    with client.stream("POST", "/api/generate", json=payload) as response:
        if response.status_code != httpx.codes.OK:
            response.read()
            raise OllamaError(f"HTTP {response.status_code}: {response.text}")
        for line in response.iter_lines():
            if not line:
                continue
            elapsed = clock() - start
            chunk = json.loads(line)
            if "error" in chunk:
                raise OllamaError(chunk["error"])
            if ttft_ns is None and (chunk.get("thinking") or chunk.get("response")):
                ttft_ns = elapsed
            if first_answer_ns is None and chunk.get("response"):
                first_answer_ns = elapsed
            answer_parts.append(chunk.get("response", ""))
            if chunk.get("done"):
                return StreamResult(ttft_ns, first_answer_ns, elapsed, "".join(answer_parts), chunk)
    raise OllamaError("stream ended without a final 'done' line")
