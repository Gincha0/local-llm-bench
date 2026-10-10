"""Run the benchmark on one tier: config entries in, one Measurement per request out."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import httpx

from local_llm_bench.config import BenchConfig, Model, Regime, RunSettings, Tier
from local_llm_bench.measure import OllamaError, StreamResult, stream_generate
from local_llm_bench.results import Measurement

SERVER_RESTART_ATTEMPTS = 60  # one poll per second: how long systemd gets to restart Ollama


def build_payload(
    model: Model, regime: Regime, run: RunSettings, tier: Tier, prompt: str
) -> dict[str, Any]:
    """The /api/generate request body for one model and regime on one tier."""
    options: dict[str, Any] = {
        "temperature": run.temperature,
        "seed": run.seed,
        "num_ctx": regime.num_ctx,
        "num_predict": run.max_tokens,
    }
    if not tier.gpu_offload:
        options["num_gpu"] = 0  # layers offloaded to the GPU; 0 keeps the whole model on CPU
    payload: dict[str, Any] = {
        "model": model.ollama_tag,
        "prompt": prompt,
        "stream": True,
        "think": run.think,
        "options": options,
    }
    if regime.json_output:
        payload["format"] = "json"
    return payload


def to_measurement(base: Measurement, result: StreamResult) -> Measurement:
    """`base` says which request this was; add the client timings and Ollama's own counters."""
    final = result.final
    return base.model_copy(
        update={
            "ttft_ns": result.ttft_ns,
            "first_answer_ns": result.first_answer_ns,
            "total_ns": result.total_ns,
            "load_ns": final.get("load_duration"),
            "prompt_tokens": final.get("prompt_eval_count"),
            "prompt_cached_tokens": final.get("prompt_eval_cached_count"),
            "prompt_eval_ns": final.get("prompt_eval_duration"),
            "gen_tokens": final.get("eval_count"),
            "gen_ns": final.get("eval_duration"),
            "answer": result.answer,
        }
    )


def tagged(prompt: str, request_no: int) -> str:
    return f"Run {request_no:02d}.\n{prompt}"


def run_tier(config: BenchConfig, tier: Tier, client: httpx.Client) -> Iterator[Measurement]:
    run = config.run
    for model in config.models:
        for regime in config.regimes:
            prompt = regime.prompt_file.read_text(encoding="utf-8")
            for request_no in range(run.warmup + run.repetitions):
                base = Measurement(
                    tier_id=tier.id,
                    model_id=model.id,
                    regime_id=regime.id,
                    rep=request_no,
                    warmup=request_no < run.warmup,
                    think=run.think,
                    started_at=datetime.now(UTC),
                )
                payload = build_payload(model, regime, run, tier, tagged(prompt, request_no))
                try:
                    result = stream_generate(client, payload)
                except (httpx.HTTPError, OllamaError) as error:
                    yield base.model_copy(update={"error": f"{type(error).__name__}: {error}"})
                    if isinstance(error, httpx.TransportError):
                        wait_for_server(client)  # Ollama itself died, e.g. killed out of memory
                    break
                yield add_memory(to_measurement(base, result), client, model)
        unload(client, model)


def add_memory(m: Measurement, client: httpx.Client, model: Model) -> Measurement:
    try:
        response = client.get("/api/ps")
        response.raise_for_status()
        loaded = response.json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return m
    for entry in loaded:
        if entry.get("model") == model.ollama_tag:
            return m.model_copy(
                update={
                    "loaded_bytes": entry.get("size"),
                    "loaded_vram_bytes": entry.get("size_vram"),
                    "loaded_context": entry.get("context_length"),
                }
            )
    return m


def unload(client: httpx.Client, model: Model) -> None:
    with contextlib.suppress(httpx.HTTPError):
        client.post("/api/generate", json={"model": model.ollama_tag, "keep_alive": 0})


def wait_for_server(client: httpx.Client) -> None:
    for _ in range(SERVER_RESTART_ATTEMPTS):
        try:
            client.get("/api/version").raise_for_status()
        except httpx.HTTPError:
            time.sleep(1)
        else:
            return
