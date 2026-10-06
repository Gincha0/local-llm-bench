"""Turn config entries into Ollama requests, and Ollama's answers into Measurements."""

from __future__ import annotations

from typing import Any

from local_llm_bench.config import Model, Regime, RunSettings, Tier
from local_llm_bench.measure import StreamResult
from local_llm_bench.results import Measurement


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
