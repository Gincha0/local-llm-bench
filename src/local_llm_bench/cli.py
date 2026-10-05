"""Command-line entry point.

`plan` validates the config and shows what a run would do. `probe` sends one timed request
to the tier's Ollama and prints what was measured: a smoke test before a full run.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import httpx

from local_llm_bench.config import BenchConfig, Model, Regime, RunSettings, Tier, load_config
from local_llm_bench.measure import stream_generate
from local_llm_bench.results import Measurement
from local_llm_bench.runner import build_payload, to_measurement

NS_PER_MS = 1_000_000


def plan(config: BenchConfig, tier_id: str) -> str:
    tier = config.tier(tier_id)
    run = config.run
    per_pair = run.warmup + run.repetitions
    total = len(config.models) * len(config.regimes) * per_pair
    lines = [
        f"tier {tier.id} ({tier.name}): {tier.os}/{tier.arch}, gpu_offload={tier.gpu_offload}",
        f"{len(config.models)} models x {len(config.regimes)} regimes x "
        f"({run.warmup} warmup + {run.repetitions} measured) = {total} requests",
    ]
    lines += [f"  model  {m.id:<20} {m.ollama_tag}" for m in config.models]
    lines += [f"  regime {r.id:<20} num_ctx={r.num_ctx}" for r in config.regimes]
    return "\n".join(lines)


def probe(run: RunSettings, tier: Tier, model: Model, regime: Regime, client: httpx.Client) -> str:
    base = Measurement(
        tier_id=tier.id,
        model_id=model.id,
        regime_id=regime.id,
        rep=0,
        warmup=False,
        think=run.think,
        started_at=datetime.now(UTC),
    )
    prompt = regime.prompt_file.read_text(encoding="utf-8")
    result = stream_generate(client, build_payload(model, regime, run, tier, prompt))
    m = to_measurement(base, result)
    cached = result.final.get("prompt_eval_cached_count", 0)
    return "\n".join(
        [
            f"{tier.id} / {model.id} / {regime.id}",
            f"ttft          {_ms(m.ttft_ns)}",
            f"first answer  {_ms(m.first_answer_ns)}",
            f"total         {_ms(m.total_ns)}",
            f"model load    {_ms(m.load_ns)}",
            f"prompt eval   {m.prompt_tokens} tokens ({cached} cached), {_rate(m.prompt_tps)}",
            f"generation    {m.gen_tokens} tokens, {_rate(m.gen_tps)}",
            "",
            result.answer.strip(),
        ]
    )


def _ms(ns: int | None) -> str:
    return "n/a" if ns is None else f"{ns / NS_PER_MS:.1f} ms"


def _rate(tps: float | None) -> str:
    return "n/a" if tps is None else f"{tps:.1f} tok/s"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llm-bench")
    parser.add_argument("--config", type=Path, default=Path("config/bench.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    plan_cmd = sub.add_parser("plan", help="validate config and print the run matrix")
    plan_cmd.add_argument("--tier", required=True)
    probe_cmd = sub.add_parser("probe", help="send one timed request and print the numbers")
    probe_cmd.add_argument("--tier", required=True)
    probe_cmd.add_argument("--model", required=True)
    probe_cmd.add_argument("--regime", default="short")

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "plan":
        print(plan(config, args.tier))
    elif args.command == "probe":
        tier = config.tier(args.tier)
        model = config.model(args.model)
        regime = config.regime(args.regime)
        with httpx.Client(base_url=tier.endpoint, timeout=config.run.timeout_s) as client:
            print(probe(config.run, tier, model, regime, client))
