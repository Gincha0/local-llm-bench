"""Command-line entry point. `plan` validates the config and shows what a run would do."""

from __future__ import annotations

import argparse
from pathlib import Path

from local_llm_bench.config import BenchConfig, load_config


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llm-bench")
    parser.add_argument("--config", type=Path, default=Path("config/bench.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    plan_cmd = sub.add_parser("plan", help="validate config and print the run matrix")
    plan_cmd.add_argument("--tier", required=True)

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "plan":
        print(plan(config, args.tier))
