# local-llm-bench

Which model class fits which device tier, and where is the cliff?

A benchmark harness that runs the same local LLMs and the same prompt suite on four
hardware tiers, from a desktop GPU down to a 4 GB ARM VPS, and measures throughput,
latency and memory. Failed runs (out of memory, timeouts) are recorded as results, not
crashes: the point where a model stops fitting is one of the findings.

> **Status:** in progress. Measurement loop and runner done; first runs next.
> Results below are placeholders until real runs land.

## Device tiers

| Tier | Machine | OS / arch | Proves |
|---|---|---|---|
| T1 | Ryzen 7 9800X3D + RTX 4070 12 GB, 32 GB | Linux / x86_64 | The ceiling |
| T2 | Same machine, CPU only | Linux / x86_64 | The GPU offload benefit, isolated |
| T3 | Laptop, Ryzen 7 7435HS + RTX 4060 8 GB, 24 GB | Windows / x86_64 | Smaller VRAM and thermal limits |
| T4 | Hetzner CAX11, 2 vCPU, 4 GB | Linux / arm64 | Non-x86, no GPU; where models stop fitting |

## What is measured

Per model, per tier, per prompt regime:

- **Time to first token**: client-side wall clock, streamed
- **Prompt-eval and generation throughput**: tokens/s, reported separately
- **Model load time**
- **Memory**: what Ollama holds for the loaded model and how much of it is in VRAM
  (`/api/ps`, read after every request). Below 100 % GPU means the model was split
  with system RAM: throughput drops and nothing reports an error
- **Output quality**: small fixed rubric, stated below

Raw durations and token counts are stored; rates are derived at report time.
Every run also writes `<tier>-<time>.env.json` next to its results: OS, CPU, RAM,
GPU and driver, Ollama version, and the harness commit, flagged if there were
uncommitted changes.

## Prompt regimes

| Regime | What it stresses |
|---|---|
| `short` | Interactive latency: TTFT dominates |
| `long_context` | Prompt-eval throughput and KV-cache memory (~3k tokens in) |
| `structured` | JSON output: format adherence under constraint |

## Results

_Pending._

## Methodology

_Pending: warmup policy, repetitions, decoding settings, thinking on/off, limitations
(cloud tiers are shared and virtualised, so run-to-run variance is higher there)._

## Running it

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/), with every
`ollama_tag` in `config/bench.yaml` pulled on the machine being measured.

```sh
uv sync
uv run llm-bench plan --tier T1                        # validate config, print the run matrix
uv run llm-bench probe --tier T1 --model qwen3-4b-q4   # one timed request: smoke test
uv run llm-bench run --tier T1                         # everything -> results/T1-<time>.jsonl + .env.json
uv run pytest && uv run ruff check && uv run mypy src tests
```

Adding a model, tier or regime is an entry in `config/bench.yaml`, not a code change.
