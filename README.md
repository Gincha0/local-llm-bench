# local-llm-bench

Which model class fits which device tier, and where is the cliff?

A benchmark harness that runs the same local LLMs and the same prompts on several hardware
tiers, from a desktop GPU down to a 4 GB ARM cloud VM, and measures throughput, latency,
memory and output correctness. Failed runs (out of memory, a crashed server) are recorded as
results, not crashes: the point where a model stops fitting is one of the findings.

> **Status:** three of four tiers measured (T1, T2, T4). T3, a Windows laptop, is next.
> Hand-graded quality for the free-text prompts follows after that.

## Device tiers

| Tier | Machine | OS / arch | Memory bandwidth (spec) | Proves |
|---|---|---|---|---|
| T1 | Ryzen 7 9800X3D + RTX 4070 12 GB, 32 GB RAM | Linux / x86_64 | 504 GB/s (GDDR6X) | The ceiling |
| T2 | Same machine, GPU not used | Linux / x86_64 | 96 GB/s (DDR5-6000, 2 channels) | What the GPU adds, isolated |
| T3 | Laptop, Ryzen 7 7435HS + RTX 4060 8 GB, 24 GB RAM | Windows / x86_64 | 256 GB/s (GDDR6) | Less VRAM: where models start to split |
| T4 | AWS EC2 t4g.medium (Graviton2), 2 vCPU, 4 GB | Linux / arm64 | none published (shared host) | Non-x86, no GPU; where models stop fitting |

Models: one family at four sizes, Qwen3 0.6B / 1.7B / 4B / 8B, each as Q4_K_M and Q8_0: eight
GGUF files from 0.5 to 8.2 GiB. Size and quantisation are the only variables.

## Results

Full table, every tier × model × prompt: [report/results.md](report/results.md), generated
from the raw results by `llm-bench report`.

![Generation speed against model size](report/gen-speed.png)

**1. Generation speed is set by memory bandwidth.** Each generated token reads every weight
once, so a tier cannot generate faster than its memory bandwidth divided by the model's size:
the dashed lines. Measured as tokens/s × file size:

- T1 reaches 380–480 GB/s for models from 1.7B up; the 8B Q8 model runs at 480 GB/s, 95 % of
  the 4070's 504 GB/s. Small models fall further below the line (0.6B Q4: 263 GB/s) because
  fixed per-token work dominates, so halving the bytes with Q4 speeds the 8B model up 1.6×
  but the 0.6B model only 1.24×.
- T2, the same machine on its CPU, reaches 62–82 GB/s of its RAM's 96 GB/s.
- **Equal bytes, equal speed:** 8B Q4 (4.9 GiB) generates 86.9 tok/s, 4B Q8 (4.1 GiB)
  96.4 tok/s. At the same memory budget, the choice between them is a quality question.

**2. Reading the prompt is compute-bound.** The GPU generates 3.3–7.0× faster than the same
machine's CPU but processes the prompt 28–48× faster. For the ~3,000-token prompt, time to
first token is 0.09–0.64 s on T1 and 2.3–29 s on T2.

![Memory each model needs against what each tier has](report/memory-fit.png)

**3. Where models stop fitting.** On the 4 GB ARM VM, 0.6B and 1.7B fit at every context
size. 4B Q4 fits with a 2k context (3.1 GiB) and was killed for lack of memory at 8k. 4B Q8 and
both 8B models do not load. Going from a 2k to an 8k context adds 0.7–0.95 GiB of KV cache.
What fits is not necessarily usable: the 3,000-token prompt takes 50–56 s before the first
token on 0.6B and 84–103 s on 1.7B.

**4. On two ARM cores, Q8 beats Q4.** 0.6B generates 42.3 tok/s at Q8 and 36.8 at Q4; 1.7B
16.2 and 15.2. The effective bandwidth is only 19–36 GB/s, so here the CPU, not memory, is the
limit, and unpacking 4-bit K-quants most likely costs more compute than the bytes it saves.
The bandwidth rule holds only while memory is the bottleneck.

**5. At small sizes, quantisation decides correctness.** The `structured` prompt asks for four
fields from a switch log as JSON; every repetition is checked against the one correct answer.

| Model | T1 | T2 | T4 |
|---|---|---|---|
| qwen3-0.6b-q4 | 4/10 | 0/10 | 1/10 |
| qwen3-0.6b-q8 | 0/10 | 0/10 | 0/10 |
| qwen3-1.7b-q4 | 0/10 | 0/10 | 0/10 |
| qwen3-1.7b-q8 | 10/10 | 10/10 | 10/10 |
| qwen3-4b-q4 | 10/10 | 10/10 | 10/10 |
| qwen3-4b-q8 | 10/10 | 10/10 | did not load |
| qwen3-8b-q4 | 10/10 | 10/10 | did not load |
| qwen3-8b-q8 | 10/10 | 10/10 | did not load |

1.7B is correct every time at Q8 and never at Q4. The most common wrong answer picks the
link-up event on the next line of the log.

**6. The prompt cache that killed the 4 GB VM.** Ollama 0.34.4 runs llama.cpp's
`llama-server`, whose RAM prompt cache is on by default. Before each new request it saves the
previous prompt's KV state (331.5 MiB for the long prompt). This benchmark makes every prompt
unique (see Methodology), so the cache can never be reused and only grows. On T4 the kernel
killed `llama-server` for lack of memory at about 3.5 GB resident, while Ollama's own memory
report still said 1.7 GiB. The save also runs inside the time to first token: with the cache on,
T1's TTFT was 12–57 % higher. The cache is disabled on every tier
(`LLAMA_ARG_CACHE_RAM=0`), and the harness now waits for the server to restart after a crash
instead of ending the run.

## What is measured

Per model, per tier, per prompt regime:

- **Time to first token:** client-side clock from sending the request to the first streamed
  token
- **Prompt and generation throughput:** tokens/s, reported separately
- **Model load time**
- **Memory:** what Ollama holds for the loaded model and how much of it is in VRAM
  (`/api/ps`, read after every request). Below 100 % GPU on a GPU tier means the model was split
  with system RAM: throughput drops and nothing reports an error
- **Correctness:** checked by code where there is one right answer; hand-graded against a
  rubric otherwise (rubrics in the config and in `report/results.md`)

Raw durations and token counts are stored; rates are derived at report time. Every run also
writes `<tier>-<time>.env.json` next to its results: OS, CPU, RAM, GPU and driver, Ollama
version, and the harness commit, flagged if there were uncommitted changes.

## Prompt regimes

| Regime | What it stresses | Quality check |
|---|---|---|
| `short` | Interactive latency: TTFT dominates | Hand-graded, 2-point rubric (pending) |
| `long_context` | Prompt processing and KV-cache memory (~3k tokens in, 8k context) | Hand-graded, 5-point rubric (pending) |
| `structured` | JSON output under a fixed schema | Automatic, every repetition |

## Methodology

- **Backend:** Ollama 0.34.4 on every tier. Models pulled by tag; sizes are the GGUF file sizes
  from the Ollama registry.
- **Repetitions:** per model and regime, 1 warmup request (absorbs model load, discarded) and
  10 measured. Tables show medians.
- **Decoding:** temperature 0, fixed seed, at most 256 generated tokens, thinking off (Qwen3
  thinks by default; the setting is recorded with every result).
- **Prompt reuse:** Ollama reuses its KV cache for a repeated prompt prefix, which makes
  repeated requests look fast. Every request therefore starts with a unique line ("Run NN."),
  and the number of reused tokens is recorded with every result as proof. Only the 6-token
  chat-template prefix is reused. Side effect: answers differ between repetitions.
- **TTFT** is the first token of any kind, measured on the client with a monotonic clock.
- **Memory** is Ollama's own reservation (weights, KV cache, buffers), not measured by the OS,
  and it does not include `llama-server`'s prompt cache.
- **Units:** sizes in GiB (1024³ bytes); `ollama ps` shows decimal GB. Bandwidth in decimal
  GB/s, as vendors state it.
- Every result in this README comes from harness commit `e83569f` with no uncommitted changes.

## Limitations

- T4 is a burstable VM on a shared host: run-to-run variance is higher, and there is no fixed
  bandwidth figure to compare against.
- SMT is off on the T1/T2 machine (8 cores, 8 threads), set in firmware.
- The desktop holds some VRAM for the display during T1.
- The client timeout applies to each read from the stream, not to the whole request.
- One T4 measurement (0.6B Q4, long_context, repetition 2) overlapped a stray second run and
  has about 4 s extra TTFT; the median is unaffected.
- Free-text answers are not graded yet, and T3 has not run yet.

## Running it

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/), with every
`ollama_tag` in `config/bench.yaml` pulled on the machine being measured.

```sh
uv sync
uv run llm-bench plan --tier T1                        # validate config, print the run matrix
uv run llm-bench probe --tier T1 --model qwen3-4b-q4   # one timed request: smoke test
uv run llm-bench run --tier T1                         # everything -> results/T1-<time>.jsonl + .env.json
uv run llm-bench report                                # results -> report/results.md + charts
uv run pytest && uv run ruff check && uv run mypy src tests
```

Adding a model, tier or regime is an entry in `config/bench.yaml`, not a code change.

## How this was built

Built with an AI assistant (Claude). The design decisions, the choice of what to measure, every
benchmark run and the review of every change are mine. Most of the code was written by the
assistant; I read each change, explained it back, and committed it myself, one change per commit.
