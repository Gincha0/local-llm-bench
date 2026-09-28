"""What one measurement records.

Raw values only: durations and counts as measured. Rates (tokens/s) are derived when
reporting, never stored, so a wrong formula can be fixed without re-running anything.
A failed run (OOM, timeout) is still a result: `error` is set and the numbers stay None.
That is how the cliff gets documented instead of crashing the run.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # identity
    tier_id: str
    model_id: str
    regime_id: str
    rep: int
    warmup: bool
    think: bool
    started_at: datetime

    # client-side wall clock (time.perf_counter)
    ttft_s: float | None = None
    wall_s: float | None = None

    # server-reported (Ollama returns nanoseconds; converted to seconds on write)
    load_s: float | None = None
    prompt_tokens: int | None = None
    prompt_eval_s: float | None = None
    gen_tokens: int | None = None
    gen_s: float | None = None

    # memory; None where the tier cannot measure it
    peak_rss_mb: float | None = None
    peak_vram_mb: float | None = None

    error: str | None = None

    @property
    def prompt_tps(self) -> float | None:
        if self.prompt_tokens is None or not self.prompt_eval_s:
            return None
        return self.prompt_tokens / self.prompt_eval_s

    @property
    def gen_tps(self) -> float | None:
        if self.gen_tokens is None or not self.gen_s:
            return None
        return self.gen_tokens / self.gen_s
