"""Benchmark configuration: what to run, where, and how.

One YAML file drives a run. Adding a model, a device tier or a prompt regime is a YAML
entry, not a code change. Unknown keys are rejected so a typo fails loudly instead of
silently falling back to a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunSettings(Strict):
    repetitions: int = Field(ge=1)
    warmup: int = Field(ge=0)
    max_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0.0)
    seed: int
    think: bool
    timeout_s: float = Field(gt=0)
    sustained_minutes: float = Field(gt=0)
    results_dir: Path


class Model(Strict):
    id: str
    family: str
    params_b: float = Field(gt=0)
    quant: str
    ollama_tag: str


class Tier(Strict):
    id: str
    name: str
    os: Literal["linux", "windows"]
    arch: Literal["x86_64", "arm64"]
    gpu_offload: bool
    memory_gb: float = Field(gt=0)
    vram_gb: float | None = None
    endpoint: str = "http://localhost:11434"
    notes: str = ""


class Regime(Strict):
    id: str
    description: str
    prompt_file: Path
    num_ctx: int = Field(ge=512)
    json_output: bool = False


class BenchConfig(Strict):
    schema_version: Literal[1]
    run: RunSettings
    models: list[Model] = Field(min_length=1)
    tiers: list[Tier] = Field(min_length=1)
    regimes: list[Regime] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unique(self) -> BenchConfig:
        for name, items in (("model", self.models), ("tier", self.tiers), ("regime", self.regimes)):
            ids = [item.id for item in items]
            dupes = {i for i in ids if ids.count(i) > 1}
            if dupes:
                raise ValueError(f"duplicate {name} id(s): {sorted(dupes)}")
        return self

    def tier(self, tier_id: str) -> Tier:
        for t in self.tiers:
            if t.id == tier_id:
                return t
        raise KeyError(f"unknown tier {tier_id!r}; known: {[t.id for t in self.tiers]}")

    def model(self, model_id: str) -> Model:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"unknown model {model_id!r}; known: {[m.id for m in self.models]}")

    def regime(self, regime_id: str) -> Regime:
        for r in self.regimes:
            if r.id == regime_id:
                return r
        raise KeyError(f"unknown regime {regime_id!r}; known: {[r.id for r in self.regimes]}")


def load_config(path: Path) -> BenchConfig:
    """Load and validate a config.

    Prompt paths are resolved against the config file's folder, not the directory the
    command was launched from, and returned absolute. Callers never re-resolve them.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = BenchConfig.model_validate(raw)
    base = path.parent.resolve()
    regimes = [r.model_copy(update={"prompt_file": base / r.prompt_file}) for r in config.regimes]
    missing = [r.prompt_file for r in regimes if not r.prompt_file.is_file()]
    if missing:
        raise FileNotFoundError(f"prompt file(s) not found: {missing}")
    return config.model_copy(update={"regimes": regimes})
