from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from local_llm_bench.cli import plan
from local_llm_bench.config import BenchConfig, load_config

CONFIG = Path(__file__).parent.parent / "config" / "bench.yaml"


def raw_config() -> dict[str, Any]:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_shipped_config_is_valid() -> None:
    config = load_config(CONFIG)
    assert [t.id for t in config.tiers] == ["T1", "T2", "T3", "T4", "T5"]


def test_unknown_key_is_rejected() -> None:
    raw = raw_config()
    raw["run"]["repetitons"] = 3  # typo must not be silently ignored
    with pytest.raises(ValidationError):
        BenchConfig.model_validate(raw)


def test_duplicate_ids_are_rejected() -> None:
    raw = raw_config()
    raw["models"].append(dict(raw["models"][0]))
    with pytest.raises(ValidationError, match="duplicate model"):
        BenchConfig.model_validate(raw)


def test_prompt_paths_do_not_depend_on_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # launch from somewhere unrelated, as a VPS service would
    config = load_config(CONFIG)
    assert all(r.prompt_file.is_absolute() and r.prompt_file.is_file() for r in config.regimes)


def test_missing_prompt_file_is_reported(tmp_path: Path) -> None:
    (tmp_path / "bench.yaml").write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "bench.yaml")


def test_plan_counts_requests() -> None:
    config = load_config(CONFIG)
    out = plan(config, "T4")
    per_pair = config.run.warmup + config.run.repetitions
    expected = len(config.models) * len(config.regimes) * per_pair
    assert f"= {expected} requests" in out


def test_unknown_tier_names_the_known_ones() -> None:
    with pytest.raises(KeyError, match="T1"):
        load_config(CONFIG).tier("T9")


def test_unknown_model_names_the_known_ones() -> None:
    with pytest.raises(KeyError, match="qwen3-4b-q4"):
        load_config(CONFIG).model("llama-404")


def test_unknown_regime_names_the_known_ones() -> None:
    with pytest.raises(KeyError, match="short"):
        load_config(CONFIG).regime("medium")
