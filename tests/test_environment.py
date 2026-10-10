from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest

from local_llm_bench import environment
from local_llm_bench.environment import capture

STARTED = datetime(2026, 10, 6, tzinfo=UTC)


def ollama(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ollama.test")


def test_ollama_version_is_recorded() -> None:
    env = capture(ollama(lambda r: httpx.Response(200, json={"version": "0.34.4"})), "T1", STARTED)
    assert env.ollama_version == "0.34.4"
    assert (env.tier_id, env.started_at) == ("T1", STARTED)


def test_unreachable_ollama_is_unknown_not_a_crash() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    assert capture(ollama(refuse), "T1", STARTED).ollama_version is None


def test_missing_tool_is_none_not_a_crash() -> None:
    assert environment._command(["no-such-tool-llm-bench"]) is None


def test_cpu_name_comes_from_lscpu(monkeypatch: pytest.MonkeyPatch) -> None:
    lscpu = "Architecture:  aarch64\nVendor ID:     ARM\nModel name:    Neoverse-N1\n"
    monkeypatch.setattr(environment, "_command", lambda args: lscpu)
    assert environment._cpu_name() == "Neoverse-N1"


@pytest.mark.parametrize(
    ("changed_files", "dirty"), [("", False), ("README.md", True), (None, None)]
)
def test_dirty_means_tracked_files_changed(
    monkeypatch: pytest.MonkeyPatch, changed_files: str | None, dirty: bool | None
) -> None:
    def fake(args: list[str]) -> str | None:
        return changed_files if "diff-index" in args else None

    monkeypatch.setattr(environment, "_command", fake)
    env = capture(ollama(lambda r: httpx.Response(404)), "T1", STARTED)
    assert env.git_dirty is dirty
