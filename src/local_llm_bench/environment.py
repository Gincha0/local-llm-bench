"""What a run ran on, saved next to its results.

The same model on the same machine gives different numbers after a driver or Ollama
update, so every results file gets a record of the machine, the software and the exact
harness commit. Anything this machine cannot report is None, never guessed.
"""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict

HERE = Path(__file__).parent  # inside the repo, so git finds it wherever the run starts


class Environment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier_id: str
    started_at: datetime
    system: str  # OS and kernel, e.g. "Linux 6.17.1-2-cachyos"
    arch: str  # as the OS names it: x86_64, aarch64, AMD64
    cpu: str | None
    cpu_count: int | None  # logical CPUs
    memory_bytes: int | None  # installed RAM
    gpu: str | None  # name, driver, VRAM from nvidia-smi; None without an NVIDIA GPU
    ollama_version: str | None
    python_version: str
    git_commit: str | None
    git_dirty: bool | None  # tracked files differ from the commit: results don't match it


def capture(client: httpx.Client, tier_id: str, started_at: datetime) -> Environment:
    changed = _command(["git", "diff-index", "--name-only", "HEAD", "--"])
    return Environment(
        tier_id=tier_id,
        started_at=started_at,
        system=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
        cpu=_cpu_name(),
        cpu_count=os.cpu_count(),
        memory_bytes=_memory_bytes(),
        gpu=_command(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        )
        or None,
        ollama_version=_ollama_version(client),
        python_version=platform.python_version(),
        git_commit=_command(["git", "rev-parse", "HEAD"]) or None,
        git_dirty=None if changed is None else changed != "",
    )


def _command(args: list[str]) -> str | None:
    """A command's output, or None if it is missing or fails: not every tier has every tool."""
    try:
        done = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
            cwd=HERE,
            env={**os.environ, "LC_ALL": "C"},  # English labels, so "Model name" parses
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def _cpu_name() -> str | None:
    """CPU model. On Linux from lscpu, which also names ARM cores; /proc/cpuinfo does not."""
    for line in (_command(["lscpu"]) or "").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "Model name":
            return value.strip()
    return platform.processor() or None


def _memory_bytes() -> int | None:
    """Installed RAM from the OS page counters. Not available on Windows."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def _ollama_version(client: httpx.Client) -> str | None:
    try:
        response = client.get("/api/version")
        response.raise_for_status()
        version = response.json().get("version")
    except (httpx.HTTPError, ValueError):
        return None
    return version if isinstance(version, str) else None
