"""The two README charts, drawn from the report's summaries.

Uses matplotlib's object API (Figure, no pyplot): no global state, no window, only files.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from local_llm_bench.config import BenchConfig, Model, Tier
from local_llm_bench.report import BYTES_PER_GIB, Run, Summary

# One colour per tier, in config order. Checked as a set for colour-blind readers (every
# pair stays apart), so a fifth tier needs a fifth colour checked against these four.
TIER_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
QUANT_MARKERS = ("o", "s", "^", "D")
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e4e0"
DPI = 150


def tier_colors(config: BenchConfig) -> dict[str, str]:
    if len(config.tiers) > len(TIER_COLORS):
        raise ValueError(f"{len(config.tiers)} tiers but only {len(TIER_COLORS)} chart colours")
    return {tier.id: color for tier, color in zip(config.tiers, TIER_COLORS, strict=False)}


def gen_speed_chart(config: BenchConfig, summaries: list[Summary], out: Path) -> None:
    """Generation speed against model file size, one colour per tier, both axes log.

    Each generated token reads every weight once, so a tier cannot generate faster than
    its memory bandwidth divided by the model's size: the dashed line. On log-log axes
    that ceiling is a straight line. Uses the smallest-context regime, where the KV cache
    adds the least to what each token has to read.
    """
    regime_id = _smallest_and_largest_context(config)[0][1]
    colors = tier_colors(config)
    sizes = {m.id: m.size_bytes / BYTES_PER_GIB for m in config.models}
    quants = sorted({m.quant for m in config.models})
    markers = dict(zip(quants, QUANT_MARKERS, strict=False))
    quant_of = {m.id: m.quant for m in config.models}
    x_min, x_max = min(sizes.values()) / 1.3, max(sizes.values()) * 1.3

    fig = Figure(figsize=(9, 6))
    ax = fig.add_subplot()
    drawn: list[Tier] = []
    for tier in config.tiers:
        points = [
            (sizes[s.model_id], s.gen_tps, markers[quant_of[s.model_id]])
            for s in summaries
            if s.tier_id == tier.id and s.regime_id == regime_id and s.gen_tps is not None
        ]
        if not points:
            continue
        drawn.append(tier)
        color = colors[tier.id]
        for x, y, marker in points:
            ax.plot(
                x,
                y,
                marker=marker,
                markersize=8,
                color=color,
                markeredgecolor="white",
                markeredgewidth=1.5,
                linestyle="none",
                zorder=3,
            )
        if tier.memory_bandwidth_gb_s:
            _ceiling(ax, tier.id, tier.memory_bandwidth_gb_s, color, x_min, x_max)
        x, y, _ = max(points)
        ax.annotate(
            tier.id,
            (x, y),
            xytext=(9, -4),
            textcoords="offset points",
            color=TEXT,
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    _log_ticks(ax.xaxis, [0.5, 1, 2, 4, 8])
    _log_ticks(ax.yaxis, [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    ax.set_xlabel("Model file size, GiB (log scale)", color=TEXT_MUTED)
    ax.set_ylabel("Generation speed, tokens/s (median, log scale)", color=TEXT_MUTED)
    _style(ax)
    _titles(
        ax,
        f"Generation speed against model size ('{regime_id}' prompt)",
        "Dashed: the ceiling the memory's spec bandwidth allows (bandwidth ÷ model size)",
    )
    handles = [
        Line2D([], [], color=colors[t.id], marker="o", linestyle="none", label=f"{t.id} {t.name}")
        for t in drawn
    ]
    handles += [
        Line2D([], [], color=TEXT_MUTED, marker=markers[q], linestyle="none", label=q)
        for q in quants
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9, labelcolor=TEXT)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")


def memory_fit_chart(
    config: BenchConfig, runs: dict[str, Run], summaries: list[Summary], out: Path
) -> None:
    """What Ollama holds for each model, one panel per tier, against what the tier has.

    Two bars per model: the smallest and the largest context window in the config, because
    the KV cache grows with the context. On a GPU tier the part that did not fit in VRAM is
    hatched: it was split off to system RAM. A cross marks a run that failed.
    """
    colors = tier_colors(config)
    tiers = [t for t in config.tiers if t.id in runs]
    contexts = _smallest_and_largest_context(config)
    found = {(s.tier_id, s.model_id, s.regime_id): s for s in summaries}

    fig = Figure(figsize=(9, 1.2 + 2.4 * len(tiers)), layout="constrained")
    axes = fig.subplots(len(tiers), 1, sharex=True, squeeze=False)[:, 0]
    any_split = False
    for ax, tier in zip(axes, tiers, strict=True):
        panel = [
            [found[(tier.id, m.id, regime_id)] for _, regime_id in contexts] for m in config.models
        ]
        split = _memory_panel(ax, tier, runs[tier.id], panel, colors[tier.id])
        any_split = any_split or split
    axes[-1].set_xticks(range(len(config.models)), [_short_name(m) for m in config.models])

    shades = [_tint(TEXT_MUTED, 0.5)] * (len(contexts) - 1) + [TEXT_MUTED]
    handles: list[Artist] = [
        Patch(color=shade, label=f"{ctx // 1024}k context ({regime_id})")
        for shade, (ctx, regime_id) in zip(shades, contexts, strict=True)
    ]
    handles.append(Line2D([], [], color=TEXT, linewidth=1, label="memory the tier has"))
    if any_split:
        handles.append(
            Patch(facecolor="white", edgecolor=TEXT_MUTED, hatch="////", label="split to RAM")
        )
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncols=len(handles),
        frameon=False,
        fontsize=9,
        labelcolor=TEXT,
    )
    fig.suptitle(
        "Memory Ollama holds per model, against what each tier has",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=13,
    )
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")


def _memory_panel(ax: Axes, tier: Tier, run: Run, panel: list[list[Summary]], color: str) -> bool:
    """One tier: panel[model][context] -> bars. Returns whether any model was split to RAM."""
    width = 0.32
    capacity, capacity_label = _capacity(tier, run)
    top = capacity
    split = False
    for i, contexts in enumerate(panel):
        failed: dict[str, list[float]] = {}
        for j, s in enumerate(contexts):
            x = i + (j - (len(contexts) - 1) / 2) * (width + 0.04)
            if s.loaded_gib is None:
                if s.failure:
                    failed.setdefault(s.failure, []).append(x)
                continue
            shade = _tint(color, 0.5) if j < len(contexts) - 1 else color
            share = s.gpu_share if tier.gpu_offload and s.gpu_share is not None else 1.0
            in_place = s.loaded_gib * share
            ax.bar(x, in_place, width, color=shade, zorder=2)
            if s.loaded_gib - in_place > 0.01:
                split = True
                ax.bar(
                    x,
                    s.loaded_gib - in_place,
                    width,
                    bottom=in_place,
                    color="white",
                    edgecolor=shade,
                    hatch="////",
                    linewidth=0,
                    zorder=2,
                )
            top = max(top, s.loaded_gib)
        for reason, xs in failed.items():
            _failed(ax, xs, reason)

    ax.axhline(capacity, color=TEXT, linewidth=1, zorder=1)
    ax.annotate(
        capacity_label,
        (0, capacity),
        xycoords=("axes fraction", "data"),
        xytext=(4, 3),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color=TEXT_MUTED,
        fontsize=8.5,
    )
    ax.set_ylim(0, top * 1.2)
    ax.set_ylabel("GiB", color=TEXT_MUTED)
    ax.set_title(f"{tier.id} {tier.name}", loc="left", color=TEXT, fontsize=10.5)
    _style(ax)
    ax.grid(axis="x", visible=False)
    return split


def _ceiling(ax: Axes, tier_id: str, gb_s: float, color: str, x_min: float, x_max: float) -> None:
    gib_s = gb_s * 1e9 / BYTES_PER_GIB
    ax.plot(
        [x_min, x_max],
        [gib_s / x_min, gib_s / x_max],
        color=color,
        linestyle=(0, (4, 3)),
        linewidth=1.5,
        zorder=2,
    )
    ax.annotate(
        f"{tier_id} ceiling: {gb_s:g} GB/s",
        (x_min, gib_s / x_min),
        xytext=(6, 6),
        textcoords="offset points",
        color=TEXT_MUTED,
        fontsize=8.5,
    )


def _capacity(tier: Tier, run: Run) -> tuple[float, str]:
    """VRAM for a GPU tier, RAM the OS reports for a CPU tier (falls back to the config)."""
    if tier.gpu_offload and tier.vram_gb is not None:
        return tier.vram_gb, f"{tier.vram_gb:g} GiB VRAM"
    if run.env.memory_bytes is not None:
        ram = run.env.memory_bytes / BYTES_PER_GIB
        return ram, f"{ram:.1f} GiB RAM (what the OS reports)"
    return tier.memory_gb, f"{tier.memory_gb:g} GiB RAM"


def _smallest_and_largest_context(config: BenchConfig) -> list[tuple[int, str]]:
    first_regime: dict[int, str] = {}
    for regime in config.regimes:
        first_regime.setdefault(regime.num_ctx, regime.id)
    chosen = sorted({min(first_regime), max(first_regime)})
    return [(ctx, first_regime[ctx]) for ctx in chosen]


def _failed(ax: Axes, xs: list[float], reason: str) -> None:
    """A cross at each failed bar's place, one label for the group."""
    ax.plot(xs, [0] * len(xs), "x", markersize=7, mew=2, color=TEXT, clip_on=False, zorder=3)
    ax.annotate(
        reason,
        (sum(xs) / len(xs), 0),
        xytext=(0, 9),
        textcoords="offset points",
        rotation=90,
        ha="center",
        va="bottom",
        color=TEXT_MUTED,
        fontsize=8,
    )


def _short_name(model: Model) -> str:
    return f"{model.params_b:g}B {model.quant.split('_')[0]}"


def _tint(color: str, amount: float) -> str:
    """Mix a colour with white: amount 0 keeps it, 1 is white."""
    r, g, b = to_rgb(color)
    return to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))


def _log_ticks(axis: Axis, values: list[float]) -> None:
    axis.set_major_locator(FixedLocator(values))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _titles(ax: Axes, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", color=TEXT, fontsize=13, pad=26)
    ax.text(0, 1.025, subtitle, transform=ax.transAxes, color=TEXT_MUTED, fontsize=9)


def _style(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
