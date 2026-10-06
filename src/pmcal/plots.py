"""Figures for the report, as static PNGs.

One look across every figure: a near-white surface, hairline gridlines,
recessive axes, thin marks, and a fixed colour per entity. Polymarket is
always blue and Kalshi always orange. Horizons, which are ordered, use one
blue ramp from light (far from resolution) to dark (close to it). Text is
never coloured by series. Every figure has a matching CSV table in
``results/tables``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

VENUE_COLOR = {"polymarket": "#2a78d6", "kalshi": "#eb6834"}
VENUE_NAME = {"polymarket": "Polymarket", "kalshi": "Kalshi"}
# categorical slots 3 and 4, for series that are not venues
EXTRA = ["#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
# ordinal blue ramp, light to dark, every step >= 2:1 against the surface
HORIZON_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b", "#0d366b"]

DPI = 150


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": INK_2,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "lines.linewidth": 1.5,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _caption(fig: Figure, text: str) -> None:
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=7.5, color=MUTED, wrap=True)


def _compact(x: float, _pos: object = None) -> str:
    if x >= 1e6:
        return f"{x / 1e6:g}M"
    if x >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def _nonneg(parts: list[object]) -> np.ndarray:
    """Error-bar lengths from [estimate - lo, hi - estimate], floored at 0.

    Percentile bootstrap intervals need not contain the point estimate (binned ECE
    is biased upwards, so its replicates sit above the estimate); the bar then
    starts at the estimate."""
    out: np.ndarray = np.clip(np.vstack([np.asarray(x, dtype=float) for x in parts]), 0, None)
    return out


def _venues(df: pd.DataFrame) -> list[str]:
    return [v for v in ("polymarket", "kalshi") if v in set(df["venue"])]


# --------------------------------------------------------------------------


def reliability(
    tables: dict[str, pd.DataFrame], title: str, path: Path, *, caption: str = ""
) -> Path:
    """Reliability diagrams, one column per venue, with the price histogram below.

    ``tables`` maps venue -> output of ``analysis.reliability_table``.
    """
    _style()
    venues = [v for v in ("polymarket", "kalshi") if v in tables]
    fig, axes = plt.subplots(
        2,
        len(venues),
        figsize=(3.6 * len(venues), 4.6),
        gridspec_kw={"height_ratios": [3.2, 1]},
        sharex=True,
        squeeze=False,
    )
    for j, v in enumerate(venues):
        t = tables[v]
        t = t[t["n"] > 0]
        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1.0, zorder=1)
        yerr = _nonneg([t["freq"] - t["freq_lo"], t["freq_hi"] - t["freq"]])
        ax.errorbar(
            t["mean_price"],
            t["freq"],
            yerr=yerr,
            color=VENUE_COLOR[v],
            lw=1.5,
            elinewidth=1.2,
            capsize=0,
            marker="o",
            ms=5,
            mec=SURFACE,
            mew=1.2,
            zorder=3,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(VENUE_NAME[v], loc="left")
        if j == 0:
            ax.set_ylabel("share that resolved YES")
        hx = axes[1, j]
        widths = t["hi_edge"] - t["lo_edge"]
        hx.bar(
            t["lo_edge"],
            t["n"],
            width=widths * 0.92,
            align="edge",
            color=VENUE_COLOR[v],
            alpha=0.35,
            lw=0,
        )
        hx.set_yscale("log")
        hx.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_compact))
        hx.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        hx.grid(axis="x", visible=False)
        hx.set_xlabel("price")
        if j == 0:
            hx.set_ylabel("contracts", labelpad=8)
    fig.suptitle(title, x=0.0, ha="left", fontsize=11, color=INK, fontweight="bold")
    if caption:
        _caption(fig, caption)
    fig.tight_layout()
    return _save(fig, path)


def reliability_by_horizon(
    tables: dict[str, dict[str, pd.DataFrame]], title: str, path: Path, *, caption: str = ""
) -> Path:
    """One panel per venue, one curve per horizon (ordinal blue ramp)."""
    _style()
    venues = [v for v in ("polymarket", "kalshi") if v in tables]
    fig, axes = plt.subplots(1, len(venues), figsize=(3.6 * len(venues), 3.8), squeeze=False)
    for j, v in enumerate(venues):
        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1.0, zorder=1)
        hs = list(tables[v])
        ramp = HORIZON_RAMP[: len(hs)]
        for h, col in zip(hs, ramp, strict=True):
            t = tables[v][h]
            t = t[t["n"] > 0]
            ax.plot(
                t["mean_price"],
                t["freq"],
                color=col,
                marker="o",
                ms=4,
                mec=SURFACE,
                mew=1.0,
                label=h,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(VENUE_NAME[v], loc="left")
        ax.set_xlabel("price")
        if j == 0:
            ax.set_ylabel("share that resolved YES")
        ax.legend(title="before resolution", title_fontsize=8, loc="upper left")
    fig.suptitle(title, x=0.0, ha="left", fontsize=11, color=INK, fontweight="bold")
    if caption:
        _caption(fig, caption)
    fig.tight_layout()
    return _save(fig, path)


def longshot(tables: dict[str, pd.DataFrame], title: str, path: Path, *, caption: str = "") -> Path:
    """Return per dollar by price bin, both venues, dodged, with intervals."""
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    venues = [v for v in ("polymarket", "kalshi") if v in tables]
    offs = np.linspace(-0.006, 0.006, len(venues)) if len(venues) > 1 else [0.0]
    ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
    for v, off in zip(venues, offs, strict=True):
        t = tables[v]
        t = t[t["n"] > 0]
        x = t["mean_price"] + off
        yerr = _nonneg([t["return"] - t["return_lo"], t["return_hi"] - t["return"]])
        ax.errorbar(
            x,
            100 * t["return"],
            yerr=100 * yerr,
            color=VENUE_COLOR[v],
            lw=1.5,
            elinewidth=1.2,
            marker="o",
            ms=5,
            mec=SURFACE,
            mew=1.2,
            label=VENUE_NAME[v],
            zorder=3,
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel("price paid")
    ax.set_ylabel("average return per $1, before fees (%)")
    ax.legend(loc="lower right")
    ax.set_title(title, loc="left")
    if caption:
        _caption(fig, caption)
    fig.tight_layout()
    return _save(fig, path)


def metric_by_level(
    df: pd.DataFrame,
    level_col: str,
    levels: Sequence[str],
    metrics: Sequence[tuple[str, str]],
    title: str,
    path: Path,
    *,
    ref_lines: dict[str, float] | None = None,
    caption: str = "",
) -> Path:
    """Small multiples, one per metric: x = level (e.g. horizon), one series per venue.

    ``df`` has columns venue, level_col, metric, metric_lo, metric_hi.
    """
    _style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 3.2), squeeze=False)
    xs = np.arange(len(levels))
    venues = _venues(df)
    offs = np.linspace(-0.08, 0.08, len(venues)) if len(venues) > 1 else [0.0]
    for ax, (m, label) in zip(axes[0], metrics, strict=True):
        if ref_lines and m in ref_lines:
            ax.axhline(ref_lines[m], color=AXIS, lw=1.0, zorder=1)
        for v, off in zip(venues, offs, strict=True):
            d = df[df["venue"] == v].set_index(level_col).reindex(list(levels))
            yerr = _nonneg([d[m] - d[f"{m}_lo"], d[f"{m}_hi"] - d[m]])
            ax.errorbar(
                xs + off,
                d[m],
                yerr=yerr,
                color=VENUE_COLOR[v],
                lw=1.5,
                elinewidth=1.2,
                marker="o",
                ms=5,
                mec=SURFACE,
                mew=1.2,
                label=VENUE_NAME[v],
            )
        ax.set_xticks(xs, list(levels))
        ax.set_title(label, loc="left", fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axes[0, 0].legend(loc="best")
    fig.suptitle(title, x=0.0, ha="left", fontsize=11, color=INK, fontweight="bold")
    if caption:
        _caption(fig, caption)
    fig.tight_layout()
    return _save(fig, path)


def dot_rows(
    df: pd.DataFrame,
    row_col: str,
    rows: Sequence[str],
    metrics: Sequence[tuple[str, str]],
    title: str,
    path: Path,
    *,
    ref_lines: dict[str, float] | None = None,
    caption: str = "",
) -> Path:
    """Horizontal dot plot: one row per category, one panel per metric, venues dodged."""
    _style()
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(3.4 * len(metrics), 0.42 * len(rows) + 1.2), squeeze=False
    )
    ys = np.arange(len(rows))[::-1]
    venues = _venues(df)
    offs = np.linspace(0.12, -0.12, len(venues)) if len(venues) > 1 else [0.0]
    for k, (ax, (m, label)) in enumerate(zip(axes[0], metrics, strict=True)):
        if ref_lines and m in ref_lines:
            ax.axvline(ref_lines[m], color=AXIS, lw=1.0, zorder=1)
        for v, off in zip(venues, offs, strict=True):
            d = df[df["venue"] == v].set_index(row_col).reindex(list(rows))
            ok = d[m].notna().to_numpy()
            xerr = _nonneg([d[m] - d[f"{m}_lo"], d[f"{m}_hi"] - d[m]])[:, ok]
            ax.errorbar(
                d[m][ok],
                (ys + off)[ok],
                xerr=xerr,
                color=VENUE_COLOR[v],
                lw=0,
                elinewidth=1.2,
                marker="o",
                ms=5,
                mec=SURFACE,
                mew=1.2,
                label=VENUE_NAME[v],
            )
        ax.set_yticks(ys, list(rows) if k == 0 else [""] * len(rows))
        ax.set_title(label, loc="left", fontsize=9.5)
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, x=0.0, ha="left", fontsize=11, color=INK, fontweight="bold")
    if caption:
        _caption(fig, caption)
    fig.tight_layout()
    return _save(fig, path)


def recalibration_gains(df: pd.DataFrame, title: str, path: Path, *, caption: str = "") -> Path:
    """Out-of-sample Brier improvement from recalibration, per venue (panels) and
    horizon (x), one series per method."""
    _style()
    d = df[df["score"] == "brier"]
    venues = _venues(d)
    horizons = [h for h in ("30d", "14d", "7d", "3d", "1d", "6h", "1h") if h in set(d["horizon"])]
    methods = ["platt", "isotonic"]
    fig, axes = plt.subplots(
        1, len(venues), figsize=(3.8 * len(venues), 3.2), squeeze=False, sharey=True
    )
    xs = np.arange(len(horizons))
    for ax, v in zip(axes[0], venues, strict=True):
        ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
        for i, m in enumerate(methods):
            g = d[(d["venue"] == v) & (d["method"] == m)].set_index("horizon").reindex(horizons)
            off = (i - 0.5) * 0.16
            yerr = _nonneg([g["gain"] - g["gain_lo"], g["gain_hi"] - g["gain"]])
            ax.errorbar(
                xs + off,
                1e3 * g["gain"],
                yerr=1e3 * yerr,
                color=EXTRA[i],
                lw=0,
                elinewidth=1.2,
                marker="o",
                ms=5,
                mec=SURFACE,
                mew=1.2,
                label={"platt": "Platt (logit-linear)", "isotonic": "isotonic"}[m],
            )
        ax.set_xticks(xs, horizons)
        ax.set_title(VENUE_NAME[v], loc="left")
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("Brier improvement (x 1000)")
    axes[0, 0].legend(loc="best")
    fig.suptitle(title, x=0.0, ha="left", fontsize=11, color=INK, fontweight="bold")
    if caption:
        _caption(fig, caption)
    fig.tight_layout()
    return _save(fig, path)


def anchor_contrast(
    tables: dict[str, pd.DataFrame], title: str, path: Path, *, caption: str = ""
) -> Path:
    """Reliability of the same contracts priced by the scheduled-time anchor and by
    the actual-close anchor. ``tables`` maps label -> reliability table."""
    _style()
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1.0, zorder=1)
    for i, (label, t) in enumerate(tables.items()):
        t = t[t["n"] > 0]
        yerr = _nonneg([t["freq"] - t["freq_lo"], t["freq_hi"] - t["freq"]])
        ax.errorbar(
            t["mean_price"],
            t["freq"],
            yerr=yerr,
            color=EXTRA[i],
            lw=1.5,
            elinewidth=1.2,
            marker="o",
            ms=5,
            mec=SURFACE,
            mew=1.2,
            label=label,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("price")
    ax.set_ylabel("share that resolved YES")
    ax.legend(loc="upper left")
    ax.set_title(title, loc="left")
    if caption:
        _caption(fig, caption)
    fig.tight_layout()
    return _save(fig, path)


def paired_scatter(
    df: pd.DataFrame, x: str, y: str, color_by: str, title: str, path: Path, *, caption: str = ""
) -> Path:
    """Scatter of two venues' prices for the same event, coloured by outcome."""
    _style()
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1.0, zorder=1)
    for i, (lab, g) in enumerate(df.groupby(color_by, sort=True)):
        ax.scatter(
            g[x],
            g[y],
            s=14,
            color=EXTRA[i],
            alpha=0.7,
            edgecolors=SURFACE,
            linewidths=0.6,
            label=str(lab),
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{VENUE_NAME.get(x.removeprefix('p_'), x)} price")
    ax.set_ylabel(f"{VENUE_NAME.get(y.removeprefix('p_'), y)} price")
    ax.legend(loc="upper left", title="resolved", title_fontsize=8)
    ax.set_title(title, loc="left")
    if caption:
        _caption(fig, caption)
    fig.tight_layout()
    return _save(fig, path)
