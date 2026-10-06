"""The study's questions, computed on the panel.

Every scalar comes with a 95% interval from a cluster bootstrap that resamples
clusters (``pmcal.stats``). A cluster is a venue event, or a whole game when the
venue splits one game over several events (``cluster_id``), optionally coarsened
(``AnalysisConfig.cluster_by``) so that contracts exposed to the same shock, such as
everything in one category resolving on the same day, are resampled together.
Comparisons between horizons, venues or methods reuse the same bootstrap
replicates for both sides, so the interval on a difference is a paired one.

The functions return tidy DataFrames; ``pmcal.report`` turns them into the
tables and figures in ``results/``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pmcal.recalibrate import walk_forward
from pmcal.stats import (
    EPS,
    bin_edges,
    bootstrap_draws,
    cluster_index,
    coarsen_clusters,
    debiased_reliability,
    decompose,
    effective_n,
    fit_logistic,
    logit,
    replicate_clusters,
    wilson_interval,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

Z995 = 2.807  # two-sided 99.5% normal quantile, for Bonferroni over 10 primary estimands

METRICS = (
    "brier",
    "reliability",
    "resolution",
    "uncertainty",
    "ece",
    "log_loss",
    "mean_price",
    "base_rate",
    "slope",
    "intercept",
)


@dataclass(frozen=True)
class AnalysisConfig:
    headline_horizon: str = "1d"
    panel_horizons: tuple[str, ...] = ("30d", "7d", "1d", "1h")
    n_bins: int = 10
    n_boot: int = 1000
    n_null: int = 200  # simulations for the ECE noise floor
    seed: int = 0
    log_clip: float = 1e-3  # probabilities clipped to [c, 1 - c] for the log score
    min_events: int = 30  # smallest subgroup (in events) reported on its own
    volume_buckets: tuple[float, ...] = (1e3, 1e4, 1e5, 1e6)
    refit_every_days: int = 7
    min_train: int = 1000
    min_train_events: int = 200
    # Extra keys that coarsen the bootstrap cluster (see ``recluster``): any of
    # "series_day", "category_day", "day", "series". Empty keeps game/event clusters.
    cluster_by: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AnalysisConfig:
        names = {f.name for f in fields(cls)}
        unknown = set(raw) - names
        if unknown:
            raise ValueError(f"unknown keys in [analysis]: {sorted(unknown)}")
        kw: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
        return cls(**kw)


# --------------------------------------------------------------------------
# Core metric vector


def metric_vector(
    p: FloatArray,
    y: FloatArray,
    edges: FloatArray,
    log_clip: float,
    w: FloatArray | None = None,
    *,
    with_fit: bool = True,
) -> FloatArray:
    d = decompose(p, y, edges, w)
    q = np.clip(p, log_clip, 1 - log_clip)
    ll = float(np.average(-(y * np.log(q) + (1 - y) * np.log1p(-q)), weights=w))
    slope = intercept = np.nan
    if with_fit and 0 < y.mean() < 1:
        f = fit_logistic(logit(p), y, w)
        if f.converged:  # separated replicates have no finite slope
            slope, intercept = f.slope, f.intercept
    return np.array(
        [
            d.brier,
            d.reliability,
            d.resolution,
            d.uncertainty,
            d.ece,
            ll,
            float(np.average(p, weights=w)),
            d.base_rate,
            slope,
            intercept,
        ]
    )


def _draws(events: NDArray[Any], cfg: AnalysisConfig, salt: int = 0) -> list[IntArray]:
    groups = cluster_index(events)
    return bootstrap_draws(groups, cfg.n_boot, np.random.default_rng(cfg.seed + salt))


def summarise(
    df: pd.DataFrame, cfg: AnalysisConfig, *, salt: int = 0, weight: str | None = None
) -> dict[str, float]:
    """Metrics with cluster-bootstrap intervals for one subset of the panel.

    ``weight`` names a column of row weights (e.g. ``sample_weight`` for
    population-weighted estimates, or ``event_weight`` = 1 / contracts in the
    event, which counts every event once)."""
    p = df["price"].to_numpy(dtype=np.float64)
    y = df["outcome"].to_numpy(dtype=np.float64)
    w = None if weight is None else df[weight].to_numpy(dtype=np.float64)
    ev = df["cluster_id"].to_numpy()
    _, ev_codes = np.unique(ev, return_inverse=True)
    edges = bin_edges(cfg.n_bins)
    est = metric_vector(p, y, edges, cfg.log_clip, w)
    draws = _draws(ev, cfg, salt)
    reps = np.array(
        [metric_vector(p[d], y[d], edges, cfg.log_clip, None if w is None else w[d]) for d in draws]
    )
    lo = np.nanquantile(reps, 0.025, axis=0)
    hi = np.nanquantile(reps, 0.975, axis=0)
    # Binned ECE and reliability are biased upwards in finite samples, and their
    # bootstrap replicates more so, so the percentile interval sits above the
    # estimate. For these two, use the basic (reverse percentile) interval
    # [2 est - q97.5, 2 est - q2.5], floored at 0. That interval is centred on the
    # bootstrap bias-corrected estimate 2 est - mean(replicates) (``*_bc``), not on
    # the plug-in estimate, which can fall above it; report the two together.
    bc: dict[str, float] = {}
    for m in ("reliability", "ece"):
        i = METRICS.index(m)
        lo[i], hi[i] = max(0.0, 2 * est[i] - hi[i]), max(0.0, 2 * est[i] - lo[i])
        bc[m] = float(min(max(0.0, 2 * est[i] - np.nanmean(reps[:, i])), hi[i]))
    out: dict[str, float] = {"n": float(len(df)), "clusters": float(pd.unique(ev).size)}
    for i, m in enumerate(METRICS):
        out[m] = float(est[i])
        out[f"{m}_lo"] = float(lo[i])
        out[f"{m}_hi"] = float(hi[i])
    out["ece_bc"] = bc["ece"]
    out["reliability_bc"] = bc["reliability"]
    # 99.5% intervals for the regression coefficients and the Brier score, for reading
    # the primary estimands with a Bonferroni correction (10 tests at 5%). With 1,000
    # draws a 0.25% quantile rests on two or three replicates, so these use the
    # bootstrap standard error: estimate +/- 2.807 se.
    for m in ("slope", "intercept", "brier"):
        i = METRICS.index(m)
        se = float(np.nanstd(reps[:, i], ddof=1))
        out[f"{m}_lo995"] = float(est[i] - Z995 * se)
        out[f"{m}_hi995"] = float(est[i] + Z995 * se)
    # Share of replicates whose calibration regression separated (no finite slope).
    # Above a few percent the slope interval is conditional on non-separation and
    # should not be reported.
    out["slope_boot_fail"] = float(np.mean(~np.isfinite(reps[:, METRICS.index("slope")])))
    # Debiased reliability. Replicates use their own cluster labels (a cluster drawn
    # twice counts twice). In the bootstrap world the true reliability is the sample's
    # plug-in value, which the debiased replicates estimate without bias, so the pivot
    # is (replicate - plug-in): interval [rd - (q97.5 - rel), rd - (q2.5 - rel)]. A
    # plain percentile interval of the replicates sits too high (it covered 82-86% in
    # simulation, this one 97-99%).
    rd = debiased_reliability(p, y, edges, w, ev_codes)
    rd_reps = np.array(
        [
            debiased_reliability(
                p[d], y[d], edges, None if w is None else w[d], replicate_clusters(d, ev_codes)
            )
            for d in draws
        ]
    )
    rel = est[METRICS.index("reliability")]
    out["reliability_db"] = float(rd)
    out["reliability_db_lo"] = float(rd - (np.nanquantile(rd_reps, 0.975) - rel))
    out["reliability_db_hi"] = float(rd - (np.nanquantile(rd_reps, 0.025) - rel))
    rd_se = float(np.nanstd(rd_reps, ddof=1))
    out["reliability_db_lo995"] = float(rd - Z995 * rd_se)
    out["reliability_db_hi995"] = float(rd + Z995 * rd_se)
    # Noise floor: the ECE expected from sampling noise alone if every price were
    # the true probability (outcomes drawn as independent Bernoulli(p)), and the
    # share of such draws with an ECE at least as large as the one observed.
    rng = np.random.default_rng(cfg.seed + 7919 + salt)
    null = np.array(
        [
            decompose(p, (rng.uniform(size=p.size) < p).astype(np.float64), edges, w).ece
            for _ in range(cfg.n_null)
        ]
    )
    out["ece_null"] = float(null.mean())
    out["ece_null_p"] = float((1 + (null >= est[METRICS.index("ece")]).sum()) / (1 + null.size))
    # cluster-robust (sandwich) CI for the slope as a cross-check on the bootstrap
    if 0 < y.mean() < 1:
        f = fit_logistic(logit(p), y, w, clusters=ev)
        out["slope_se_cluster"] = float(f.se[1]) if f.converged else np.nan
        out["intercept_se_cluster"] = float(f.se[0]) if f.converged else np.nan
    return out


def summarise_by(
    df: pd.DataFrame,
    by: list[str],
    cfg: AnalysisConfig,
    *,
    min_events: int | None = None,
    weight: str | None = None,
) -> pd.DataFrame:
    rows = []
    floor = cfg.min_events if min_events is None else min_events
    for i, (key, g) in enumerate(df.groupby(by, sort=True, observed=True)):
        if g["cluster_id"].nunique() < floor:
            continue
        key_t = key if isinstance(key, tuple) else (key,)
        stats = summarise(g, cfg, salt=i, weight=weight)
        rows.append({**dict(zip(by, key_t, strict=True)), **stats})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Reliability diagram data


def reliability_table(df: pd.DataFrame, edges: FloatArray) -> pd.DataFrame:
    """Per bin: count, events, mean price, observed frequency and a 95% Wilson
    interval computed with the design-effect-adjusted effective sample size, so
    a bin full of outcomes from a few big multi-outcome events gets a wide band."""
    p = df["price"].to_numpy(dtype=np.float64)
    y = df["outcome"].to_numpy(dtype=np.float64)
    ev = df["cluster_id"].to_numpy()
    k = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for j in range(len(edges) - 1):
        m = k == j
        n = int(m.sum())
        if n == 0:
            rows.append({"bin": j, "lo_edge": edges[j], "hi_edge": edges[j + 1], "n": 0})
            continue
        ne = effective_n(y[m], ev[m])
        freq = float(y[m].mean())
        lo, hi = wilson_interval(freq * ne, ne)
        rows.append(
            {
                "bin": j,
                "lo_edge": float(edges[j]),
                "hi_edge": float(edges[j + 1]),
                "n": n,
                "clusters": int(pd.unique(ev[m]).size),
                "n_eff": ne,
                "mean_price": float(p[m].mean()),
                "freq": freq,
                "freq_lo": lo,
                "freq_hi": hi,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Favourite-longshot bias


def symmetrise(df: pd.DataFrame) -> pd.DataFrame:
    """Each contract twice: as priced, and as its complement (1 - p, 1 - y).

    On the symmetrised sample the calibration intercept is zero by construction,
    so the slope isolates the favourite-longshot pattern from any preference for
    the YES side. Both copies share the event id, so the bootstrap keeps them
    together.
    """
    comp = df.copy()
    comp["price"] = 1.0 - comp["price"]
    comp["outcome"] = 1.0 - comp["outcome"]
    comp["side"] = "complement"
    base = df.copy()
    base["side"] = "as listed"
    return pd.concat([base, comp], ignore_index=True)


def longshot_returns(df: pd.DataFrame, edges: FloatArray, cfg: AnalysisConfig) -> pd.DataFrame:
    """Average return on a dollar spent buying contracts priced in each bin,
    before fees: ``mean(y) / mean(p) - 1`` with a cluster-bootstrap interval.

    Run on the symmetrised sample this covers both buying YES at p and buying
    NO at 1 - p. A favourite-longshot bias shows up as returns that fall the
    cheaper the contract.
    """
    p = df["price"].to_numpy(dtype=np.float64)
    y = df["outcome"].to_numpy(dtype=np.float64)
    ev = df["cluster_id"].to_numpy()
    K = len(edges) - 1
    k = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, K - 1)

    def stat(idx: IntArray) -> FloatArray:
        kk = k[idx]
        n = np.bincount(kk, minlength=K).astype(float)
        sp = np.bincount(kk, weights=p[idx], minlength=K)
        sy = np.bincount(kk, weights=y[idx], minlength=K)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, sy / sp - 1.0, np.nan)

    est = stat(np.arange(p.size))
    reps = np.array([stat(d) for d in _draws(ev, cfg, salt=17)])
    lo = np.nanquantile(reps, 0.025, axis=0)
    hi = np.nanquantile(reps, 0.975, axis=0)
    n = np.bincount(k, minlength=K)
    events = [int(pd.unique(ev[k == j]).size) for j in range(K)]
    mean_p = [float(p[k == j].mean()) if n[j] else np.nan for j in range(K)]
    return pd.DataFrame(
        {
            "lo_edge": edges[:-1],
            "hi_edge": edges[1:],
            "n": n,
            "clusters": events,
            "mean_price": mean_p,
            "return": est,
            "return_lo": lo,
            "return_hi": hi,
        }
    )


# --------------------------------------------------------------------------
# Paired comparisons


def paired_differences(
    df: pd.DataFrame,
    key: str,
    levels: Iterable[str],
    base: str,
    cfg: AnalysisConfig,
    metrics: tuple[str, ...] = ("brier", "ece", "log_loss", "slope"),
    *,
    balanced: bool = True,
) -> pd.DataFrame:
    """Metric(level) - metric(base) on the same contracts, paired by event.

    With ``balanced`` (the default) ``df`` must be balanced: every contract appears
    once per level of ``key`` (e.g. every horizon in the panel). With
    ``balanced=False`` levels may cover different contracts (``nested_panel``); the
    pairing is then by cluster: each replicate draws clusters once and every level
    is recomputed on its own rows from those clusters. The bootstrap resamples
    clusters and recomputes every level on the same draw.
    """
    if not balanced:
        return _paired_unbalanced(df, key, list(levels), base, cfg, metrics)
    edges = bin_edges(cfg.n_bins)
    wide = {
        lv: df[df[key] == lv].sort_values(["venue", "contract_id"]).reset_index(drop=True)
        for lv in [base, *levels]
    }
    ref = wide[base]
    for lv, w in wide.items():
        if not (w["contract_id"].to_numpy() == ref["contract_id"].to_numpy()).all():
            raise ValueError(f"paired_differences: level {lv} is not aligned with {base}")
    ev = ref["cluster_id"].to_numpy()
    y = ref["outcome"].to_numpy(dtype=np.float64)
    prices = {lv: w["price"].to_numpy(dtype=np.float64) for lv, w in wide.items()}
    idx_m = [METRICS.index(m) for m in metrics]

    def vec(idx: IntArray) -> FloatArray:
        out = []
        for lv in [base, *levels]:
            out.append(metric_vector(prices[lv][idx], y[idx], edges, cfg.log_clip)[idx_m])
        return np.concatenate(out)

    est = vec(np.arange(y.size))
    reps = np.array([vec(d) for d in _draws(ev, cfg, salt=29)])
    rows = _paired_rows(key, [base, *levels], metrics, est, reps)
    out = pd.DataFrame(rows)
    out.attrs["n"] = int(y.size)
    out.attrs["clusters"] = int(pd.unique(ev).size)
    return out


BIASED_METRICS = ("ece", "reliability")  # plug-in binned metrics, biased upwards


def _paired_rows(
    key: str,
    all_lv: list[str],
    metrics: tuple[str, ...],
    est: FloatArray,
    reps: NDArray[np.float64],
) -> list[dict[str, Any]]:
    """Rows of ``paired_differences`` from the estimate vector and its replicates
    (level-major, metric-minor; the base level first). Percentile intervals, except
    basic intervals for the upward-biased binned metrics (as in ``summarise``), for
    both the levels and the differences: a level with fewer contracts or more
    mid-range prices has more bias, so even a difference of plug-in ECEs is not
    centred on the true difference."""

    def interval(e: float, r: FloatArray, basic: bool) -> tuple[float, float]:
        q_lo, q_hi = np.nanquantile(r, 0.025), np.nanquantile(r, 0.975)
        if basic:
            return float(2 * e - q_hi), float(2 * e - q_lo)
        return float(q_lo), float(q_hi)

    m = len(metrics)
    base = all_lv[0]
    rows: list[dict[str, Any]] = []
    for i, lv in enumerate(all_lv):
        for j, met in enumerate(metrics):
            col = i * m + j
            basic = met in BIASED_METRICS
            lo, hi = interval(est[col], reps[:, col], basic)
            if basic:
                lo, hi = max(0.0, lo), max(0.0, hi)
            row: dict[str, Any] = {key: lv, "metric": met, "value": est[col], "lo": lo, "hi": hi}
            if lv != base:
                d_est = est[col] - est[j]
                row["diff_vs_base"] = d_est
                dr = reps[:, col] - reps[:, j]
                row["diff_lo"], row["diff_hi"] = interval(d_est, dr, basic)
                dse = float(np.nanstd(dr, ddof=1))
                row["diff_lo995"] = float(d_est - Z995 * dse)
                row["diff_hi995"] = float(d_est + Z995 * dse)
            rows.append(row)
    return rows


def _paired_unbalanced(
    df: pd.DataFrame,
    key: str,
    levels: list[str],
    base: str,
    cfg: AnalysisConfig,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    edges = bin_edges(cfg.n_bins)
    all_lv = [base, *levels]
    _, codes = np.unique(df["cluster_id"].to_numpy(), return_inverse=True)
    G = int(codes.max()) + 1 if codes.size else 0
    idx_m = [METRICS.index(m) for m in metrics]
    per: dict[str, tuple[FloatArray, FloatArray, list[IntArray]]] = {}
    for lv in all_lv:
        m = (df[key] == lv).to_numpy()
        rows = np.flatnonzero(m)
        c = codes[m]
        order = np.argsort(c, kind="stable")
        starts = np.searchsorted(c[order], np.arange(G + 1))
        groups = [order[starts[g] : starts[g + 1]] for g in range(G)]
        p = df["price"].to_numpy(dtype=np.float64)[rows]
        y = df["outcome"].to_numpy(dtype=np.float64)[rows]
        per[lv] = (p, y, groups)

    def vec(pick: IntArray | None) -> FloatArray:
        out = []
        for lv in all_lv:
            p, y, groups = per[lv]
            if pick is None:
                ii = np.arange(p.size)
            else:
                parts = [groups[g] for g in pick]
                ii = np.concatenate(parts) if parts else np.zeros(0, np.int64)
            if ii.size == 0:
                out.append(np.full(len(metrics), np.nan))
                continue
            out.append(metric_vector(p[ii], y[ii], edges, cfg.log_clip)[idx_m])
        return np.concatenate(out)

    est = vec(None)
    rng = np.random.default_rng(cfg.seed + 29)
    reps = np.array([vec(rng.integers(0, G, size=G)) for _ in range(cfg.n_boot)])
    rows_out = _paired_rows(key, all_lv, metrics, est, reps)
    for row in rows_out:
        sel = df[key] == row[key]
        row["n"] = int(sel.sum())
        row["level_clusters"] = int(df.loc[sel, "cluster_id"].nunique())
    out = pd.DataFrame(rows_out)
    out.attrs["n"] = int(df[["venue", "contract_id"]].drop_duplicates().shape[0])
    out.attrs["clusters"] = G
    return out


def balanced_panel(df: pd.DataFrame, horizons: tuple[str, ...]) -> pd.DataFrame:
    """Contracts that have a usable snapshot at every one of ``horizons``.

    Conditions the long-horizon forecasts on the market still being open (and live)
    at the short horizons, which is outcome-dependent for markets that can resolve
    early; see ``nested_panel`` for the ex-ante alternative."""
    sub = df[df["horizon"].isin(horizons)]
    counts = sub.groupby(["venue", "contract_id"])["horizon"].nunique()
    full = counts[counts == len(horizons)].index
    keyed = sub.set_index(["venue", "contract_id"])
    return keyed.loc[keyed.index.isin(full)].reset_index()


def nested_panel(df: pd.DataFrame, horizons: tuple[str, ...]) -> pd.DataFrame:
    """Contracts usable at the longest horizon ``horizons[0]``, with every later
    horizon at which they are still in ``df`` (open, priced and live then).

    Unlike ``balanced_panel`` this never conditions an earlier forecast on what
    happened later. Requiring a live price at 1h before the scheduled time keeps
    only markets that had not resolved early, and for "by date" markets an early
    resolution is mostly a YES, so the balanced panel's long-horizon prices are
    scored on a population enriched for NO outcomes. Here the population is fixed by
    an ex-ante criterion (priced at the first horizon) and shrinks at later horizons
    only by information known at each of them (the market had already closed, or
    its price had gone stale). Pair horizons with
    ``paired_differences(..., balanced=False)``.
    """
    sub = df[df["horizon"].isin(horizons)]
    first = sub.loc[sub["horizon"] == horizons[0], ["venue", "contract_id"]].drop_duplicates()
    return sub.merge(first, on=["venue", "contract_id"], how="inner")


# --------------------------------------------------------------------------
# Bootstrap clusters


CLUSTER_KEYS = ("series_day", "category_day", "day", "series")


def cluster_keys(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` with the columns ``recluster`` can coarsen by: the scheduled UTC day,
    and that day within the series (or, without a series, the cluster) and within
    the category."""
    out = df.copy()
    day = pd.to_datetime(out["scheduled_ts"].astype("int64"), unit="s", utc=True).dt.strftime(
        "%Y-%m-%d"
    )
    series = out["series_id"].astype("string").fillna(out["cluster_id"].astype("string"))
    venue = out["venue"].astype("string") if "venue" in out else ""
    out["day"] = day
    out["series"] = venue + "|" + series
    out["series_day"] = out["series"] + "|" + day
    out["category_day"] = out["category"].astype("string") + "|" + day
    return out


def recluster(df: pd.DataFrame, by: Iterable[str]) -> pd.Series:
    """Bootstrap clusters that nest ``cluster_id`` and each column in ``by`` (from
    ``cluster_keys``), as string labels on ``df``'s index.

    Each game/event cluster takes a single value of every key: the one of its
    earliest-scheduled row (e.g. a multi-day "by date" ladder joins the category-day
    of its first deadline). Taking every row's own value instead would chain: one
    event with deadlines on many days links all of those days, and through other
    such events a whole category's year becomes one cluster (it put 75-90% of
    politics and world contracts in a single cluster)."""
    by = tuple(by)
    unknown = set(by) - set(CLUSTER_KEYS)
    if unknown:
        raise ValueError(f"unknown cluster keys {sorted(unknown)}; choose from {CLUSTER_KEYS}")
    if not by:
        return df["cluster_id"].astype("string")
    k = cluster_keys(df)
    first = k.sort_values("scheduled_ts", kind="stable").groupby("cluster_id", sort=False).head(1)
    first = first.set_index("cluster_id")
    base = k["cluster_id"]
    keys = [base.map(first[c]).to_numpy() for c in by]
    lab = coarsen_clusters(base.astype(str).to_numpy(), *keys)
    return pd.Series([f"cc{i}" for i in lab], index=df.index, dtype="string")


def clustering_sensitivity(
    df: pd.DataFrame,
    cfg: AnalysisConfig,
    schemes: dict[str, tuple[str, ...]] | None = None,
    metrics: tuple[str, ...] = ("brier", "reliability_db", "ece", "slope", "intercept"),
) -> pd.DataFrame:
    """The same subset summarised under different bootstrap clusterings: the number
    of clusters and the 95% interval width of each metric, and the width relative to
    the game/event clustering. ``schemes`` maps a label to ``recluster`` keys; the
    special key "event" resamples venue events instead of games."""
    if schemes is None:
        schemes = {
            "event": ("event",),
            "game/event": (),
            "+ series-day": ("series_day",),
            "+ category-day": ("category_day",),
            "+ day": ("day",),
        }
    rows = []
    for label, keys in schemes.items():
        event = keys == ("event",)
        cl = df["event_id"].astype("string") if event else recluster(df, keys)
        s = summarise(df.assign(cluster_id=cl), cfg)
        row: dict[str, Any] = {"clustering": label, "clusters": int(s["clusters"])}
        for m in metrics:
            row[m] = s[m]
            row[f"{m}_lo"] = s[f"{m}_lo"]
            row[f"{m}_hi"] = s[f"{m}_hi"]
            row[f"{m}_width"] = s[f"{m}_hi"] - s[f"{m}_lo"]
        rows.append(row)
    out = pd.DataFrame(rows)
    base = out.loc[out["clustering"] == "game/event"]
    if len(base):
        for m in metrics:
            out[f"{m}_width_ratio"] = out[f"{m}_width"] / float(base[f"{m}_width"].iloc[0])
    return out


# --------------------------------------------------------------------------
# Liquidity buckets


def volume_bucket(v: pd.Series, edges: tuple[float, ...]) -> pd.Series:
    labels = []
    bounds = [*edges, np.inf]
    for lo, hi in pairwise(bounds):
        labels.append(f"{_money(lo)}+" if not np.isfinite(hi) else f"{_money(lo)}-{_money(hi)}")
    cut = pd.cut(v.astype(float), bins=bounds, labels=labels, right=False)
    return cut.astype("string")


def _money(x: float) -> str:
    if x >= 1e6:
        return f"${x / 1e6:g}M"
    if x >= 1e3:
        return f"${x / 1e3:g}k"
    return f"${x:g}"


def activity_bucket(df: pd.DataFrame) -> pd.Series:
    """Ex-ante activity in the 24h before the snapshot, comparable across venues:
    no trade in 24h / some / busy. Uses only information available at t_snap,
    unlike lifetime volume, which includes trading after the snapshot."""
    age = df["last_trade_age_s"].astype(float)
    n_tr = df["n_trades_24h"].astype("Float64")
    vol = df["volume_24h"].astype(float)
    busy = (n_tr >= 50).fillna(False) | (vol >= 1000).fillna(False)
    traded = (age <= 86400).fillna(False) | (n_tr >= 1).fillna(False) | (vol > 0).fillna(False)
    out = pd.Series(np.where(busy, "busy", np.where(traded, "traded", "no trade")), index=df.index)
    return out.astype("string")


# --------------------------------------------------------------------------
# Out-of-sample recalibration


def recalibration(df: pd.DataFrame, cfg: AnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward Platt and isotonic recalibration, fitted per venue and horizon.

    Returns (scores, refits). Scores compare the raw price with each recalibrated
    forecast on exactly the same test rows, with paired cluster-bootstrap
    intervals on the improvement (positive = recalibration helps).
    """
    rows = []
    refit_rows = []
    for i, ((venue, horizon), g) in enumerate(df.groupby(["venue", "horizon"], sort=True)):
        g = g.reset_index(drop=True)
        res = walk_forward(
            g.assign(event_id=g["cluster_id"]),
            refit_every_s=cfg.refit_every_days * 86400,
            min_train=cfg.min_train,
            min_train_events=cfg.min_train_events,
        )
        pr = res.predictions
        if pr.empty:
            continue
        refit_rows.append(res.refits.assign(venue=str(venue), horizon=str(horizon)))
        test = g.loc[pr.index]
        y = test["outcome"].to_numpy(dtype=np.float64)
        ev = test["cluster_id"].to_numpy()
        forecasts = {m: pr[m].to_numpy(dtype=np.float64) for m in ("raw", "platt", "isotonic")}
        c = cfg.log_clip

        def scores(
            idx: IntArray,
            forecasts: dict[str, FloatArray] = forecasts,
            y: FloatArray = y,
            c: float = c,
        ) -> FloatArray:
            yy = y[idx]
            out = []
            for m in ("raw", "platt", "isotonic"):
                q = forecasts[m][idx]
                qc = np.clip(q, c, 1 - c)
                out.append(np.mean((q - yy) ** 2))
                out.append(np.mean(-(yy * np.log(qc) + (1 - yy) * np.log1p(-qc))))
            return np.array(out)

        est = scores(np.arange(y.size))
        reps = np.array([scores(d) for d in _draws(ev, cfg, salt=100 + i)])
        for mi, m in enumerate(("platt", "isotonic"), start=1):
            for si, s in enumerate(("brier", "log_loss")):
                raw_col, m_col = si, 2 * mi + si
                gain = reps[:, raw_col] - reps[:, m_col]
                rows.append(
                    {
                        "venue": venue,
                        "horizon": horizon,
                        "method": m,
                        "score": s,
                        "n_test": int(y.size),
                        "events_test": int(pd.unique(ev).size),
                        "raw": est[raw_col],
                        "recalibrated": est[m_col],
                        "gain": est[raw_col] - est[m_col],
                        "gain_lo": np.nanquantile(gain, 0.025),
                        "gain_hi": np.nanquantile(gain, 0.975),
                        "gain_lo995": est[raw_col]
                        - est[m_col]
                        - Z995 * float(np.nanstd(gain, ddof=1)),
                        "gain_hi995": est[raw_col]
                        - est[m_col]
                        + Z995 * float(np.nanstd(gain, ddof=1)),
                        "rel_gain_pct": 100 * (est[raw_col] - est[m_col]) / est[raw_col],
                    }
                )
    refits = pd.concat(refit_rows, ignore_index=True) if refit_rows else pd.DataFrame()
    return pd.DataFrame(rows), refits


# --------------------------------------------------------------------------
# Convenience


def clip_prob(p: FloatArray) -> FloatArray:
    return np.clip(p, EPS, 1 - EPS)


StatFn = Callable[[IntArray], FloatArray]
