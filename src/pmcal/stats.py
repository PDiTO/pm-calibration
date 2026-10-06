"""Calibration statistics with event-clustered uncertainty.

Everything here takes forecasts ``p`` in [0, 1], binary outcomes ``y`` and a
cluster label per row. The cluster is the event: the outcomes of one
multi-outcome event are mutually exclusive (exactly one resolves YES), so
treating them as independent rows would understate every standard error.
Confidence intervals therefore come from a cluster bootstrap that resamples
whole events with replacement.

Definitions (Murphy, 1973; Bröcker, 2009):

- Brier score ``BS = mean((p - y)^2)``.
- With forecasts grouped into bins k with n_k rows, mean forecast pbar_k and
  observed frequency obar_k, and overall base rate obar:
  ``REL = sum_k n_k (pbar_k - obar_k)^2 / N`` (reliability, lower is better),
  ``RES = sum_k n_k (obar_k - obar)^2 / N`` (resolution, higher is better),
  ``UNC = obar (1 - obar)`` (uncertainty).
  Binning adds two within-bin terms, so ``BS = REL - RES + UNC + WBV - WBC``
  with ``WBV = sum_k sum_{i in k} (p_i - pbar_k)^2 / N`` (within-bin variance)
  and ``WBC = 2 sum_k sum_{i in k} (p_i - pbar_k)(y_i - obar_k) / N``
  (within-bin covariance). Both are returned, so the identity holds exactly.
- ECE ``= sum_k n_k |pbar_k - obar_k| / N``.
- Both REL and ECE are biased upwards: even perfectly calibrated prices give a
  positive value from sampling noise in each bin's observed frequency. Because
  ``E[(obar_k - pbar_k)^2] = (true gap_k)^2 + Var(obar_k)``, subtracting a
  cluster-robust estimate of ``Var(obar_k)`` from each bin's term gives an
  approximately unbiased reliability (``debiased_reliability``; Ferro and Fricker,
  2012, with clustered rather than independent outcomes). It can be negative when
  prices are calibrated. There is no such exact correction for the absolute value
  in the ECE.
- Calibration regression: ``logit P(y = 1) = a + b logit(p)``. ``b > 1`` means
  prices are too close to 0.5 (underconfident), ``b < 1`` too extreme.
  A favourite-longshot bias, where cheap contracts are overpriced and expensive
  ones underpriced, shows up as ``b > 1``: the true probability is further from
  0.5 than the price.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import special, stats

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

EPS = 1e-4  # clip probabilities into [EPS, 1 - EPS] before taking logs or logits
MAX_COEF = 50.0  # |a| or |b| beyond this in a logistic fit is taken as separation


def _clip(p: FloatArray) -> FloatArray:
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)


def logit(p: FloatArray) -> FloatArray:
    q = _clip(p)
    return np.log(q) - np.log1p(-q)


def brier(p: FloatArray, y: FloatArray, w: FloatArray | None = None) -> float:
    return float(np.average((np.asarray(p) - np.asarray(y)) ** 2, weights=w))


def log_loss(p: FloatArray, y: FloatArray, w: FloatArray | None = None) -> float:
    q = _clip(p)
    ll = -(y * np.log(q) + (1.0 - y) * np.log1p(-q))
    return float(np.average(ll, weights=w))


# --------------------------------------------------------------------------
# Binning


def bin_edges(n_bins: int = 10, scheme: str = "uniform") -> FloatArray:
    """Fixed bin edges on [0, 1]. ``"fine_tails"`` splits the outer deciles further,
    since that's where favourite-longshot effects live."""
    if scheme == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if scheme == "fine_tails":
        inner = np.linspace(0.1, 0.9, 9)
        return np.unique(np.concatenate([[0.0, 0.02, 0.05], inner, [0.95, 0.98, 1.0]]))
    raise ValueError(f"unknown bin scheme {scheme!r}")


def assign_bins(p: FloatArray, edges: FloatArray) -> IntArray:
    """Bin index for each p, with p == 1 put in the last bin."""
    k = np.searchsorted(edges, p, side="right") - 1
    return np.clip(k, 0, len(edges) - 2).astype(np.int64)


@dataclass(frozen=True)
class BinTable:
    edges: FloatArray
    n: FloatArray  # (weighted) count per bin
    mean_p: FloatArray  # mean forecast per bin
    freq: FloatArray  # observed frequency per bin
    n_clusters: IntArray  # distinct events per bin


def bin_table(
    p: FloatArray,
    y: FloatArray,
    edges: FloatArray,
    w: FloatArray | None = None,
    clusters: Sequence[object] | NDArray[np.generic] | None = None,
) -> BinTable:
    k = assign_bins(p, edges)
    K = len(edges) - 1
    ww = np.ones_like(p) if w is None else np.asarray(w, dtype=np.float64)
    n = np.bincount(k, weights=ww, minlength=K)
    sp = np.bincount(k, weights=ww * p, minlength=K)
    sy = np.bincount(k, weights=ww * y, minlength=K)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_p = np.where(n > 0, sp / n, np.nan)
        freq = np.where(n > 0, sy / n, np.nan)
    if clusters is None:
        nc = np.bincount(k, minlength=K).astype(np.int64)
    else:
        c = np.asarray(clusters)
        nc = np.array([np.unique(c[k == j]).size for j in range(K)], dtype=np.int64)
    return BinTable(edges=edges, n=n, mean_p=mean_p, freq=freq, n_clusters=nc)


@dataclass(frozen=True)
class Decomposition:
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    within_bin_variance: float
    within_bin_covariance: float
    ece: float
    base_rate: float
    n: float

    def check(self) -> float:
        """Residual of the identity BS = REL - RES + UNC + WBV - WBC (should be ~0)."""
        return self.brier - (
            self.reliability
            - self.resolution
            + self.uncertainty
            + self.within_bin_variance
            - self.within_bin_covariance
        )


def decompose(
    p: FloatArray, y: FloatArray, edges: FloatArray, w: FloatArray | None = None
) -> Decomposition:
    """Murphy decomposition of the Brier score over fixed bins, exact via the
    within-bin terms, plus the binned ECE."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ww = np.ones_like(p) if w is None else np.asarray(w, dtype=np.float64)
    N = float(ww.sum())
    k = assign_bins(p, edges)
    K = len(edges) - 1
    n = np.bincount(k, weights=ww, minlength=K)
    with np.errstate(invalid="ignore", divide="ignore"):
        pbar = np.where(n > 0, np.bincount(k, weights=ww * p, minlength=K) / n, 0.0)
        obar_k = np.where(n > 0, np.bincount(k, weights=ww * y, minlength=K) / n, 0.0)
    obar = float((ww * y).sum() / N)
    rel = float((n * (pbar - obar_k) ** 2).sum() / N)
    res = float((n * (obar_k - obar) ** 2).sum() / N)
    unc = obar * (1.0 - obar)
    dp = p - pbar[k]
    wbv = float((ww * dp**2).sum() / N)
    wbc = float(2.0 * (ww * dp * (y - obar_k[k])).sum() / N)
    ece = float((n * np.abs(pbar - obar_k)).sum() / N)
    bs = float((ww * (p - y) ** 2).sum() / N)
    return Decomposition(bs, rel, res, unc, wbv, wbc, ece, obar, N)


def debiased_reliability(
    p: FloatArray,
    y: FloatArray,
    edges: FloatArray,
    w: FloatArray | None = None,
    clusters: Sequence[object] | NDArray[np.generic] | None = None,
) -> float:
    """Reliability with the sampling noise of each bin's frequency subtracted.

    ``REL_d = sum_k (n_k / N) [(obar_k - pbar_k)^2 - V_k]`` where ``V_k`` is a
    cluster-robust (CR1) estimate of the variance of the bin's mean residual
    ``mean(y - p)``. Approximately unbiased for the population reliability whatever
    its size, unlike the plug-in REL, so its bootstrap distribution is centred on
    the estimate and percentile intervals are usable. Negative values mean the
    observed gaps are smaller than noise alone would produce. Bins with a single
    cluster cannot estimate their own noise and contribute zero.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ww = np.ones_like(p) if w is None else np.asarray(w, dtype=np.float64)
    if clusters is None:
        inv = np.arange(p.size)
    else:
        _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    N = float(ww.sum())
    k = assign_bins(p, edges)
    out = 0.0
    for j in np.unique(k):
        m = k == j
        wj = ww[m]
        nj = float(wj.sum())
        if nj <= 0:
            continue
        r = y[m] - p[m]
        gap = float((wj * r).sum() / nj)
        _, cj = np.unique(inv[m], return_inverse=True)
        G = int(cj.max()) + 1
        if G < 2:
            continue
        s = np.bincount(cj, weights=wj * (r - gap), minlength=G)
        var = float((s**2).sum() / nj**2) * G / (G - 1)
        out += nj / N * (gap**2 - var)
    return out


# --------------------------------------------------------------------------
# Logistic calibration regression


@dataclass(frozen=True)
class CalibrationFit:
    intercept: float  # a
    slope: float  # b
    cov: FloatArray  # 2x2 covariance of (a, b), cluster-robust if clusters given
    n: int
    n_clusters: int
    # False when Newton's method did not settle or the coefficients ran off to
    # infinity (complete or quasi-complete separation, common in small bootstrap
    # replicates). Estimates from such a fit are meaningless.
    converged: bool = True

    @property
    def se(self) -> FloatArray:
        return np.sqrt(np.diag(self.cov))

    def predict(self, p: FloatArray) -> FloatArray:
        return np.asarray(special.expit(self.intercept + self.slope * logit(p)), dtype=np.float64)


def fit_logistic(
    x: FloatArray,
    y: FloatArray,
    w: FloatArray | None = None,
    *,
    clusters: Sequence[object] | NDArray[np.generic] | None = None,
    l2: float = 0.0,
) -> CalibrationFit:
    """Logistic regression of y on [1, x] by Newton's method.

    The covariance is the cluster-robust sandwich ``H^-1 (sum_g s_g s_g^T) H^-1``
    with the usual G/(G-1) small-sample factor when clusters are given, the
    inverse Hessian otherwise. ``l2`` adds a small ridge on the slope's
    deviation from 1 (prior "prices are calibrated"), used only to keep the
    walk-forward fit stable on tiny training sets.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ww = np.ones_like(x) if w is None else np.asarray(w, dtype=np.float64)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    prior = np.array([0.0, 1.0])
    R = np.diag([0.0, l2])
    converged = False
    for _ in range(100):
        mu = special.expit(X @ beta)
        g = X.T @ (ww * (y - mu)) - R @ (beta - prior)
        H = (X * (ww * mu * (1 - mu))[:, None]).T @ X + R
        step = np.linalg.solve(H + 1e-12 * np.eye(2), g)
        beta = beta + step
        if not np.all(np.isfinite(beta)):
            break
        if np.max(np.abs(step)) < 1e-10:
            converged = True
            break
    # Under separation the likelihood keeps rising as |b| grows; Newton then either
    # fails to settle or settles (on a flat likelihood) at absurd coefficients.
    converged = converged and bool(np.all(np.abs(beta) < MAX_COEF))
    if not np.all(np.isfinite(beta)):
        nan2 = np.full((2, 2), np.nan)
        return CalibrationFit(math.nan, math.nan, nan2, len(x), 0, converged=False)
    mu = special.expit(X @ beta)
    H = (X * (ww * mu * (1 - mu))[:, None]).T @ X + R
    Hinv = np.linalg.inv(H)
    scores = X * (ww * (y - mu))[:, None]
    if clusters is not None:
        c = np.asarray(clusters)
        _, inv = np.unique(c, return_inverse=True)
        G = int(inv.max()) + 1
        S = np.zeros((G, 2))
        np.add.at(S, inv, scores)
        meat = S.T @ S * (G / max(G - 1, 1))
        cov = Hinv @ meat @ Hinv
    else:
        G = len(x)
        cov = Hinv
    return CalibrationFit(float(beta[0]), float(beta[1]), cov, len(x), G, converged)


def calibration_regression(
    p: FloatArray,
    y: FloatArray,
    clusters: Sequence[object] | NDArray[np.generic] | None = None,
    w: FloatArray | None = None,
) -> CalibrationFit:
    return fit_logistic(logit(p), y, w, clusters=clusters)


# --------------------------------------------------------------------------
# Cluster bootstrap


@dataclass(frozen=True)
class Interval:
    estimate: float
    lo: float
    hi: float

    def fmt(self, digits: int = 3, scale: float = 1.0) -> str:
        e, lo, hi = (v * scale for v in (self.estimate, self.lo, self.hi))
        return f"{e:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def cluster_index(clusters: Sequence[object] | NDArray[np.generic]) -> list[IntArray]:
    """Row indices grouped by cluster, in a stable order."""
    c = np.asarray(clusters)
    _, inv = np.unique(c, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    return [g.astype(np.int64) for g in np.split(order, bounds)]


def bootstrap_draws(
    groups: list[IntArray], n_boot: int, rng: np.random.Generator
) -> list[IntArray]:
    """Row indices for each bootstrap replicate: resample whole clusters."""
    G = len(groups)
    sizes = np.array([g.size for g in groups])
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, G, size=G)
        idx = np.concatenate([groups[i] for i in pick]) if sizes[pick].sum() else np.zeros(0, int)
        out.append(idx.astype(np.int64))
    return out


def replicate_clusters(idx: IntArray, clusters: Sequence[object] | NDArray[np.generic]) -> IntArray:
    """Cluster labels for the rows of one bootstrap replicate from ``bootstrap_draws``.

    A cluster drawn twice is two independent clusters in the replicate, so statistics
    that need cluster labels (cluster-robust variances) must not merge the copies.
    Each drawn cluster is a contiguous block of increasing row indices, so a new
    block starts wherever the original label changes or the row index does not
    increase.
    """
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64)
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    lab = inv[idx]
    new = np.ones(idx.size, dtype=bool)
    new[1:] = (lab[1:] != lab[:-1]) | (idx[1:] <= idx[:-1])
    return (np.cumsum(new) - 1).astype(np.int64)


def coarsen_clusters(*keys: Sequence[object] | NDArray[np.generic]) -> IntArray:
    """Clusters that nest every one of ``keys``: rows sharing a value of any key are
    in the same cluster (connected components). For example the game or event
    together with (category, scheduled day) gives clusters that keep an event's
    outcomes together and also put everything in one category resolving on the same
    day together, for common shocks such as one crypto price path or one news day.
    Missing values (None/NaN) in a key link nothing.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(keys[0])
    rows: list[IntArray] = []
    cols: list[IntArray] = []
    offset = 0
    for key in keys:
        s = pd.Series(np.asarray(key, dtype=object))
        if len(s) != n:
            raise ValueError("coarsen_clusters: keys differ in length")
        codes, uniq = pd.factorize(s, use_na_sentinel=True)
        ok = codes >= 0
        rows.append(np.flatnonzero(ok).astype(np.int64))
        cols.append(n + offset + codes[ok].astype(np.int64))
        offset += len(uniq)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    size = n + offset
    g = coo_matrix((np.ones(r.size), (r, c)), shape=(size, size))
    _, lab = connected_components(g, directed=False)
    _, out = np.unique(lab[:n], return_inverse=True)
    return out.astype(np.int64)


def cluster_bootstrap(
    stat: Callable[[IntArray], float | FloatArray],
    clusters: Sequence[object] | NDArray[np.generic],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    draws: list[IntArray] | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Percentile intervals for a (vector) statistic of the rows, resampling clusters.

    ``stat`` receives an index array into the original rows and returns a scalar
    or a 1-d array. Returns (estimate, lo, hi) arrays. Pass ``draws`` to reuse
    the same replicates across statistics, which makes differences between them
    paired.
    """
    groups = cluster_index(clusters)
    n = sum(g.size for g in groups)
    est = np.atleast_1d(np.asarray(stat(np.arange(n)), dtype=np.float64))
    if draws is None:
        draws = bootstrap_draws(groups, n_boot, np.random.default_rng(seed))
    reps = np.array([np.atleast_1d(np.asarray(stat(d), dtype=np.float64)) for d in draws])
    lo = np.nanquantile(reps, alpha / 2, axis=0)
    hi = np.nanquantile(reps, 1 - alpha / 2, axis=0)
    return est, lo, hi


# --------------------------------------------------------------------------
# Per-bin intervals that respect clustering without a full bootstrap


def wilson_interval(k: float, n: float, z: float = 1.959964) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def effective_n(y: FloatArray, clusters: Sequence[object] | NDArray[np.generic]) -> float:
    """Kish-style effective sample size of a mean of y under clustering:
    n / design effect, with the design effect estimated from the cluster sums."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 2:
        return float(n)
    groups = cluster_index(clusters)
    if len(groups) == n:
        return float(n)
    m = y.mean()
    var_iid = y.var(ddof=1) / n
    if var_iid <= 0:
        return float(n)
    sums = np.array([(y[g] - m).sum() for g in groups])
    G = len(groups)
    var_cl = (sums**2).sum() / n**2 * G / max(G - 1, 1)
    deff = max(var_cl / var_iid, 1e-9)
    return float(n / deff)


def binomial_test_two_sided(k: int, n: int, p: float) -> float:
    return float(stats.binomtest(k, n, p).pvalue)


__all__ = [
    "EPS",
    "BinTable",
    "CalibrationFit",
    "Decomposition",
    "Interval",
    "assign_bins",
    "bin_edges",
    "bin_table",
    "bootstrap_draws",
    "brier",
    "calibration_regression",
    "cluster_bootstrap",
    "cluster_index",
    "coarsen_clusters",
    "debiased_reliability",
    "decompose",
    "effective_n",
    "fit_logistic",
    "log_loss",
    "logit",
    "replicate_clusters",
    "wilson_interval",
]
