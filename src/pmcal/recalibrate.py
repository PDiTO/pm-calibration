"""Out-of-sample recalibration, walk-forward in information time.

The question is whether a simple map from price to probability, learned from
markets that have already resolved, improves forecasts on markets that have
not. Two maps:

- ``platt``: logit-linear, ``q = expit(a + b logit(p))`` (Platt scaling with the
  price's logit as the single feature).
- ``isotonic``: the monotone step function from pool-adjacent-violators on the
  price, linearly interpolated between block centres and shrunk towards the
  price where training data are thin.

No fit may use an outcome that was not public when the forecast was made. The
walk-forward refits every ``refit_every_s`` seconds at times T_0 < T_1 < ...
The model fitted at T_k sees only contracts with ``resolved_ts < T_k``, and is
applied only to snapshots with ``T_k <= t_snap < T_{k+1}``. So every training
outcome was known before every forecast it is applied to. Splitting on the
outcome's resolution date alone would not guarantee that for long horizons: a
30-day-ahead snapshot is taken a month before its own resolution, when outcomes
that resolved in between were still unknown.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import special

from pmcal.stats import fit_logistic, logit

FloatArray = NDArray[np.float64]


def pav(
    x: FloatArray, y: FloatArray, w: FloatArray | None = None
) -> tuple[FloatArray, NDArray[np.intp]]:
    """Weighted isotonic (non-decreasing) regression by pool-adjacent-violators.

    Returns the fitted value for each input row, in the input order, and the
    sort order used. Ties in x are pooled first so the fit is a function of x.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ww = np.ones_like(x) if w is None else np.asarray(w, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    xs, ys, ws = x[order], y[order], ww[order]
    # pool ties
    ux, start = np.unique(xs, return_index=True)
    wsum = np.add.reduceat(ws, start)
    ysum = np.add.reduceat(ws * ys, start)
    vals: list[float] = []
    wts: list[float] = []
    cnt: list[int] = []
    for i in range(ux.size):
        v, wt, c = ysum[i] / wsum[i], wsum[i], 1
        while vals and vals[-1] > v:
            pv, pw, pc = vals.pop(), wts.pop(), cnt.pop()
            v = (pv * pw + v * wt) / (pw + wt)
            wt += pw
            c += pc
        vals.append(v)
        wts.append(wt)
        cnt.append(c)
    fitted_u = np.repeat(np.array(vals), np.array(cnt))
    counts = np.diff(np.append(start, xs.size))
    fitted_sorted = np.repeat(fitted_u, counts)
    out = np.empty_like(fitted_sorted)
    out[order] = fitted_sorted
    return out, order


@dataclass(frozen=True)
class IsotonicMap:
    """Piecewise-linear interpolation through the PAV block centres."""

    knots_x: FloatArray
    knots_y: FloatArray
    knots_w: FloatArray
    shrink: float  # pseudo-count pulling each block towards its mean price

    def __call__(self, p: FloatArray) -> FloatArray:
        p = np.asarray(p, dtype=np.float64)
        return np.asarray(np.interp(p, self.knots_x, self.knots_y), dtype=np.float64)


def fit_isotonic(p: FloatArray, y: FloatArray, *, shrink: float = 10.0) -> IsotonicMap:
    """Isotonic map from price to probability.

    Each PAV block's value is shrunk towards the block's mean price with
    ``shrink`` pseudo-observations, so a block of three contracts that all
    resolved NO does not map a 20% price to 0%. The block centres (mean price)
    are the interpolation knots, with (0, 0) and (1, 1) added at the ends.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fitted, order = pav(p, y)
    fs = fitted[order]
    ps = p[order]
    # block boundaries where the fitted value changes
    cut = np.flatnonzero(np.diff(fs) != 0) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [fs.size]])
    kx, ky, kw = [], [], []
    for s, e in zip(starts, ends, strict=True):
        n = e - s
        mp = float(ps[s:e].mean())
        val = float(fs[s])
        kx.append(mp)
        ky.append((n * val + shrink * mp) / (n + shrink))
        kw.append(float(n))
    kx_a = np.array([0.0, *kx, 1.0])
    ky_a = np.maximum.accumulate(np.array([0.0, *ky, 1.0]))  # shrinkage can break monotonicity
    kw_a = np.array([0.0, *kw, 0.0])
    # merge duplicate x knots (possible at the ends)
    ux, idx = np.unique(kx_a, return_index=True)
    return IsotonicMap(ux, ky_a[idx], kw_a[idx], shrink)


Recalibrator = Callable[[FloatArray], FloatArray]


def fit_platt(p: FloatArray, y: FloatArray, *, l2: float = 1.0) -> Recalibrator:
    fit = fit_logistic(logit(p), y, l2=l2)

    def apply(q: FloatArray) -> FloatArray:
        return np.asarray(special.expit(fit.intercept + fit.slope * logit(q)), dtype=np.float64)

    return apply


FITTERS: dict[str, Callable[[FloatArray, FloatArray], Recalibrator]] = {
    "platt": lambda p, y: fit_platt(p, y),
    "isotonic": lambda p, y: fit_isotonic(p, y),
}


@dataclass(frozen=True)
class WalkForwardResult:
    predictions: pd.DataFrame  # one row per test snapshot: raw and recalibrated forecasts
    refits: pd.DataFrame  # one row per refit: time, training size


def walk_forward(
    df: pd.DataFrame,
    *,
    methods: tuple[str, ...] = ("platt", "isotonic"),
    refit_every_s: int = 7 * 86400,
    min_train: int = 500,
    min_train_events: int = 100,
    group_cols: tuple[str, ...] = (),
) -> WalkForwardResult:
    """Walk-forward recalibration.

    ``df`` needs columns ``price``, ``outcome``, ``t_snap``, ``resolved_ts`` and
    ``event_id``, plus any ``group_cols``. With ``group_cols`` (for example
    ``("venue",)``) a separate map is fitted per group, on that group's history
    only. The returned predictions are indexed by ``df``'s index labels.
    Snapshots before the first refit with enough training data are not scored
    by any method, so all methods, including the raw price, are compared on
    exactly the same rows.
    """
    need = {"price", "outcome", "t_snap", "resolved_ts", "event_id", *group_cols}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"walk_forward: missing columns {sorted(missing)}")
    groups = [((), df)] if not group_cols else list(df.groupby(list(group_cols), sort=True))
    preds: list[pd.DataFrame] = []
    refits: list[dict[str, object]] = []
    for key, g in groups:
        g = g.assign(row=g.index).reset_index(drop=True)
        t0 = int(g["t_snap"].min())
        t1 = int(g["t_snap"].max()) + 1
        grid = np.arange(t0 - t0 % refit_every_s, t1 + refit_every_s, refit_every_s)
        res_ts = g["resolved_ts"].to_numpy(dtype=np.float64)
        t_snap = g["t_snap"].to_numpy(dtype=np.float64)
        p = g["price"].to_numpy(dtype=np.float64)
        y = g["outcome"].to_numpy(dtype=np.float64)
        ev = g["event_id"].to_numpy()
        for T, T_next in pairwise(grid):
            test = (t_snap >= T) & (t_snap < T_next)
            if not test.any():
                continue
            train = res_ts < T
            n_tr = int(train.sum())
            n_ev = int(np.unique(ev[train]).size) if n_tr else 0
            if n_tr < min_train or n_ev < min_train_events:
                continue
            out = g.loc[test, ["row"]].copy()
            out["refit_ts"] = int(T)
            out["raw"] = p[test]
            for m in methods:
                f = FITTERS[m](p[train], y[train])
                out[m] = f(p[test])
            preds.append(out)
            refits.append(
                {
                    "group": "|".join(map(str, key)) if group_cols else "all",
                    "refit_ts": int(T),
                    "n_train": n_tr,
                    "n_train_events": n_ev,
                    "n_test": int(test.sum()),
                }
            )
    if not preds:
        empty = pd.DataFrame(columns=["row", "refit_ts", "raw", *methods])
        return WalkForwardResult(empty, pd.DataFrame(refits))
    pr = pd.concat(preds, ignore_index=True).set_index("row").sort_index()
    return WalkForwardResult(pr, pd.DataFrame(refits))
