"""Price of a contract at a fixed horizon before an anchor time.

Both venue pipelines turn their raw price data into a ``PricePath`` and call
``take_snapshots``, so the rules below apply identically to Polymarket and
Kalshi:

- The snapshot time is ``t = anchor_ts - horizon_s``. For the main analysis the
  anchor is the ex-ante scheduled time (``scheduled_ts``), which does not depend
  on the outcome. The alternative anchor, the actual close, is outcome-dependent
  for markets that resolve early and is kept only to show how much it distorts
  the results.
- A snapshot is only taken while the market was open: ``open_ts <= t < close_ts``.
  Whether a market is still open at ``t`` is known at ``t``, so this filter does
  not condition on the future.
- The price is the latest observation at or before ``t``, never after, and is
  rejected if older than ``max_age_s``.
- Where bid and ask are known (Kalshi), the price is the mid when the spread is at
  most ``max_spread``; otherwise the last trade. Where only a price series is
  known (Polymarket's ``prices-history``), that series is used as is.
- Staleness diagnostics (moves and volume in the 24 hours up to ``t``, age of
  the last price change) are recorded so the analysis can filter thin markets.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from pmcal.schema import HORIZONS

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

STATUSES: Final = (
    "ok",
    "not_open",  # t is before the market opened
    "closed",  # t is at or after the actual close
    "no_data",  # no observation at or before t
    "too_old",  # the latest observation before t is older than max_age_s
    "bad_price",  # observation exists but no usable price (e.g. empty book, no trades)
)

DAY: Final = 86400
HOUR: Final = 3600


def default_max_age(horizon_s: int) -> int:
    """How old the observation behind a snapshot may be: h/4, clamped to [1h, 1d]."""
    return int(min(DAY, max(HOUR, horizon_s // 4)))


@dataclass(frozen=True)
class PricePath:
    """Observations for one contract, sorted by ``ts`` (Unix seconds).

    ``price`` is the venue's reference series (Polymarket) or NaN (Kalshi, where
    bid/ask/last are given instead). Optional arrays have the same length as
    ``ts`` and use NaN for "unknown at this observation". ``volume`` is the native
    volume traded in the interval ending at ``ts``.
    """

    ts: IntArray
    price: FloatArray
    bid: FloatArray | None = None
    ask: FloatArray | None = None
    last: FloatArray | None = None
    volume: FloatArray | None = None

    def __post_init__(self) -> None:
        n = self.ts.shape[0]
        for name in ("price", "bid", "ask", "last", "volume"):
            a = getattr(self, name)
            if a is not None and a.shape[0] != n:
                raise ValueError(f"{name} has length {a.shape[0]}, ts has {n}")
        if n > 1 and np.any(np.diff(self.ts) < 0):
            raise ValueError("ts must be sorted")

    @classmethod
    def from_points(cls, ts: list[int] | IntArray, price: list[float] | FloatArray) -> PricePath:
        t = np.asarray(ts, dtype=np.int64)
        p = np.asarray(price, dtype=np.float64)
        order = np.argsort(t, kind="stable")
        return cls(ts=t[order], price=p[order])

    @classmethod
    def merge(cls, *paths: PricePath) -> PricePath:
        """Concatenate paths (e.g. a daily and a fine series) and drop duplicate timestamps.

        Where two paths have an observation at the same timestamp the later
        argument wins, so pass the finer-grained path last.
        """
        paths = tuple(p for p in paths if p.ts.size)
        if not paths:
            return cls(ts=np.zeros(0, np.int64), price=np.zeros(0))
        fields = ("bid", "ask", "last", "volume")
        has = {f: any(getattr(p, f) is not None for p in paths) for f in fields}
        ts = np.concatenate([p.ts for p in paths])
        cols: dict[str, FloatArray] = {"price": np.concatenate([p.price for p in paths])}
        for f in fields:
            if has[f]:
                cols[f] = np.concatenate(
                    [
                        getattr(p, f) if getattr(p, f) is not None else np.full(p.ts.size, np.nan)
                        for p in paths
                    ]
                )
        # stable sort, then keep the last occurrence of each timestamp
        order = np.argsort(ts, kind="stable")
        ts = ts[order]
        keep = np.ones(ts.size, dtype=bool)
        keep[:-1] = ts[1:] != ts[:-1]
        out: dict[str, Any] = {k: v[order][keep] for k, v in cols.items()}
        return cls(ts=ts[keep], **out)


@dataclass(frozen=True)
class Snapshot:
    horizon: str
    horizon_s: int
    t_snap: int
    status: str
    price: float = math.nan
    price_source: str = ""
    obs_ts: int | None = None
    bid: float = math.nan
    ask: float = math.nan
    last: float = math.nan
    volume_24h: float = math.nan
    n_moves_24h: int | None = None
    last_move_age_s: float = math.nan
    # Filled in by the venue module from trade prints, where it has them
    # (use dataclasses.replace on the returned Snapshot).
    n_trades_24h: int | None = None
    last_trade_age_s: float = math.nan

    def as_row(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "horizon_s": self.horizon_s,
            "t_snap": self.t_snap,
            "status": self.status,
            "price": self.price,
            "price_source": self.price_source,
            "obs_ts": self.obs_ts,
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "volume_24h": self.volume_24h,
            "n_moves_24h": self.n_moves_24h,
            "last_move_age_s": self.last_move_age_s,
            "n_trades_24h": self.n_trades_24h,
            "last_trade_age_s": self.last_trade_age_s,
        }


def _valid_prob(x: float) -> bool:
    return math.isfinite(x) and 0.0 <= x <= 1.0


def _reference_series(path: PricePath, max_spread: float) -> FloatArray:
    """The price used at each observation: the mid if quotes are tight, else last trade,
    else the venue's series."""
    ref = path.price.astype(np.float64, copy=True)
    if path.last is not None:
        use = np.isfinite(path.last)
        ref = np.where(use, path.last, ref)
    if path.bid is not None and path.ask is not None:
        bid, ask = path.bid, path.ask
        tight = (
            np.isfinite(bid)
            & np.isfinite(ask)
            & (bid > 0.0)
            & (ask < 1.0)
            & (ask >= bid)
            & (ask - bid <= max_spread)
        )
        ref = np.where(tight, 0.5 * (bid + ask), ref)
    return ref


def snapshot_at(
    path: PricePath,
    t: int,
    *,
    horizon: str,
    horizon_s: int,
    open_ts: int | None,
    close_ts: int | None,
    max_age_s: int | None = None,
    max_spread: float = 0.10,
    window_s: int = DAY,
) -> Snapshot:
    """The contract's price at time ``t``, using only observations at or before ``t``."""
    max_age = default_max_age(horizon_s) if max_age_s is None else max_age_s

    def mk(status: str, **kw: Any) -> Snapshot:
        return Snapshot(horizon=horizon, horizon_s=horizon_s, t_snap=t, status=status, **kw)

    if open_ts is not None and t < open_ts:
        return mk("not_open")
    if close_ts is not None and t >= close_ts:
        return mk("closed")
    i = int(np.searchsorted(path.ts, t, side="right")) - 1
    if i < 0:
        return mk("no_data")
    if t - int(path.ts[i]) > max_age:
        return mk("too_old", obs_ts=int(path.ts[i]))

    # The latest observation at or before t that carries a usable price.
    ref = _reference_series(path, max_spread)
    usable = np.flatnonzero((ref[: i + 1] >= 0.0) & (ref[: i + 1] <= 1.0))
    if usable.size == 0:
        return mk("bad_price", obs_ts=int(path.ts[i]))
    j = int(usable[-1])
    obs_ts = int(path.ts[j])
    if t - obs_ts > max_age:
        return mk("too_old", obs_ts=obs_ts)
    price = float(ref[j])
    bid = float(path.bid[j]) if path.bid is not None else math.nan
    ask = float(path.ask[j]) if path.ask is not None else math.nan
    last = float(path.last[j]) if path.last is not None else math.nan
    tight = _valid_prob(bid) and _valid_prob(ask) and 0 < bid <= ask < 1 and ask - bid <= max_spread
    if tight:
        source = "mid"
    elif math.isfinite(last):
        source = "last"
    else:
        source = "series"

    # Staleness diagnostics over (t - window_s, t], on the reference series
    # forward-filled so that observations without a usable price are not moves.
    lo = int(np.searchsorted(path.ts, t - window_s, side="right"))
    filled = _ffill(np.where((ref >= 0.0) & (ref <= 1.0), ref, np.nan)[: i + 1])
    d = np.diff(filled)
    moved = np.isfinite(d) & (d != 0)
    n_moves = int(np.count_nonzero(moved[max(lo - 1, 0) :]))
    changed = np.flatnonzero(moved)
    if changed.size:
        last_move_age = float(t - path.ts[changed[-1] + 1])
    else:
        last_move_age = float(t - path.ts[int(usable[0])])
    vol = math.nan
    if path.volume is not None:
        v = path.volume[lo : i + 1]
        vol = float(np.nansum(v)) if v.size else 0.0

    return mk(
        "ok",
        price=price,
        price_source=source,
        obs_ts=obs_ts,
        bid=bid,
        ask=ask,
        last=last,
        volume_24h=vol,
        n_moves_24h=n_moves,
        last_move_age_s=last_move_age,
    )


def _ffill(a: FloatArray) -> FloatArray:
    if a.size == 0:
        return a
    idx = np.where(np.isfinite(a), np.arange(a.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    # leading NaNs stay NaN
    first = np.flatnonzero(np.isfinite(a))
    if first.size == 0:
        return np.full_like(a, np.nan)
    out[: first[0]] = np.nan
    return out


def take_snapshots(
    path: PricePath,
    *,
    anchor_ts: int,
    open_ts: int | None,
    close_ts: int | None,
    horizons: Mapping[str, int] = HORIZONS,
    max_spread: float = 0.10,
) -> list[Snapshot]:
    """One snapshot per horizon, at ``anchor_ts - horizon_s``."""
    return [
        snapshot_at(
            path,
            anchor_ts - h,
            horizon=label,
            horizon_s=h,
            open_ts=open_ts,
            close_ts=close_ts,
            max_spread=max_spread,
        )
        for label, h in horizons.items()
    ]
