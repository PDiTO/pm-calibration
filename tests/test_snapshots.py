from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pmcal.snapshots import PricePath, default_max_age, snapshot_at, take_snapshots

H = 3600
D = 86400


def _path(ts: list[int], price: list[float], **kw: list[float]) -> PricePath:
    extra = {k: np.asarray(v, dtype=float) for k, v in kw.items()}
    return PricePath(
        ts=np.asarray(ts, dtype=np.int64), price=np.asarray(price, dtype=float), **extra
    )


def test_uses_last_observation_at_or_before_t() -> None:
    p = _path([0, 600, 1200], [0.2, 0.3, 0.4])
    s = snapshot_at(p, 900, horizon="x", horizon_s=H, open_ts=0, close_ts=10_000)
    assert s.status == "ok"
    assert s.price == 0.3
    assert s.obs_ts == 600
    s = snapshot_at(p, 1200, horizon="x", horizon_s=H, open_ts=0, close_ts=10_000)
    assert s.price == 0.4


def test_open_close_and_staleness() -> None:
    p = _path([0, 600], [0.2, 0.3])
    kw: dict[str, Any] = {"horizon": "x", "horizon_s": H}
    assert snapshot_at(p, -1, open_ts=0, close_ts=10_000, **kw).status == "not_open"
    assert snapshot_at(p, 10_000, open_ts=0, close_ts=10_000, **kw).status == "closed"
    assert snapshot_at(p, 600 + H + 1, open_ts=0, close_ts=99_999, **kw).status == "too_old"
    assert snapshot_at(p, 600 + H, open_ts=0, close_ts=99_999, **kw).status == "ok"
    empty = _path([5000], [0.5])
    assert snapshot_at(empty, 100, open_ts=0, close_ts=10_000, **kw).status == "no_data"


def test_max_age_scales_with_horizon() -> None:
    assert default_max_age(H) == H
    assert default_max_age(D) == 6 * H
    assert default_max_age(30 * D) == D


def test_mid_when_tight_else_last() -> None:
    nan = math.nan
    p = _path(
        [0, 60, 120],
        [nan, nan, nan],
        bid=[0.40, 0.10, 0.0],
        ask=[0.44, 0.60, 1.0],
        last=[0.41, 0.35, nan],
        volume=[5, 0, 0],
    )
    kw: dict[str, Any] = {"horizon": "x", "horizon_s": H, "open_ts": 0, "close_ts": 10_000}
    s0 = snapshot_at(p, 0, **kw)
    assert s0.price_source == "mid"
    assert s0.price == pytest.approx(0.42)
    s1 = snapshot_at(p, 60, **kw)
    assert s1.price_source == "last"
    assert s1.price == 0.35
    # empty book and no trade print: fall back to the previous usable observation
    s2 = snapshot_at(p, 120, **kw)
    assert s2.status == "ok"
    assert s2.obs_ts == 60
    assert s2.price == 0.35
    assert s2.volume_24h == 5


def test_staleness_diagnostics() -> None:
    ts = [0, 3600, 7200, 10800, 14400]
    p = _path(ts, [0.5, 0.5, 0.6, 0.6, 0.6])
    s = snapshot_at(p, 14400, horizon="x", horizon_s=H, open_ts=0, close_ts=99_999)
    assert s.n_moves_24h == 1
    assert s.last_move_age_s == 14400 - 7200
    flat = _path(ts, [0.5] * 5)
    s = snapshot_at(flat, 14400, horizon="x", horizon_s=H, open_ts=0, close_ts=99_999)
    assert s.n_moves_24h == 0
    assert s.last_move_age_s == 14400


def test_merge_prefers_later_paths_on_ties() -> None:
    daily = _path([0, D, 2 * D], [0.1, 0.2, 0.3])
    fine = _path([D, D + 600], [0.25, 0.26])
    m = PricePath.merge(daily, fine)
    assert m.ts.tolist() == [0, D, D + 600, 2 * D]
    assert m.price.tolist() == [0.1, 0.25, 0.26, 0.3]


@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(1, 60),
    seed=st.integers(0, 10_000),
    cut=st.integers(0, 100 * H),
)
def test_snapshot_never_looks_ahead(n: int, seed: int, cut: int) -> None:
    """Replacing every observation after t with different data leaves the
    snapshot at t unchanged."""
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(0, 100 * H, n))
    pr = rng.uniform(size=n)
    vol = rng.uniform(size=n)
    a = PricePath(ts=ts, price=pr, volume=vol)
    pr2 = np.where(ts > cut, rng.uniform(size=n), pr)
    b = PricePath(ts=ts, price=pr2, volume=np.where(ts > cut, 99.0, vol))
    kw: dict[str, Any] = {"horizon": "x", "horizon_s": 6 * H, "open_ts": 0, "close_ts": 200 * H}
    sa, sb = snapshot_at(a, cut, **kw), snapshot_at(b, cut, **kw)
    assert sa.as_row() == pytest.approx(sb.as_row(), nan_ok=True) or sa == sb


def test_take_snapshots_horizons() -> None:
    ts = list(range(0, 40 * D, H))
    p = _path(ts, [0.5] * len(ts))
    snaps = take_snapshots(p, anchor_ts=35 * D, open_ts=0, close_ts=36 * D)
    by = {s.horizon: s for s in snaps}
    assert by["30d"].t_snap == 5 * D
    assert all(s.status == "ok" for s in snaps)
    early = take_snapshots(p, anchor_ts=35 * D, open_ts=0, close_ts=34 * D)
    by = {s.horizon: s for s in early}
    assert by["1h"].status == "closed"
    assert by["7d"].status == "ok"
