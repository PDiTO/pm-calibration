from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import special

from pmcal.recalibrate import fit_isotonic, fit_platt, pav, walk_forward
from pmcal.stats import brier, logit


@settings(max_examples=50, deadline=None)
@given(n=st.integers(1, 200), seed=st.integers(0, 10_000))
def test_pav_matches_sklearn(n: int, seed: int) -> None:
    iso = pytest.importorskip("sklearn.isotonic")
    rng = np.random.default_rng(seed)
    x = rng.choice(np.linspace(0, 1, 15), n)  # many ties
    y = rng.uniform(size=n)
    w = rng.uniform(0.5, 2, n)
    ours, _ = pav(x, y, w)
    ref = iso.IsotonicRegression().fit(x, y, sample_weight=w).predict(x)
    np.testing.assert_allclose(ours, ref, atol=1e-10)


def test_isotonic_map_is_monotone_and_shrunk() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=500)
    y = (rng.uniform(size=500) < p).astype(float)
    f = fit_isotonic(p, y, shrink=10)
    grid = np.linspace(0, 1, 101)
    out = f(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert out[0] == 0.0
    assert out[-1] == 1.0
    # a tiny all-NO block at a 30% price should not be mapped to 0
    p2 = np.array([0.3, 0.3, 0.3, 0.7, 0.7, 0.7])
    y2 = np.array([0, 0, 0, 1, 1, 1.0])
    f2 = fit_isotonic(p2, y2, shrink=10)
    assert 0.15 < f2(np.array([0.3]))[0] < 0.3


def test_platt_recovers_squash() -> None:
    rng = np.random.default_rng(1)
    q = rng.uniform(0.05, 0.95, 50_000)
    y = (rng.uniform(size=q.size) < q).astype(float)
    p = special.expit(0.5 * logit(q))
    f = fit_platt(p, y, l2=0.0)
    assert brier(f(p), y) < brier(p, y) - 1e-3
    np.testing.assert_allclose(f(p), q, atol=0.02)


def _panel(n: int = 4000, seed: int = 2, squash: float = 0.6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t_snap = np.sort(rng.integers(0, 200 * 86400, n))
    resolved = t_snap + rng.integers(3600, 40 * 86400, n)
    q = rng.uniform(0.02, 0.98, n)
    y = (rng.uniform(size=n) < q).astype(float)
    return pd.DataFrame(
        {
            "price": special.expit(squash * logit(q)),
            "outcome": y,
            "t_snap": t_snap,
            "resolved_ts": resolved,
            "event_id": [f"e{i // 2}" for i in range(n)],
        }
    )


def test_walk_forward_never_trains_on_unresolved_outcomes() -> None:
    df = _panel()
    res = walk_forward(df, min_train=200, min_train_events=50)
    pr = res.predictions
    assert len(pr) > 0
    # every scored snapshot falls in its refit's window, after the refit time
    t = df.loc[pr.index, "t_snap"].to_numpy()
    assert np.all(t >= pr["refit_ts"].to_numpy())
    assert np.all(t < pr["refit_ts"].to_numpy() + 7 * 86400)
    # training sizes only count outcomes resolved before the refit
    for r in res.refits.itertuples():
        assert r.n_train == int((df["resolved_ts"] < r.refit_ts).sum())


def test_walk_forward_is_invariant_to_future_outcomes() -> None:
    """Scrambling the outcomes of contracts that resolve after T must not change
    any prediction made before T."""
    df = _panel()
    T = 120 * 86400
    a = walk_forward(df, min_train=200, min_train_events=50).predictions
    df2 = df.copy()
    late = df2["resolved_ts"] >= T
    df2.loc[late, "outcome"] = 1.0 - df2.loc[late, "outcome"]
    b = walk_forward(df2, min_train=200, min_train_events=50).predictions
    early = df.loc[a.index, "t_snap"] < T
    cols = ["raw", "platt", "isotonic"]
    pd.testing.assert_frame_equal(a.loc[early, cols], b.loc[early, cols])


def test_walk_forward_improves_squashed_prices() -> None:
    df = _panel(n=20_000, squash=0.5)
    pr = walk_forward(df, min_train=500, min_train_events=200).predictions
    y = df.loc[pr.index, "outcome"].to_numpy()
    assert brier(pr["platt"].to_numpy(), y) < brier(pr["raw"].to_numpy(), y)
    assert brier(pr["isotonic"].to_numpy(), y) < brier(pr["raw"].to_numpy(), y)


def test_walk_forward_groups_fit_separately() -> None:
    df = pd.concat([_panel(seed=3).assign(venue="a"), _panel(seed=4).assign(venue="b")])
    df = df.reset_index(drop=True)
    res = walk_forward(df, min_train=200, min_train_events=50, group_cols=("venue",))
    assert set(res.refits["group"]) == {"a", "b"}
    assert res.predictions.index.is_unique
