from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import special

from pmcal.stats import (
    bin_edges,
    bin_table,
    brier,
    calibration_regression,
    cluster_bootstrap,
    cluster_index,
    decompose,
    effective_n,
    fit_logistic,
    log_loss,
    logit,
    wilson_interval,
)


def _calibrated(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.01, 0.99, n)
    y = (rng.uniform(size=n) < p).astype(float)
    return p, y


@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(5, 400),
    seed=st.integers(0, 10_000),
    scheme=st.sampled_from(["uniform", "fine_tails"]),
    weighted=st.booleans(),
)
def test_brier_decomposition_identity_is_exact(
    n: int, seed: int, scheme: str, weighted: bool
) -> None:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    p[: n // 5] = rng.choice([0.0, 1.0, 0.5], n // 5)  # edges and ties
    y = (rng.uniform(size=n) < rng.uniform(0, 1, n)).astype(float)
    w = rng.uniform(0.1, 3.0, n) if weighted else None
    d = decompose(p, y, bin_edges(10, scheme), w)
    assert d.check() == pytest.approx(0.0, abs=1e-12)
    assert d.brier == pytest.approx(brier(p, y, w), abs=1e-12)
    assert d.reliability >= 0
    assert d.resolution >= 0
    assert 0 <= d.ece <= 1


def test_decomposition_on_calibrated_forecasts() -> None:
    p, y = _calibrated(200_000)
    d = decompose(p, y, bin_edges(10))
    assert d.reliability < 2e-4
    assert d.ece < 0.01
    # uncertainty of a uniform-p population is ~ 1/4, resolution ~ var(p) = 0.08
    assert d.uncertainty == pytest.approx(0.25, abs=0.005)
    assert d.resolution == pytest.approx(np.var(p), abs=0.005)


def test_decomposition_detects_miscalibration() -> None:
    rng = np.random.default_rng(1)
    q = rng.uniform(0.05, 0.95, 100_000)
    y = (rng.uniform(size=q.size) < q).astype(float)
    p = special.expit(0.5 * logit(q))  # prices squashed towards 0.5
    d = decompose(p, y, bin_edges(10))
    assert d.reliability > 5e-3
    fit = calibration_regression(p, y)
    assert fit.slope == pytest.approx(2.0, rel=0.05)  # inverse of the squash
    assert fit.intercept == pytest.approx(0.0, abs=0.05)


def test_logistic_matches_sklearn() -> None:
    sk = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(2)
    x = rng.normal(size=3000)
    y = (rng.uniform(size=x.size) < special.expit(0.3 + 1.4 * x)).astype(float)
    ours = fit_logistic(x, y)
    ref = sk.LogisticRegression(C=1e12, tol=1e-12, max_iter=10_000).fit(x[:, None], y)
    assert ours.intercept == pytest.approx(ref.intercept_[0], abs=1e-5)
    assert ours.slope == pytest.approx(ref.coef_[0, 0], abs=1e-5)


def test_cluster_robust_se_grows_with_duplicated_clusters() -> None:
    p, y = _calibrated(2000, seed=3)
    x = logit(p)
    base = fit_logistic(x, y, clusters=np.arange(x.size))
    # every observation duplicated 5 times inside its own cluster: same point
    # estimate, and the cluster-robust SE should not shrink by sqrt(5)
    xx, yy = np.repeat(x, 5), np.repeat(y, 5)
    cl = np.repeat(np.arange(x.size), 5)
    dup = fit_logistic(xx, yy, clusters=cl)
    naive = fit_logistic(xx, yy)
    assert dup.slope == pytest.approx(base.slope, rel=1e-6)
    assert dup.se[1] == pytest.approx(base.se[1], rel=0.01)
    naive_base = fit_logistic(x, y)
    assert naive.se[1] == pytest.approx(naive_base.se[1] / np.sqrt(5), rel=1e-6)


def test_cluster_bootstrap_widens_for_correlated_rows() -> None:
    """Mutually exclusive outcomes: one YES per event. Resampling rows as if
    independent gives intervals that are too narrow; resampling events does not."""
    rng = np.random.default_rng(4)
    n_events, k = 300, 8
    clusters = np.repeat(np.arange(n_events), k)
    # a shared event-level shock makes outcomes within an event correlated
    shock = np.repeat(rng.normal(size=n_events), k)
    y = (rng.uniform(size=n_events * k) < special.expit(-1 + 2 * shock)).astype(float)

    def mean(idx: np.ndarray) -> float:
        return float(y[idx].mean())

    _, lo_c, hi_c = cluster_bootstrap(mean, clusters, n_boot=400, seed=0)
    _, lo_i, hi_i = cluster_bootstrap(mean, np.arange(y.size), n_boot=400, seed=0)
    assert (hi_c - lo_c)[0] > 1.5 * (hi_i - lo_i)[0]


def test_cluster_index_partitions_rows() -> None:
    c = np.array(["b", "a", "b", "c", "a", "b"])
    groups = cluster_index(c)
    assert sorted(np.concatenate(groups).tolist()) == list(range(6))
    assert sorted(len(g) for g in groups) == [1, 2, 3]


def test_effective_n() -> None:
    rng = np.random.default_rng(5)
    y = rng.uniform(size=1000)
    assert effective_n(y, np.arange(1000)) == 1000
    yy = np.repeat(y[:100], 10)
    ne = effective_n(yy, np.repeat(np.arange(100), 10))
    assert 70 < ne < 130


def test_bin_table_and_wilson() -> None:
    p = np.array([0.05, 0.15, 0.15, 0.95, 1.0])
    y = np.array([0, 1, 0, 1, 1.0])
    t = bin_table(p, y, bin_edges(10), clusters=["a", "b", "b", "c", "d"])
    assert t.n.tolist() == [1, 2, 0, 0, 0, 0, 0, 0, 0, 2]
    assert t.n_clusters.tolist()[:2] == [1, 1]
    assert t.freq[1] == 0.5
    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi
    assert wilson_interval(0, 10)[0] == 0.0


def test_log_loss_is_finite_at_extremes() -> None:
    assert np.isfinite(log_loss(np.array([0.0, 1.0]), np.array([1.0, 0.0])))


def test_debiased_reliability_is_about_unbiased_under_clustering() -> None:
    """Plug-in REL is positive even for calibrated prices; the debiased version is
    centred on the truth, for calibrated and for miscalibrated prices, with outcomes
    correlated within clusters."""
    from pmcal.stats import debiased_reliability

    rng = np.random.default_rng(11)
    edges = bin_edges(10)
    n_clusters, k = 200, 5
    cl = np.repeat(np.arange(n_clusters), k)
    for squash, truth in ((1.0, 0.0), (0.7, None)):
        plug, deb = [], []
        for _ in range(200):
            shock = np.repeat(rng.normal(0, 0.5, n_clusters), k)
            q = special.expit(logit(rng.uniform(0.02, 0.98, cl.size)) + shock)
            p = special.expit(squash * logit(q))
            y = (rng.uniform(size=q.size) < q).astype(float)
            plug.append(decompose(p, y, edges).reliability)
            deb.append(debiased_reliability(p, y, edges, clusters=cl))
        if truth is None:  # population value from one very large sample
            shock = np.repeat(rng.normal(0, 0.5, 200_000), 5)
            q = special.expit(logit(rng.uniform(0.02, 0.98, shock.size)) + shock)
            p = special.expit(squash * logit(q))
            kb = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
            n = np.bincount(kb, minlength=10)
            gap = (np.bincount(kb, p, 10) - np.bincount(kb, q, 10)) / n
            truth = float((n * gap**2).sum() / n.sum())
        se = np.std(deb) / np.sqrt(len(deb))
        assert np.mean(plug) > truth + 5 * se  # the plug-in is visibly biased
        assert abs(np.mean(deb) - truth) < 4 * se + 2e-5


def test_debiased_reliability_matches_iid_formula_without_clusters() -> None:
    from pmcal.stats import debiased_reliability

    p = np.array([0.1, 0.1, 0.15, 0.6, 0.65, 0.7])
    y = np.array([0, 1, 0, 1, 1, 0.0])
    edges = np.array([0.0, 0.5, 1.0])
    expected = 0.0
    for m in (p < 0.5, p >= 0.5):
        r = y[m] - p[m]
        expected += m.sum() / p.size * (r.mean() ** 2 - r.var(ddof=1) / m.sum())
    assert debiased_reliability(p, y, edges) == pytest.approx(expected, rel=1e-12)


def test_replicate_clusters_counts_a_repeated_cluster_twice() -> None:
    from pmcal.stats import bootstrap_draws, replicate_clusters

    c = np.array(["a", "a", "b", "c", "c", "c"])
    g = cluster_index(c)  # a: [0, 1], b: [2], c: [3, 4, 5]
    idx = np.concatenate([g[0], g[0], g[2], g[1], g[1]])
    assert replicate_clusters(idx, c).tolist() == [0, 0, 1, 1, 2, 2, 2, 3, 4]
    rng = np.random.default_rng(0)
    for d in bootstrap_draws(g, 20, rng):
        lab = replicate_clusters(d, c)
        assert lab.max() + 1 == len(g)  # one label per drawn cluster


def test_coarsen_clusters_nests_every_key() -> None:
    from pmcal.stats import coarsen_clusters

    event = np.array(["e1", "e1", "e2", "e3", "e4", "e5"])
    day = np.array(["d1", "d2", "d2", None, "d5", "d5"], dtype=object)
    lab = coarsen_clusters(event, day)
    # e1 links d1 and d2, d2 links e2; e3 has no day; e4 and e5 share d5
    assert lab[0] == lab[1] == lab[2]
    assert len({lab[0], lab[3], lab[4]}) == 3
    assert lab[4] == lab[5]


def test_fit_logistic_flags_separation() -> None:
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    y = np.array([0, 0, 0, 1, 1, 1.0])
    assert not fit_logistic(x, y).converged
    p, yy = _calibrated(500, seed=9)
    assert fit_logistic(logit(p), yy).converged
