from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pmcal.analysis import (
    AnalysisConfig,
    activity_bucket,
    balanced_panel,
    longshot_returns,
    paired_differences,
    recalibration,
    reliability_table,
    summarise,
    summarise_by,
    symmetrise,
    volume_bucket,
)
from pmcal.panel import build_panel, main_mask, selection_table
from pmcal.stats import bin_edges
from tests.synthetic import make_tables

CFG = AnalysisConfig(n_boot=60, min_train=300, min_train_events=60)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    parts = []
    for i, v in enumerate(("polymarket", "kalshi")):
        c, s = make_tables(v, n_events=500, seed=i)
        parts.append(build_panel(c, s))
    return pd.concat(parts, ignore_index=True)


def test_panel_flags(panel: pd.DataFrame) -> None:
    assert {"live", "usable", "early_close"} <= set(panel.columns)
    m = main_mask(panel)
    assert m.sum() > 0
    assert (panel.loc[m, "anchor"] == "scheduled").all()
    assert panel.loc[m, "price"].notna().all()


def test_summarise_recovers_longshot_slope(panel: pd.DataFrame) -> None:
    df = panel[main_mask(panel) & (panel["horizon"] == "1d")]
    s = summarise(symmetrise(df), CFG)
    assert s["slope"] > 1.0
    assert s["slope_lo"] <= s["slope"] <= s["slope_hi"]
    assert s["brier_lo"] <= s["brier"] <= s["brier_hi"]
    assert s["intercept"] == pytest.approx(0.0, abs=1e-6)


def test_summarise_by_and_reliability(panel: pd.DataFrame) -> None:
    df = panel[main_mask(panel)]
    t = summarise_by(df, ["venue", "horizon"], CFG)
    assert set(t["venue"]) == {"polymarket", "kalshi"}
    rt = reliability_table(df[df["horizon"] == "1d"], bin_edges(10))
    ok = rt[rt["n"] > 0]
    assert (ok["freq_lo"] <= ok["freq"]).all()
    assert (ok["freq"] <= ok["freq_hi"]).all()
    assert (ok["n_eff"] <= ok["n"] * 1.5).all()


def test_longshot_returns(panel: pd.DataFrame) -> None:
    df = symmetrise(panel[main_mask(panel) & (panel["horizon"] == "1d")])
    r = longshot_returns(df, bin_edges(10), CFG)
    assert len(r) == 10
    # with slope > 1, the cheapest bin loses money and the dearest makes some
    assert r["return"].iloc[0] < 0
    assert r["return"].iloc[-1] > r["return"].iloc[0]


def test_paired_horizons(panel: pd.DataFrame) -> None:
    df = panel[main_mask(panel)]
    bal = balanced_panel(df, ("7d", "1d"))
    assert bal.groupby(["venue", "contract_id"]).size().eq(2).all()
    d = paired_differences(bal, "horizon", ["1d"], "7d", CFG)
    row = d[(d["horizon"] == "1d") & (d["metric"] == "brier")].iloc[0]
    # information arrives over time: 1d should beat 7d on Brier
    assert row["diff_vs_base"] < 0
    assert row["diff_lo"] <= row["diff_vs_base"] <= row["diff_hi"]


def test_recalibration_runs(panel: pd.DataFrame) -> None:
    df = panel[main_mask(panel) & panel["horizon"].isin(["1d"])]
    scores, refits = recalibration(df, CFG)
    assert not scores.empty
    assert set(scores["method"]) == {"platt", "isotonic"}
    assert (scores["gain_lo"] <= scores["gain_hi"]).all()
    assert (refits["n_train"] >= CFG.min_train).all()


def test_buckets(panel: pd.DataFrame) -> None:
    b = volume_bucket(pd.Series([500.0, 5e3, 5e5, 5e7]), (1e3, 1e4, 1e5, 1e6))
    assert b.tolist()[1:] == ["$1k-$10k", "$100k-$1M", "$1M+"]
    assert pd.isna(b.iloc[0])
    a = activity_bucket(panel.head(50))
    assert set(a) <= {"no trade", "traded", "busy"}


def test_selection_table(panel: pd.DataFrame) -> None:
    c = panel.drop_duplicates(["venue", "contract_id"])
    t = selection_table(c, panel, ("7d", "1d"))
    assert "1d: confirmed quote" in t.index
    assert np.all(t.loc["resolved"] > 0)


def test_symmetrised_slope_absorbs_a_yes_lean() -> None:
    """With listed prices skewed cheap and the YES side overpriced at every price
    (a < 0, b = 1: no longshot bias), forcing a = 0 by symmetrising produces b > 1.
    The as-listed fit of a and b together recovers b = 1. This is why the report
    takes the longshot slope from the as-listed fit."""
    from scipy import special

    from pmcal.stats import fit_logistic, logit

    rng = np.random.default_rng(7)
    p = special.expit(rng.normal(-1.8, 1.6, 40_000))  # mostly cheap, like listed YES
    y = (rng.uniform(size=p.size) < special.expit(-0.5 + logit(p))).astype(float)
    listed = fit_logistic(logit(p), y)
    sym = symmetrise(pd.DataFrame({"price": p, "outcome": y}))
    sym_fit = fit_logistic(logit(sym["price"].to_numpy()), sym["outcome"].to_numpy())
    assert listed.slope == pytest.approx(1.0, abs=0.04)
    assert listed.intercept == pytest.approx(-0.5, abs=0.06)
    assert sym_fit.slope > 1.06


def test_summarise_intervals_contain_their_estimates(panel: pd.DataFrame) -> None:
    df = panel[main_mask(panel) & (panel["horizon"] == "1d")]
    s = summarise(df, CFG)
    assert s["ece_lo"] <= s["ece_bc"] <= s["ece_hi"]
    assert s["ece_bc"] <= s["ece"] + 1e-12
    assert s["reliability_db_lo"] <= s["reliability_db"] <= s["reliability_db_hi"]
    assert s["reliability_db"] <= s["reliability"]
    assert 0.0 <= s["slope_boot_fail"] <= 1.0


def test_metric_vector_drops_separated_slopes() -> None:
    from pmcal.analysis import METRICS, metric_vector

    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1, 1.0])
    v = metric_vector(p, y, bin_edges(10), 1e-3)
    assert np.isnan(v[METRICS.index("slope")])
    assert np.isnan(v[METRICS.index("intercept")])
    assert np.isfinite(v[METRICS.index("brier")])


def test_nested_panel_does_not_condition_on_later_horizons(panel: pd.DataFrame) -> None:
    from pmcal.analysis import nested_panel

    df = panel[main_mask(panel)]
    hs = ("7d", "1d")
    nest = nested_panel(df, hs)
    first = df[df["horizon"] == "7d"]
    # every contract priced 7d out is kept at 7d, including those closed by 1d
    assert len(nest[nest["horizon"] == "7d"]) == len(first)
    closed_later = set(first["contract_id"]) - set(df.loc[df["horizon"] == "1d", "contract_id"])
    assert closed_later  # the synthetic panel has early closers
    assert closed_later <= set(nest.loc[nest["horizon"] == "7d", "contract_id"])
    # later horizons only hold contracts that were priced at the first one
    assert set(nest["contract_id"]) <= set(first["contract_id"])
    bal = balanced_panel(df, hs)
    assert len(bal) < len(nest)


def test_paired_unbalanced_matches_balanced_estimates(panel: pd.DataFrame) -> None:
    df = panel[main_mask(panel)]
    bal = balanced_panel(df, ("7d", "1d"))
    a = paired_differences(bal, "horizon", ["1d"], "7d", CFG)
    b = paired_differences(bal, "horizon", ["1d"], "7d", CFG, balanced=False)
    pd.testing.assert_series_equal(a["value"], b["value"])
    pd.testing.assert_series_equal(
        a["diff_vs_base"].dropna().reset_index(drop=True),
        b["diff_vs_base"].dropna().reset_index(drop=True),
    )
    for d in (a, b):
        ok = d.dropna(subset=["diff_vs_base"])
        assert ((ok["diff_lo"] <= ok["diff_vs_base"]) | (ok["metric"] == "ece")).all()
    from pmcal.analysis import nested_panel

    nest = nested_panel(df, ("7d", "1d"))
    c = paired_differences(nest, "horizon", ["1d"], "7d", CFG, balanced=False)
    row = c[(c["horizon"] == "1d") & (c["metric"] == "brier")].iloc[0]
    assert row["diff_vs_base"] < 0
    assert row["diff_lo"] <= row["diff_vs_base"] <= row["diff_hi"]
    assert set(c["n"]) == {int((nest["horizon"] == h).sum()) for h in ("7d", "1d")}


def test_recluster_and_clustering_sensitivity(panel: pd.DataFrame) -> None:
    from pmcal.analysis import clustering_sensitivity, recluster

    df = panel[main_mask(panel) & (panel["horizon"] == "1d")]
    base = df["cluster_id"].nunique()
    coarse = recluster(df, ("category_day",))
    assert coarse.nunique() < base
    # every original cluster sits inside one coarse cluster
    assert (df.assign(cc=coarse).groupby("cluster_id")["cc"].nunique() == 1).all()
    with pytest.raises(ValueError, match="unknown cluster keys"):
        recluster(df, ("week",))
    t = clustering_sensitivity(df, CFG)
    assert t.loc[t["clustering"] == "game/event", "slope_width_ratio"].iloc[0] == 1.0
    assert t["clusters"].iloc[-1] <= base


def test_recluster_does_not_chain_multi_day_events() -> None:
    """A "by date" ladder with deadlines on several days must not merge every other
    event on those days (and, through further ladders, a whole category) into one
    cluster: each game/event cluster joins the category-day of its first deadline."""
    from pmcal.analysis import recluster

    day = 86400
    t0 = 1_770_000_000
    rows = [("ladder", t0 + d * day) for d in range(5)]  # one event, five deadlines
    rows += [(f"e{d}", t0 + d * day) for d in range(5)]  # one ordinary event per day
    df = pd.DataFrame(rows, columns=["cluster_id", "scheduled_ts"])
    df["venue"] = "polymarket"
    df["series_id"] = None
    df["category"] = "politics"
    cc = recluster(df, ("category_day",))
    by_cluster = df.assign(cc=cc).groupby("cluster_id")["cc"].nunique()
    assert (by_cluster == 1).all()  # nests the base clusters
    assert cc[df["cluster_id"] == "ladder"].iloc[0] == cc[df["cluster_id"] == "e0"].iloc[0]
    assert cc.nunique() == 5  # e1..e4 stay apart from each other and from the ladder


def test_confirmed_quote_rule() -> None:
    """Kalshi: a quote counts when it is tight on the log-odds scale. Polymarket: the
    midpoint counts when a trade in the last 24h sits within half that width."""
    from pmcal.panel import build_panel

    c, s = make_tables("kalshi", n_events=5, seed=3)
    s = s[(s["anchor"] == "scheduled") & (s["status"] == "ok")].head(4).copy()
    s["price"] = [0.50, 0.50, 0.03, 0.03]
    s["bid"] = [0.45, 0.30, 0.025, 0.01]
    s["ask"] = [0.55, 0.70, 0.035, 0.05]
    s["last"] = [0.52, 0.60, 0.03, 0.04]
    s["last_trade_age_s"] = [100.0, 100.0, 100.0, 3 * 86400.0]
    p = build_panel(c, s).set_index("price_mid", append=True)
    rules = p["price_rule"].tolist()
    # 45/55 is 0.40 wide in log-odds (tight); 30/70 is 1.7 (wide, priced at the trade);
    # 2.5/3.5c is 0.35 (tight); 1/5c is 1.6 and the trade is stale
    assert rules == ["quote", "last trade", "quote", "stale"]
    assert p["price"].tolist()[:3] == pytest.approx([0.50, 0.60, 0.03])

    c2, s2 = make_tables("polymarket", n_events=5, seed=4)
    s2 = s2[(s2["anchor"] == "scheduled") & (s2["status"] == "ok")].head(3).copy()
    s2["price"] = [0.50, 0.03, 0.03]
    s2["last"] = [0.53, 0.031, 0.045]
    s2["last_trade_age_s"] = 100.0
    p2 = build_panel(c2, s2)
    # 0.50 vs 0.53: 0.12 apart (within 0.2); 3c vs 3.1c: 0.03; 3c vs 4.5c: 0.42 (no)
    assert p2["price_rule"].tolist() == ["quote", "quote", "last trade"]
    assert p2["price"].tolist() == pytest.approx([0.50, 0.03, 0.045])
