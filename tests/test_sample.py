from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from pmcal.config import Config, FrameConfig
from pmcal.sample import draw_sample, frame_mask

T = 1_770_000_000  # Feb 2026


def _index() -> pd.DataFrame:
    rows: list[tuple[str | None, str, str, int]] = []
    # series A: 20 events in one month, 1 contract each; series B: 2 events of 30 contracts
    for e in range(20):
        rows.append(("A", f"a{e}", f"a{e}-0", T + e * 3600))
    for e in range(2):
        for j in range(30):
            rows.append(("B", f"b{e}", f"b{e}-{j}", T + e * 3600))
    for e in range(5):  # one-offs, no series
        rows.append((None, f"o{e}", f"o{e}-0", T))
    df = pd.DataFrame(rows, columns=["series_id", "event_id", "contract_id", "scheduled_ts"])
    df["venue"] = "polymarket"
    df["volume_usd"] = 5000.0
    df["volume_native"] = 5000.0
    df["is_short_crypto"] = False
    df["open_ts"] = df["scheduled_ts"] - 86400
    df["category"] = "sports"
    return df


def test_caps_and_weights() -> None:
    cfg = Config()
    cfg = replace(
        cfg, sample=replace(cfg.sample, max_events_per_series_month=3, max_contracts_per_event=12)
    )
    s = draw_sample(_index(), cfg)
    per_series = s.groupby("series_id", dropna=False)["event_id"].nunique()
    assert per_series["A"] == 3
    assert per_series["B"] == 2
    assert s["series_id"].isna().sum() == 5  # one-offs all kept
    assert s.groupby("event_id").size().max() == 12
    a = s[s["series_id"] == "A"]
    np.testing.assert_allclose(a["sample_weight"], 20 / 3)
    b = s[s["series_id"] == "B"]
    np.testing.assert_allclose(b["sample_weight"], 30 / 12)
    # deterministic
    s2 = draw_sample(_index().sample(frac=1, random_state=1), cfg)
    assert set(s["contract_id"]) == set(s2["contract_id"])


FLOOR_1K = Config(frame=FrameConfig(min_volume_usd=1000.0))


def test_frame_excludes_inplay_and_low_volume() -> None:
    idx = _index()
    idx.loc[0, "open_ts"] = idx.loc[0, "scheduled_ts"]  # opened at the scheduled time
    idx.loc[1, "volume_usd"] = 10.0
    idx["volume_missing"] = False
    idx.loc[2, "volume_usd"] = np.nan
    idx.loc[2, "volume_missing"] = True
    m = frame_mask(idx, FLOOR_1K)
    assert not m.iloc[0]
    assert not m.iloc[1]
    assert m.iloc[2]
    # the default frame has no lifetime floor beyond "traded at all"
    idx.loc[3, "volume_usd"] = 0.0
    m0 = frame_mask(idx, Config())
    assert m0.iloc[1]
    assert not m0.iloc[3]


def test_kalshi_floor_uses_contracts() -> None:
    idx = _index()
    idx["venue"] = "kalshi"
    idx["volume_usd"] = np.nan  # not known at listing time
    idx["volume_native"] = 2000.0
    idx.loc[0, "volume_native"] = 500.0
    m = frame_mask(idx, FLOOR_1K)
    assert not m.iloc[0]
    assert m.iloc[1:].all()


def test_per_venue_caps() -> None:
    cfg = Config()
    cfg = replace(
        cfg,
        sample=replace(
            cfg.sample,
            venue={"kalshi": {"max_events_per_series_month": 1, "max_contracts_per_event": 5}},
        ),
    )
    idx = _index()
    pm = draw_sample(idx, cfg)
    k = draw_sample(idx.assign(venue="kalshi", volume_native=5000.0), cfg)
    assert k.groupby("series_id")["event_id"].nunique().max() == 1
    assert k.groupby("event_id").size().max() == 5
    assert pm.groupby("series_id")["event_id"].nunique().max() == 3


def test_forced_contracts_bypass_caps() -> None:
    from pmcal.sample import forced_from_matches

    cfg = replace(
        Config(),
        sample=replace(Config().sample, max_events_per_series_month=1, max_contracts_per_event=2),
    )
    idx = _index()
    base = draw_sample(idx, cfg)
    target = [f"b0-{j}" for j in range(30)] + [f"a{e}-0" for e in range(20)]
    s = draw_sample(idx, cfg, force_contracts=target)
    assert set(target) <= set(s["contract_id"])
    added = s[~s["contract_id"].isin(base["contract_id"])]
    assert added["forced"].all()
    assert (added["sample_weight"] == 1.0).all()
    assert not s.loc[s["contract_id"].isin(base["contract_id"]), "forced"].any()
    m = pd.DataFrame(
        {
            "pair_id": [f"p{i}" for i in range(40)],
            "event_key": ["g1"] * 30 + ["g2"] * 10,
            "polymarket_contract_id": [f"x{i}" for i in range(40)],
            "kalshi_contract_id": [f"k{i}" for i in range(40)],
        }
    )
    pmf = forced_from_matches(m, "polymarket", seed=1, max_pairs_per_key=12)
    kf = forced_from_matches(m, "kalshi", seed=1, max_pairs_per_key=12)
    assert len(pmf) == 22
    assert {p[1:] for p in pmf} == {k[1:] for k in kf}  # same pairs on both venues
