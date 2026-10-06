"""Cross-venue matching, offline.

The fixtures in ``fixtures/match`` are real rows of the two venue indices (October
2026): an MLB game with its moneyline and one total, an EPL game (three-way), the
June and September 2026 Fed decisions (Polymarket listed September twice), February
2026 CPI tails, the Grammys' Best New Artist, a women's college basketball game that
must not be matched to another game at the same time with similar names, an MTV VMA
category that must not be matched to a different category, and one day of Los Angeles
temperature brackets (same airport, different resolution source: excluded; on this day
the two venues resolved one bracket differently).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pmcal import match as mt
from pmcal.config import Config, StudyConfig
from pmcal.http import CachedClient, canonical_url
from pmcal.venues import kalshi, polymarket

FIX = Path(__file__).parent / "fixtures" / "match"


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pm_rows = json.loads((FIX / "polymarket_rows.json").read_text())
    k_rows = json.loads((FIX / "kalshi_rows.json").read_text())
    pm = polymarket.coerce(pd.DataFrame(pm_rows), polymarket.INDEX_COLUMNS)
    k = kalshi.to_contracts_frame(k_rows)
    meta = json.loads((FIX / "kalshi_series.json").read_text())
    return pm, k, meta


@pytest.fixture(scope="module")
def matched() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pm, k, meta = _frames()
    pairs = mt.find_pairs(pm, k, meta)
    return mt.pairs_frame(pairs, pm, k), pm, k


# -- normalisation ------------------------------------------------------------------


def test_fold_and_tokens() -> None:
    assert mt.fold("Atlético Nott'm (FL) São Paulo") == "atletico nottm fl sao paulo"
    assert mt.fold("Łódź & Co.") == "lodz and co"
    assert mt.name_tokens("Ohio St.") == ("ohio", "state")
    assert mt.name_tokens("St. Louis Blues") == ("saint", "louis", "blues")
    assert mt.name_tokens("England Under-19s") == ("england", "u19")
    assert mt.name_tokens("Arsenal FC") == ("arsenal",)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Arsenal", "Arsenal FC"),
        ("Man City", "Manchester City FC"),
        ("PSG", "Paris Saint-Germain"),
        ("Ohio St.", "Ohio State Buckeyes"),
        ("TheMongolz", "The MongolZ"),
        ("Zhang Zhizhen", "Zhizhen Zhang"),
        ("Jirí Procházka", "Jiri Prochazka"),
        ("SE Missouri St.", "Southeast Missouri State Redhawks"),
    ],
)
def test_names_that_match(a: str, b: str) -> None:
    assert mt.name_similarity(a, b) >= mt.MIN_GAME_SCORE


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Kentucky", "Eastern Kentucky Colonels"),  # a different school
        ("Eastern Illinois", "Western Illinois Leathernecks"),
        ("Texas", "Texas A&M Aggies"),
        ("Washington", "Washington State Cougars"),
        ("Spain", "Spain U21"),  # youth side
        ("Arsenal", "Arsenal Women"),
        ("Real Madrid", "Real Madrid Castilla"),
        ("Karolina Pliskova", "Kristyna Pliskova"),  # twins
        ("Manchester City", "Manchester United"),
    ],
)
def test_names_that_do_not_match(a: str, b: str) -> None:
    assert mt.name_similarity(a, b) < mt.MIN_TEAM_SCORE


def test_us_teams() -> None:
    assert mt.us_team("nba", "Trail Blazers") == "trail blazers"
    assert mt.us_team("nba", "Portland") == "trail blazers"
    assert mt.us_team("nba", "Los Angeles C", "LAC") == "clippers"
    assert mt.us_team("nba", "Los Angeles") is None  # Lakers or Clippers
    assert mt.us_team("mlb", "A's", "ATH") == "athletics"
    assert mt.us_team("mlb", "Chicago WS") == "white sox"
    assert mt.us_team("nhl", "BOS Bruins") == "bruins"
    assert mt.us_team("nhl", "St. Louis") == "blues"
    assert mt.us_team("nfl", "New York J") == "jets"


def test_split_matchup() -> None:
    assert mt.split_matchup("Counter-Strike: FaZe vs TheMongolz (BO3) - BLAST Bounty") == (
        "FaZe",
        "TheMongolz",
    )
    assert mt.split_matchup("Belgium vs Senegal: Regulation Time Moneyline") == (
        "Belgium",
        "Senegal",
    )
    assert mt.split_matchup("Miami (FL) at Ohio St. Winner?") == ("Miami (FL)", "Ohio St.")
    assert mt.split_matchup("Liverpool FC vs. Leeds United FC - More Markets") == (
        "Liverpool FC",
        "Leeds United FC",
    )
    assert mt.split_matchup(
        "UFC Fight Night: Chase Hooper vs. Lance Gibson Jr. (Lightweight, Main Card)"
    ) == ("Chase Hooper", "Lance Gibson Jr.")
    assert mt.split_matchup("Who will win the Lyon mayoral election?") is None


# -- direction and dates ----------------------------------------------------------------


def test_direction_mapping() -> None:
    assert mt.map_to_polymarket(0.3, True) == 0.3
    assert mt.map_to_polymarket(0.3, False) == pytest.approx(0.7)
    assert mt.map_to_polymarket(1.0, False) == 0.0
    assert math.isnan(mt.map_to_polymarket(math.nan, False))


def test_date_tolerance() -> None:
    assert mt.within(0, mt.GAME_TOL_S, mt.GAME_TOL_S)
    assert not mt.within(0, mt.GAME_TOL_S + 1, mt.GAME_TOL_S)
    assert not mt.within(None, 0, mt.GAME_TOL_S)
    assert not mt.within(float("nan"), 0, mt.GAME_TOL_S)


def _game(venue: str, start: int, names: tuple[str, str], sport: str = "baseball") -> mt.Game:
    return mt.Game(venue, f"{venue}-{start}", sport, False, None, start, names, False, {"a": 0})


def test_game_start_tolerance() -> None:
    t = 1_780_000_000
    p = [_game("polymarket", t, ("Lotte Giants", "Kiwoom Heroes"))]
    near = [_game("kalshi", t + mt.GAME_TOL_S, ("Lotte Giants", "Kiwoom Heroes"))]
    far = [_game("kalshi", t + mt.GAME_TOL_S + 60, ("Lotte Giants", "Kiwoom Heroes"))]
    assert len(mt.match_games(p, near)) == 1
    assert mt.match_games(p, far) == []
    # a tennis match may be timed at the start of the session on one venue
    pt = [_game("polymarket", t, ("Arthur Cazaux", "Martin Landaluce"), "tennis")]
    kt = [_game("kalshi", t + 8 * 3600, ("Martin Landaluce", "Arthur Cazaux"), "tennis")]
    ((_, _, _, swapped),) = mt.match_games(pt, kt)
    assert swapped


def test_doubleheader_goes_to_the_closer_start() -> None:
    t = 1_780_000_000
    p = [_game("polymarket", t, ("Yankees", "Red Sox"))]
    k = [
        _game("kalshi", t + 2 * 3600, ("Yankees", "Red Sox")),
        _game("kalshi", t + 600, ("Yankees", "Red Sox")),
    ]
    ((_, kg, _, _),) = mt.match_games(p, k)
    assert kg.start == t + 600


# -- brackets ---------------------------------------------------------------------------


def test_brackets() -> None:
    assert mt.open_bracket("≥1.1%") == ("ge", 1.1)
    assert mt.open_bracket("0.6%+") == ("ge", 0.6)
    assert mt.open_bracket("0.0% or less") == ("le", 0.0)
    assert mt.open_bracket("≤3.9%") == ("le", 3.9)
    assert mt.open_bracket("0.4%") is None
    assert mt.fed_bracket_pm(
        "Will the Fed decrease interest rates by 50+ bps after the June 2026 meeting?"
    ) == (("cut", 50), (2026, 6))
    assert mt.fed_bracket_pm(
        "Will there be no change in Fed interest rates after the March 2026 meeting?"
    ) == (("hold", 0), (2026, 3))
    assert mt.temperature_bracket("76-77°F") == (76.0, 77.0)
    assert mt.temperature_bracket("96° to 97°") == (96.0, 97.0)
    assert mt.temperature_bracket("66° or above") == (66.0, math.inf)
    assert mt.temperature_bracket("23°F or below") == (-math.inf, 23.0)


def test_event_titles() -> None:
    assert mt.title_coverage("VMA Best Pop 2026", "MTV Video Music Awards: Best Pop") >= 0.6
    assert (
        mt.title_coverage(
            "Oscars 2026: Best Picture Winner", "Which film will win the most Oscars?"
        )
        < mt.MIN_TITLE_COVERAGE
    )
    assert mt.title_coverage("Dutch Grand Prix: Driver Fastest Lap", "Dutch Grand Prix Winner") < (
        mt.MIN_TITLE_COVERAGE
    )
    assert mt.title_qualifiers("Portugal Presidential Election: 1st Round Winner") != (
        mt.title_qualifiers("Portugal Presidential Election")
    )
    assert "nominee" in mt.title_qualifiers("The Actor Awards Nominations: Best Ensemble")
    assert mt.title_days("Maine Democratic Senate nominee on July 27?") == {("jul", 27)}


def test_is_live() -> None:
    assert mt.is_live("mid", math.nan, 0)
    assert mt.is_live("last", 3600.0, 0)
    # a moving midpoint alone is not evidence of a tradable price (same rule as panel)
    assert not mt.is_live("series", math.nan, 2)
    assert not mt.is_live("last", 2 * 86400.0, 0)


def test_polymarket_price_rule() -> None:
    assert mt.polymarket_price(0.50, 0.12, 3600.0, 0.10) == 0.12  # wide book: use the trade
    assert mt.polymarket_price(0.50, 0.45, 3600.0, 0.10) == 0.50  # close: keep the mid
    assert mt.polymarket_price(0.50, 0.12, 3 * 86400.0, 0.10) == 0.50  # stale trade: mid


# -- real rows --------------------------------------------------------------------------


def _pair(df: pd.DataFrame, k_contract: str) -> pd.Series:
    rows = df[df["kalshi_contract_id"] == k_contract]
    assert len(rows) == 1, k_contract
    return rows.iloc[0]


def test_moneyline_total_and_three_way(
    matched: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    df, _, _ = matched
    ml = df[df["kalshi_event_id"] == "KXMLBGAME-26JUN191905CINNYY"]
    assert len(ml) == 1  # Polymarket prices one side of the game
    r = ml.iloc[0]
    assert r["family"] == "sports_moneyline"
    assert r["same_direction"]
    side = r["polymarket_label"]  # the side chosen by hash
    assert mt.us_team("mlb", side) == mt.us_team(
        "mlb", r["kalshi_label"], r["kalshi_contract_id"][-3:]
    )
    tot = _pair(df, "KXMLBTOTAL-26JUN191905CINNYY-11")
    assert tot["family"] == "sports_total"
    assert tot["same_direction"] == (tot["polymarket_label"] == "Over")
    assert tot["event_key"] == r["event_key"] == "MLB-26JUN191905CINNYY"
    epl = df[df["kalshi_event_id"] == "KXEPLGAME-26APR10WHUWOL"]
    assert sorted(epl["kalshi_label"]) == ["Tie", "West Ham", "Wolverhampton"]
    assert epl["same_direction"].all()
    tie = epl[epl["kalshi_label"] == "Tie"].iloc[0]
    assert "draw" in tie["polymarket_title"]


def test_fed_and_duplicate_event(matched: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    df, _, _ = matched
    fed = df[df["family"] == "fed_decision"]
    assert set(fed["event_key"]) == {"fed:2026-06", "fed:2026-09"}
    assert len(fed) == 10
    # Polymarket listed the September meeting twice; the later listing is kept (the
    # first one settled in May, months before the meeting)
    sep = fed[fed["event_key"] == "fed:2026-09"]
    assert set(sep["polymarket_event_id"]) == {"481717"}
    assert _pair(df, "KXFEDDECISION-26SEP-C26")["polymarket_title"].startswith(
        "Will the Fed decrease interest rates by 50+ bps"
    )


def test_cpi_tails(matched: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    df, _, _ = matched
    cpi = df[df["family"] == "cpi"].set_index("kalshi_label")
    # "0.4% or more" is Kalshi's "above 0.3%"; "0.0% or less" is not "above 0.0%"
    assert cpi.loc["Above 0.3%", "same_direction"]
    assert "0.4% or more" in str(cpi.loc["Above 0.3%", "polymarket_title"])
    low = cpi[~cpi["same_direction"]]
    assert len(low) == 1
    assert "or less" in low.iloc[0]["polymarket_title"]


def test_awards(matched: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    df, _, _ = matched
    aw = df[df["kalshi_event_id"] == "KXGRAMBNA-68"]
    assert len(aw) >= 6
    assert (aw["family"] == "awards").all()
    for _, r in aw.iterrows():
        assert (
            mt.name_similarity(r["polymarket_title"].split(" win ")[0][5:], r["kalshi_label"]) > 0.8
        )


def test_negatives(matched: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    df, pm, k = matched
    # "Eastern Kentucky vs. North Alabama (W)" is not "Kentucky at Alabama" (W), same tip-off
    assert not df["kalshi_event_id"].str.startswith("KXNCAAWBGAME").any()
    # VMA "Best Art Direction" is not "Best Visual Effects"
    assert not (df["kalshi_event_id"] == "KXVMA-VIS26").any()
    # LA temperature: same airport, but Wunderground vs the NWS climate report, and on
    # this day the venues resolved the 70-71°F bracket differently
    wx = mt.pairs_frame(mt.weather_pairs(pm, k), pm, k)
    assert len(wx) >= 3
    assert (wx["outcome_agree"] == 0.0).any()
    assert not df["family"].str.startswith("weather").any()


def test_outcomes_agree(matched: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    df, _, _ = matched
    resolved = df["outcome_agree"].notna()
    assert resolved.sum() >= 20
    assert (df.loc[resolved, "outcome_agree"] == 1.0).all()
    assert df["pair_id"].is_unique
    assert df["polymarket_contract_id"].is_unique
    assert df["kalshi_contract_id"].is_unique


def test_complement_outcome_mapping() -> None:
    pm, k, _ = _frames()
    i = int(np.flatnonzero(pm["event_id"] == "590713")[0])
    j = int(np.flatnonzero(k["contract_id"] == "KXMLBTOTAL-26JUN191905CINNYY-11")[0])
    for same in (True, False):
        df = mt.pairs_frame([mt.Pair(i, j, same, "x", "e", 1.0)], pm, k)
        ko = float(k["outcome"].to_numpy()[j])
        assert df["kalshi_outcome_mapped"].iat[0] == (ko if same else 1.0 - ko)


# -- matched snapshots from the cache ------------------------------------------------------


def _write_cache(client: CachedClient, path: str, params: Any, data: Any) -> None:
    url = canonical_url(client.base_url, path, params)
    CachedClient._write(client.cache_path(url), url, 200, data)


def test_matched_snapshots_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    matched: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    df, pm, k = matched
    cfg = Config(study=StudyConfig(data_dir=str(tmp_path)))
    monkeypatch.delenv("PMCAL_DATA_DIR", raising=False)
    tot = df[df["kalshi_contract_id"] == "KXMLBTOTAL-26JUN191905CINNYY-11"]
    (cfg.interim_dir()).mkdir(parents=True)
    tot.to_parquet(cfg.interim_dir() / mt.MATCHES_FILE)
    prow = pm[pm["contract_id"] == tot["polymarket_contract_id"].iat[0]]
    krow = k[k["contract_id"] == tot["kalshi_contract_id"].iat[0]]
    prow.to_parquet(cfg.interim_dir() / "polymarket_index.parquet")
    krow.to_parquet(cfg.interim_dir() / "kalshi_index.parquet")
    p, kr = prow.iloc[0], krow.iloc[0]
    anchor = min(int(p["scheduled_ts"]), int(kr["scheduled_ts"]))
    # Polymarket: a flat 0.40 series every 10 minutes over the last 8 days
    pts = [{"t": t, "p": 0.40} for t in range(anchor - 8 * 86400, anchor, 600)]
    with polymarket.clob_client(cfg) as c:
        for params in polymarket.price_requests(
            str(p["pm_token_id"]), int(p["open_ts"]), int(p["scheduled_ts"]), int(p["close_ts"])
        ):
            _write_cache(c, "prices-history", params, {"history": pts})
    # Kalshi: hourly candles with a tight 0.58/0.62 book and a trade every hour
    candles = [
        {
            "end_period_ts": t,
            "yes_bid": {"close": 0.58},
            "yes_ask": {"close": 0.62},
            "price": {"close": 0.60, "previous": 0.60, "mean": 0.60},
            "volume": 10,
        }
        for t in range(anchor - 40 * 86400, anchor + 3600, 3600)
    ]
    with kalshi.make_client(cfg) as c:
        for req in kalshi.contract_requests(kr, cfg):
            _write_cache(c, req.path, req.params, {"candlesticks": candles})
    monkeypatch.setenv("PMCAL_OFFLINE", "1")
    out = mt.build_matched_snapshots(cfg, workers=1)
    assert len(out) == len(cfg.snapshots.horizons)
    assert (cfg.processed_dir() / mt.SNAPSHOTS_FILE).exists()
    h1 = out[out["horizon"] == "1h"].iloc[0]
    assert h1["t_snap"] == anchor - 3600
    assert h1["p_polymarket"] == pytest.approx(0.40)
    expected = 0.60 if tot["same_direction"].iat[0] else 0.40
    assert h1["p_kalshi"] == pytest.approx(expected)
    assert h1["live_kalshi"]
    assert h1["price_source_kalshi"] == "mid"
    assert h1["outcome"] == tot["polymarket_outcome"].iat[0]


def test_core_caps_games_per_league_month() -> None:
    t = 1_780_000_000
    rows = [
        {"family": "sports_moneyline", "event_key": f"MLB-G{g}", "kalshi_scheduled_ts": t}
        for g in range(10)
    ]
    rows += [{"family": "sports_total", "event_key": "MLB-G0", "kalshi_scheduled_ts": t}]
    rows += [{"family": "fed_decision", "event_key": "fed:2026-06", "kalshi_scheduled_ts": t}]
    df = mt.mark_core(pd.DataFrame(rows), Config())
    games = df.loc[df["core"] & (df["family"] == "sports_moneyline"), "event_key"]
    assert len(games) == Config().sample.max_events_per_series_month
    # a game's lines go with the game
    assert df.loc[df["family"] == "sports_total", "core"].iat[0] == ("MLB-G0" in set(games))
    assert df.loc[df["family"] == "fed_decision", "core"].all()


def test_find_matches_writes_all_and_core(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PMCAL_DATA_DIR", raising=False)
    pm, k, _ = _frames()
    cfg = Config(study=StudyConfig(data_dir=str(tmp_path)))
    cfg.interim_dir().mkdir(parents=True)
    pm.to_parquet(cfg.interim_dir() / "polymarket_index.parquet")
    k.to_parquet(cfg.interim_dir() / "kalshi_index.parquet")
    core = mt.find_matches(cfg)  # no cached series list: sports come from the tickers
    every = pd.read_parquet(cfg.interim_dir() / mt.ALL_MATCHES_FILE)
    written = pd.read_parquet(cfg.interim_dir() / mt.MATCHES_FILE)
    assert len(written) == len(core)
    assert set(written["pair_id"]) <= set(every["pair_id"])
    assert set(every.loc[every["core"].astype(bool), "pair_id"]) == set(written["pair_id"])
    assert {"fed_decision", "sports_moneyline"} <= set(every["family"])
