"""Polymarket pipeline, offline, on trimmed real API responses.

``fixtures/polymarket/markets_keyset.json`` holds 25 markets from Gamma's
``/markets/keyset?include_tag=true`` (descriptions and unused fields dropped),
chosen to cover Yes/No, team-name and Over/Under binaries, a neg-risk soccer event,
non-sports markets that carry a ``gameStartTime``, crypto up/down series of several
lengths, AMM-era markets, 50-50 and pending resolutions, and an early close.
``prices_history_3874448.json`` and ``trades_3874448.json`` are the CLOB price
history and data-api trades of the early-closed market (Alcaraz to reach the US Open
semifinals, closed 1.6 days before its end date). HTTP is served by
``httpx.MockTransport`` from these files.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest

from pmcal.config import Config, FrameConfig, StudyConfig
from pmcal.http import CachedClient
from pmcal.sample import frame_mask
from pmcal.schema import CATEGORIES, CONTRACT_COLUMNS, SNAPSHOT_COLUMNS
from pmcal.snapshots import PricePath, default_max_age
from pmcal.venues import polymarket as pm

FIXTURES = Path(__file__).parent / "fixtures" / "polymarket"
DAY = 86400

ALCARAZ = "3874448"  # early close; has price and trade fixtures
TRUMP_FAV = "518586"  # Yes/No, politics, 2025
MLB = "3743676"  # team-name moneyline, endDate a week after the game
SPREAD = "3035500"
TOTAL = "3035504"
LIMACHE_WIN = "2992279"  # neg-risk soccer moneyline; title names the original date
LIMACHE_DRAW = "2992281"
WEATHER = "4529742"  # non-sports market with a gameStartTime
SHUTDOWN = "623595"  # gameStartTime equals the day the outcome became known
BTC_HOURLY = "4523144"
BTC_5M = "4551368"
BTC_15M = "4551374"
BTC_DAILY = "528107"
TRUMP_UPDOWN = "669493"  # "Up or Down", but not crypto
LOL = "3899519"  # esports, closed before its (edited) game start
MENTIONS = "527695"  # endDate before the market was created
ECB = "2587797"
BOXING = "3886759"  # resolved 50-50
YANG = "12016"  # AMM era: payouts with 1e-7 residuals, market-level category only
MAYWEATHER = "161491"  # AMM era: final pool prices 0.90/0.10
PERU = "126965"  # AMM era: "Global Politics" + election
OTHER = "949819"  # "Other" placeholder of an augmented neg-risk event, unresolved
BTC_STRIKE_HOURLY = "3970517"  # "Bitcoin above 79,800 on August 29, 7AM ET?", hourly series
BTC_STRIKE_WEEKLY = "3779840"  # weekly strike series


def _markets() -> dict[str, dict[str, Any]]:
    page = json.loads((FIXTURES / "markets_keyset.json").read_text())
    return {m["id"]: m for m in page["markets"]}


@pytest.fixture(scope="module")
def markets() -> dict[str, dict[str, Any]]:
    return _markets()


@pytest.fixture(scope="module")
def rows(markets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {k: pm.parse_market(m) for k, m in markets.items()}


def _cfg(tmp_path: Path, start: str = "2026-08-01", end: str = "2026-09-30") -> Config:
    return Config(
        study=StudyConfig(start=start, end=end, data_dir=str(tmp_path / "data")),
        frame=FrameConfig(),  # the study's rule: traded at all, no lifetime-volume floor
    )


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PMCAL_DATA_DIR", raising=False)
    monkeypatch.delenv("PMCAL_OFFLINE", raising=False)


# -- parsing helpers --------------------------------------------------------------


def test_parse_ts_formats() -> None:
    assert pm.parse_ts("2025-01-14T22:46:28Z") == 1736894788
    assert pm.parse_ts("2025-01-14 22:46:28+00") == 1736894788
    assert pm.parse_ts("2025-01-14 22:46:28.00179+00") == 1736894788
    assert pm.parse_ts("2025-01-14") == 1736812800
    assert pm.parse_ts(None) is None
    assert pm.parse_ts("") is None
    assert pm.parse_ts("not a date") is None


# -- outcome ----------------------------------------------------------------------


def test_outcome_parsing_rules() -> None:
    assert pm.parse_resolution(["1", "0"], 0, "resolved") == ("resolved", 1.0)
    assert pm.parse_resolution(["1", "0"], 1, "resolved") == ("resolved", 0.0)
    # AMM-era residuals are within tolerance...
    amm = ["0.00000009792476954804243452949098140021029", "0.9999999020752304519575654705090186"]
    assert pm.parse_resolution(amm, 1, None) == ("resolved", 1.0)
    # ...a closed CLOB market quoting half a tick from 1 is not a resolution
    assert pm.parse_resolution(["0.9995", "0.0005"], 0, None)[0] == "split"
    status, outcome = pm.parse_resolution(["0.5", "0.5"], 0, "resolved")
    assert status == "split"
    assert np.isnan(outcome)
    assert pm.parse_resolution(["0.32", "0.68"], 0, "resolved")[0] == "split"
    assert pm.parse_resolution(["0", "0"], 0, "resolved")[0] == "void"
    assert pm.parse_resolution(["1", "0"], 0, "proposed")[0] == "unresolved"
    assert pm.parse_resolution(["1", "0"], 0, "disputed")[0] == "unresolved"
    assert pm.parse_resolution([], 0, None)[0] == "unresolved"
    assert pm.parse_resolution(["x", "0"], 0, None)[0] == "unresolved"


def test_outcome_parsing_on_fixtures(rows: dict[str, dict[str, Any]]) -> None:
    r = rows[TRUMP_FAV]  # outcomePrices ["0", "1"]: No won
    assert (r["resolution_status"], r["outcome"], r["outcome_label"]) == ("resolved", 0.0, "Yes")
    assert rows[BOXING]["resolution_status"] == "split"
    assert rows[YANG]["resolution_status"] == "resolved"
    assert rows[YANG]["outcome"] == 0.0
    assert rows[MAYWEATHER]["resolution_status"] == "split"
    other = rows[OTHER]
    assert other["resolution_status"] == "unresolved"
    assert other["pm_neg_risk_other"] is True
    for r in rows.values():
        assert r["resolution_status"] in {"resolved", "split", "void", "unresolved"}
        assert np.isnan(r["outcome"]) == (r["resolution_status"] != "resolved")


# -- side -------------------------------------------------------------------------


def test_side_yes_no() -> None:
    assert pm.choose_side(["Yes", "No"], "0xabc") == 0
    assert pm.choose_side(["No", "Yes"], "0xabc") == 1


def test_side_hash_is_deterministic_and_balanced() -> None:
    teams = ["Home", "Away"]
    assert pm.choose_side(teams, "0xAbC") == pm.choose_side(teams, "0xabc")
    ids = [f"0x{i:064x}" for i in range(4000)]
    first = np.mean([pm.choose_side(teams, c) == 0 for c in ids])
    assert 0.46 < first < 0.54


def test_side_on_fixtures(
    markets: dict[str, dict[str, Any]], rows: dict[str, dict[str, Any]]
) -> None:
    for mid in (MLB, SPREAD, TOTAL, BOXING, PERU, BTC_HOURLY):
        m, r = markets[mid], rows[mid]
        outcomes = json.loads(m["outcomes"])
        side = outcomes.index(r["outcome_label"])
        assert side == pm.choose_side(outcomes, m["conditionId"])
        if r["pm_has_clob"]:
            assert r["contract_id"] == json.loads(m["clobTokenIds"])[side]
        if r["resolution_status"] == "resolved":
            assert r["outcome"] == round(float(json.loads(m["outcomePrices"])[side]))
    assert rows[TOTAL]["outcome_label"] in {"Over", "Under"}


# -- category ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tags", "market_category", "title", "slugs", "expected"),
    [
        (["sports", "nfl"], None, "", (), ("sports", "tags")),
        (["nfl", "business", "sports"], None, "Who will buy the Seahawks?", (), ("sports", "tags")),
        (["politics", "geopolitics", "ukraine"], None, "Ceasefire by June?", (), ("world", "tags")),
        (
            ["world", "ukraine"],
            None,
            "Ukraine presidential election held?",
            (),
            ("politics", "tags"),
        ),
        (["ecb", "economy", "politics"], None, "", (), ("economics", "tags")),
        (["economy", "politics"], None, "Government shutdown?", (), ("politics", "tags")),
        (["mentions", "politics", "trump"], None, "", (), ("culture", "tags")),
        (["pop-culture", "politics", "tweets-markets"], None, "", (), ("culture", "tags")),
        (["business", "finance", "tech", "economy"], None, "", (), ("finance", "tags")),
        (["crypto", "bitcoin", "politics"], None, "", (), ("crypto", "tags")),
        (["weather", "paris"], None, "", (), ("weather", "tags")),
        (["ai", "openai"], None, "", (), ("science_tech", "tags")),
        (["earnings", "equities"], None, "", (), ("finance", "tags")),
        # tickers are tags: AFL is Aflac here, not Australian football
        (["earnings", "equities", "finance", "afl"], None, "", (), ("finance", "tags")),
        (["all"], "US-current-affairs", "", (), ("politics", "market_category")),
        (["all"], "Global Politics", "Netanyahu PM?", (), ("world", "market_category")),
        (
            ["all"],
            "Global Politics",
            "Peru Presidential Election?",
            (),
            ("politics", "market_category"),
        ),
        (["all"], None, "", ("nfl-thursday-colts-vs-broncos",), ("sports", "slug")),
        (["all"], None, "", ("fifa-world-cup-winner",), ("sports", "slug")),
        (["all", "something"], None, "", ("will-it-happen",), ("other", "none")),
    ],
)
def test_category_mapping(
    tags: list[str],
    market_category: str | None,
    title: str,
    slugs: tuple[str, ...],
    expected: tuple[str, str],
) -> None:
    assert pm.categorize(tags, market_category, title, slugs) == expected


def test_category_on_fixtures(rows: dict[str, dict[str, Any]]) -> None:
    expected = {
        TRUMP_FAV: "politics",
        MLB: "sports",
        LOL: "sports",
        WEATHER: "weather",
        ECB: "economics",
        MENTIONS: "culture",
        BTC_HOURLY: "crypto",
        SHUTDOWN: "politics",
        YANG: "politics",
        PERU: "politics",
    }
    for mid, cat in expected.items():
        assert rows[mid]["category"] == cat, rows[mid]["title"]
    for r in rows.values():
        assert r["category"] in CATEGORIES
    assert rows[YANG]["venue_category"].startswith("category:US-current-affairs")
    assert "mlb" in rows[MLB]["venue_category"].split("|")


# -- anchor -----------------------------------------------------------------------


def test_anchor_game_start_for_sports(
    markets: dict[str, dict[str, Any]], rows: dict[str, dict[str, Any]]
) -> None:
    r = rows[MLB]
    gst = pm.parse_ts(markets[MLB]["gameStartTime"])
    assert (r["anchor_source"], r["scheduled_ts"], r["event_start_ts"]) == ("game_start", gst, gst)
    # the moneyline's endDate is a week after the game: it must not be the anchor
    assert r["pm_end_date_ts"] - r["scheduled_ts"] == 7 * DAY


def test_anchor_ignores_game_start_outside_sports(
    markets: dict[str, dict[str, Any]], rows: dict[str, dict[str, Any]]
) -> None:
    for mid in (WEATHER, SHUTDOWN):
        r = rows[mid]
        assert markets[mid].get("gameStartTime")
        assert r["anchor_source"] == "end_date"
        assert r["scheduled_ts"] == pm.parse_ts(markets[mid]["endDate"])
        assert r["event_start_ts"] is None
    # the shutdown ladder's gameStartTime is when the outcome became known
    assert rows[SHUTDOWN]["pm_game_start_ts"] > rows[SHUTDOWN]["close_ts"]


def test_anchor_rules() -> None:
    assert pm.choose_anchor(100, None, 50, True) == (50, "game_start", 50)
    assert pm.choose_anchor(100, 40, 50, True) == (40, "game_start", 40)
    assert pm.choose_anchor(100, 40, 50, False) == (100, "end_date", None)
    assert pm.choose_anchor(100, None, None, True) == (100, "end_date", None)


def test_anchor_flags(rows: dict[str, dict[str, Any]]) -> None:
    assert "game_start_after_close" in rows[LOL]["pm_anchor_flags"]
    assert "scheduled_before_created" in rows[MENTIONS]["pm_anchor_flags"]
    assert "title_date_mismatch" in rows[LIMACHE_WIN]["pm_anchor_flags"]
    assert rows[MLB]["pm_anchor_flags"] == ""
    assert rows[ALCARAZ]["pm_anchor_flags"] == ""


def test_open_and_close(
    markets: dict[str, dict[str, Any]], rows: dict[str, dict[str, Any]]
) -> None:
    m, r = markets[ALCARAZ], rows[ALCARAZ]
    assert r["open_ts"] == pm.parse_ts(m["acceptingOrdersTimestamp"])
    assert r["close_ts"] == r["resolved_ts"] == pm.parse_ts(m["closedTime"])
    # an AMM-era market has no acceptingOrdersTimestamp/startDate: createdAt
    assert rows[YANG]["open_ts"] == pm.parse_ts(markets[YANG]["createdAt"])
    assert rows[YANG]["pm_has_clob"] is False


# -- short crypto -----------------------------------------------------------------


def test_short_crypto_detection(rows: dict[str, dict[str, Any]]) -> None:
    for mid, interval in ((BTC_5M, 300), (BTC_15M, 900), (BTC_HOURLY, 3600)):
        assert rows[mid]["is_short_crypto"] is True
        assert rows[mid]["pm_updown_interval_s"] == interval
    assert rows[BTC_DAILY]["is_short_crypto"] is False
    assert rows[BTC_DAILY]["category"] == "crypto"
    assert rows[TRUMP_UPDOWN]["is_short_crypto"] is False
    # any crypto series with sub-daily recurrence, not only "Up or Down"
    assert rows[BTC_STRIKE_HOURLY]["pm_series_recurrence"] == "hourly"
    assert rows[BTC_STRIKE_HOURLY]["is_short_crypto"] is True
    assert rows[BTC_STRIKE_WEEKLY]["pm_series_recurrence"] == "weekly"
    assert rows[BTC_STRIKE_WEEKLY]["is_short_crypto"] is False


def test_short_crypto_rule() -> None:
    base: dict[str, Any] = {
        "is_crypto": True,
        "recurrence": None,
        "series_slug": None,
        "tags": [],
        "updown_interval": None,
        "open_ts": 0,
        "scheduled_ts": 10 * pm.DAY,
    }
    assert not pm.short_crypto(**base)
    assert not pm.short_crypto(**{**base, "is_crypto": False, "recurrence": "hourly"})
    for rec in ("5m", "15m", "hourly", "4h"):
        assert pm.short_crypto(**{**base, "recurrence": rec})
    for rec in ("daily", "weekly", "monthly"):
        assert not pm.short_crypto(**{**base, "recurrence": rec, "tags": ["1h"]})
    # no recurrence: slug suffix, interval tag, up/down interval, then scheduled life
    assert pm.short_crypto(**{**base, "series_slug": "eth-multi-strikes-hourly"})
    assert not pm.short_crypto(**{**base, "series_slug": "eth-up-or-down-daily"})
    assert pm.short_crypto(**{**base, "tags": ["crypto", "15M"]})
    assert pm.short_crypto(**{**base, "updown_interval": 3600})
    assert not pm.short_crypto(**{**base, "updown_interval": pm.DAY})
    assert pm.short_crypto(**{**base, "scheduled_ts": 4 * 3600})
    assert not pm.short_crypto(**{**base, "scheduled_ts": 5 * 3600})


def test_updown_interval_from_question() -> None:
    q4h = "Bitcoin Up or Down - September 15, 8:00AM-12:00PM ET"
    assert pm.updown_interval_s(q4h, None, [], None, None) == 4 * 3600
    assert pm.updown_interval_s("XRP Up or Down - May 2, 11PM ET", None, [], None, None) == 3600
    assert pm.updown_interval_s("Bitcoin Up or Down on March 14?", None, [], None, None) is None
    assert pm.updown_interval_s("Will BTC go up?", "btc-up-or-down-5m", [], None, None) is None
    assert pm.updown_interval_s("ETH Up or Down", "eth-up-or-down-4h", [], None, None) == 14400


# -- the index --------------------------------------------------------------------


def test_assemble_index(tmp_path: Path, rows: dict[str, dict[str, Any]]) -> None:
    cfg = _cfg(tmp_path)
    df = pm.assemble_index(list(rows.values()), cfg)
    assert list(df.columns) == list(pm.INDEX_COLUMNS)
    for col, dtype in pm.INDEX_COLUMNS.items():
        assert str(df[col].dtype) == dtype, col
    assert set(CONTRACT_COLUMNS) <= set(df.columns)
    extra = set(df.columns) - set(CONTRACT_COLUMNS) - {"volume_missing"}
    assert all(c.startswith("pm_") for c in extra)
    assert not df["volume_missing"].any()
    assert (df["pm_volume_source"] == "gamma").all()
    by_id = df.set_index("pm_market_id")
    # filtered on the scheduled time, not the close
    sched = df["scheduled_ts"].astype("int64")
    assert ((sched >= cfg.study.start_ts) & (sched < cfg.study.end_ts)).all()
    assert TRUMP_FAV not in by_id.index
    assert MLB in by_id.index  # game Aug 27; its endDate (Sep 3) also in window
    # short crypto stays in the index, out of the frame
    assert by_id.loc[BTC_5M, "is_short_crypto"]
    assert not by_id.loc[BTC_5M, "pm_in_frame"]
    assert by_id.loc[MLB, "pm_in_frame"]
    # the neg-risk soccer moneyline (3 outcomes) is one event; the spread and total
    # are in the game's "More Markets" event, which shares the gameId
    ev = by_id.loc[LIMACHE_WIN, "event_id"]
    assert (df["event_id"] == ev).sum() == 3
    assert by_id.loc[LIMACHE_WIN, "n_in_event"] == 3
    assert by_id.loc[SPREAD, "event_id"] != ev
    assert by_id.loc[SPREAD, "pm_game_id"] == by_id.loc[LIMACHE_WIN, "pm_game_id"]
    # ...so they share the bootstrap cluster; non-game markets cluster on the event
    assert by_id.loc[SPREAD, "cluster_id"] == "game:90105687"
    assert by_id.loc[LIMACHE_WIN, "cluster_id"] == "game:90105687"
    assert by_id.loc[ECB, "cluster_id"] == by_id.loc[ECB, "event_id"]
    assert by_id.loc[LIMACHE_DRAW, "mutually_exclusive"]
    assert not by_id.loc[SPREAD, "mutually_exclusive"]
    assert by_id.loc[BTC_5M, "n_in_event"] == 0  # excluded from the frame
    # never traded: the placeholder is not in the index
    assert OTHER not in by_id.index
    assert (df["volume_usd"] > 0).all()
    # traded at all is enough: no lifetime-volume floor
    thin = {**rows[MLB], "pm_market_id": "thin", "volume_usd": 8.0, "volume_native": 8.0}
    zero = {**rows[MLB], "pm_market_id": "zero", "volume_usd": 0.0, "volume_native": 0.0}
    idx2 = pm.assemble_index([*rows.values(), thin, zero], cfg).set_index("pm_market_id")
    assert idx2.loc["thin", "pm_in_frame"]
    assert "zero" not in idx2.index
    # pm_in_frame is pmcal.sample.frame_mask
    assert (df["pm_in_frame"].astype(bool) == frame_mask(df, cfg)).all()


def _all_markets() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in ("markets_keyset.json", "markets_null_volume.json"):
        out.extend(json.loads((FIXTURES / name).read_text())["markets"])
    return out


def _gamma_transport(log: list[dict[str, str]] | None = None) -> httpx.MockTransport:
    """Gamma's /markets/keyset over the fixtures, one page per query. Like the real API,
    a volume floor drops markets whose volume is null, and ``order=closedTime`` sorts
    on the close."""
    markets = _all_markets()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/markets/keyset"
        q = dict(request.url.params)
        if log is not None:
            log.append(q)
        assert q["closed"] == "true"
        assert q["include_tag"] == "true"
        lo, hi = pm.parse_ts(q["end_date_min"]), pm.parse_ts(q["end_date_max"])
        assert lo is not None
        assert hi is not None
        ms = [m for m in markets if lo <= (pm.parse_ts(m["endDate"]) or 0) <= hi]
        if "volume_num_min" in q:
            vmin = float(q["volume_num_min"])
            ms = [m for m in ms if m.get("volumeNum") is not None and m["volumeNum"] >= vmin]
        if q.get("order") == "closedTime":
            ms.sort(key=lambda m: pm.parse_ts(m.get("closedTime")) or 0)
            if q["ascending"] == "false":
                ms.reverse()
        return httpx.Response(200, json={"markets": ms})

    return httpx.MockTransport(handler)


def _live_volume_transport() -> httpx.MockTransport:
    lv = json.loads((FIXTURES / "live_volume.json").read_text())

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/live-volume"
        return httpx.Response(200, json=lv.get(request.url.params["id"], [{"total": 0}]))

    return httpx.MockTransport(handler)


def _patch_clients(monkeypatch: pytest.MonkeyPatch, **transports: httpx.MockTransport) -> None:
    urls = {"gamma": pm.GAMMA_URL, "clob": pm.CLOB_URL, "data": pm.DATA_URL}
    for host, transport in transports.items():

        def make(c: Config, host: str = host, t: httpx.MockTransport = transport) -> CachedClient:
            return CachedClient(
                urls[host], c.raw_dir(pm.VENUE) / host, rate_per_s=1000, transport=t
            )

        monkeypatch.setattr(pm, f"{host}_client", make)


def test_build_index_offline_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    _patch_clients(monkeypatch, gamma=_gamma_transport(), data=_live_volume_transport())
    df = pm.build_index(cfg, workers=2)
    assert (cfg.interim_dir() / pm.INDEX_FILE).exists()
    assert MLB in set(df["pm_market_id"])
    monkeypatch.setenv("PMCAL_OFFLINE", "1")
    again = pm.build_index(cfg, workers=2)
    pd.testing.assert_frame_equal(df, again)


def test_plan_windows_volume_gap() -> None:
    lo, hi = pm.parse_ts("2026-01-01"), pm.parse_ts("2026-06-01")
    assert lo is not None
    assert hi is not None
    plan = pm.plan_windows(lo, hi)
    u_lo = pm.VOLUME_GAP[0] - pm.LIST_MARGIN_BEFORE
    u_hi = pm.VOLUME_GAP[1] + pm.LIST_MARGIN_AFTER
    plain = sorted((w for w in plan if not w.sweep), key=lambda w: w.lo)
    # the plain windows tile the range, without the floor exactly on the gap's range
    assert plain[0].lo == lo
    assert plain[-1].hi == hi
    assert all(a.hi == b.lo for a, b in itertools.pairwise(plain))
    for w in plain:
        assert w.floor == (w.hi <= u_lo or w.lo >= u_hi)
    assert min(w.lo for w in plain if not w.floor) == u_lo
    assert max(w.hi for w in plain if not w.floor) == u_hi
    # each floored window is also swept in closedTime order, towards the gap
    sweeps = [w for w in plan if w.sweep]
    assert all(not w.floor and w.hi - w.lo <= pm.SWEEP_WINDOW for w in sweeps)
    assert {w.sweep for w in sweeps if w.hi <= u_lo} == {"desc"}
    assert {w.sweep for w in sweeps if w.lo >= u_hi} == {"asc"}
    assert sum(w.hi - w.lo for w in sweeps) == sum(w.hi - w.lo for w in plain if w.floor)


def test_list_params_floor() -> None:
    w = pm.ListWindow(0, pm.DAY)
    # "traded at all": a tiny floor, sent as a plain decimal string
    assert pm._list_params(w, 0.0)["volume_num_min"] == "0.000001"
    assert pm._list_params(w, 1000.0)["volume_num_min"] == 1000.0
    assert "volume_num_min" not in {
        k for k, v in pm._list_params(pm.ListWindow(0, pm.DAY, floor=False), 0.0).items() if v
    }
    v = pd.Series([0.0, 0.5, 2000.0, np.nan])
    assert pm._volume_ok(v, 0.0).tolist() == [False, True, True, True]
    assert pm._volume_ok(v, 1000.0).tolist() == [False, False, True, True]


def test_past_gap() -> None:
    before = {"closedTime": "2026-02-20 00:00:00+00"}
    inside = {"closedTime": "2026-03-10 00:00:00+00"}
    after = {"closedTime": "2026-04-02 00:00:00+00"}
    assert not pm._past_gap([before, inside], "asc")
    assert pm._past_gap([inside, after], "asc")
    assert not pm._past_gap([after, inside], "desc")
    assert pm._past_gap([inside, before], "desc")
    assert pm._past_gap([{"closedTime": None}], "asc")
    assert pm._past_gap([], "desc")


def test_null_volume_parsed_as_missing() -> None:
    m = json.loads((FIXTURES / "markets_null_volume.json").read_text())["markets"][0]
    r = pm.parse_market(m)
    assert r["volume_missing"] is True
    assert np.isnan(r["volume_usd"])
    assert r["pm_volume_source"] == "missing"
    # outside the gap a null volume means the market never traded
    late = pm.parse_market({**m, "closedTime": "2026-04-10 12:00:00+00"})
    assert late["volume_missing"] is False
    assert late["volume_usd"] == 0.0
    assert late["pm_volume_source"] == "gamma_null"


def test_build_index_fills_volume_gap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, start="2026-03-01", end="2026-03-31")
    queries: list[dict[str, str]] = []
    _patch_clients(monkeypatch, gamma=_gamma_transport(queries), data=_live_volume_transport())
    df = pm.build_index(cfg, workers=2)
    nv = df.loc[df["volume_missing"].astype(bool)].set_index("pm_market_id")
    # both null-volume markets are found (a floored listing would drop them)...
    assert set(nv.index) == {"1421535", "1421540"}
    # ...and get /live-volume's figure
    assert nv.loc["1421535", "volume_usd"] == pytest.approx(8359.278306)
    assert nv.loc["1421540", "volume_native"] == pytest.approx(5821.577376)
    assert (nv["pm_volume_source"] == "live_volume").all()
    assert nv["pm_in_frame"].all()
    # the gap's endDate range was listed without the floor, the rest swept
    assert any("volume_num_min" not in q and "order" not in q for q in queries)
    assert any(q.get("order") == "closedTime" for q in queries)


def test_fill_missing_volume_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, start="2026-03-01", end="2026-03-31")
    _patch_clients(monkeypatch, data=_live_volume_transport())
    rows = pd.DataFrame([pm.parse_market(m) for m in _all_markets()])
    rows = rows.loc[rows["volume_missing"]].copy()
    other = rows.iloc[[0]].assign(pm_market_id="x", market_id="0xnot-listed")
    unknown = rows.iloc[[0]].assign(pm_market_id="y", event_id="555")
    rows = pd.concat([rows, other, unknown], ignore_index=True)
    # nothing cached yet: a cache-only fill leaves NaN
    assert pm.fill_missing_volume(rows, cfg)["volume_usd"].isna().all()
    pm.fill_missing_volume(rows, cfg, fetch=True, workers=2)
    monkeypatch.setenv("PMCAL_OFFLINE", "1")
    out = pm.fill_missing_volume(rows, cfg).set_index("pm_market_id")
    vol = out["volume_usd"].astype("float64")
    assert vol["1421535"] == pytest.approx(8359.278306)
    assert vol["x"] == 0.0  # not listed in a non-empty response
    assert np.isnan(vol["y"])  # the event has no volume data
    assert out["volume_missing"].all()


# -- prices -----------------------------------------------------------------------


def test_fine_windows() -> None:
    o, s = 0, 100 * DAY
    lb = pm.FINE_LOOKBACK
    assert pm.fine_windows(o, s, None) == [(s - lb, s)]
    # early close: one window ending at the close covers both anchors
    assert pm.fine_windows(o, s, s - 3 * DAY) == [(s - 3 * DAY - lb, s - 3 * DAY)]
    assert pm.fine_windows(o, s, s - 30 * DAY) == [(s - 30 * DAY - lb, s - 30 * DAY)]
    # closed hours after a game start: extended to the close
    assert pm.fine_windows(o, s, s + 3 * 3600) == [(s - lb, s + 3 * 3600)]
    # closed long after: two windows
    assert pm.fine_windows(o, s, s + 20 * DAY) == [(s - lb, s), (s + 12 * DAY, s + 20 * DAY)]
    # windows ending before the market opened are skipped
    assert pm.fine_windows(s + DAY, s, None) == []
    for wins in (pm.fine_windows(o, s, c) for c in (None, s - DAY, s + DAY, s + 9 * DAY)):
        assert all(b - a <= pm.MAX_PRICE_WINDOW for a, b in wins)


def test_split_window() -> None:
    assert pm.split_window(0, 40 * DAY) == [
        (0, 15 * DAY),
        (15 * DAY, 30 * DAY),
        (30 * DAY, 40 * DAY),
    ]
    assert pm.split_window(5, 5) == []


def test_history_to_path() -> None:
    data = {
        "history": [
            {"t": 300, "p": 0.3},
            {"t": 100, "p": "0.1"},
            {"p": 0.9},
            {"t": 200, "p": None},
        ]
    }
    path = pm.history_to_path(data)
    assert path.ts.tolist() == [100, 200, 300]
    assert path.price[0] == 0.1
    assert np.isnan(path.price[1])
    assert pm.history_to_path({"history": []}).ts.size == 0
    assert pm.history_to_path(None).ts.size == 0
    daily = pm.history_to_path({"history": [{"t": 0, "p": 0.5}, {"t": 600, "p": 0.6}]})
    fine = pm.history_to_path({"history": [{"t": 600, "p": 0.65}, {"t": 1200, "p": 0.7}]})
    merged = PricePath.merge(daily, fine)
    assert merged.ts.tolist() == [0, 600, 1200]
    assert merged.price.tolist() == [0.5, 0.65, 0.7]


def test_history_fixture_to_path() -> None:
    fx = json.loads((FIXTURES / "prices_history_3874448.json").read_text())
    daily = pm.history_to_path(fx["daily"]["response"])
    fine = pm.history_to_path(fx["fine"]["response"])
    assert daily.ts.size == len(fx["daily"]["response"]["history"])
    assert np.all(np.diff(daily.ts) > 0)
    assert np.all((fine.price >= 0) & (fine.price <= 1))
    # daily points sit at about 00:00 UTC; the fine series is on a 10-minute grid
    assert np.all(daily.ts % DAY < 60)
    assert np.median(np.diff(fine.ts)) == pytest.approx(600, abs=2)


def test_parse_trades_maps_other_token() -> None:
    data = [
        {"asset": "B", "price": 0.3, "timestamp": 20},
        {"asset": "A", "price": 0.6, "timestamp": 10},
    ]
    tr = pm.parse_trades(data, "A")
    assert tr.ts.tolist() == [10, 20]
    assert tr.price.tolist() == pytest.approx([0.6, 0.7])


# -- fetch + snapshots end to end ---------------------------------------------------


def _price_transport() -> httpx.MockTransport:
    fx = json.loads((FIXTURES / "prices_history_3874448.json").read_text())
    token = str(fx["daily"]["params"]["market"])
    fine = fx["fine"]["response"]["history"]

    def handler(request: httpx.Request) -> httpx.Response:
        q = request.url.params
        if q["market"] != token:
            return httpx.Response(200, json={"history": []})
        if q.get("interval") == "max":
            assert q["fidelity"] == "1440"
            return httpx.Response(200, json=fx["daily"]["response"])
        a, b = int(q["startTs"]), int(q["endTs"])
        if b - a > pm.MAX_PRICE_WINDOW:
            return httpx.Response(400, json={"error": "interval is too long"})
        return httpx.Response(200, json={"history": [p for p in fine if a <= p["t"] <= b]})

    return httpx.MockTransport(handler)


def _trades_transport() -> httpx.MockTransport:
    trades = json.loads((FIXTURES / "trades_3874448.json").read_text())

    def handler(request: httpx.Request) -> httpx.Response:
        q = request.url.params
        start, end, limit = int(q.get("start", 0)), int(q["end"]), int(q["limit"])
        sel = [t for t in trades if t["conditionId"] == q["market"]]
        sel = [t for t in sel if start <= t["timestamp"] <= end]
        sel.sort(key=lambda t: -t["timestamp"])  # newest first, like the API
        return httpx.Response(200, json=sel[:limit])

    return httpx.MockTransport(handler)


@pytest.fixture
def mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    cfg = _cfg(tmp_path)
    _patch_clients(monkeypatch, clob=_price_transport(), data=_trades_transport())
    return cfg


def _index(cfg: Config, rows: dict[str, dict[str, Any]], ids: list[str]) -> pd.DataFrame:
    df = pm.assemble_index(list(rows.values()), cfg)
    return df.loc[df["pm_market_id"].isin(ids)].reset_index(drop=True)


def test_fetch_then_snapshots_offline(
    mocked: Config, monkeypatch: pytest.MonkeyPatch, rows: dict[str, dict[str, Any]]
) -> None:
    cfg = mocked
    contracts = _index(cfg, rows, [ALCARAZ, MLB])
    assert len(contracts) == 2
    pm.fetch_prices(contracts, cfg, workers=2)

    monkeypatch.setenv("PMCAL_OFFLINE", "1")
    snaps = pm.build_snapshots(contracts, cfg)
    assert list(snaps.columns) == list(SNAPSHOT_COLUMNS) + list(pm.PM_SNAPSHOT_COLUMNS)
    for col, dtype in SNAPSHOT_COLUMNS.items():
        assert str(snaps[col].dtype) == dtype, col
    n_h = len(cfg.snapshots.horizons)
    assert len(snaps) == 2 * 2 * n_h
    assert (snaps["pm_cache_missing"] == 0).all()

    a = snaps.loc[snaps["contract_id"] == rows[ALCARAZ]["contract_id"]]
    sched = a.loc[a["anchor"] == "scheduled"].set_index("horizon")
    close = a.loc[a["anchor"] == "close"].set_index("horizon")
    r = rows[ALCARAZ]
    assert (sched["t_snap"] == r["scheduled_ts"] - sched["horizon_s"]).all()
    assert (close["t_snap"] == r["close_ts"] - close["horizon_s"]).all()
    # closed 1.6 days early: the scheduled anchor's short horizons are after the close
    assert sched.loc["1d", "status"] == "closed"
    assert sched.loc["1h", "status"] == "closed"
    assert sched.loc["30d", "status"] == "not_open"
    assert sched.loc["3d", "status"] == "ok"
    assert sched.loc["7d", "status"] == "ok"
    # the leaky close anchor has a price right up to the close
    assert close.loc["1h", "status"] == "ok"
    assert close.loc["6h", "status"] == "ok"

    ok = a.loc[a["status"] == "ok"]
    assert (ok["obs_ts"] <= ok["t_snap"]).all()
    ages = ok["t_snap"] - ok["obs_ts"]
    assert all(ag <= default_max_age(int(h)) for ag, h in zip(ages, ok["horizon_s"], strict=True))
    assert (ok["price_source"] == "series").all()

    # trade prints on the ok scheduled snapshots only
    s_ok = sched.loc[sched["status"] == "ok"]
    assert s_ok["n_trades_24h"].notna().all()
    assert close["n_trades_24h"].isna().all()
    trades = json.loads((FIXTURES / "trades_3874448.json").read_text())
    for _, row in s_ok.iterrows():
        t = int(row["t_snap"])
        n = sum(1 for x in trades if t - DAY < x["timestamp"] <= t)
        assert row["n_trades_24h"] == n
        before = [x["timestamp"] for x in trades if x["timestamp"] <= t]
        if before:
            assert row["last_trade_age_s"] == t - max(before)
            assert 0 <= row["last"] <= 1
        else:
            assert np.isnan(row["last_trade_age_s"])

    # the other contract has no price data in the mock: no_data inside its life
    b = snaps.loc[snaps["contract_id"] == rows[MLB]["contract_id"]]
    assert set(b["status"]) <= {"no_data", "not_open", "closed"}
    assert "no_data" in set(b["status"])
    assert (b["pm_path_n"] == 0).all()

    # public helpers read the same cache
    row = contracts.loc[contracts["pm_market_id"] == ALCARAZ].iloc[0]
    path = pm.load_price_path(row, cfg)
    assert path.ts.size == a["pm_path_n"].iloc[0] > 0


def test_fetch_windows_and_load_trades_window(
    mocked: Config, rows: dict[str, dict[str, Any]]
) -> None:
    cfg = mocked
    contracts = _index(cfg, rows, [ALCARAZ])
    row = contracts.iloc[0]
    assert pm.load_price_path(row, cfg).ts.size == 0  # nothing cached yet
    close = int(row["close_ts"])
    extra = [(close - 20 * DAY, close)]  # longer than 15 days: split into pieces
    pm.fetch_windows(row, extra, cfg)
    path = pm.load_price_path(row, cfg, extra_windows=extra)
    assert path.ts.size > 0
    assert path.ts.max() <= close

    t = close - 2 * DAY
    n0, age0 = pm.load_trades_window(row, t, cfg)  # cache only: not fetched yet
    assert n0 is None
    assert np.isnan(age0)
    n, age = pm.load_trades_window(row, t, cfg, fetch=True)
    trades = json.loads((FIXTURES / "trades_3874448.json").read_text())
    assert n == sum(1 for x in trades if t - DAY < x["timestamp"] <= t)
    before = [x["timestamp"] for x in trades if x["timestamp"] <= t]
    assert before
    assert age == t - max(before)
    # now cached
    assert pm.load_trades_window(row, t, cfg) == (n, age)


def test_snapshot_rows_without_close_time(rows: dict[str, dict[str, Any]], tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    rec = dict(rows[ALCARAZ])
    rec["close_ts"] = None
    res = pm.contract_snapshots(rec, cfg, lambda h, p, q: (False, None))
    close_rows = [r for r in res.rows if r["anchor"] == "close"]
    assert len(close_rows) == len(cfg.snapshots.horizons)
    assert {r["status"] for r in close_rows} == {"no_data"}
    assert res.missing > 0


def test_contract_ref_from_series(rows: dict[str, dict[str, Any]], tmp_path: Path) -> None:
    df = pm.assemble_index([rows[ALCARAZ]], _cfg(tmp_path))
    ref = pm.ContractRef.from_row(df.iloc[0])
    assert ref.has_clob
    assert ref.token_id == rows[ALCARAZ]["pm_token_id"]
    assert ref.close_ts == rows[ALCARAZ]["close_ts"]
