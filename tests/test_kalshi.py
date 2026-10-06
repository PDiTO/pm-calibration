"""Kalshi pipeline, offline, from trimmed real API responses in ``fixtures/kalshi``.

The fixtures are real responses captured from the public trade API in October 2026
(long text fields such as the rules dropped). Requests are served by an
``httpx.MockTransport`` so the real ``CachedClient`` (cache, offline mode) is used.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import numpy as np
import pandas as pd
import pytest

from pmcal.config import Config, FrameConfig, StudyConfig
from pmcal.http import CachedClient, OfflineCacheMiss
from pmcal.schema import CATEGORIES, CONTRACT_COLUMNS, SNAPSHOT_COLUMNS
from pmcal.venues import kalshi

FIX = Path(__file__).parent / "fixtures" / "kalshi"


def _load(name: str) -> Any:
    return json.loads((FIX / name).read_text())


def _utc(s: str) -> int:
    ts = kalshi.parse_ts(s)
    assert ts is not None
    return ts


SERIES = _load("series.json")
MARKETS = _load("markets.json")
EVENTS = _load("events.json")
CANDLES_HIST = _load("candles_historical.json")
CANDLES_LIVE = _load("candles_live.json")

BY_TICKER: dict[str, dict[str, Any]] = {
    m["ticker"]: {**m, "_tier": tier} for tier in ("historical", "live") for m in MARKETS[tier]
}
EVENT_BY_TICKER: dict[str, dict[str, Any]] = {e["event_ticker"]: e for e in EVENTS["events"]}
SERIES_ROW: dict[str, dict[str, Any]] = {
    s["ticker"]: {
        "series_id": s["ticker"],
        "venue_category": s["category"],
        "frequency": s["frequency"],
        "series_volume": float(s["volume_fp"]),
    }
    for s in SERIES["series"]
}

HIST_CANDLE_TICKER = CANDLES_HIST["ticker"]
LIVE_CANDLE_TICKER = CANDLES_LIVE["ticker"]


def _series_of(event_ticker: str) -> str:
    return str(EVENT_BY_TICKER[event_ticker]["series_ticker"])


def _row(ticker: str) -> dict[str, Any]:
    m = BY_TICKER[ticker]
    ev = EVENT_BY_TICKER.get(m["event_ticker"])
    series = SERIES_ROW[_series_of(m["event_ticker"])]
    starts = kalshi.milestone_starts(EVENTS["milestones"])
    return kalshi.market_row(m, ev, series, event_start=starts.get(m["event_ticker"]))


# --------------------------------------------------------------------------- mock API


class FakeKalshi:
    """Serves market and event data the way the API does, enough for the pipeline:
    the events listing honours ``min_close_ts`` (an event is kept if any market
    closes at or after it) and nests live-tier markets; the historical listing is
    newest-first by open time and ignores time filters; ``limit``/``cursor`` page."""

    def __init__(
        self,
        historical: list[dict[str, Any]],
        live: list[dict[str, Any]],
        events: list[dict[str, Any]],
        milestones: list[dict[str, Any]],
        *,
        unlisted_events: frozenset[str] = frozenset(),
    ) -> None:
        self.historical, self.live = historical, live
        self.events = {e["event_ticker"]: e for e in events}
        self.milestones = milestones
        self.unlisted = unlisted_events
        self.seen: list[str] = []

    def _series(self, m: dict[str, Any]) -> str:
        ev = self.events.get(m["event_ticker"])
        return str(ev["series_ticker"]) if ev else m["event_ticker"].split("-")[0]

    @staticmethod
    def _page(items: list[Any], q: dict[str, str], key: str, **extra: Any) -> httpx.Response:
        start = int(q.get("cursor") or 0)
        limit = int(q.get("limit") or 100)
        chunk = items[start : start + limit]
        cursor = str(start + limit) if start + limit < len(items) else ""
        return httpx.Response(200, json={"cursor": cursor, key: chunk, **extra})

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(str(request.url))
        url = urlparse(str(request.url))
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        path = url.path.removeprefix("/trade-api/v2")
        if path == "/series":
            return httpx.Response(200, json=SERIES)
        if path == "/historical/cutoff":
            return httpx.Response(200, json={"market_settled_ts": "2026-08-07T00:00:00Z"})
        if path == "/historical/markets":
            if "event_ticker" in q:
                ms = [m for m in self.historical if m["event_ticker"] == q["event_ticker"]]
            else:
                ms = [m for m in self.historical if self._series(m) == q["series_ticker"]]
            ms = sorted(ms, key=lambda m: m["open_time"], reverse=True)
            return self._page(ms, q, "markets")
        if path == "/events":
            lo = int(q.get("min_close_ts") or 0)
            everything = self.historical + self.live
            evs = []
            for et, e in sorted(self.events.items(), reverse=True):
                if e["series_ticker"] != q["series_ticker"] or et in self.unlisted:
                    continue
                closes = [_utc(m["close_time"]) for m in everything if m["event_ticker"] == et]
                if not closes or max(closes) < lo:
                    continue
                nested = [m for m in self.live if m["event_ticker"] == et]
                evs.append({**e, "markets": nested} if q.get("with_nested_markets") else e)
            tickers = {e["event_ticker"] for e in evs}
            ms = [
                m
                for m in self.milestones
                if q.get("with_milestones") == "true"
                and tickers & set(m["primary_event_tickers"] + m["related_event_tickers"])
            ]
            return self._page(evs, q, "events", milestones=ms)
        if path.startswith("/events/"):
            et = path.removeprefix("/events/")
            if et in self.events:
                return httpx.Response(200, json={"event": self.events[et], "markets": []})
        if path == f"/historical/markets/{HIST_CANDLE_TICKER}/candlesticks":
            return httpx.Response(200, json=CANDLES_HIST)
        if path.endswith(f"/markets/{LIVE_CANDLE_TICKER}/candlesticks") and path.startswith(
            "/series"
        ):
            return httpx.Response(200, json=CANDLES_LIVE)
        return httpx.Response(404, json={"error": {"code": "not_found"}})


def _client(api: FakeKalshi, cfg: Config) -> CachedClient:
    return CachedClient(
        kalshi.BASE_URL,
        cfg.raw_dir("kalshi") / "http",
        rate_per_s=1000.0,
        transport=httpx.MockTransport(api.handler),
    )


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.delenv("PMCAL_DATA_DIR", raising=False)
    monkeypatch.delenv("PMCAL_OFFLINE", raising=False)
    return Config(study=StudyConfig(start="2021-01-01", end="2026-09-30", data_dir=str(tmp_path)))


@pytest.fixture
def mock_api(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Route make_client through the fixture API; returns the list of requested URLs."""
    api = FakeKalshi(MARKETS["historical"], MARKETS["live"], EVENTS["events"], EVENTS["milestones"])
    monkeypatch.setattr(kalshi, "make_client", lambda cfg: _client(api, cfg))
    return api.seen


# --------------------------------------------------------------------------- series filter


def test_series_filter(cfg: Config, mock_api: list[str]) -> None:
    with kalshi.make_client(cfg) as client:
        series = kalshi.fetch_series(client)
    reasons = {r["series_id"]: kalshi.series_skip_reason(r, cfg) for r in kalshi._records(series)}
    assert reasons == {
        "KXHIGHNY": "",
        "KXMLBGAME": "",
        "KXNBAGAME": "",
        "KXFED": "",
        "KXINX": "",
        "KXTRUMPADMINLEAVE": "",
        "KXIPOOPENAI": "",
        "KXBTC15M": "short_crypto",  # 15-minute crypto: studied elsewhere
        "KXBTCD": "short_crypto",  # hourly crypto
        "KXTEMPNYCH": "short",  # hourly temperature ladder
        "KXMVENFLSINGLEGAME": "combo",
        "KXTXSTATEWIDEDEMS": "",  # 906 contracts in its lifetime: traded, so listed
        "HIGHNY": "no_volume",  # legacy ticker: its markets now sit under KXHIGHNY
    }
    assert set(kalshi.select_series(series, cfg)["series_id"]) == {
        k for k, v in reasons.items() if v == ""
    }

    # With a floor, a series whose lifetime volume is below it cannot hold a market
    # that passes it.
    floor = Config(study=cfg.study, frame=FrameConfig(min_volume_usd=1000.0))
    rows = {r["series_id"]: r for r in kalshi._records(series)}
    assert kalshi.series_skip_reason(rows["KXTXSTATEWIDEDEMS"], floor) == "low_volume"
    assert kalshi.series_skip_reason(rows["HIGHNY"], floor) == "no_volume"
    assert kalshi.series_skip_reason(rows["KXHIGHNY"], floor) == ""

    keep_crypto = Config(study=cfg.study, frame=FrameConfig(exclude_short_crypto=False))
    assert kalshi.series_skip_reason(rows["KXBTC15M"], keep_crypto) == ""
    assert kalshi.is_short_crypto("Crypto", "fifteen_min")
    assert kalshi.series_skip_reason(rows["KXTEMPNYCH"], keep_crypto) == "short"


# --------------------------------------------------------------------------- categories


@pytest.mark.parametrize(
    ("venue", "canonical"),
    [
        ("Sports", "sports"),
        ("Politics", "politics"),
        ("Elections", "politics"),
        ("Economics", "economics"),
        ("Financials", "finance"),
        ("Companies", "finance"),
        ("Commodities", "finance"),
        ("Crypto", "crypto"),
        ("Entertainment", "culture"),
        ("Mentions", "culture"),
        ("Social", "culture"),
        ("Science and Technology", "science_tech"),
        ("AI", "science_tech"),
        ("Health", "science_tech"),
        ("Climate and Weather", "weather"),
        ("World", "world"),
        ("Transportation", "other"),
        ("Exotics", "other"),
        ("Education", "other"),
        ("Business", "other"),
        ("", "other"),
    ],
)
def test_category_mapping(venue: str, canonical: str) -> None:
    assert kalshi.map_category(venue) == canonical
    assert canonical in CATEGORIES


def test_category_in_rows() -> None:
    assert _row("HIGHNY-24AUG15-B85.5")["category"] == "weather"
    assert _row("HIGHNY-24AUG15-B85.5")["venue_category"] == "Climate and Weather"
    assert _row("KXFED-26SEP-T3.25")["category"] == "economics"
    assert _row("KXTRUMPADMINLEAVE-26DEC31-GBOV")["category"] == "politics"


# --------------------------------------------------------------------------- outcomes


def test_outcome_parsing() -> None:
    assert kalshi.parse_outcome(BY_TICKER["KXMLBGAME-26AUG061610DETSEA-DET"]) == (1.0, "resolved")
    assert kalshi.parse_outcome(BY_TICKER["KXMLBGAME-26AUG061610DETSEA-SEA"]) == (0.0, "resolved")
    # A postponed game settled at a fair price: result "scalar", value 0.49.
    out, status = kalshi.parse_outcome(BY_TICKER["KXMLBGAME-26JUN061935BOSNYY-BOS"])
    assert status == "split"
    assert math.isnan(out)

    base = dict(BY_TICKER["HIGHNY-24AUG15-B85.5"])
    assert kalshi.parse_outcome(base) == (1.0, "resolved")
    for result, status in (("void", "void"), ("", "void")):
        out, got = kalshi.parse_outcome(base | {"result": result, "settlement_value_dollars": None})
        assert got == status
        assert math.isnan(out)
    out, got = kalshi.parse_outcome(base | {"status": "active", "result": ""})
    assert got == "unresolved"
    assert math.isnan(out)


def test_outcome_fields_in_row() -> None:
    row = _row("KXMLBGAME-26AUG061610DETSEA-DET")
    assert row["outcome"] == 1.0
    assert row["resolution_status"] == "resolved"
    assert row["outcome_label"] == "Detroit"
    assert row["event_title"] == "Detroit vs Seattle"
    assert row["mutually_exclusive"] is True
    assert row["resolved_ts"] == _utc("2026-08-06T23:03:11.96736Z")


# --------------------------------------------------------------------------- anchors


def test_ticker_start_ts() -> None:
    # Summer (EDT, UTC-4) and winter (EST, UTC-5).
    assert kalshi.ticker_start_ts("KXMLBGAME-26AUG061610DETSEA") == _utc("2026-08-06T20:10:00Z")
    assert kalshi.ticker_start_ts("KXIPLGAME-26MAR291000RRGT") == _utc("2026-03-29T14:00:00Z")
    assert kalshi.ticker_start_ts("KXT20MATCH-26JAN081200NCHWAR") == _utc("2026-01-08T17:00:00Z")
    # Date-only tickers, other suffixes and impossible times give nothing.
    assert kalshi.ticker_start_ts("KXNBAGAME-26OCT20OKCSAS") is None
    assert kalshi.ticker_start_ts("KXINX-26OCT05H1600") is None
    assert kalshi.ticker_start_ts("KXMLBGAME-26AUG062610DETSEA") is None
    assert kalshi.ticker_start_ts("KXFED-26SEP") is None


def test_ticker_date_ts() -> None:
    assert kalshi.ticker_date_ts("KXTRUMPADMINLEAVE-26DEC31") == _utc("2026-12-31T05:00:00Z")
    assert kalshi.ticker_date_ts("KXMENTION-POWE26MAR19") == _utc("2026-03-19T04:00:00Z")
    # The event ticker has no date here; the market ticker does.
    assert kalshi.ticker_date_ts("KXIPOOPENAI") is None
    assert kalshi.ticker_date_ts("KXIPOOPENAI", "KXIPOOPENAI-26MAY01") == _utc(
        "2026-05-01T04:00:00Z"
    )
    for t in ("KXPGATOUR-MAST27", "KXFED-26SEP", "KXX-26FEB30"):
        assert kalshi.ticker_date_ts(t) is None


def test_anchor_sports_game_start_from_ticker() -> None:
    row = _row("KXMLBGAME-26AUG061610DETSEA-DET")
    assert row["anchor_source"] == "game_start"
    assert row["k_event_start_source"] == "ticker"
    assert row["scheduled_ts"] == row["event_start_ts"] == _utc("2026-08-06T20:10:00Z")
    # expected_expiration_time is the expected *end* (start + 3h), not the start.
    assert row["k_expected_expiration_ts"] == _utc("2026-08-06T23:10:00Z")
    assert row["close_ts"] == _utc("2026-08-06T23:00:04Z")


def test_anchor_sports_game_start_from_milestone() -> None:
    ticker = next(t for t in BY_TICKER if t.startswith("KXNBAGAME-"))
    row = _row(ticker)
    milestone = next(
        m for m in EVENTS["milestones"] if row["event_id"] in m["primary_event_tickers"]
    )
    assert kalshi.ticker_start_ts(row["event_id"]) is None
    assert row["anchor_source"] == "game_start"
    assert row["k_event_start_source"] == "milestone"
    assert row["scheduled_ts"] == _utc(milestone["start_date"])
    assert row["scheduled_ts"] < row["close_ts"]


def test_anchor_close_time_when_market_cannot_close_early() -> None:
    m = BY_TICKER["HIGHNY-21AUG06-T86"]
    # On 2021 markets expected_expiration_time is just the open time: unusable.
    assert m["expected_expiration_time"] == m["open_time"]
    row = _row("HIGHNY-21AUG06-T86")
    assert row["k_can_close_early"] is False
    assert row["anchor_source"] == "close_time"
    assert row["scheduled_ts"] == row["close_ts"] == _utc("2021-08-07T03:59:00Z")


def test_anchor_event_strike_date() -> None:
    row = _row("HIGHNY-24AUG15-B85.5")
    assert row["anchor_source"] == "strike_date"
    assert row["scheduled_ts"] == _utc("2024-08-16T03:59:00Z") == row["close_ts"]
    fed = _row("KXFED-26SEP-T3.25")
    assert fed["anchor_source"] == "strike_date"
    # occurrence_datetime is filled in after the fact (to the millisecond) and ignored.
    assert fed["k_occurrence_ts"] != fed["scheduled_ts"]


def test_early_close_keeps_the_ex_ante_anchor(cfg: Config) -> None:
    """A "leaves before the end of 2026" market that resolved YES in February:
    close_time moved to the actual close; expected_expiration_time did not move but
    equals latest_expiration_time (deadline + 7 days), so the ticker's deadline is
    used."""
    m = BY_TICKER["KXTRUMPADMINLEAVE-26DEC31-GBOV"]
    assert m["expected_expiration_time"] == m["latest_expiration_time"] == "2027-01-07T15:00:00Z"
    row = _row("KXTRUMPADMINLEAVE-26DEC31-GBOV")
    assert row["outcome"] == 1.0
    assert row["anchor_source"] == "ticker_date"
    assert row["scheduled_ts"] == _utc("2026-12-31T05:00:00Z")
    assert row["close_ts"] == _utc("2026-02-03T17:16:49Z")
    assert row["k_closed_early"] is True
    # The frame is defined on the scheduled time: this contract is out of a window
    # ending 2026-09-30 although it traded and resolved inside it.
    df = kalshi.to_contracts_frame([row])
    assert not kalshi.frame_mask(df, cfg).iloc[0]


def test_anchor_stale_expected_expiration() -> None:
    """Every deadline of this event shares expected_expiration_time 2027-06-01, later
    than the market's own latest_expiration_time: inconsistent, so not used. The
    market ticker carries the deadline (before May 1, 2026), where it closed (NO)."""
    row = _row("KXIPOOPENAI-26MAY01")
    assert row["k_expected_expiration_ts"] > row["k_latest_expiration_ts"]
    assert row["anchor_source"] == "ticker_date"
    assert row["scheduled_ts"] == _utc("2026-05-01T04:00:00Z")
    assert abs(row["scheduled_ts"] - row["close_ts"]) <= 60

    # Without a date in the ticker: the latest expiration, ex ante but late.
    m = BY_TICKER["KXIPOOPENAI-26MAY01"] | {"ticker": "KXIPOOPENAI-MAYFIRST"}
    a = kalshi.scheduled_anchor(m, None, venue_category="Financials")
    assert a.source == "latest_expiration"
    assert a.scheduled_ts == row["k_latest_expiration_ts"]


def test_anchor_expected_expiration_only_when_informative() -> None:
    m = dict(BY_TICKER["KXTRUMPADMINLEAVE-26DEC31-GBOV"])
    m["expected_expiration_time"] = "2027-01-01T05:00:00Z"  # before latest: informative
    a = kalshi.scheduled_anchor(m, None, venue_category="Politics")
    assert (a.source, a.scheduled_ts) == ("expected_expiration", _utc("2027-01-01T05:00:00Z"))
    m["expected_expiration_time"] = m["open_time"]  # 2021-style placeholder
    a = kalshi.scheduled_anchor(m, None, venue_category="Politics")
    assert a.source == "ticker_date"
    m["can_close_early"] = False
    a = kalshi.scheduled_anchor(m, None, venue_category="Politics")
    assert a.source == "close_time"


def test_milestone_starts_primary_related_and_ambiguous() -> None:
    nba = next(m for m in EVENTS["milestones"] if m["type"] == "basketball_game")
    starts = kalshi.milestone_starts(EVENTS["milestones"])
    start = _utc(nba["start_date"])
    for t in nba["primary_event_tickers"]:
        assert starts[t].ts == start
    # Player props and mentions are listed as related events of the game.
    related_only = set(nba["related_event_tickers"]) - set(nba["primary_event_tickers"])
    assert related_only
    for t in related_only:
        assert starts[t] == kalshi.EventStart(start, "Sports", "basketball_game", nba["id"])
    # A future linked to two games with different starts gets no start.
    other = nba | {
        "id": "other-game",
        "start_date": "2026-05-07T00:00:00Z",
        "primary_event_tickers": ["KXNBAGAME-26MAY06NYKPHI"],
        "related_event_tickers": ["KXNBAEAST-26", "KXNBAGAME-26MAY06NYKPHI"],
    }
    first = nba | {"related_event_tickers": [*nba["related_event_tickers"], "KXNBAEAST-26"]}
    starts = kalshi.milestone_starts([first, other])
    assert "KXNBAEAST-26" not in starts
    assert starts["KXNBAGAME-26MAY06NYKPHI"].ts == _utc("2026-05-07T00:00:00Z")


def test_anchor_mentions_event_start() -> None:
    """Mentions markets ("what will X say during Y") are anchored at Y's start."""
    m = dict(BY_TICKER["KXTRUMPADMINLEAVE-26DEC31-GBOV"])
    start = kalshi.EventStart(_utc("2026-02-03T15:00:00Z"), "Mentions", "one_off_milestone")
    a = kalshi.scheduled_anchor(m, None, venue_category="Mentions", event_start=start)
    assert (a.source, a.scheduled_ts, a.event_start_source) == (
        "event_start",
        start.ts,
        "milestone",
    )
    game = kalshi.EventStart(start.ts, "Sports", "soccer_tournament_multi_leg")
    a = kalshi.scheduled_anchor(m, None, venue_category="Mentions", event_start=game)
    assert a.source == "game_start"
    # Other categories keep their scheduled close but record the start.
    a = kalshi.scheduled_anchor(m, None, venue_category="Politics", event_start=start)
    assert a.source == "ticker_date"
    assert a.event_start_ts == start.ts


def test_frame_mask_uses_scheduled_time_and_volume(cfg: Config) -> None:
    rows = [_row(t) for t in BY_TICKER]
    rows.append(
        _row("HIGHNY-24AUG15-B83.5") | {"contract_id": "NEVER-TRADED", "volume_native": 0.0}
    )
    df = kalshi.to_contracts_frame(rows)
    sched = df["scheduled_ts"]
    in_window = (sched >= cfg.study.start_ts) & (sched < cfg.study.end_ts)
    low = df["contract_id"] == "HIGHNY-24AUG15-B83.5"  # 193 contracts traded
    never = df["contract_id"] == "NEVER-TRADED"

    # Default: no floor beyond "traded at least once".
    assert cfg.frame.min_volume_usd == 0.0
    mask = kalshi.frame_mask(df, cfg)
    expected = in_window & (df["volume_native"] > 0)
    assert mask.tolist() == expected.fillna(False).tolist()
    assert mask[low].all()
    assert not mask[never].any()

    # An explicit floor, in contracts.
    floor = Config(study=cfg.study, frame=FrameConfig(min_volume_usd=1000.0))
    mask = kalshi.frame_mask(df, floor)
    expected = in_window & (df["volume_native"] >= 1000.0)
    assert mask.tolist() == expected.fillna(False).tolist()
    assert not mask[low].any()


# --------------------------------------------------------------------------- index


def test_build_index(cfg: Config, mock_api: list[str]) -> None:
    df = kalshi.build_index(cfg, workers=2, series=["KXMLBGAME", "KXHIGHNY", "KXTRUMPADMINLEAVE"])
    assert list(df.columns) == list(CONTRACT_COLUMNS) + list(kalshi.K_COLUMNS)
    for col, dtype in CONTRACT_COLUMNS.items():
        assert str(df[col].dtype) == dtype, col
    assert (cfg.interim_dir() / "kalshi_index.parquet").exists()
    assert df["contract_id"].is_unique
    assert set(df["category"]) <= set(CATEGORIES)
    # In: the 2021 and 2024 weather markets (also the 193-contract strike: no floor
    # beyond having traded), the MLB games. Out: the 2027-scheduled early closer.
    assert "HIGHNY-24AUG15-B83.5" in set(df["contract_id"])
    assert (df["volume_native"] > 0).all()
    assert "KXTRUMPADMINLEAVE-26DEC31-GBOV" not in set(df["contract_id"])
    det = df.set_index("contract_id").loc["KXMLBGAME-26AUG061610DETSEA-DET"]
    assert det["n_in_event"] == 2
    assert det["k_tier"] == "historical"
    live = df[df["k_tier"] == "live"]
    assert len(live) >= 1
    # Milestones are asked for on sports (and mentions) series only.
    events_urls = [u for u in mock_api if "/events?" in u]
    assert all(("with_milestones" in u) == ("KXMLBGAME" in u) for u in events_urls)
    assert all("min_close_ts" in u and "with_nested_markets=true" in u for u in events_urls)
    # The live MLB game came nested in its event, not from a market listing.
    assert not any("/markets?" in u and "/historical/" not in u for u in mock_api)
    # Second run: everything from the cache.
    n = len(mock_api)
    kalshi.build_index(cfg, workers=2, series=["KXMLBGAME", "KXHIGHNY", "KXTRUMPADMINLEAVE"])
    assert len(mock_api) == n


# --------------------------------------------------------------------------- candles


def test_historical_candles_to_path() -> None:
    raw = CANDLES_HIST["candlesticks"]
    c = kalshi.parse_candles(raw)
    path = kalshi.candles_to_path(c)
    assert path.bid is not None
    assert path.ask is not None
    assert path.last is not None
    assert path.volume is not None
    assert path.ts.tolist() == sorted(x["end_period_ts"] for x in raw)
    assert np.all(np.isnan(path.price))
    for x in raw:
        i = int(np.searchsorted(path.ts, x["end_period_ts"]))
        assert path.bid[i] == float(x["yes_bid"]["close"])
        assert path.ask[i] == float(x["yes_ask"]["close"])
        assert path.volume[i] == float(x["volume"])
        trade = x["price"]["close"]
        expected_last = float(trade) if trade is not None else x["price"]["previous"]
        if expected_last is None:
            assert math.isnan(path.last[i])
        else:
            assert path.last[i] == float(expected_last)
    # The format has hours without trades (close null, previous set).
    assert any(x["price"]["close"] is None and x["price"]["previous"] for x in raw)


def test_live_candles_to_path() -> None:
    raw = CANDLES_LIVE["candlesticks"]
    c = kalshi.parse_candles(raw)
    path = kalshi.candles_to_path(c)
    assert path.last is not None
    assert path.bid is not None
    assert path.volume is not None
    for x in raw:
        i = int(np.searchsorted(path.ts, x["end_period_ts"]))
        p = x["price"]
        assert path.bid[i] == float(x["yes_bid"]["close_dollars"])
        assert path.volume[i] == float(x["volume_fp"])
        if "close_dollars" in p:
            assert path.last[i] == float(p["close_dollars"])
        elif "previous_dollars" in p:
            assert path.last[i] == float(p["previous_dollars"])
        else:
            assert math.isnan(path.last[i])
    assert any("close_dollars" not in x["price"] for x in raw)
    assert c.traded.any()
    assert not c.traded.all()


def test_last_trade_age() -> None:
    c = kalshi.parse_candles(CANDLES_HIST["candlesticks"])
    traded = c.ts[c.traded]
    t = int(traded[3]) + 1800
    assert kalshi.last_trade_age(c, t) == 1800.0
    assert math.isnan(kalshi.last_trade_age(c, int(c.ts[0]) - 1))


def test_candle_windows() -> None:
    h, d = kalshi.HOUR, kalshi.DAY
    max_h = 30 * d
    # Short-lived market: one window from just before the open to the close.
    o, s, c = 1_000_000_000, 1_000_000_000 + 2 * d + 1234, 1_000_000_000 + 2 * d + 1234
    (w,) = kalshi.candle_windows(o, c, s, max_h)
    assert w[0] <= o - h
    assert w[1] >= c
    assert w[0] % h == 0
    assert w[1] % h == 0
    # Closed early, 100 days before schedule: only the hours before the close matter.
    o, c = 1_000_000_000, 1_000_000_000 + 200 * d
    (w,) = kalshi.candle_windows(o, c, c + 100 * d, max_h)
    assert w[1] >= c > w[1] - h
    assert w[0] <= c - 31 * d
    # Ran far past a stale schedule: two windows, each within the request limit.
    ws = kalshi.candle_windows(o, c, o + 10 * d, max_h)
    assert len(ws) == 2
    assert all(b - a <= kalshi.MAX_CANDLES_PER_REQUEST * h for a, b in ws)
    # Long windows are split.
    ws = kalshi.align_windows([(0, 6000 * h)])
    assert ws == [(0, 5000 * h), (5000 * h, 6000 * h)]


def test_requests_by_tier(cfg: Config) -> None:
    hist = _row("HIGHNY-24AUG15-B85.5")
    (r,) = kalshi.contract_requests(hist, cfg)
    assert r.path == "/historical/markets/HIGHNY-24AUG15-B85.5/candlesticks"
    assert r.params["period_interval"] == 60
    live = _row(LIVE_CANDLE_TICKER)
    (r,) = kalshi.contract_requests(live, cfg)
    assert r.path == f"/series/KXMLBGAME/markets/{LIVE_CANDLE_TICKER}/candlesticks"


# --------------------------------------------------------------------------- end to end


def _contracts(*tickers: str) -> pd.DataFrame:
    return kalshi.to_contracts_frame([_row(t) for t in tickers])


def test_fetch_then_snapshots_offline(
    cfg: Config, mock_api: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    contracts = _contracts(HIST_CANDLE_TICKER, LIVE_CANDLE_TICKER, "HIGHNY-24AUG15-B83.5")
    kalshi.fetch_prices(contracts, cfg, workers=2)
    n = len(mock_api)
    kalshi.fetch_prices(contracts, cfg, workers=2)  # idempotent
    assert len(mock_api) == n

    monkeypatch.setenv("PMCAL_OFFLINE", "1")
    snaps = kalshi.build_snapshots(contracts, cfg)
    assert list(snaps.columns) == list(SNAPSHOT_COLUMNS)
    assert {c: str(t) for c, t in snaps.dtypes.items()} == SNAPSHOT_COLUMNS
    n_h = len(cfg.snapshots.horizons)
    assert len(snaps) == 3 * 2 * n_h
    assert len(mock_api) == n  # nothing fetched

    sched = snaps[(snaps["anchor"] == "scheduled")].set_index(["contract_id", "horizon"])
    # Weather market (2024): prices up to the strike date, closed for the long horizons.
    w = sched.loc[HIST_CANDLE_TICKER]
    assert w.loc["1h", "status"] == "ok"
    assert w.loc["1h", "price_source"] in ("mid", "last")
    assert w.loc["1h", "obs_ts"] <= w.loc["1h", "t_snap"]
    assert w.loc["1h", "last_trade_age_s"] >= 0
    assert w.loc["3d", "status"] == "not_open"
    # The market without candles (the API had none: 404) gets no_data while open.
    nd = sched.loc["HIGHNY-24AUG15-B83.5"]
    assert set(nd["status"]) <= {"no_data", "not_open"}
    assert (nd["status"] == "no_data").any()
    # Live MLB game: the scheduled anchor is the first pitch.
    g = sched.loc[LIVE_CANDLE_TICKER]
    start = int(contracts.set_index("contract_id").loc[LIVE_CANDLE_TICKER, "scheduled_ts"])
    assert int(g.loc["1h", "t_snap"]) == start - 3600
    assert g.loc["1h", "status"] == "ok"
    # Anchor "close" exists for every contract and horizon.
    assert (snaps["anchor"] == "close").sum() == 3 * n_h


def test_load_price_path_and_extra_windows(
    cfg: Config, mock_api: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    contracts = _contracts(HIST_CANDLE_TICKER)
    row = contracts.iloc[0]
    assert kalshi.load_price_path(row, cfg).ts.size == 0  # nothing cached yet
    kalshi.fetch_prices(contracts, cfg)
    path = kalshi.load_price_path(row, cfg)
    # every real candle is in the path, and quiet hours in between are filled in
    real_ts = {int(c["end_period_ts"]) for c in CANDLES_HIST["candlesticks"]}
    assert real_ts <= set(path.ts.tolist())
    assert path.ts.size >= len(CANDLES_HIST["candlesticks"])
    assert set(np.diff(path.ts).tolist()) <= {3600}

    # An extra window is fetched once, remembered, and read back offline.
    extra = (int(row["open_ts"]) - 5 * kalshi.DAY, int(row["open_ts"]) - 3 * kalshi.DAY)
    kalshi.fetch_windows(row, [extra], cfg)
    n = len(mock_api)
    reqs = kalshi.contract_requests(row, cfg)
    assert len(reqs) == 2
    monkeypatch.setenv("PMCAL_OFFLINE", "1")
    assert kalshi.load_price_path(row, cfg).ts.size == path.ts.size  # same candles served
    kalshi.fetch_windows(row, [extra], cfg)  # cached: no request, no error offline
    assert len(mock_api) == n
    with kalshi.make_client(cfg) as client, pytest.raises(OfflineCacheMiss):
        client.get("/historical/markets/NOT-FETCHED/candlesticks")


def test_estimate_volume_usd(
    cfg: Config, mock_api: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    contracts = _contracts(HIST_CANDLE_TICKER, "HIGHNY-24AUG15-B83.5")
    assert contracts["volume_usd"].isna().all()  # the index leaves it empty
    kalshi.fetch_prices(contracts, cfg)
    monkeypatch.setenv("PMCAL_OFFLINE", "1")
    out = kalshi.estimate_volume_usd(contracts, cfg)
    assert list(out.columns) == list(contracts.columns)
    assert contracts["volume_usd"].isna().all()  # a copy, the input is untouched
    vol = out["volume_usd"]
    c = kalshi.parse_candles(CANDLES_HIST["candlesticks"])
    ok = np.isfinite(c.mean) & (c.volume > 0)
    vwap = float((c.volume[ok] * c.mean[ok]).sum() / c.volume[ok].sum())
    assert vol.iloc[0] == pytest.approx(float(contracts["volume_native"].iloc[0]) * vwap)
    assert 0 < vol.iloc[0] < float(contracts["volume_native"].iloc[0])
    assert math.isnan(vol.iloc[1])


# --------------------------------------------------------------------------- window listing


def _window_cfg(tmp_path: Path) -> Config:
    return Config(study=StudyConfig(start="2026-01-01", end="2026-09-30", data_dir=str(tmp_path)))


def test_listing_lower_bound(tmp_path: Path) -> None:
    cfg = _window_cfg(tmp_path)
    # 30-day longest horizon + 5 days before 2026-01-01.
    assert kalshi.listing_lower_bound(cfg) == _utc("2025-11-27T00:00:00Z")


def _synthetic_market(ticker: str, opened: str, closed: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "event_ticker": ticker.rsplit("-", 1)[0],
        "open_time": opened,
        "close_time": closed,
        "status": "finalized",
        "result": "no",
        "volume_fp": "5000.00",
    }


def test_list_series_window_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Step 1 finds the events in range, step 2 pages the historical tier down to the
    frontier, step 3 looks up the event whose only market opened before it, and an
    event missing from the events listing is recovered from step 2."""
    monkeypatch.delenv("PMCAL_OFFLINE", raising=False)
    monkeypatch.setattr(kalshi, "HIST_PAGE_LIMIT", 1)
    cfg = _window_cfg(tmp_path)
    m = _synthetic_market
    historical = [
        m("KXTEST-26APR-A", "2026-03-01T00:00:00Z", "2026-04-01T00:00:00Z"),
        m("KXTEST-26APR-B", "2026-02-28T00:00:00Z", "2026-04-01T00:00:00Z"),
        m("KXTEST-25JUN-A", "2025-05-01T00:00:00Z", "2025-06-01T00:00:00Z"),  # before L
        m("KXTEST-ORPH-A", "2025-04-01T00:00:00Z", "2026-03-25T00:00:00Z"),  # unlisted event
        m("KXTEST-OLD-A", "2022-01-01T00:00:00Z", "2022-02-01T00:00:00Z"),  # past frontier
        m("KXTEST-26MAR-A", "2021-06-01T00:00:00Z", "2026-03-01T00:00:00Z"),  # long-lived
    ]
    live = [m("KXTEST-26SEP-A", "2026-08-10T00:00:00Z", "2026-09-15T00:00:00Z")]
    events = [
        {"event_ticker": t, "series_ticker": "KXTEST", "title": t, "mutually_exclusive": False}
        for t in ("KXTEST-26APR", "KXTEST-25JUN", "KXTEST-ORPH", "KXTEST-OLD", "KXTEST-26MAR")
    ] + [{"event_ticker": "KXTEST-26SEP", "series_ticker": "KXTEST", "title": "sep"}]
    api = FakeKalshi(historical, live, events, [], unlisted_events=frozenset({"KXTEST-ORPH"}))
    with _client(api, cfg) as client:
        lst = kalshi.list_series_window(
            client, "KXTEST", min_close_ts=kalshi.listing_lower_bound(cfg), with_milestones=False
        )
    tiers = {x["ticker"]: x["_tier"] for x in lst.markets}
    assert tiers == {
        "KXTEST-26APR-A": "historical",
        "KXTEST-26APR-B": "historical",
        "KXTEST-ORPH-A": "historical",
        "KXTEST-26MAR-A": "historical",
        "KXTEST-26SEP-A": "live",
    }
    assert set(lst.events) == {"KXTEST-26APR", "KXTEST-26MAR", "KXTEST-26SEP", "KXTEST-ORPH"}
    assert lst.events["KXTEST-ORPH"]["title"] == "KXTEST-ORPH"
    assert "markets" not in lst.events["KXTEST-26SEP"]
    # Paging stopped at the first page opened before the frontier (page 5 of 6).
    assert lst.requests == Counter(
        event_pages=1, historical_pages=5, event_lookups=1, orphan_events=1
    )


def test_list_series_window_inactive_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PMCAL_OFFLINE", raising=False)
    cfg = _window_cfg(tmp_path)
    old = [_synthetic_market("KXTEST-24JUN-A", "2024-05-01T00:00:00Z", "2024-06-01T00:00:00Z")]
    events = [{"event_ticker": "KXTEST-24JUN", "series_ticker": "KXTEST", "title": "old"}]
    api = FakeKalshi(old, [], events, [])
    with _client(api, cfg) as client:
        lst = kalshi.list_series_window(
            client, "KXTEST", min_close_ts=kalshi.listing_lower_bound(cfg), with_milestones=False
        )
    assert lst.markets == []
    assert lst.requests == Counter(event_pages=1)  # one request for a series with nothing


def test_build_index_2026_window(
    tmp_path: Path, mock_api: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PMCAL_OFFLINE", raising=False)
    cfg = _window_cfg(tmp_path)
    df = kalshi.build_index(cfg, workers=2, series=["KXMLBGAME", "KXHIGHNY", "KXFED"])
    ids = set(df["contract_id"])
    assert not any(t.startswith("HIGHNY-") for t in ids)  # 2021 and 2024
    assert {"KXMLBGAME-26AUG061610DETSEA-DET", "KXFED-26SEP-T3.25", LIVE_CANDLE_TICKER} <= ids
    # No historical-tier listing for the weather series: nothing of it is in range.
    assert not any("historical/markets?" in u and "KXHIGHNY" in u for u in mock_api)
    sched = df["scheduled_ts"]
    assert ((sched >= cfg.study.start_ts) & (sched < cfg.study.end_ts)).all()


# --------------------------------------------------------------------------- clusters


def test_cluster_id() -> None:
    nba = next(m for m in EVENTS["milestones"] if m["type"] == "basketball_game")
    starts = kalshi.milestone_starts(EVENTS["milestones"])
    game = f"game:{nba['id']}"
    # Moneyline, spread and player props of one game share the cluster.
    for t in nba["primary_event_tickers"] + nba["related_event_tickers"]:
        assert kalshi.cluster_id(t, starts.get(t)) == game
    row = _row(next(t for t in BY_TICKER if t.startswith("KXNBAGAME-")))
    assert row["cluster_id"] == game
    # MLB: start from the ticker, cluster from the milestone.
    det = _row("KXMLBGAME-26AUG061610DETSEA-DET")
    assert det["cluster_id"].startswith("game:")
    assert det["cluster_id"] == _row("KXMLBGAME-26AUG061610DETSEA-SEA")["cluster_id"]
    # Without a sports milestone the event is its own cluster.
    assert _row("HIGHNY-24AUG15-B85.5")["cluster_id"] == "HIGHNY-24AUG15"
    speech = kalshi.EventStart(1, "Mentions", "one_off_milestone", "m-1")
    assert kalshi.cluster_id("KXTRUMPMENTION-26SEP16", speech) == "KXTRUMPMENTION-26SEP16"
    two_games = kalshi.EventStart(1, "Sports", "basketball_game", None)
    assert kalshi.cluster_id("KXNBAEAST-26", two_games) == "KXNBAEAST-26"


def test_fill_hourly_carries_state_through_quiet_hours() -> None:
    import numpy as np

    from pmcal.snapshots import PricePath, snapshot_at
    from pmcal.venues.kalshi import fill_hourly

    h = 3600
    sparse = PricePath(
        ts=np.array([0, 5 * h], dtype=np.int64),
        price=np.full(2, np.nan),
        bid=np.array([0.40, 0.50]),
        ask=np.array([0.44, 0.54]),
        last=np.array([0.41, 0.52]),
        volume=np.array([10.0, 3.0]),
    )
    full = fill_hourly(sparse, until=8 * h)
    assert full.ts.tolist() == [i * h for i in range(9)]
    assert full.bid is not None
    assert full.volume is not None
    assert full.bid[:5].tolist() == [0.40] * 5
    assert full.bid[5:].tolist() == [0.50] * 4
    assert full.volume.tolist() == [10.0, 0, 0, 0, 0, 3.0, 0, 0, 0]
    # a snapshot 3.5 hours after the last change is a live mid, not "too_old"
    s = snapshot_at(full, 8 * h + 1800 - 1, horizon="1h", horizon_s=h, open_ts=0, close_ts=10 * h)
    assert s.status == "ok"
    assert s.price_source == "mid"
    assert s.price == pytest.approx(0.52)
    s_sparse = snapshot_at(
        sparse, 8 * h + 1800 - 1, horizon="1h", horizon_s=h, open_ts=0, close_ts=10 * h
    )
    assert s_sparse.status == "too_old"
