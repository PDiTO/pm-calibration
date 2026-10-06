"""Kalshi: the sampling frame, hourly candlesticks and snapshots, from the public trade API.

Only unauthenticated read endpoints are used, all through ``pmcal.http.CachedClient``
so a second run reads everything from disk (and ``PMCAL_OFFLINE=1`` proves it).

Listing (``build_index``)
-------------------------
The global settled-market listing is unusable at scale: it is dominated by
multivariate combo markets and by hourly strike ladders (thousands of index,
temperature and commodity strikes settle every hour). So the frame is built per
series, from the series list. Skipped series:

- ``frequency`` ``hourly`` or ``fifteen_min``: short-duration ladders (the short
  crypto ones are studied elsewhere). ``frequency`` is the series' *current*
  setting, so a few series that are hourly now but listed daily markets in earlier
  years (``KXBTC``, ``KXBTCD``, ``KXETHD``, ``KXINXU``) are skipped with those
  daily markets;
- combo series (``KXMVE...``);
- series that never traded (lifetime ``volume_fp`` 0), and, if a volume floor is
  set, series whose lifetime volume is below it (no market in them can pass it).
  Legacy tickers such as ``HIGHNY`` show zero volume because their markets are
  filed under today's ``KX`` series (``HIGHNY-21AUG06-T86`` is a ``KXHIGHNY``
  market).

Markets live in two tiers: those settled before ``/historical/cutoff``'s
``market_settled_ts`` only in the historical tier, the newer ones only in the live
tier. Each remaining series is listed in three steps (``list_series_window``) so
that only the study period is read, not the series' whole history:

1. ``/events?series_ticker=S&min_close_ts=L&with_nested_markets=true`` (plus
   ``with_milestones=true`` for sports and mentions series). The close filter is
   honoured for events of both tiers and keeps an event if any of its markets
   closes at or after ``L``; one empty page shows that a series has nothing in
   range. The nested markets are the event's live-tier markets, as full market
   objects; historical-tier markets are never nested.
2. ``/historical/markets?series_ticker=S``, which accepts no time filter and only
   one event per request, is sorted newest-first by open time. It is paged until a
   page holds only markets opened more than 730 days before ``L``, keeping the
   markets of step 1's events.
3. ``/historical/markets?event_ticker=E`` for each step-1 event that may have
   markets beyond that point: events neither nested nor seen in step 2, and events
   with a market (nested or seen) opened within 60 days of it.

A few events (2 in 600 series surveyed) are missing from the events listing
although ``/events/{ticker}`` has them; their historical markets are kept in step
2 if they close at or after ``L``, and the event is fetched by ticker.

``L`` is the study start minus the longest horizon minus 5 days. An event is only
missed if *every* one of its markets closed before ``L``; such markets are closed at
every scheduled-anchor snapshot time (``t_snap >= scheduled - 30d >= start - 30d >
close``), so leaving them out removes no priced snapshot, whatever their outcome.
It does leave them out of the leaky close-anchor contrast and out of counts of
early closes. Markets of a listed event are all kept, including any that closed
before ``L``.

Clusters (``cluster_id``): Kalshi splits one game over many events (moneyline,
spread, totals, player props, announcer mentions), all linked to the game's
milestone. Events linked to exactly one sports milestone share the cluster
``game:<milestone id>``; every other event is its own cluster.

Frame rule (see ``frame_mask``)
-------------------------------
- The *scheduled* time (``scheduled_ts``, below) is in ``[study.start_ts, study.end_ts)``.
  Filtering on the actual close would let in long-dated markets that resolved early
  (mostly YES) without their unresolved peers.
- ``volume_fp`` (contracts traded) > 0, and >= ``frame.min_volume_usd`` if a floor
  is set (the default is 0). Lifetime volume includes trading after any snapshot
  and upsets attract volume, so a higher floor would select on the outcome.
  Contracts, not dollars: a dollar floor on the YES side would favour contracts
  that drifted towards YES. The listing has no dollar volume, so the index leaves
  ``volume_usd`` empty; ``estimate_volume_usd`` fills it from the cached candles.

Outcomes: Kalshi has no void result. A cancelled or postponed event settles as
``result == "scalar"`` at a fair price (e.g. 0.49), which is mapped to "split".

Anchor (``scheduled_ts``)
-------------------------
The horizon anchor has to be fixed before the outcome is known. Kalshi's date
fields differ in how far they can be trusted for that (checked on the live API in
October 2026; ``scheduled_anchor`` has the exact precedence):

- ``close_time`` moves to the actual close when a market closes early, so it is
  only ex ante when ``can_close_early`` is false.
- ``occurrence_datetime`` is often filled in after the fact (to the millisecond of
  the announcement), so it is never used.
- ``expected_expiration_time`` is set at creation and does not move when a market
  closes early, but it is often not an expected time at all: on many series it
  equals ``latest_expiration_time`` (the deadline plus a 7- or 14-day settlement
  buffer), it is shared by every deadline of an event, and on 2021-22 markets it
  equals ``open_time``. It is used only when it is after the open and strictly
  before the latest expiration.
- The event's ``strike_date``, where present (weather, index, Fed events), is the
  scheduled end of what the event measures.
- Tickers carry ex-ante dates: ``-26DEC31`` is 2026-12-31, ``KXMLBGAME-26AUG061610``
  is 2026-08-06 16:10 US Eastern.
- For sports and mentions markets the anchor is the start of the underlying game,
  match, speech or call, so "1h before" means one hour before kickoff on both
  venues. It comes from a time in the ticker, else from the event's milestone.
  Milestones list the events they belong to as primary or related; a start is used
  only when it is unambiguous (futures linked to several games get none).

Prices
------
Hourly candlesticks over ``[min(scheduled, close) - 31 days, close]``, from the
per-market endpoint of the market's tier (historical-tier markets have no batch
endpoint; the live tier's batch endpoint is not used so that cache keys depend on
the contract alone, see ``fetch_prices``). The API allows 5,000 candles per request,
so this is one request per contract. A candle is only emitted for an hour in which
the book or trades changed, so an hour without a candle had no activity. Each
candle becomes one ``PricePath`` observation: bid/ask = the YES bid/ask at the
close of the hour, last = the hour's closing trade price or, in an hour without
trades, the previous trade price. Hours without a candle carry the previous state
forward (``fill_hourly``), up to the market's close. The snapshot rules themselves live in
``pmcal.snapshots`` so both venues are treated alike.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pmcal.config import Config
from pmcal.http import CachedClient, Params
from pmcal.schema import CATEGORIES, CONTRACT_COLUMNS, SNAPSHOT_COLUMNS
from pmcal.snapshots import FloatArray, IntArray, PricePath, Snapshot, take_snapshots

log = logging.getLogger(__name__)

VENUE: Final = "kalshi"
BASE_URL: Final = "https://api.elections.kalshi.com/trade-api/v2"
# The basic tier is documented at about 20 reads/s, but measured in October 2026 the
# historical endpoints answered 429 to 12% of requests at a steady 5/s and 46% at
# 8/s, and none at 4/s. Each 429 also halves the client's rate for a minute, so 4/s
# is both the polite and the fastest setting.
RATE_PER_S: Final = 4.0

HOUR: Final = 3600
DAY: Final = 86400
CANDLE_PERIOD_MIN: Final = 60
MAX_CANDLES_PER_REQUEST: Final = 5000  # per request; more is answered with 400

SHORT_FREQUENCIES: Final = frozenset({"hourly", "fifteen_min"})
COMBO_SERIES_PREFIX: Final = "KXMVE"
EASTERN: Final = ZoneInfo("America/New_York")

# A market is treated as having closed early (for the k_closed_early flag and the
# report) when it closed more than this long before its scheduled time.
EARLY_CLOSE_MARGIN_S: Final = DAY

CATEGORY_MAP: Final[dict[str, str]] = {
    "Sports": "sports",
    "Politics": "politics",
    "Elections": "politics",
    "Economics": "economics",
    "Financials": "finance",
    "Companies": "finance",
    "Commodities": "finance",
    "Crypto": "crypto",
    "Entertainment": "culture",
    "Mentions": "culture",
    "Social": "culture",
    "Science and Technology": "science_tech",
    "AI": "science_tech",
    "Health": "science_tech",
    "Climate and Weather": "weather",
    "World": "world",
}
assert set(CATEGORY_MAP.values()) <= set(CATEGORIES)

# Extra columns of the index, all prefixed k_ as the schema allows.
K_COLUMNS: Final[dict[str, str]] = {
    "k_series_frequency": "string",
    "k_tier": "string",  # "live" or "historical": which endpoints serve the market
    "k_market_type": "string",
    "k_strike_type": "string",
    "k_can_close_early": "boolean",
    "k_status": "string",
    "k_result": "string",
    "k_settlement_value": "float64",
    "k_expected_expiration_ts": "Int64",
    "k_latest_expiration_ts": "Int64",
    "k_occurrence_ts": "Int64",  # audit only: often filled in after the fact
    "k_strike_date_ts": "Int64",  # the event's strike_date, where it has one
    "k_event_start_source": "string",  # "ticker" or "milestone" when event_start_ts is set
    "k_closed_early": "boolean",  # close_ts < scheduled_ts - 1 day
}


# --------------------------------------------------------------------------- helpers


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], df.to_dict("records"))


def make_client(cfg: Config) -> CachedClient:
    return CachedClient(BASE_URL, cfg.raw_dir(VENUE) / "http", rate_per_s=RATE_PER_S, burst=1)


def parse_ts(value: object) -> int | None:
    """ISO-8601 (``2026-08-06T23:00:04.123Z``) to integer Unix seconds, None if missing."""
    if not isinstance(value, str) or not value:
        return None
    try:
        d = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.UTC)
    if d.year < 2000:  # "0001-01-01T00:00:00Z" placeholders
        return None
    return math.floor(d.timestamp())


def _float(value: object) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def _paginate(
    client: CachedClient, path: str, key: str, params: Params, *, extra_keys: Sequence[str] = ()
) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    """Follow ``cursor`` to the end. Returns the items under ``key`` and, for each of
    ``extra_keys``, everything found under it on all pages."""
    items: list[dict[str, Any]] = []
    extras: dict[str, list[Any]] = {k: [] for k in extra_keys}
    cursor: str | None = None
    seen: set[str] = set()
    while True:
        page = client.get_json(path, {**params, "cursor": cursor}) or {}
        got = page.get(key) or []
        items.extend(got)
        for k in extra_keys:
            extras[k].extend(page.get(k) or [])
        cursor = page.get("cursor") or None
        if not cursor or not got or cursor in seen:
            break
        seen.add(cursor)
    return items, extras


class _Progress:
    """Thread-safe progress logging with rate and ETA."""

    def __init__(self, what: str, total: int, every_s: float = 15.0) -> None:
        self.what, self.total, self.every_s = what, total, every_s
        self.done = 0
        self.t0 = self._last = time.monotonic()
        self._lock = threading.Lock()

    def step(self, n: int = 1) -> None:
        with self._lock:
            self.done += n
            now = time.monotonic()
            if now - self._last < self.every_s and self.done < self.total:
                return
            self._last = now
            rate = self.done / max(now - self.t0, 1e-9)
            eta = (self.total - self.done) / rate if rate > 0 else math.inf
            log.info(
                "%s: %d/%d (%.1f%%), %.2f/s, ETA %.0f min",
                self.what,
                self.done,
                self.total,
                100 * self.done / max(self.total, 1),
                rate,
                eta / 60,
            )


# --------------------------------------------------------------------------- series


def fetch_series(client: CachedClient) -> pd.DataFrame:
    """Every series with its category, frequency and lifetime volume (one request)."""
    data = client.get_json("/series", {"include_volume": True}) or {}
    rows = [
        {
            "series_id": s.get("ticker"),
            "series_title": s.get("title"),
            "venue_category": s.get("category") or "",
            "frequency": s.get("frequency") or "",
            "series_volume": _float(s.get("volume_fp")),
            "tags": ",".join(s.get("tags") or []),
        }
        for s in data.get("series") or []
    ]
    return pd.DataFrame(rows)


def map_category(venue_category: str | None) -> str:
    return CATEGORY_MAP.get(venue_category or "", "other")


def is_short_crypto(venue_category: str | None, frequency: str | None) -> bool:
    return venue_category == "Crypto" and frequency in SHORT_FREQUENCIES


def series_skip_reason(row: Mapping[str, Any], cfg: Config) -> str:
    """Why a series is left out of the frame, or "" if it is listed.

    Non-crypto hourly and 15-minute series are always skipped (strike ladders that
    settle every hour). Short crypto series are skipped when
    ``frame.exclude_short_crypto`` is set. A series that never traded
    (``volume_fp == 0``) is skipped; so is one whose lifetime volume is below
    ``frame.min_volume_usd`` when a floor is set, as none of its markets can pass it.
    """
    ticker = str(row["series_id"])
    if ticker.startswith(COMBO_SERIES_PREFIX):
        return "combo"
    freq = str(row.get("frequency") or "")
    if freq in SHORT_FREQUENCIES:
        crypto = is_short_crypto(str(row.get("venue_category") or ""), freq)
        if not crypto or cfg.frame.exclude_short_crypto:
            return "short_crypto" if crypto else "short"
    volume = float(row.get("series_volume") or 0.0)
    if not volume > 0:
        return "no_volume"
    if not volume >= cfg.frame.min_volume_usd:
        return "low_volume"
    return ""


def select_series(series: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The series to list, with a log line counting what was skipped and why."""
    reasons = series.apply(lambda r: series_skip_reason(r, cfg), axis=1)
    counts = Counter(reasons)
    short_crypto = series[reasons == "short_crypto"]
    log.info(
        "series: %d total, %d listed; skipped %d short (hourly/15-min, non-crypto), "
        "%d short crypto (%.3g contracts lifetime), %d combo, %d never traded, "
        "%d below the volume floor",
        len(series),
        counts[""],
        counts["short"],
        counts["short_crypto"],
        float(short_crypto["series_volume"].sum()),
        counts["combo"],
        counts["no_volume"],
        counts["low_volume"],
    )
    return series[reasons == ""].reset_index(drop=True)


# --------------------------------------------------------------------------- listing

# Markets that closed this long before the study start, plus the longest horizon,
# cannot have a scheduled-anchor snapshot in the window (see the module docstring).
LIST_PAD_S: Final = 5 * DAY
# The historical listing is paged back to markets opened this long before the
# listing bound; events that may reach further back are looked up one by one.
HIST_DEPTH_S: Final = 730 * DAY
HIST_SLACK_S: Final = 60 * DAY
HIST_PAGE_LIMIT: Final = 1000
EVENT_PAGE_LIMIT: Final = 200
SETTLED_STATUSES: Final = frozenset({"finalized", "settled"})


def listing_lower_bound(cfg: Config) -> int:
    """``L``: events are listed if any of their markets closed at or after it."""
    return cfg.study.start_ts - max(cfg.snapshots.horizon_seconds().values()) - LIST_PAD_S


@dataclass
class SeriesListing:
    """What the listing found for one series."""

    events: dict[str, dict[str, Any]]  # event ticker -> event (without nested markets)
    markets: list[dict[str, Any]]  # every market of those events, tagged ``_tier``
    milestones: list[dict[str, Any]]
    requests: Counter[str] = dataclasses.field(default_factory=Counter)


def _get(
    client: CachedClient, path: str, params: Params, requests: Counter[str], kind: str
) -> dict[str, Any]:
    requests[kind] += 1
    data = client.get_json(path, params)
    return data if isinstance(data, dict) else {}


def list_series_window(
    client: CachedClient, series_ticker: str, *, min_close_ts: int, with_milestones: bool
) -> SeriesListing:
    """Every market of the series' events that have a market closing at or after
    ``min_close_ts``, from both tiers. See the module docstring for the three steps.

    Milestone pages are much larger, so they are only asked for where event start
    times are used (sports and mentions series).
    """
    req: Counter[str] = Counter()
    params: Params = {
        "series_ticker": series_ticker,
        "limit": EVENT_PAGE_LIMIT,
        "min_close_ts": min_close_ts,
        "with_nested_markets": True,
    }
    if with_milestones:
        params["with_milestones"] = True
    events: dict[str, dict[str, Any]] = {}
    milestones: list[dict[str, Any]] = []
    markets: dict[str, dict[str, Any]] = {}
    earliest_open: dict[str, int] = {}

    def add(m: Mapping[str, Any], tier: str) -> None:
        et = str(m.get("event_ticker"))
        markets[str(m["ticker"])] = {**m, "_tier": tier}
        o = parse_ts(m.get("open_time"))
        if o is not None:
            earliest_open[et] = min(o, earliest_open.get(et, o))

    # 1. Events with a market closing at or after the bound; live markets nested.
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        page = _get(client, "/events", {**params, "cursor": cursor}, req, "event_pages")
        got = page.get("events") or []
        for e in got:
            nested = e.get("markets") or []
            events[str(e["event_ticker"])] = {k: v for k, v in e.items() if k != "markets"}
            for m in nested:
                add(m, "live")
        milestones.extend(page.get("milestones") or [])
        cursor = page.get("cursor") or None
        if not cursor or not got or cursor in seen_cursors:
            break
        seen_cursors.add(cursor)
    if not events:
        # Nothing closes in range. (Step 2 is skipped, so an event missing from the
        # events listing in a series with no listed event would be missed.)
        return SeriesListing({}, [], milestones, req)
    nested_events = set(earliest_open)

    # 2. The historical tier, newest first by open time, down to the frontier.
    frontier = min_close_ts - HIST_DEPTH_S
    seen_hist: set[str] = set()
    orphans: dict[str, list[dict[str, Any]]] = {}
    complete = False
    cursor, seen_cursors = None, set()
    while True:
        page = _get(
            client,
            "/historical/markets",
            {"series_ticker": series_ticker, "limit": HIST_PAGE_LIMIT, "cursor": cursor},
            req,
            "historical_pages",
        )
        got = page.get("markets") or []
        for m in got:
            et = str(m.get("event_ticker"))
            if et in events:
                # A market that moved tiers between the two listings is kept once,
                # as historical.
                add(m, "historical")
                seen_hist.add(et)
            elif (parse_ts(m.get("close_time")) or 0) >= min_close_ts:
                orphans.setdefault(et, []).append(m)
        cursor = page.get("cursor") or None
        if not cursor or not got or cursor in seen_cursors:
            complete = True
            break
        seen_cursors.add(cursor)
        opens = [o for o in (parse_ts(m.get("open_time")) for m in got) if o is not None]
        if opens and max(opens) < frontier:
            break

    # 3. Events that may have historical markets beyond the frontier.
    if not complete:
        todo = sorted(
            et
            for et in events
            if (et not in seen_hist and et not in nested_events)
            or earliest_open.get(et, frontier) < frontier + HIST_SLACK_S
        )
        for et in todo:
            cursor, seen_cursors = None, set()
            while True:
                page = _get(
                    client,
                    "/historical/markets",
                    {"event_ticker": et, "limit": HIST_PAGE_LIMIT, "cursor": cursor},
                    req,
                    "event_lookups",
                )
                got = page.get("markets") or []
                for m in got:
                    add(m, "historical")
                cursor = page.get("cursor") or None
                if not cursor or not got or cursor in seen_cursors:
                    break
                seen_cursors.add(cursor)

    # A few events are missing from the events listing although /events/{ticker}
    # has them; their historical markets were kept in step 2 by their own close.
    for et, ms in sorted(orphans.items()):
        ev = _get(client, f"/events/{et}", {}, req, "orphan_events").get("event")
        events[et] = dict(ev) if isinstance(ev, Mapping) else {"event_ticker": et}
        for m in ms:
            add(m, "historical")
    return SeriesListing(events, list(markets.values()), milestones, req)


# --------------------------------------------------------------------------- parsing

_MONTHS: Final = {
    m: i + 1
    for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    )
}
_MON = "|".join(_MONTHS)
_TICKER_START = re.compile(rf"^(\d{{2}})({_MON})(\d{{2}})(\d{{2}})(\d{{2}})(?=[A-Z])")
_TICKER_DATE = re.compile(rf"(?<![0-9])(\d{{2}})({_MON})(\d{{2}})")


def _eastern_ts(yy: str, mon: str, dd: str, hh: int = 0, mm: int = 0) -> int | None:
    try:
        local = dt.datetime(2000 + int(yy), _MONTHS[mon], int(dd), hh, mm, tzinfo=EASTERN)
    except (ValueError, KeyError):
        return None
    return int(local.timestamp())


def ticker_start_ts(event_ticker: str) -> int | None:
    """Scheduled start encoded in an event ticker, if any.

    ``KXMLBGAME-26AUG061610DETSEA`` -> 2026-08-06 16:10 US Eastern. MLB, cricket and
    a few other series put the time in the ticker; most give the date only
    (``KXNBAGAME-26OCT20OKCSAS``), which returns None.
    """
    _, _, rest = event_ticker.partition("-")
    m = _TICKER_START.match(rest)
    if not m:
        return None
    yy, mon, dd, hh, mm = m.groups()
    if int(hh) > 23 or int(mm) > 59:
        return None
    return _eastern_ts(yy, mon, dd, int(hh), int(mm))


def ticker_date_ts(*tickers: str) -> int | None:
    """Start (00:00 US Eastern) of the first ``YYMONDD`` date found after the series
    prefix of the given tickers, tried in order.

    ``KXTRUMPADMINLEAVE-26DEC31`` -> 2026-12-31, ``KXIPOOPENAI-26MAY01`` (a market
    ticker) -> 2026-05-01. The date is assigned when the market is created, so it is
    ex ante; the start of the day errs early, never into the event.
    """
    for t in tickers:
        _, _, rest = t.partition("-")
        m = _TICKER_DATE.search(rest)
        if m:
            yy, mon, dd = m.groups()
            return _eastern_ts(yy, mon, dd)
    return None


@dataclass(frozen=True)
class EventStart:
    """Start of the game, match, speech or call behind an event, from a milestone."""

    ts: int
    category: str  # the milestone's category, e.g. "Sports", "Mentions"
    kind: str  # the milestone type, e.g. "baseball_game", "one_off_milestone"
    milestone_id: str | None = None  # None if several milestones share the start


def milestone_starts(milestones: Iterable[Mapping[str, Any]]) -> dict[str, EventStart]:
    """event ticker -> start of the milestone it belongs to.

    An event can be a milestone's primary event (``KXNBAGAME-...``) or a related one
    (player props, mentions of the game). Primary links win; a start is only kept
    if it is unambiguous, so a future linked to several games (a conference winner
    listed under each game of a series) gets none. The milestone id is kept when
    exactly one milestone is linked; it identifies the game across series.
    """
    links: dict[str, dict[str, set[tuple[int, str, str, str]]]] = {"primary": {}, "related": {}}
    seen: set[str] = set()
    for m in milestones:
        mid = str(m.get("id") or id(m))
        if mid in seen:
            continue
        seen.add(mid)
        start = parse_ts(m.get("start_date"))
        if start is None:
            continue
        val = (start, str(m.get("category") or ""), str(m.get("type") or ""), mid)
        for kind, key in (
            ("primary", "primary_event_tickers"),
            ("related", "related_event_tickers"),
        ):
            for t in m.get(key) or []:
                links[kind].setdefault(t, set()).add(val)
    out: dict[str, EventStart] = {}
    for kind in ("related", "primary"):  # primary overwrites related
        for t, vals in links[kind].items():
            starts = {v[0] for v in vals}
            if len(starts) == 1:
                ts, cat, typ, mid = sorted(vals)[0]
                out[t] = EventStart(ts, cat, typ, mid if len(vals) == 1 else None)
            elif kind == "primary":
                out.pop(t, None)
    return out


def parse_outcome(market: Mapping[str, Any]) -> tuple[float, str]:
    """(outcome, resolution_status) of the market's YES side.

    "yes" -> 1.0 and "no" -> 0.0 (resolved); a settlement value strictly between 0 and
    1, or a scalar result, -> "split"; "void"/"cancelled" or a void status -> "void";
    anything not finalized -> "unresolved".
    """
    status = str(market.get("status") or "").lower()
    result = str(market.get("result") or "").lower()
    if result in ("void", "voided", "cancelled", "canceled") or status in (
        "voided",
        "void",
        "cancelled",
        "canceled",
    ):
        return math.nan, "void"
    if status not in ("finalized", "settled"):
        return math.nan, "unresolved"
    value = _float(market.get("settlement_value_dollars"))
    if result == "yes" and (math.isnan(value) or value == 1.0):
        return 1.0, "resolved"
    if result == "no" and (math.isnan(value) or value == 0.0):
        return 0.0, "resolved"
    if result in ("yes", "no", "scalar") or (math.isfinite(value) and 0.0 <= value <= 1.0):
        return math.nan, "split"
    return math.nan, "void" if result == "" else "unresolved"


@dataclass(frozen=True)
class Anchor:
    scheduled_ts: int | None
    source: str
    event_start_ts: int | None = None
    event_start_source: str | None = None  # "ticker" or "milestone"


# Categories whose markets are about something that happens at a known time (a
# game, a speech): anchored at its start rather than at the scheduled close.
EVENT_START_CATEGORIES: Final = frozenset({"Sports", "Mentions"})


def scheduled_anchor(
    market: Mapping[str, Any],
    event: Mapping[str, Any] | None,
    *,
    venue_category: str,
    event_start: EventStart | None = None,
) -> Anchor:
    """The ex-ante time the outcome is due, and where it came from.

    Precedence:

    1. Sports and mentions markets: the start of the game or event, from a time in
       the event ticker (sports) or else the event's milestone ("game_start" for a
       sports milestone, "event_start" otherwise).
    2. ``can_close_early`` false: ``close_time`` ("close_time"). The market cannot
       close before it, so it does not depend on the outcome.
    3. The event's ``strike_date`` ("strike_date"), if not before ``open_time``.
    4. ``expected_expiration_time`` ("expected_expiration"), if after ``open_time``
       and strictly before ``latest_expiration_time`` (where the two are equal it is
       the deadline plus the settlement buffer, not an expected time).
    5. A date in the event or market ticker ("ticker_date"), 00:00 US Eastern, if
       not before the open (some tickers carry the creation date instead).
    6. ``latest_expiration_time`` ("latest_expiration"): ex ante but late by the
       settlement buffer (7-14 days), so short horizons fall after the close.
    7. The actual close ("close_time_fallback"); outcome-dependent if the market
       closed early, so the analysis should treat these as leaky.
    """
    open_ts = parse_ts(market.get("open_time"))
    close_ts = parse_ts(market.get("close_time"))
    exp_ts = parse_ts(market.get("expected_expiration_time"))
    latest_ts = parse_ts(market.get("latest_expiration_time"))
    event_ticker = str(market.get("event_ticker") or "")

    tick = ticker_start_ts(event_ticker) if venue_category == "Sports" else None
    start_ts, start_src = (tick, "ticker") if tick is not None else (None, None)
    if start_ts is None and event_start is not None:
        start_ts, start_src = event_start.ts, "milestone"
    if venue_category in EVENT_START_CATEGORIES and start_ts is not None:
        is_game = tick is not None or (event_start is not None and event_start.category == "Sports")
        source = "game_start" if is_game else "event_start"
        return Anchor(start_ts, source, start_ts, start_src)

    def anchor(ts: int | None, source: str) -> Anchor:
        return Anchor(ts, source, start_ts, start_src)

    if market.get("can_close_early") is False and close_ts is not None:
        return anchor(close_ts, "close_time")

    strike = parse_ts((event or {}).get("strike_date"))
    if strike is not None and (open_ts is None or strike >= open_ts):
        return anchor(strike, "strike_date")

    if (
        exp_ts is not None
        and (open_ts is None or exp_ts > open_ts)
        and (latest_ts is None or exp_ts < latest_ts)
    ):
        return anchor(exp_ts, "expected_expiration")

    date_ts = ticker_date_ts(event_ticker, str(market.get("ticker") or ""))
    if date_ts is not None and (open_ts is None or date_ts >= open_ts):
        return anchor(date_ts, "ticker_date")

    if latest_ts is not None and (open_ts is None or latest_ts > open_ts):
        return anchor(latest_ts, "latest_expiration")
    return anchor(close_ts, "close_time_fallback")


def cluster_id(event_ticker: str, event_start: EventStart | None) -> str:
    """The bootstrap cluster: the game for any event linked to exactly one sports
    milestone (moneyline, spread, totals, props and announcer mentions of one game
    share it), otherwise the event itself."""
    if event_start is not None and event_start.category == "Sports" and event_start.milestone_id:
        return f"game:{event_start.milestone_id}"
    return event_ticker


def market_row(
    market: Mapping[str, Any],
    event: Mapping[str, Any] | None,
    series: Mapping[str, Any],
    *,
    event_start: EventStart | None = None,
) -> dict[str, Any]:
    """One contract row (CONTRACT_COLUMNS + K_COLUMNS) for a listed market."""
    venue_category = str(series.get("venue_category") or "")
    frequency = str(series.get("frequency") or "")
    outcome, status = parse_outcome(market)
    anchor = scheduled_anchor(market, event, venue_category=venue_category, event_start=event_start)
    close_ts = parse_ts(market.get("close_time"))
    volume = _float(market.get("volume_fp"))
    closed_early = (
        anchor.scheduled_ts is not None
        and close_ts is not None
        and close_ts < anchor.scheduled_ts - EARLY_CLOSE_MARGIN_S
    )
    event = event or {}
    return {
        "venue": VENUE,
        "contract_id": market["ticker"],
        "market_id": market["ticker"],
        "event_id": market.get("event_ticker"),
        "cluster_id": cluster_id(str(market.get("event_ticker") or ""), event_start),
        "series_id": series.get("series_id"),
        "title": market.get("title"),
        "event_title": event.get("title"),
        "outcome_label": market.get("yes_sub_title") or market.get("title"),
        "category": map_category(venue_category),
        "venue_category": venue_category,
        "n_in_event": None,  # filled in once the frame is known
        "mutually_exclusive": event.get("mutually_exclusive"),
        "open_ts": parse_ts(market.get("open_time")),
        "close_ts": close_ts,
        "scheduled_ts": anchor.scheduled_ts,
        "anchor_source": anchor.source,
        "event_start_ts": anchor.event_start_ts,
        "resolved_ts": parse_ts(market.get("settlement_ts")),
        "outcome": outcome,
        "resolution_status": status,
        "volume_usd": math.nan,  # see estimate_volume_usd
        "volume_native": volume,
        "liquidity_usd": math.nan,  # Kalshi's listing has no liquidity figure
        "is_short_crypto": is_short_crypto(venue_category, frequency),
        "k_series_frequency": frequency,
        "k_tier": market.get("_tier"),
        "k_market_type": market.get("market_type"),
        "k_strike_type": market.get("strike_type"),
        "k_can_close_early": market.get("can_close_early"),
        "k_status": market.get("status"),
        "k_result": market.get("result"),
        "k_settlement_value": _float(market.get("settlement_value_dollars")),
        "k_expected_expiration_ts": parse_ts(market.get("expected_expiration_time")),
        "k_latest_expiration_ts": parse_ts(market.get("latest_expiration_time")),
        "k_occurrence_ts": parse_ts(market.get("occurrence_datetime")),
        "k_strike_date_ts": parse_ts(event.get("strike_date")),
        "k_event_start_source": anchor.event_start_source,
        "k_closed_early": closed_early,
    }


def is_combo(market: Mapping[str, Any]) -> bool:
    return bool(market.get("mve_selected_legs") or market.get("mve_collection_ticker")) or str(
        market.get("ticker") or ""
    ).startswith(COMBO_SERIES_PREFIX)


def frame_mask(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Rows in the sampling frame: scheduled in the study window, traded at least once
    (``volume_fp > 0``) and at or above ``frame.min_volume_usd`` contracts if a floor
    is set."""
    sched = df["scheduled_ts"]
    in_window = sched.notna() & (sched >= cfg.study.start_ts) & (sched < cfg.study.end_ts)
    vol = df["volume_native"].fillna(0.0)
    enough = (vol > 0) & (vol >= cfg.frame.min_volume_usd)
    return (in_window & enough).fillna(False).astype(bool)


def to_contracts_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Rows -> DataFrame with exactly CONTRACT_COLUMNS + K_COLUMNS and their dtypes."""
    cols = {**CONTRACT_COLUMNS, **K_COLUMNS}
    return pd.DataFrame(rows, columns=list(cols)).astype(cols)


# --------------------------------------------------------------------------- index


def _index_series(
    client: CachedClient, series: Mapping[str, Any], min_close_ts: int
) -> tuple[list[dict[str, Any]], Counter[str]]:
    ticker = str(series["series_id"])
    want = series.get("venue_category") in EVENT_START_CATEGORIES
    lst = list_series_window(client, ticker, min_close_ts=min_close_ts, with_milestones=want)
    counts: Counter[str] = Counter(lst.requests)
    counts["series_active"] += bool(lst.events)
    counts["events_listed"] += len(lst.events)
    starts = milestone_starts(lst.milestones)
    rows = []
    for m in lst.markets:
        if is_combo(m):
            counts["combo"] += 1
            continue
        if str(m.get("status") or "") not in SETTLED_STATUSES:
            counts["not_settled"] += 1
            continue
        counts[f"tier_{m['_tier']}"] += 1
        et = str(m.get("event_ticker"))
        rows.append(market_row(m, lst.events.get(et), series, event_start=starts.get(et)))
    counts["listed"] += len(rows)
    return rows, counts


def build_index(
    cfg: Config, *, workers: int = 4, series: Sequence[str] | None = None
) -> pd.DataFrame:
    """Every settled market in the sampling frame (metadata only), one row per market.

    ``series`` restricts the listing to the given series tickers (smoke tests); by
    default every series that passes ``series_skip_reason`` is listed, with
    ``list_series_window``. Void and split markets stay in the index with their
    ``resolution_status``. ``volume_native`` is ``volume_fp`` (contracts, the frame
    floor); ``volume_usd`` is left NaN for ``estimate_volume_usd``. Written to
    ``interim/kalshi_index.parquet``.
    """
    t0 = time.monotonic()
    with make_client(cfg) as client:
        all_series = fetch_series(client)
        chosen = select_series(all_series, cfg)
        if series is not None:
            wanted = set(series)
            missing = wanted - set(all_series["series_id"])
            if missing:
                log.warning("unknown series: %s", sorted(missing))
            skipped = wanted - set(chosen["series_id"]) - missing
            if skipped:
                log.info("requested but outside the frame rule: %s", sorted(skipped))
            chosen = chosen[chosen["series_id"].isin(wanted)].reset_index(drop=True)
        cutoff = (client.get_json("/historical/cutoff") or {}).get("market_settled_ts")
        lower = listing_lower_bound(cfg)
        log.info(
            "listing %d series: events with a market closing on or after %s (historical cutoff %s)",
            len(chosen),
            dt.datetime.fromtimestamp(lower, dt.UTC).date(),
            cutoff,
        )

        progress = _Progress("series listed", len(chosen))
        rows: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        lock = threading.Lock()

        failed: list[Mapping[str, Any]] = []

        def work(s: Mapping[str, Any]) -> None:
            try:
                r, c = _index_series(client, s, lower)
            except Exception as e:  # retried once below, after the others
                log.warning("listing %s failed: %s", s["series_id"], e)
                with lock:
                    failed.append(s)
            else:
                with lock:
                    rows.extend(r)
                    counts.update(c)
            progress.step()

        records = _records(chosen)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for fut in [pool.submit(work, s) for s in records]:
                fut.result()
        for s in list(failed):  # a series that fails twice aborts the index
            r, c = _index_series(client, s, lower)
            rows.extend(r)
            counts.update(c)
        log.info(
            "listing done in %.0f min: %d series with events in range; requests: %d event "
            "pages, %d historical pages, %d event lookups; http %s",
            (time.monotonic() - t0) / 60,
            counts["series_active"],
            counts["event_pages"],
            counts["historical_pages"],
            counts["event_lookups"],
            client.stats,
        )

    df = to_contracts_frame(rows)
    mask = frame_mask(df, cfg)
    sched = df["scheduled_ts"]
    in_window = sched.notna() & (sched >= cfg.study.start_ts) & (sched < cfg.study.end_ts)
    log.info(
        "markets: %d settled listed (%d historical, %d live), %d not settled and %d combos "
        "dropped; %d scheduled in the window, %d of those below the volume floor; "
        "%d in the frame",
        counts["listed"],
        counts["tier_historical"],
        counts["tier_live"],
        counts["not_settled"],
        counts["combo"],
        int(in_window.sum()),
        int((in_window & ~mask).sum()),
        int(mask.sum()),
    )
    df = df[mask].drop_duplicates("contract_id")
    df = df.sort_values(["series_id", "event_id", "contract_id"]).reset_index(drop=True)
    df["n_in_event"] = df.groupby("event_id")["contract_id"].transform("size").astype("Int64")
    out = cfg.interim_dir() / "kalshi_index.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info(
        "wrote %s: %d contracts, %d events, %d clusters in %.0fs; resolution %s",
        out,
        len(df),
        df["event_id"].nunique(),
        df["cluster_id"].nunique(),
        time.monotonic() - t0,
        dict(Counter(df["resolution_status"])),
    )
    return df


# --------------------------------------------------------------------------- candles


@dataclass(frozen=True)
class CandleRequest:
    """One per-market request for hourly candles over ``[start_ts, end_ts]``."""

    path: str
    start_ts: int
    end_ts: int

    @property
    def params(self) -> Params:
        return {
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "period_interval": CANDLE_PERIOD_MIN,
        }


def _opt_int(v: Any) -> int | None:
    return None if v is None or pd.isna(v) else int(v)


def _merge_intervals(iv: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for lo, hi in sorted(iv):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def align_windows(windows: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Hour-align ``(start_ts, end_ts)`` windows outwards, merge overlaps, and split any
    window longer than the API's 5,000 candles per request."""
    aligned = [((lo // HOUR) * HOUR, -((-hi) // HOUR) * HOUR) for lo, hi in windows if hi > lo]
    out: list[tuple[int, int]] = []
    step = MAX_CANDLES_PER_REQUEST * HOUR
    for lo, hi in _merge_intervals(aligned):
        out.extend((a, min(a + step, hi)) for a in range(lo, hi, step))
    return out


def candle_windows(
    open_ts: int | None,
    close_ts: int | None,
    scheduled_ts: int | None,
    max_horizon_s: int,
) -> list[tuple[int, int]]:
    """Hour-aligned ``(start_ts, end_ts)`` windows of hourly candles a contract needs.

    Each anchor (scheduled and close) needs the hours from ``anchor - max_horizon - 1d``
    (the extra day is the largest staleness allowance) up to ``min(anchor, close)``:
    nothing after the close is used and nothing before the open exists. For almost
    every market the two merge into one window of at most 62 days, i.e. one request.
    """
    if close_ts is None:
        return []
    pad = max_horizon_s + DAY
    iv = []
    for anchor in (scheduled_ts, close_ts):
        if anchor is None:
            continue
        lo = anchor - pad
        if open_ts is not None:
            lo = max(lo, open_ts - HOUR)
        iv.append((lo, min(anchor, close_ts)))
    return align_windows(iv)


def _endpoints(row: Mapping[str, Any] | pd.Series) -> tuple[str, str]:
    """(primary, fallback) per-market candle paths. The tier decides the primary; the
    other tier's endpoint is the fallback for a market that moved tiers after it was
    listed (the historical cutoff advances)."""
    ticker, series = row["contract_id"], row["series_id"]
    hist = f"/historical/markets/{ticker}/candlesticks"
    live = f"/series/{series}/markets/{ticker}/candlesticks"
    return (live, hist) if row.get("k_tier") == "live" else (hist, live)


def _extra_windows_file(cfg: Config, ticker: str) -> Path:
    return cfg.raw_dir(VENUE) / "extra_windows" / f"{ticker}.json"


def _read_extra_windows(cfg: Config, ticker: str) -> list[tuple[int, int]]:
    p = _extra_windows_file(cfg, ticker)
    if not p.exists():
        return []
    return [(int(a), int(b)) for a, b in json.loads(p.read_text())]


_extra_lock = threading.Lock()


def _add_extra_windows(cfg: Config, ticker: str, windows: list[tuple[int, int]]) -> None:
    p = _extra_windows_file(cfg, ticker)
    with _extra_lock:
        merged = sorted(set(_read_extra_windows(cfg, ticker)) | set(windows))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".tmp{threading.get_ident()}")
        tmp.write_text(json.dumps(merged))
        os.replace(tmp, p)


def contract_requests(row: Mapping[str, Any] | pd.Series, cfg: Config) -> list[CandleRequest]:
    """The candle requests of one contract: the planned windows plus any extra windows
    added with ``fetch_windows``. Depends on the row alone, so the cache keys can be
    rebuilt for any contract without knowing which others were fetched with it."""
    max_h = max(cfg.snapshots.horizon_seconds().values())
    windows = candle_windows(
        _opt_int(row["open_ts"]),
        _opt_int(row["close_ts"]),
        _opt_int(row["scheduled_ts"]),
        max_h,
    )
    windows = sorted(set(windows) | set(_read_extra_windows(cfg, str(row["contract_id"]))))
    primary, _ = _endpoints(row)
    return [CandleRequest(primary, lo, hi) for lo, hi in windows]


def _fallback(r: CandleRequest, row: Mapping[str, Any] | pd.Series) -> CandleRequest:
    primary, other = _endpoints(row)
    return CandleRequest(other if r.path == primary else primary, r.start_ts, r.end_ts)


def _needs_fetch(client: CachedClient, r: CandleRequest, alt: CandleRequest) -> bool:
    hit = client.cached(r.path, r.params)
    if hit is None:
        return True
    return hit.status == 404 and client.cached(alt.path, alt.params) is None


def _fetch_one(client: CachedClient, r: CandleRequest, alt: CandleRequest) -> None:
    if client.get(r.path, r.params).status == 404:
        client.get(alt.path, alt.params)


def fetch_prices(contracts: pd.DataFrame, cfg: Config, *, workers: int = 6) -> None:
    """Download the hourly candlesticks of every contract into the cache (idempotent).

    One per-market request per window, usually one per contract. (The live tier
    also has a batch endpoint, but its cache keys would depend on which markets
    were fetched together; per-market keys let ``load_price_path`` rebuild any
    contract's prices from the row alone.)
    """
    jobs: list[tuple[CandleRequest, CandleRequest]] = []
    n_planned = 0
    with make_client(cfg) as client:
        for row in _records(contracts):
            for r in contract_requests(row, cfg):
                n_planned += 1
                alt = _fallback(r, row)
                if _needs_fetch(client, r, alt):
                    jobs.append((r, alt))
        log.info(
            "candles: %d contracts, %d requests planned, %d already cached, %d to fetch",
            len(contracts),
            n_planned,
            n_planned - len(jobs),
            len(jobs),
        )
        if not jobs:
            return
        progress = _Progress("candle requests", len(jobs))
        failures: list[str] = []
        lock = threading.Lock()

        def work(job: tuple[CandleRequest, CandleRequest]) -> None:
            try:
                _fetch_one(client, *job)
            except Exception as e:  # keep going; a re-run retries whatever is missing
                with lock:
                    failures.append(f"{job[0].path}: {e}")
            progress.step()

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for fut in [pool.submit(work, j) for j in jobs]:
                fut.result()
        log.info("candles done: http %s, %d failures", client.stats, len(failures))
        for f in failures[:10]:
            log.warning("candle request failed: %s", f)


def fetch_windows(
    row: Mapping[str, Any] | pd.Series,
    windows: list[tuple[int, int]],
    cfg: Config,
    *,
    client: CachedClient | None = None,
) -> None:
    """Download extra hourly candle windows for one contract into the cache.

    The windows are hour-aligned, split to the per-request limit and remembered in
    ``raw/kalshi/extra_windows/<ticker>.json`` so that ``load_price_path`` and
    ``build_snapshots`` include them from then on.
    """
    aligned = align_windows(windows)
    if not aligned:
        return
    _add_extra_windows(cfg, str(row["contract_id"]), aligned)
    primary, _ = _endpoints(row)
    own = client is None
    c = client or make_client(cfg)
    try:
        for lo, hi in aligned:
            r = CandleRequest(primary, lo, hi)
            alt = _fallback(r, row)
            if _needs_fetch(c, r, alt):
                _fetch_one(c, r, alt)
    finally:
        if own:
            c.close()


# --------------------------------------------------------------------------- candles -> path


def _field(d: object, key: str) -> float:
    """``d[key]`` or ``d[key + "_dollars"]`` as a float (live candles use the suffix)."""
    if not isinstance(d, Mapping):
        return math.nan
    v = d.get(key)
    if v is None:
        v = d.get(f"{key}_dollars")
    return _float(v)


def _volume(c: Mapping[str, Any]) -> float:
    v = c.get("volume")
    if v is None:
        v = c.get("volume_fp")
    return _float(v)


@dataclass(frozen=True)
class Candles:
    """Hourly candles of one market, sorted by end time."""

    ts: IntArray  # end_period_ts
    bid: FloatArray  # YES bid at the close of the hour
    ask: FloatArray  # YES ask at the close of the hour
    close: FloatArray  # last trade in the hour, NaN if none
    previous: FloatArray  # last trade before the hour, NaN if never
    mean: FloatArray  # mean trade price in the hour
    volume: FloatArray  # contracts traded in the hour

    @property
    def traded(self) -> NDArray[np.bool_]:
        return np.isfinite(self.close) | (np.nan_to_num(self.volume) > 0)


def parse_candles(candles: Iterable[Mapping[str, Any]]) -> Candles:
    """Live (``*_dollars``, ``volume_fp``) and historical (bare keys) formats alike.

    In an hour without trades the live format gives ``price: {"previous_dollars": ..}``
    (or ``{}`` before the first trade) and the historical one ``close: null`` with
    ``previous`` set.
    """
    by_ts: dict[int, Mapping[str, Any]] = {}
    for c in candles:
        t = c.get("end_period_ts")
        if t is not None:
            by_ts[int(t)] = c
    ts = sorted(by_ts)
    cs = [by_ts[t] for t in ts]

    def col(f: Any) -> FloatArray:
        return np.array([f(c) for c in cs], dtype=np.float64)

    return Candles(
        ts=np.array(ts, dtype=np.int64),
        bid=col(lambda c: _field(c.get("yes_bid"), "close")),
        ask=col(lambda c: _field(c.get("yes_ask"), "close")),
        close=col(lambda c: _field(c.get("price"), "close")),
        previous=col(lambda c: _field(c.get("price"), "previous")),
        mean=col(lambda c: _field(c.get("price"), "mean")),
        volume=col(_volume),
    )


def candles_to_path(c: Candles, until: int | None = None) -> PricePath:
    """ts = end of the hour; bid/ask = YES quotes at the close of the hour; last = the
    hour's closing trade, else the previous trade; volume = contracts; price = NaN.

    Kalshi emits a candle only for an hour in which the book or trades changed, so an
    hour without a candle kept the previous hour's state. With ``until`` the path is
    forward-filled onto a regular hourly grid from the first candle up to ``until``
    (the market's close): quotes and last trade carry forward, volume is 0. Without
    this, a quiet but well-quoted market would look like missing data ("too_old").
    """
    last = np.where(np.isfinite(c.close), c.close, c.previous)
    path = PricePath(
        ts=c.ts,
        price=np.full(c.ts.size, np.nan),
        bid=c.bid,
        ask=c.ask,
        last=last,
        volume=c.volume,
    )
    if until is None or c.ts.size == 0:
        return path
    return fill_hourly(path, until)


def fill_hourly(path: PricePath, until: int) -> PricePath:
    """Forward-fill a sparse hourly PricePath onto every hour from its first
    observation to ``until`` (inclusive of the last full hour before it)."""
    start = int(path.ts[0])
    end = max(int(path.ts[-1]), int(until))
    grid = np.arange(start, end + 1, 3600, dtype=np.int64)
    ts = np.union1d(grid, path.ts)
    src = np.searchsorted(path.ts, ts, side="right") - 1  # last real candle at or before
    real = np.isin(ts, path.ts)
    bid, ask, last, vol = path.bid, path.ask, path.last, path.volume
    if bid is None or ask is None or last is None or vol is None:
        raise ValueError("fill_hourly needs bid, ask, last and volume")
    return PricePath(
        ts=ts,
        price=np.full(ts.size, np.nan),
        bid=bid[src],
        ask=ask[src],
        last=last[src],
        volume=np.where(real, vol[src], 0.0),
    )


def merge_candles(parts: Sequence[Candles]) -> Candles:
    """Concatenate candle sets, sorted by time, one candle per timestamp."""
    parts = [p for p in parts if p.ts.size]
    if not parts:
        return parse_candles([])
    if len(parts) == 1:
        return parts[0]
    names = [f.name for f in dataclasses.fields(Candles)]
    cat = {f: np.concatenate([getattr(p, f) for p in parts]) for f in names}
    order = np.argsort(cat["ts"], kind="stable")
    ts = cat["ts"][order]
    keep = np.ones(ts.size, dtype=bool)
    keep[:-1] = ts[1:] != ts[:-1]
    return Candles(**{f: cat[f][order][keep] for f in names})


def last_trade_age(c: Candles, t: int) -> float:
    """Seconds from the end of the last candle at or before ``t`` that had a trade, to ``t``.

    Candle granularity: the trade happened somewhere in that hour, so this can
    understate the age by up to an hour. NaN if no candle in the fetched windows
    (at most 31 days before the anchor) had a trade.
    """
    i = int(np.searchsorted(c.ts, t, side="right"))
    idx = np.flatnonzero(c.traded[:i])
    return float(t - c.ts[idx[-1]]) if idx.size else math.nan


def _load_contract_candles(
    client: CachedClient, row: Mapping[str, Any] | pd.Series, cfg: Config
) -> tuple[Candles, int]:
    """A contract's merged candles from the cache only, and how many of its requests
    were not cached at all."""
    parts: list[Candles] = []
    misses = 0
    for r in contract_requests(row, cfg):
        alt = _fallback(r, row)
        hit = client.cached(r.path, r.params)
        if hit is None or not 200 <= hit.status < 300:
            alt_hit = client.cached(alt.path, alt.params)
            if hit is None and alt_hit is None:
                misses += 1
            hit = alt_hit
        if hit is not None and 200 <= hit.status < 300 and isinstance(hit.data, Mapping):
            parts.append(parse_candles(hit.data.get("candlesticks") or []))
    return merge_candles(parts), misses


def load_price_path(
    row: Mapping[str, Any] | pd.Series, cfg: Config, *, client: CachedClient | None = None
) -> PricePath:
    """A contract's PricePath rebuilt purely from the cache (planned windows plus any
    added with ``fetch_windows``). Empty if nothing is cached. Never touches the
    network, so it works under ``PMCAL_OFFLINE=1``."""
    own = client is None
    c = client or make_client(cfg)
    try:
        candles, _ = _load_contract_candles(c, row, cfg)
    finally:
        if own:
            c.close()
    return candles_to_path(candles, until=_opt_int(row["close_ts"]))


def contract_snapshots(
    row: Mapping[str, Any] | pd.Series,
    candles: Candles,
    horizons: Mapping[str, int],
    max_spread: float,
) -> list[dict[str, Any]]:
    """SNAPSHOT_COLUMNS rows for one contract: anchors "scheduled" and "close".

    ``last_trade_age_s`` comes from the candles, at candle (hour) granularity;
    ``n_trades_24h`` is left empty (candles give volume, not trade counts).
    """
    open_ts, close_ts = _opt_int(row["open_ts"]), _opt_int(row["close_ts"])
    path = candles_to_path(candles, until=close_ts)
    rows = []
    for anchor, anchor_ts in (("scheduled", _opt_int(row["scheduled_ts"])), ("close", close_ts)):
        if anchor_ts is None:
            snaps = [
                Snapshot(horizon=h, horizon_s=s, t_snap=-1, status="no_data")
                for h, s in horizons.items()
            ]
        else:
            snaps = take_snapshots(
                path,
                anchor_ts=anchor_ts,
                open_ts=open_ts,
                close_ts=close_ts,
                horizons=horizons,
                max_spread=max_spread,
            )
        for s in snaps:
            if s.status in ("ok", "too_old", "bad_price"):
                s = dataclasses.replace(s, last_trade_age_s=last_trade_age(candles, s.t_snap))
            rec = s.as_row()
            if anchor_ts is None:
                rec["t_snap"] = None
            rows.append(
                {"venue": VENUE, "contract_id": row["contract_id"], "anchor": anchor, **rec}
            )
    return rows


def build_snapshots(contracts: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """SNAPSHOT_COLUMNS rows for every contract x anchor x horizon, from the cache only.

    Anchor "scheduled" uses ``scheduled_ts`` (the main analysis); anchor "close" uses
    the actual ``close_ts`` and is deliberately leaky (it moves with early
    resolution), kept to show the artifact. Contracts without price data get
    "no_data" rows (or "not_open"/"closed" where the snapshot time is outside the
    trading window). Prices come from the same loader as ``load_price_path``.
    """
    horizons = cfg.snapshots.horizon_seconds()
    rows: list[dict[str, Any]] = []
    misses = 0
    with make_client(cfg) as client:
        for row in _records(contracts):
            candles, m = _load_contract_candles(client, row, cfg)
            misses += m
            rows.extend(contract_snapshots(row, candles, horizons, cfg.snapshots.max_spread))
    if misses:
        log.warning("%d candle requests not in the cache; run fetch_prices first", misses)
    df = pd.DataFrame(rows, columns=list(SNAPSHOT_COLUMNS))
    return df.astype(SNAPSHOT_COLUMNS)


def estimate_volume_usd(contracts: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """A copy of ``contracts`` with ``volume_usd`` estimated from the cached candles.

    Lifetime traded notional on the YES side: ``volume_fp x VWAP``, where the VWAP
    (sum of volume x mean trade price over volume) is taken over the fetched candle
    windows. For most markets the windows cover the whole life, so this is the
    exact notional; for long-lived ones it scales the lifetime contract count by the
    recent average price. NaN without trades in the windows. Reads the cache only,
    so it works under ``PMCAL_OFFLINE=1``. (The frame floor is on contracts,
    ``volume_native``; this is for describing liquidity.)
    """
    out = []
    with make_client(cfg) as client:
        for row in _records(contracts):
            c, _ = _load_contract_candles(client, row, cfg)
            val = math.nan
            ok = np.isfinite(c.mean) & np.isfinite(c.volume) & (c.volume > 0)
            vol = float(c.volume[ok].sum())
            if vol > 0:
                vwap = float((c.volume[ok] * c.mean[ok]).sum()) / vol
                val = float(row["volume_native"]) * vwap
            out.append(val)
    res = contracts.copy()
    res["volume_usd"] = pd.Series(out, index=contracts.index, dtype="float64")
    return res
