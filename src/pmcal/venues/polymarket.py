"""Polymarket: the sampling frame (Gamma), price paths (CLOB), trade prints (data-api)
and fixed-horizon snapshots.

Why it is built this way
------------------------
Listing. Gamma's offset pagination stops at 10,000 rows, so the frame is listed with
``/markets/keyset`` and its cursor. The listing is done per *market*, not per event,
because an event's ``endDate`` can be months away from its markets' own end dates
("Will X be indicted by ...?" ladders, MVP races), and only market-level filters
(``closed``, ``end_date_*``, ``volume_num_min``) match the frame definition.
``include_tag=true`` returns the parent event's tags on each market (checked to be
identical to ``/events`` tags), and the parent event and its series come nested,
so no second request per event is needed. Pages are capped at 100 markets
(``limit`` > 100 is rejected or resets the connection). The study window is split
into calendar months and a window whose first page is full is bisected (down to one
day) so that many windows can be paged in parallel under the shared rate limit.

Frame. A market is in the frame if its *scheduled* time (``scheduled_ts``) falls in
the study window, never its actual close: filtering on the close would admit
long-dated markets that resolved early, which are mostly YES outcomes. Gamma
filters on ``endDate``, and a sports moneyline's ``endDate`` is often a week or two
after the game, so the listing runs from ``LIST_MARGIN_BEFORE`` before the window
to ``LIST_MARGIN_AFTER`` after it, and rows are filtered on ``scheduled_ts``
afterwards. The frame has no lifetime-volume floor, only "traded at all" (a floor
on lifetime volume selects on the outcome, since upsets attract volume), which
Gamma applies server-side as ``volume_num_min=TRADED_MIN``: exactly the markets with
``volumeNum > 0``. That drops markets whose volume is null; outside the gap below
these are markets that never traded, except about 0.3% of non-crypto ones (checked
on /live-volume, mostly a few dollars), which are left out because finding them
would mean listing ten times as many pages.

Volume gap. Gamma returns null volume fields for markets that closed between about
2026-03-05 and 2026-03-26 (``VOLUME_GAP``), so a server-side volume floor silently
drops them. Outside the gap a null volume means the market never traded, and it is
read as 0. EndDate windows where such markets close on schedule are listed without
the floor; every other window is also swept in ``closedTime`` order (ascending after
the gap, descending before it) until the closes pass the gap, which finds long-dated
markets that resolved early in March and late resolutions. Their volume comes from
data-api ``/live-volume?id=<event>`` (one request per event; it equals ``volumeNum``
wherever both exist and lists only markets that traded), fetched while building the
index for the rows that can be in the frame. ``volume_missing`` marks rows that
closed in the gap without a Gamma volume.

Anchor. For a sports or esports game the ex-ante anchor is the game start
(``gameStartTime``, else the event's ``startTime``); trading continues in play until
resolution, so anchoring at the end date or the close would put in-play prices in
the sample. Everything else uses ``endDate``. Gamma's ``endDate`` and
``gameStartTime`` are current values: a postponed game carries its new start time.
``pm_anchor_flags`` records anchors that look broken.

Outcome. ``outcomePrices`` holds the final payout vector after resolution for CLOB
markets ("1"/"0", or "0.5"/"0.5" for a 50-50 split). Markets whose UMA status says
the answer is still pending are "unresolved" whatever their prices say. AMM-era
markets (2020-22) store the pool's final prices instead, which differ from 0/1 by
up to about 1e-5, so payouts are read with a tolerance ``PAYOUT_TOL`` that is still
below half a CLOB tick (5e-4), so a closed but unsettled CLOB market quoting 0.9995
is never taken for a resolution.

Short crypto. Crypto markets from series with sub-daily recurrence (5m, 15m,
hourly, 4h; "Up or Down" and "above X at <time>" alike) are flagged
``is_short_crypto`` and kept out of the frame, matching Kalshi, where those series
are excluded outright. See ``short_crypto`` for the fallbacks when a market has no
series recurrence. Daily and longer crypto markets stay in.

Side. Yes/No markets are priced on the Yes token. For other binaries (team names,
Over/Under, Up/Down) one side is picked by a hash of the conditionId, so ordering
conventions such as "favourite first" cannot leak into the sample.

Prices. ``GET clob/prices-history`` returns ``{"t", "p"}`` points. ``p`` is the
order-book midpoint, sampled on a regular grid whether or not anything traded. It is
not the UI's "mid unless the spread is over 10c, then last trade": with a live book
at 0.21/0.88 the series showed 0.545 while the last trade was 0.83. Wide-spread
mids are therefore noisy, and the trade prints below are what can tell a quiet
but well-quoted market from a dead one. ``interval=max`` works only at daily
fidelity (finer fidelities return an empty list), and explicit ``startTs``/``endTs``
windows may span at most 15 days. Per contract:

- one daily series over the whole life (points at about 00:00 UTC), which serves
  the 30d and 14d horizons;
- one 10-minute series covering ``[scheduled - 8d, min(scheduled, close)]`` and
  ``[close - 8d, close]`` (one window when they fit in 15 days, two otherwise),
  which serves the 7d to 1h horizons on both anchors. A market that closed before
  its scheduled time needs only ``[close - 8d, close]``, which contains the
  scheduled window; a sports market that closed hours after the game start gets
  ``[start - 8d, close]`` so that the leaky close anchor sees its in-play prices.

Trades. For each "ok" snapshot on the scheduled anchor (horizons in
``TRADE_HORIZONS``), ``GET data-api/trades`` gives the number of taker trades in
the 24 hours up to ``t_snap`` (``n_trades_24h``), the age of the last trade at or
before ``t_snap`` (``last_trade_age_s``) and its price, mapped to the contract's
side (``last``). One request covers ``[min t_snap - 24h, max t_snap]``. If it
returns a full page (``TRADES_PAGE`` trades), the window is split into one request
per snapshot, and then ``n_trades_24h`` is capped at ``TRADES_PAGE``. If no trade
falls before a snapshot inside the window, one ``limit=1`` request ending at the
window start finds the previous trade. Trades of both tokens count.

Everything goes through ``CachedClient`` (one per host), so a re-run makes no
requests, and ``build_snapshots`` reads the cache only (it works with
``PMCAL_OFFLINE=1``). ``fetch_prices`` and ``build_snapshots`` share one function
that plans and reads the requests, so they always use the same cache keys.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import logging
import math
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field, replace
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from pmcal.config import Config
from pmcal.http import CachedClient, HTTPStatusError, Params
from pmcal.sample import frame_mask
from pmcal.schema import CONTRACT_COLUMNS, SNAPSHOT_COLUMNS
from pmcal.snapshots import FloatArray, IntArray, PricePath, Snapshot, take_snapshots

log = logging.getLogger(__name__)

VENUE: Final = "polymarket"
GAMMA_URL: Final = "https://gamma-api.polymarket.com"
CLOB_URL: Final = "https://clob.polymarket.com"
DATA_URL: Final = "https://data-api.polymarket.com"
# Documented limits (docs.polymarket.com, "Rate Limits"): Gamma /markets 30/s,
# CLOB /prices-history 100/s, data-api /trades 20/s. We stay well below them.
GAMMA_RATE: Final = 4.0
CLOB_RATE: Final = 20.0
DATA_RATE: Final = 10.0
INDEX_FILE: Final = "polymarket_index.parquet"

DAY: Final = 86400
HOUR: Final = 3600

PAGE_LIMIT: Final = 100  # Gamma keyset maximum
# Server-side floor meaning "traded at all" when the configured floor is 0: it drops
# markets whose volume is null or 0 (never traded, outside VOLUME_GAP) and keeps every
# market with volume > 0. Sent as a string, as str(1e-06) would be "1e-06".
TRADED_MIN: Final = "0.000001"
MIN_SPLIT_WINDOW: Final = DAY  # listing windows are bisected down to this length
LIST_MARGIN_BEFORE: Final = 7 * DAY  # endDate earlier than the game start (postponed games)
LIST_MARGIN_AFTER: Final = 21 * DAY  # sports moneylines end up to ~2 weeks after the game

# Gamma returns null volume fields (volumeNum, volume, volumeClob, also the event's) for
# markets that closed in this period (5-9k a day from 03-05 to 03-26), so a server-side
# volume floor silently drops them. Outside it, null volume means the market never
# traded (191 of 204 sampled had zero volume on /live-volume, one had over $1k).
# Windows whose markets can close in it are listed without the floor; windows outside
# are also swept in closedTime order for early and late closers. The volume is then
# taken from data-api /live-volume (one request per event, the same figure as
# volumeNum where both exist).
VOLUME_GAP: Final = (1772323200, 1774569600)  # closedTime in [2026-03-01, 2026-03-27)
SWEEP_WINDOW: Final = 7 * DAY

FINE_FIDELITY_MIN: Final = 10
FINE_LOOKBACK: Final = 8 * DAY
MAX_PRICE_WINDOW: Final = 15 * DAY  # CLOB rejects longer startTs/endTs windows

TRADES_PAGE: Final = 1000  # data-api allows up to 10,000; 1,000 trades are ~0.75 MB
# Horizons whose scheduled-anchor snapshots get trade statistics (None = all).
TRADE_HORIZONS: Final[frozenset[str] | None] = None

EARLY_CLOSE: Final = DAY  # close more than this before scheduled_ts counts as early
SHORT_CRYPTO_MAX_S: Final = 4 * HOUR
PAYOUT_TOL: Final = 1e-4
PENDING_UMA: Final = frozenset({"proposed", "disputed", "challenged", "pending"})

PM_COLUMNS: Final[dict[str, str]] = {
    "pm_market_id": "string",  # Gamma market id
    "pm_condition_id": "string",
    "pm_token_id": "string",  # CLOB token of the chosen side (= contract_id when priced)
    "pm_other_token_id": "string",
    "pm_has_clob": "boolean",  # order-book market with token ids (AMM-era markets are not)
    "pm_neg_risk": "boolean",
    "pm_neg_risk_market_id": "string",  # the mutually exclusive group within the event
    "pm_neg_risk_augmented": "boolean",
    "pm_neg_risk_other": "boolean",  # the "Other" placeholder of an augmented event
    "pm_series_slug": "string",
    "pm_series_recurrence": "string",  # e.g. "5m", "hourly", "daily", "weekly"
    # Sports games are split over several events (main, "More Markets", "Exact Score",
    # "Halftime Result"...) that share the event's gameId: a better bootstrap cluster.
    "pm_game_id": "string",
    "pm_tags": "string",  # event tag slugs, "|"-joined
    "pm_category_source": "string",  # "tags", "market_category", "slug" or "none"
    "pm_end_date_ts": "Int64",
    "pm_game_start_ts": "Int64",  # raw gameStartTime (any category)
    "pm_event_start_ts": "Int64",  # raw event startTime (any category)
    "pm_created_ts": "Int64",
    "pm_sports_market_type": "string",
    "pm_group_item_title": "string",
    "pm_outcomes": "string",  # raw JSON
    "pm_outcome_prices": "string",  # raw JSON
    "pm_uma_status": "string",
    "pm_market_type": "string",  # "normal" or "scalar" (old AMM markets)
    "pm_updown_interval_s": "Int64",  # reference interval of an "Up or Down" market
    "pm_anchor_flags": "string",  # comma-separated warnings, see _anchor_flags
    # "gamma" (volumeNum), "gamma_null" (null outside VOLUME_GAP: read as 0),
    # "live_volume" (filled), "missing" (in the gap, not filled)
    "pm_volume_source": "string",
    "pm_in_frame": "boolean",  # pmcal.sample.frame_mask on this index
}
# volume_missing: Gamma reported no volume (see VOLUME_GAP). volume_usd is then filled
# from /live-volume where available and NaN otherwise (pmcal.sample keeps NaN rows in
# the frame and any floor is applied after fill_missing_volume).
INDEX_COLUMNS: Final[dict[str, str]] = {
    **CONTRACT_COLUMNS,
    "volume_missing": "boolean",
    **PM_COLUMNS,
}

PM_SNAPSHOT_COLUMNS: Final[dict[str, str]] = {
    "pm_path_n": "Int64",  # observations in the merged price path
    "pm_cache_missing": "Int64",  # planned requests not found in the cache
}

# -- small parsers --------------------------------------------------------------


def parse_ts(value: Any) -> int | None:
    """Unix seconds from Gamma's assorted timestamp formats, or None.

    Handles "2025-01-14T22:46:28Z", "2025-03-03 18:30:50+00", fractional seconds of
    any length, and bare dates (taken as UTC midnight).
    """
    if value is None or value == "":
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return int(value)
    try:
        d = dt.datetime.fromisoformat(str(value).strip())
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.UTC)
    return int(d.timestamp())


def _json_list(value: Any) -> list[Any]:
    """Gamma stores outcomes, prices and token ids as JSON strings."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    try:
        out = json.loads(value)
    except (TypeError, ValueError):
        return []
    return out if isinstance(out, list) else []


def _iso(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm(s: Any) -> str:
    return str(s or "").strip().lower().replace(" ", "-").strip("-")


# -- side and outcome -----------------------------------------------------------


def choose_side(outcomes: Sequence[str], condition_id: str) -> int:
    """Index of the outcome the contract refers to.

    "Yes" for Yes/No markets. Otherwise a deterministic hash of the conditionId, so
    the side is unrelated to how the venue orders outcomes (favourite first, home
    team first, Over before Under).
    """
    norm = [str(o).strip().lower() for o in outcomes]
    if len(norm) == 2 and set(norm) == {"yes", "no"}:
        return norm.index("yes")
    if not norm:
        return 0
    h = hashlib.sha256(f"pmcal-side:{condition_id.lower()}".encode()).digest()
    return int.from_bytes(h[:8], "big") % len(norm)


def parse_resolution(prices: Sequence[Any], side: int, uma_status: str | None) -> tuple[str, float]:
    """``(resolution_status, outcome)`` from the payout vector of a closed market.

    - pending UMA status, or no payout vector: "unresolved";
    - exactly one payout of 1 and the rest 0 (within ``PAYOUT_TOL``): "resolved",
      outcome 1.0 if that is ``side`` else 0.0;
    - all payouts 0: "void";
    - anything else (50-50 and other fractional payouts): "split".
    """
    if _norm(uma_status) in PENDING_UMA:
        return "unresolved", math.nan
    try:
        p = [float(x) for x in prices]
    except (TypeError, ValueError):
        return "unresolved", math.nan
    if not p or not all(math.isfinite(x) for x in p):
        return "unresolved", math.nan
    ones = [i for i, x in enumerate(p) if abs(x - 1.0) <= PAYOUT_TOL]
    zeros = [i for i, x in enumerate(p) if abs(x) <= PAYOUT_TOL]
    if len(ones) == 1 and len(zeros) == len(p) - 1:
        return "resolved", 1.0 if ones[0] == side else 0.0
    if len(zeros) == len(p):
        return "void", math.nan
    return "split", math.nan


# -- category -------------------------------------------------------------------


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


# Only unambiguous slugs: tickers are tags too ("afl" is Aflac on earnings markets), and
# genuine games carry "sports"/"esports" anyway.
SPORTS_TAGS: Final = _words(
    """sports esports games nfl nba mlb nhl wnba mls ncaa ncaab ncaaf cfb cbb
    college-football college-basketball march-madness soccer football epl premier-league
    la-liga laliga serie-a bundesliga ligue-1 ucl champions-league uel uecl europa-league
    fifa world-cup fifa-world-cup copa-america euros tennis atp wta itf golf pga pga-tour
    liv-golf ufc mma boxing f1 formula-1 formula1 nascar indycar motorsport cricket ipl
    rugby nrl chess olympics cycling horse-racing darts snooker table-tennis
    volleyball handball baseball basketball hockey npb kbo cs2 csgo counter-strike
    league-of-legends dota-2 dota2 valorant overwatch rocket-league call-of-duty
    super-bowl nba-finals stanley-cup world-series wimbledon us-open french-open
    australian-open grand-slam"""
)
# Word and post counts ("Will X say Y during Z?", "How many tweets ...?") are about
# speech, not policy, so they go to culture before the politics rules see them.
SPEECH_TAGS: Final = _words(
    "mentions mention-markets tweets tweet-markets tweets-markets elon-tweets "
    "truth-social-posts trump-speech earnings-calls"
)
WEATHER_TAGS: Final = _words(
    """weather climate temperature daily-temperature global-temperature climate-change
    hurricanes hurricane snow rain precipitation"""
)
CRYPTO_TAGS: Final = _words(
    """crypto crypto-prices bitcoin btc ethereum eth solana sol xrp ripple dogecoin bnb
    memecoins memecoin nfts nft defi stablecoins airdrops airdrop token-launch fdv
    pre-market"""
)
# Data releases and central banks: economics even when also tagged politics.
MACRO_TAGS: Final = _words(
    """fed fed-rates fomc federal-reserve interest-rates global-rates ecb boe boj
    bank-of-england bank-of-japan central-banks inflation cpi pce ppi gdp jobs
    jobs-report nfp unemployment payrolls jobless-claims macro-indicators recession
    economic-data pmi ism retail-sales"""
)
WORLD_TAGS: Final = _words(
    """geopolitics world world-affairs global-affairs global-politics middle-east
    ukraine russia ukraine-russia russia-ukraine israel gaza iran nato war
    foreign-policy ceasefire military taiwan china-taiwan north-korea syria"""
)
POLITICS_TAGS: Final = _words(
    """politics us-politics elections election global-elections world-elections
    us-elections us-election presidential-election midterms midterm-elections primaries
    congress senate house scotus supreme-court courts trump trump-presidency biden
    government polls polling approval cabinet governor mayor mayoral-election
    democrats republicans gop legislation executive-orders"""
)
ECON_GENERIC_TAGS: Final = _words("economy economics economic-policy macro")
FINANCE_TAGS: Final = _words(
    """finance stocks equities stock-market business companies company earnings ipo
    ipos commodities oil gold silver forex fx sp500 s-p-500 nasdaq dow indices
    treasuries bonds yields mergers acquisitions"""
)
CULTURE_TAGS: Final = (
    _words(
        """pop-culture culture entertainment music movies film tv television awards oscars
    grammys emmys golden-globes celebrities celebrity box-office youtube streaming
    netflix spotify billboard eurovision reality-tv social-media religion pope
    video-games gaming art books fashion royals royal-family"""
    )
    | SPEECH_TAGS
)
SCIENCE_TECH_TAGS: Final = _words(
    """science tech technology ai artificial-intelligence openai chatgpt llm space
    spacex nasa starship health covid coronavirus covid-19 pandemic medicine biotech
    fda disease vaccines bird-flu measles"""
)
ELECTION_RE: Final = re.compile(
    r"\b(elections?|primary|primaries|nominee|nomination|referendum|runoff|mayoral|"
    r"gubernatorial|electoral|caucus(es)?)\b",
    re.IGNORECASE,
)

# Pre-2023 markets have no useful tags ("all") but a market-level category.
MARKET_CATEGORY_MAP: Final[dict[str, str]] = {
    "us-current-affairs": "politics",
    "politics": "politics",
    "global-politics": "world",
    "crypto": "crypto",
    "nfts": "crypto",
    "sports": "sports",
    "nba": "sports",
    "nfl": "sports",
    "olympics": "sports",
    "chess": "sports",
    "pop-culture": "culture",
    "art": "culture",
    "business": "finance",
    "science": "science_tech",
    "tech": "science_tech",
    "coronavirus": "science_tech",
}


def _is_election(tags: Iterable[str], title: str) -> bool:
    return any("election" in t or t in {"primaries", "midterms"} for t in tags) or bool(
        ELECTION_RE.search(title)
    )


def _match_rules(tags: frozenset[str], title: str) -> str | None:
    """First matching rule, in this order:

    1. sports (sports/esports/games tags, league and competition slugs)
    2. culture for word and post counts (mentions, tweets)
    3. weather (weather, climate)
    4. crypto
    5. economics for data releases and central banks (fed, cpi, gdp, jobs, ecb...)
    6. world (geopolitics, middle-east, ukraine...) unless it is about an election
    7. politics (politics, elections, government, courts, trump...)
    8. finance (stocks, companies, earnings, commodities...)
    9. economics for the generic economy tags
    10. culture (pop culture, entertainment, awards...)
    11. science_tech (science, tech, ai, space, health)
    """
    if tags & SPORTS_TAGS:
        return "sports"
    if tags & SPEECH_TAGS:
        return "culture"
    if tags & WEATHER_TAGS:
        return "weather"
    if tags & CRYPTO_TAGS:
        return "crypto"
    if tags & MACRO_TAGS:
        return "economics"
    if tags & WORLD_TAGS:
        return "politics" if _is_election(tags, title) else "world"
    if tags & POLITICS_TAGS or any("election" in t for t in tags):
        return "politics"
    if tags & FINANCE_TAGS:
        return "finance"
    if tags & ECON_GENERIC_TAGS:
        return "economics"
    if tags & CULTURE_TAGS:
        return "culture"
    if tags & SCIENCE_TECH_TAGS:
        return "science_tech"
    return None


def _slug_tokens(*slugs: str | None) -> frozenset[str]:
    out: set[str] = set()
    for s in slugs:
        parts = [p for p in _norm(s).split("-") if p]
        out.update(parts)
        out.update(f"{a}-{b}" for a, b in itertools.pairwise(parts))
    return frozenset(out)


def categorize(
    tags: Iterable[str],
    market_category: str | None = None,
    title: str = "",
    slugs: Sequence[str | None] = (),
) -> tuple[str, str]:
    """``(category, source)``: the canonical category and where it came from.

    Event tags are tried first with the precedence in ``_match_rules``. Markets whose
    tags match nothing (old markets are tagged only "all") fall back to the old
    market-level ``category`` (``MARKET_CATEGORY_MAP``; "Global Politics" becomes
    politics when the question is about an election), then to the same rules applied
    to the words of the market and event slugs. Otherwise "other".
    """
    t = frozenset(_norm(x) for x in tags if x)
    cat = _match_rules(t, title)
    if cat:
        return cat, "tags"
    mc = MARKET_CATEGORY_MAP.get(_norm(market_category))
    if mc:
        if mc == "world" and _is_election((), title):
            mc = "politics"
        return mc, "market_category"
    cat = _match_rules(_slug_tokens(*slugs), title)
    if cat:
        return cat, "slug"
    return "other", "none"


# -- anchor and short crypto ----------------------------------------------------

_UPDOWN_SUFFIX: Final = re.compile(r"-(5m|15m|30m|1h|hourly|4h|daily|1d)$")
_SUFFIX_S: Final = {
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": HOUR,
    "hourly": HOUR,
    "4h": 4 * HOUR,
    "daily": DAY,
    "1d": DAY,
}
_Q_RANGE: Final = re.compile(
    r"(\d{1,2})(?::(\d{2}))?\s*([AP]M)\s*-\s*(\d{1,2})(?::(\d{2}))?\s*([AP]M)\s*ET", re.I
)
_Q_HOURLY: Final = re.compile(r",\s*\d{1,2}(?::\d{2})?\s*[AP]M\s*ET\b", re.I)
_Q_ISO_DATE: Final = re.compile(r"\b(20\d\d-\d\d-\d\d)\b")


def _minutes(h: str, m: str | None, ampm: str) -> int:
    hh = int(h) % 12 + (12 if ampm.upper() == "PM" else 0)
    return hh * 60 + int(m or 0)


def updown_interval_s(
    question: str,
    series_slug: str | None,
    tags: Iterable[str],
    event_start_ts: int | None,
    end_ts: int | None,
) -> int | None:
    """Length of the price interval an "Up or Down" market is about, or None.

    Taken from the series slug ("btc-up-or-down-15m"), then the interval tags
    ("5M", "15M", "1H", "4H"), then the event's start-to-end span, then the question
    ("8:00AM-8:15AM ET" or ", 7AM ET" for hourly).
    """
    if "up or down" not in question.lower():
        return None
    m = _UPDOWN_SUFFIX.search(_norm(series_slug))
    if m:
        return _SUFFIX_S[m.group(1)]
    for t in tags:
        s = _SUFFIX_S.get(_norm(t))
        if s and _norm(t) not in {"daily", "hourly"}:
            return s
    if event_start_ts and end_ts and 0 < end_ts - event_start_ts <= DAY:
        return end_ts - event_start_ts
    r = _Q_RANGE.search(question)
    if r:
        a = _minutes(r.group(1), r.group(2), r.group(3))
        b = _minutes(r.group(4), r.group(5), r.group(6))
        return ((b - a) % (24 * 60)) * 60 or None
    if _Q_HOURLY.search(question):
        return HOUR
    return None


SUB_DAILY: Final = frozenset({"5m", "15m", "30m", "1h", "hourly", "4h"})
_INTERVAL_TAGS: Final = frozenset({"5m", "15m", "30m", "1h", "4h"})


def short_crypto(
    *,
    is_crypto: bool,
    recurrence: str | None,
    series_slug: str | None,
    tags: Iterable[str],
    updown_interval: int | None,
    open_ts: int | None,
    scheduled_ts: int | None,
) -> bool:
    """Whether a market is short-horizon crypto, excluded from the frame like Kalshi's
    hourly and 15-minute series.

    Any crypto market from a series with sub-daily recurrence (5m, 15m, hourly, 4h),
    whether "Up or Down" or "above X at <time>". When the series' recurrence is
    unknown: the series slug's suffix ("-15m", "-hourly"), an interval tag ("5M",
    "15M", "1H", "4H"), an "Up or Down" interval of at most 4h, and finally a scheduled
    life (open to scheduled time) of at most 4h. Daily and longer series stay in.
    """
    if not is_crypto:
        return False
    rec = _norm(recurrence)
    if rec:
        return rec in SUB_DAILY
    m = _UPDOWN_SUFFIX.search(_norm(series_slug))
    if m:
        return _SUFFIX_S[m.group(1)] <= SHORT_CRYPTO_MAX_S
    if any(_norm(t) in _INTERVAL_TAGS for t in tags):
        return True
    if updown_interval is not None:
        return updown_interval <= SHORT_CRYPTO_MAX_S
    if open_ts is not None and scheduled_ts is not None:
        return scheduled_ts - open_ts <= SHORT_CRYPTO_MAX_S
    return False


def choose_anchor(
    end_ts: int | None,
    game_start_ts: int | None,
    event_start_ts: int | None,
    is_sports: bool,
) -> tuple[int | None, str, int | None]:
    """``(scheduled_ts, anchor_source, event_start_ts)``.

    Sports and esports markets anchor at the game start: the market's
    ``gameStartTime``, else the event's ``startTime``. Everything else anchors at
    ``endDate``. ``gameStartTime`` is not trusted outside sports: weather markets set
    it to the start of the measured day, and the 2025 government-shutdown ladder set
    it to the day the shutdown actually ended, which would leak the outcome.
    """
    if is_sports:
        start = game_start_ts if game_start_ts is not None else event_start_ts
        if start is not None:
            return start, "game_start", start
    return end_ts, "end_date", None


def _anchor_flags(
    *,
    scheduled_ts: int | None,
    anchor_source: str,
    open_ts: int | None,
    created_ts: int | None,
    close_ts: int | None,
    category: str,
    question: str,
    sports_market_type: str | None,
) -> str:
    """Comma-separated warnings about the anchor (empty when nothing looks wrong).

    - ``scheduled_before_created``/``scheduled_before_open``: endDate (or game start)
      earlier than the market's creation or opening, e.g. an endDate set by mistake;
    - ``close_before_open``: closedTime earlier than the opening;
    - ``game_start_after_close``: the market resolved before its game start
      (postponed or cancelled games, typically 50-50);
    - ``sports_game_no_start``: looks like a game but has no start time, so the
      endDate anchor may be hours or a day away from kick-off;
    - ``title_date_mismatch``: the question names a YYYY-MM-DD date more than two days
      from the scheduled time (rescheduled games keep their original title);
    - ``competition_start``: a game-start anchor on a market without a sports market
      type, i.e. the start of a multi-day competition (golf tournament winner, Olympic
      medal markets whose start is the opening ceremony) rather than of one game.
    """
    flags: list[str] = []
    if scheduled_ts is not None:
        if created_ts is not None and scheduled_ts < created_ts:
            flags.append("scheduled_before_created")
        if open_ts is not None and scheduled_ts < open_ts:
            flags.append("scheduled_before_open")
    if close_ts is not None and open_ts is not None and close_ts < open_ts:
        flags.append("close_before_open")
    if (
        anchor_source == "game_start"
        and close_ts is not None
        and scheduled_ts is not None
        and close_ts < scheduled_ts
    ):
        flags.append("game_start_after_close")
    if (
        anchor_source == "end_date"
        and category == "sports"
        and (sports_market_type or re.search(r"\bvs\.?\b", question, re.I))
    ):
        flags.append("sports_game_no_start")
    if anchor_source == "game_start" and not sports_market_type:
        flags.append("competition_start")
    m = _Q_ISO_DATE.search(question)
    if m and scheduled_ts is not None:
        named = parse_ts(m.group(1))
        if named is not None and abs(scheduled_ts - named) > 2 * DAY:
            flags.append("title_date_mismatch")
    return ",".join(flags)


# -- one market -> one contract row ---------------------------------------------


def _tag_slugs(tags: Any) -> list[str]:
    out: list[str] = []
    for t in tags or []:
        if isinstance(t, Mapping):
            s = t.get("slug") or t.get("label")
            if s:
                out.append(_norm(s))
    return out


def _gamma_volume(m: Mapping[str, Any]) -> float:
    """Lifetime volume from ``volumeNum`` (else ``volume``), NaN when Gamma has none."""
    for key in ("volumeNum", "volume"):
        v = m.get(key)
        if v is None or v == "":
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            return x
    return math.nan


def parse_market(m: Mapping[str, Any]) -> dict[str, Any]:
    """One Gamma market (as returned by ``/markets/keyset?include_tag=true``) as a
    contract row with ``INDEX_COLUMNS`` (``n_in_event`` and ``pm_in_frame`` are
    filled in later, once the whole frame is known)."""
    events = m.get("events") or []
    ev: Mapping[str, Any] = events[0] if events and isinstance(events[0], Mapping) else {}
    question = str(m.get("question") or "")
    tags = _tag_slugs(m.get("tags")) or _tag_slugs(ev.get("tags"))
    outcomes = [str(o) for o in _json_list(m.get("outcomes"))]
    prices = _json_list(m.get("outcomePrices"))
    tokens = [str(t) for t in _json_list(m.get("clobTokenIds"))]
    cond = str(m.get("conditionId") or "")
    market_id = str(m.get("id") or "")
    side = choose_side(outcomes, cond or market_id)
    status, outcome = parse_resolution(prices, side, m.get("umaResolutionStatus"))

    created = parse_ts(m.get("createdAt"))
    open_ts = parse_ts(m.get("acceptingOrdersTimestamp")) or parse_ts(m.get("startDate")) or created
    close_ts = parse_ts(m.get("closedTime"))
    end_ts = parse_ts(m.get("endDate"))
    game_start = parse_ts(m.get("gameStartTime"))
    ev_start = parse_ts(ev.get("startTime"))

    series = [x for x in ev.get("series") or [] if isinstance(x, Mapping)]
    series_slug = ev.get("seriesSlug") or (series[0].get("slug") if series else None)
    match = [x for x in series if x.get("slug") == series_slug] or series
    recurrence = (match[0].get("recurrence") or None) if match else None
    category, cat_source = categorize(
        tags, m.get("category"), question, (m.get("slug"), ev.get("slug"), series_slug)
    )
    scheduled, anchor_source, event_start = choose_anchor(
        end_ts, game_start, ev_start, category == "sports"
    )
    interval = updown_interval_s(question, series_slug, tags, ev_start, end_ts)
    is_crypto = category == "crypto" or bool(set(tags) & CRYPTO_TAGS)
    is_short = short_crypto(
        is_crypto=is_crypto,
        recurrence=recurrence,
        series_slug=series_slug,
        tags=tags,
        updown_interval=interval,
        open_ts=open_ts,
        scheduled_ts=scheduled,
    )

    has_clob = bool(m.get("enableOrderBook")) and len(tokens) == len(outcomes) and bool(tokens)
    token = tokens[side] if side < len(tokens) else None
    other = tokens[1 - side] if len(tokens) == 2 else None
    contract_id = token or f"{cond or market_id}:{side}"
    neg_risk = bool(m.get("negRisk"))
    volume = _gamma_volume(m)
    in_gap = close_ts is not None and VOLUME_GAP[0] <= close_ts < VOLUME_GAP[1]
    volume_missing = math.isnan(volume) and in_gap
    if math.isnan(volume) and not in_gap:
        volume_source = "gamma_null"  # never traded
        volume = 0.0
    else:
        volume_source = "missing" if volume_missing else "gamma"
    game_id = _opt_str(ev.get("gameId"))
    event_id = str(ev.get("id")) if ev.get("id") is not None else f"m{market_id}"
    venue_category = "|".join(([f"category:{m['category']}"] if m.get("category") else []) + tags)
    flags = _anchor_flags(
        scheduled_ts=scheduled,
        anchor_source=anchor_source,
        open_ts=open_ts,
        created_ts=created,
        close_ts=close_ts,
        category=category,
        question=question,
        sports_market_type=m.get("sportsMarketType"),
    )
    return {
        "venue": VENUE,
        "contract_id": contract_id,
        "market_id": cond or market_id,
        "event_id": event_id,
        # "game:" keeps game ids apart from event ids (the two id spaces overlap)
        "cluster_id": f"game:{game_id}" if game_id else event_id,
        "series_id": series_slug,
        "title": question,
        "event_title": ev.get("title"),
        "outcome_label": outcomes[side] if side < len(outcomes) else None,
        "category": category,
        "venue_category": venue_category,
        "n_in_event": None,
        "mutually_exclusive": neg_risk,
        "open_ts": open_ts,
        "close_ts": close_ts,
        "scheduled_ts": scheduled,
        "anchor_source": anchor_source,
        "event_start_ts": event_start,
        "resolved_ts": close_ts,
        "outcome": outcome,
        "resolution_status": status,
        "volume_usd": volume,
        "volume_native": volume,
        "liquidity_usd": math.nan,
        "is_short_crypto": is_short,
        "volume_missing": volume_missing,
        "pm_market_id": market_id,
        "pm_condition_id": cond or None,
        "pm_token_id": token,
        "pm_other_token_id": other,
        "pm_has_clob": has_clob,
        "pm_neg_risk": neg_risk,
        "pm_neg_risk_market_id": m.get("negRiskMarketID") if neg_risk else None,
        "pm_neg_risk_augmented": bool(ev.get("negRiskAugmented")),
        "pm_neg_risk_other": bool(m.get("negRiskOther")),
        "pm_series_slug": series_slug,
        "pm_series_recurrence": recurrence,
        "pm_game_id": game_id,
        "pm_tags": "|".join(tags),
        "pm_category_source": cat_source,
        "pm_end_date_ts": end_ts,
        "pm_game_start_ts": game_start,
        "pm_event_start_ts": ev_start,
        "pm_created_ts": created,
        "pm_sports_market_type": m.get("sportsMarketType"),
        "pm_group_item_title": m.get("groupItemTitle") or None,
        "pm_outcomes": json.dumps(outcomes),
        "pm_outcome_prices": json.dumps([str(p) for p in prices]) if prices else None,
        "pm_uma_status": m.get("umaResolutionStatus"),
        "pm_market_type": m.get("marketType"),
        "pm_updown_interval_s": interval,
        "pm_anchor_flags": flags,
        "pm_volume_source": volume_source,
        "pm_in_frame": None,
    }


def coerce(df: pd.DataFrame, columns: Mapping[str, str]) -> pd.DataFrame:
    """``df`` with exactly ``columns`` (missing ones added as null), in that order and
    with those dtypes."""
    out = pd.DataFrame(index=df.index)
    for col, dtype in columns.items():
        s = df[col] if col in df.columns else pd.Series(None, index=df.index, dtype="object")
        if dtype in ("Int64", "float64"):
            s = pd.to_numeric(s, errors="coerce")
        if dtype == "Int64":
            s = s.round()
        out[col] = s.astype(cast(Any, dtype))
    return out.reset_index(drop=True)


def _volume_ok(volume: pd.Series, floor: float) -> pd.Series:
    """Traded (volume > 0) and at or above the floor, or volume unknown (filled later)."""
    v = pd.to_numeric(volume, errors="coerce")
    return ((v > 0) & (v >= floor)) | v.isna()


def assemble_index(rows: pd.DataFrame | Iterable[Mapping[str, Any]], cfg: Config) -> pd.DataFrame:
    """Index rows from parsed markets: dedupe, keep scheduled times inside the study
    window and markets that traded (volume > 0 and at or above the configured floor,
    or volume unknown), flag the frame with
    ``pmcal.sample.frame_mask``, count ``n_in_event``.

    ``n_in_event`` counts the event's markets that are in the frame, whatever their
    resolution status.
    """
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame.from_records(list(rows))
    if df.empty:
        return coerce(df, INDEX_COLUMNS)
    df = df.drop_duplicates("pm_market_id", keep="first")
    sched = pd.to_numeric(df["scheduled_ts"], errors="coerce")
    in_window = (sched >= cfg.study.start_ts) & (sched < cfg.study.end_ts)
    traded = _volume_ok(df["volume_usd"], cfg.frame.min_volume_usd)
    df = coerce(df.loc[in_window & traded], INDEX_COLUMNS)
    df["pm_in_frame"] = frame_mask(df, cfg).astype("boolean")
    counts = df.loc[df["pm_in_frame"].astype(bool)].groupby("event_id").size()
    df["n_in_event"] = df["event_id"].map(counts).fillna(0).astype("Int64")
    df = df.sort_values(["scheduled_ts", "event_id", "pm_market_id"], kind="stable")
    return df.reset_index(drop=True)


# -- listing --------------------------------------------------------------------


def gamma_client(cfg: Config) -> CachedClient:
    return CachedClient(GAMMA_URL, cfg.raw_dir(VENUE) / "gamma", rate_per_s=GAMMA_RATE, burst=2)


def clob_client(cfg: Config) -> CachedClient:
    return CachedClient(CLOB_URL, cfg.raw_dir(VENUE) / "clob", rate_per_s=CLOB_RATE, burst=4)


def data_client(cfg: Config) -> CachedClient:
    return CachedClient(DATA_URL, cfg.raw_dir(VENUE) / "data", rate_per_s=DATA_RATE, burst=4)


def month_windows(lo: int, hi: int) -> list[tuple[int, int]]:
    """``[lo, hi)`` cut at calendar-month boundaries (UTC)."""
    out: list[tuple[int, int]] = []
    cur = lo
    while cur < hi:
        d = dt.datetime.fromtimestamp(cur, dt.UTC)
        nxt_month = dt.datetime(d.year + d.month // 12, d.month % 12 + 1, 1, tzinfo=dt.UTC)
        nxt = min(int(nxt_month.timestamp()), hi)
        out.append((cur, nxt))
        cur = nxt
    return out


@dataclass(frozen=True)
class ListWindow:
    """One listing request family: markets with endDate in ``[lo, hi]``.

    ``floor``: apply the volume floor server-side. ``sweep``: "asc" or "desc" to page
    in closedTime order only until the closes pass ``VOLUME_GAP`` (early closers of
    later windows, late closers of earlier ones).
    """

    lo: int
    hi: int
    floor: bool = True
    sweep: str | None = None

    @property
    def kind(self) -> str:
        return "sweep" if self.sweep else ("floor" if self.floor else "open")


def plan_windows(lo: int, hi: int) -> list[ListWindow]:
    """Listing windows for endDates in ``[lo, hi)``.

    Calendar months, cut at the edges of ``[gap start - LIST_MARGIN_BEFORE, gap end +
    LIST_MARGIN_AFTER]``: inside, no volume floor; outside, the floor plus weekly
    closedTime sweeps (ascending after the gap range, descending before it).
    """
    u_lo, u_hi = VOLUME_GAP[0] - LIST_MARGIN_BEFORE, VOLUME_GAP[1] + LIST_MARGIN_AFTER
    out: list[ListWindow] = []
    sweeps: list[ListWindow] = []
    for a, b in month_windows(lo, hi):
        cuts = sorted({a, b} | {c for c in (u_lo, u_hi) if a < c < b})
        for x, y in itertools.pairwise(cuts):
            inside = x >= u_lo and y <= u_hi
            out.append(ListWindow(x, y, floor=not inside))
            if not inside:
                direction = "asc" if x >= u_hi else "desc"
                sweeps.extend(
                    ListWindow(c, min(c + SWEEP_WINDOW, y), floor=False, sweep=direction)
                    for c in range(x, y, SWEEP_WINDOW)
                )
    return out + sweeps


def _list_params(w: ListWindow, min_volume: float) -> Params:
    params: Params = {
        "limit": PAGE_LIMIT,
        "closed": True,
        "include_tag": True,
        "end_date_min": _iso(w.lo),
        "end_date_max": _iso(w.hi),
        "volume_num_min": (min_volume if min_volume > 0 else TRADED_MIN) if w.floor else None,
    }
    if w.sweep:
        params["order"] = "closedTime"
        params["ascending"] = w.sweep == "asc"
    return params


def _past_gap(markets: Sequence[Mapping[str, Any]], direction: str) -> bool:
    """Whether a closedTime-ordered page has run past the volume gap."""
    if not markets:
        return True
    last = parse_ts(markets[-1].get("closedTime"))
    if last is None:
        return True
    return last >= VOLUME_GAP[1] if direction == "asc" else last < VOLUME_GAP[0]


@dataclass
class _Listed:
    window: ListWindow
    rows: pd.DataFrame | None = None
    pages: int = 0
    split: list[ListWindow] = field(default_factory=list)


def _halves(w: ListWindow) -> list[ListWindow]:
    mid = w.lo + (w.hi - w.lo) // 2 // HOUR * HOUR
    return [replace(w, hi=mid), replace(w, lo=mid)]


def _list_window(
    client: CachedClient, w: ListWindow, cfg: Config, keep: Callable[[pd.DataFrame], Any]
) -> _Listed:
    """The markets of one window, or a split if the window is dense.

    A plain window longer than ``MIN_SPLIT_WINDOW`` whose first page is full is
    bisected instead of paged, so dense periods become many short windows listed in
    parallel. A sweep pages until it passes the gap, and is bisected if Gamma fails
    on it (ordering a long window can return 500).
    """
    params = _list_params(w, cfg.frame.min_volume_usd)
    try:
        page = client.get_json("markets/keyset", params) or {}
    except HTTPStatusError:
        if w.sweep and w.hi - w.lo > MIN_SPLIT_WINDOW:
            return _Listed(w, pages=1, split=_halves(w))
        raise
    cursor = page.get("next_cursor")
    if cursor and not w.sweep and w.hi - w.lo > MIN_SPLIT_WINDOW:
        return _Listed(w, pages=1, split=_halves(w))
    markets = list(page.get("markets") or [])
    rows = [parse_market(m) for m in markets]
    pages, seen = 1, set()
    while cursor and not (w.sweep and _past_gap(markets, w.sweep)):
        if cursor in seen:
            raise RuntimeError(f"Gamma returned the same cursor twice for {w}")
        seen.add(cursor)
        page = client.get_json("markets/keyset", {**params, "after_cursor": cursor}) or {}
        markets = list(page.get("markets") or [])
        rows.extend(parse_market(m) for m in markets)
        cursor = page.get("next_cursor")
        pages += 1
    df = pd.DataFrame.from_records(rows) if rows else pd.DataFrame()
    return _Listed(w, rows=keep(df) if not df.empty else df, pages=pages)


def list_markets(
    cfg: Config, *, workers: int = 4, client: CachedClient | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parsed rows for every closed market that traded (volume > 0 and at or above the
    configured floor, or unknown) whose endDate is within the listing margins of the
    study window. Rows whose
    scheduled time is outside the study window are dropped as each window is parsed
    (and counted)."""
    lo = cfg.study.start_ts - LIST_MARGIN_BEFORE
    hi = cfg.study.end_ts + LIST_MARGIN_AFTER
    stats: Counter[str] = Counter()
    window_counts: list[tuple[int, int]] = []

    def keep(df: pd.DataFrame) -> pd.DataFrame:
        s = pd.to_numeric(df["scheduled_ts"], errors="coerce")
        ok = (s >= cfg.study.start_ts) & (s < cfg.study.end_ts)
        ok &= _volume_ok(df["volume_usd"], cfg.frame.min_volume_usd)
        window_counts.append((len(df), int(ok.sum())))
        return df.loc[ok]

    own = client is None
    gc = client or gamma_client(cfg)
    frames: list[pd.DataFrame] = []
    t0 = time.monotonic()
    plan = plan_windows(lo, hi)
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            pending: dict[Future[_Listed], ListWindow] = {
                pool.submit(_list_window, gc, w, cfg, keep): w for w in plan
            }
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    pending.pop(fut)
                    res = fut.result()
                    stats["pages"] += res.pages
                    if res.split:
                        stats["splits"] += 1
                        for w in res.split:
                            pending[pool.submit(_list_window, gc, w, cfg, keep)] = w
                        continue
                    stats["windows"] += 1
                    kind = res.window.kind
                    stats[f"windows_{kind}"] += 1
                    stats[f"pages_{kind}"] += res.pages
                    if res.rows is not None and not res.rows.empty:
                        frames.append(res.rows)
                    if stats["windows"] % 50 == 0:
                        log.info(
                            "listing: %d windows done, %d pending, %d pages, %d markets, %.0fs",
                            stats["windows"],
                            len(pending),
                            stats["pages"],
                            sum(n for n, _ in window_counts),
                            time.monotonic() - t0,
                        )
    finally:
        if own:
            gc.close()
    stats["listed"] = sum(n for n, _ in window_counts)
    stats["kept"] = sum(k for _, k in window_counts)
    stats["seconds"] = int(time.monotonic() - t0)
    stats.update({f"http_{k}": v for k, v in gc.stats.items()})
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not rows.empty:
        rows = rows.drop_duplicates("pm_market_id", keep="first").reset_index(drop=True)
    return rows, dict(stats)


def _live_volume_params(event_id: str) -> Params:
    return {"id": event_id}


def _event_volumes(data: Any) -> dict[str, float] | None:
    """conditionId -> volume from a /live-volume response; None when it has nothing.

    The response lists only markets that traded, so a market of a non-empty
    response that is not listed has volume 0.
    """
    if not isinstance(data, list) or not data or not isinstance(data[0], Mapping):
        return None
    markets = data[0].get("markets") or []
    out: dict[str, float] = {}
    for m in markets:
        if isinstance(m, Mapping) and m.get("market"):
            try:
                out[str(m["market"]).lower()] = float(m.get("value"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
    return out or None


def fill_missing_volume(
    contracts: pd.DataFrame, cfg: Config, *, fetch: bool = False, workers: int = 8
) -> pd.DataFrame:
    """``contracts`` with ``volume_usd``/``volume_native`` filled for rows whose volume
    Gamma did not report (``volume_missing``), from data-api ``/live-volume`` (one
    request per event; the figure equals Gamma's ``volumeNum`` where both exist).

    Reads the cache only unless ``fetch``; rows whose event is not cached or has no
    volume data keep NaN. ``pm_volume_source`` becomes "live_volume" where filled.
    The volume floor is not applied here.
    """
    out = contracts.copy()
    if "volume_missing" not in out.columns or out.empty:
        return out
    miss = out["volume_missing"].fillna(False).astype(bool) & out["volume_usd"].isna()
    events = sorted(
        {str(e) for e in out.loc[miss, "event_id"].dropna() if str(e).isdigit()}, key=int
    )
    if not events:
        return out
    vols: dict[str, dict[str, float] | None] = {}
    with data_client(cfg) as c:
        if fetch:
            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futs = {
                    pool.submit(c.get_json, "live-volume", _live_volume_params(e)): e
                    for e in events
                }
                for k, fut in enumerate(as_completed(futs), 1):
                    e = futs[fut]
                    try:
                        vols[e] = _event_volumes(fut.result())
                    except Exception as ex:  # left missing; retried on the next run
                        log.warning("live-volume for event %s failed: %s", e, ex)
                        vols[e] = None
                    if k % 2000 == 0 or k == len(events):
                        el = time.monotonic() - t0
                        log.info(
                            "live-volume: %d/%d events, %.1f/s, ETA %.1f min",
                            k,
                            len(events),
                            k / el,
                            (len(events) - k) / (k / el) / 60,
                        )
        else:
            for e in events:
                r = c.cached("live-volume", _live_volume_params(e))
                vols[e] = _event_volumes(r.data) if r is not None and r.status == 200 else None
    filled = 0
    for i in out.index[miss]:
        ev = vols.get(str(out.at[i, "event_id"]))
        if ev is None:
            continue
        cond = str(out.at[i, "market_id"]).lower()
        v = ev.get(cond, 0.0)
        out.at[i, "volume_usd"] = v
        out.at[i, "volume_native"] = v
        if "pm_volume_source" in out.columns:
            out.at[i, "pm_volume_source"] = "live_volume"
        filled += 1
    log.info(
        "volume: %d of %d rows without Gamma volume filled from /live-volume (%d events)",
        filled,
        int(miss.sum()),
        len(events),
    )
    return out


def build_index(cfg: Config, *, workers: int = 4) -> pd.DataFrame:
    """Every closed Polymarket market in the sampling frame (metadata only), one row
    per contract, written to ``interim/polymarket_index.parquet``.

    Void, split and unresolved markets stay in the index with their
    ``resolution_status``; short crypto up/down and in-play markets stay with
    ``pm_in_frame=False``. Markets that closed in ``VOLUME_GAP`` without a Gamma
    volume and can be in the frame get their volume from ``/live-volume`` (one
    request per event at the data-api rate, ``workers`` threads) before the floor is
    applied; any still unknown stay with NaN volume and ``volume_missing``.
    """
    rows, stats = list_markets(cfg, workers=workers)
    if not rows.empty:
        miss = rows["volume_missing"].astype(bool)
        # Only rows that can be in the frame need a volume: short crypto and in-play
        # markets are out of it whatever they traded.
        cand = miss & frame_mask(rows.assign(volume_missing=True), cfg)
        log.info(
            "%d listed rows closed in VOLUME_GAP without Gamma volume; %d can be in the "
            "frame (%d events), filling them from /live-volume",
            int(miss.sum()),
            int(cand.sum()),
            rows.loc[cand, "event_id"].nunique(),
        )
        if cand.any():
            filled = fill_missing_volume(rows.loc[cand], cfg, fetch=True, workers=workers)
            rows.loc[cand, ["volume_usd", "volume_native", "pm_volume_source"]] = filled[
                ["volume_usd", "volume_native", "pm_volume_source"]
            ]
    df = assemble_index(rows, cfg)
    out = cfg.interim_dir() / INDEX_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info(
        "polymarket index: listed %d markets (%d pages, %d windows, %ds; %s); "
        "%d rows written to %s",
        stats.get("listed", 0),
        stats.get("pages", 0),
        stats.get("windows", 0),
        stats.get("seconds", 0),
        ", ".join(
            f"{k}: {stats.get(f'windows_{k}', 0)} windows/{stats.get(f'pages_{k}', 0)} pages"
            for k in ("floor", "open", "sweep")
        ),
        len(df),
        out,
    )
    log_index_summary(df)
    return df


def log_index_summary(df: pd.DataFrame, top: int = 30) -> None:
    """Counts the frame decisions rest on, and the most common unmapped tags."""
    if df.empty:
        log.info("polymarket index is empty")
        return
    short = df["is_short_crypto"].astype(bool)
    log.info(
        "short crypto up/down: %d (%s); without CLOB: %d",
        int(short.sum()),
        "excluded from the frame" if not df.loc[short, "pm_in_frame"].any() else "in frame",
        int((~df["pm_has_clob"].astype(bool)).sum()),
    )
    for col in (
        "resolution_status",
        "anchor_source",
        "category",
        "pm_category_source",
        "pm_volume_source",
        "pm_in_frame",
    ):
        log.info("%s: %s", col, df[col].value_counts(dropna=False).to_dict())
    flags = Counter(f for s in df["pm_anchor_flags"].dropna() for f in str(s).split(",") if f)
    log.info("anchor flags: %s", dict(flags))
    _warn_thin_days(df.loc[df["pm_in_frame"].astype(bool)])
    other = df.loc[df["category"] == "other", "pm_tags"].dropna()
    unmapped = Counter(t for s in other for t in str(s).split("|") if t)
    log.info("top unmapped tags (category 'other'): %s", unmapped.most_common(top))


def _warn_thin_days(frame: pd.DataFrame, share: float = 0.25) -> None:
    """Warn about days with far fewer frame markets than usual.

    A gap in Gamma's data shows up this way: before ``VOLUME_GAP`` was handled, the
    days whose markets closed in early-to-mid March 2026 had a tenth of the usual
    count because their volume fields were null.
    """
    if frame.empty:
        return
    day = pd.to_datetime(frame["scheduled_ts"].astype("int64"), unit="s", utc=True).dt.floor("D")
    counts = day.value_counts().sort_index()
    full = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
    n = counts.reindex(full, fill_value=0).to_numpy()
    med = float(np.median(n))
    thin = [(d, int(k)) for d, k in zip(full, n, strict=True) if k < share * med]
    if thin:
        log.warning(
            "%d of %d days have fewer than %.0f%% of the median %.0f frame markets/day: %s",
            len(thin),
            len(n),
            100 * share,
            med,
            ", ".join(f"{d:%Y-%m-%d}={k}" for d, k in thin),
        )


# -- prices and trades ----------------------------------------------------------

Row = Mapping[str, Any] | pd.Series


def _opt_bool(x: Any) -> bool:
    return False if x is None or x is pd.NA else bool(x)


def _opt_int(x: Any) -> int | None:
    if x is None or x is pd.NA:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
        return int(x)
    except (TypeError, ValueError):
        return None


def _opt_str(x: Any) -> str | None:
    if x is None or x is pd.NA or (isinstance(x, float) and math.isnan(x)):
        return None
    return str(x)


@dataclass(frozen=True)
class ContractRef:
    """The fields of an index row that the price and trade requests depend on."""

    contract_id: str
    token_id: str | None
    condition_id: str | None
    open_ts: int | None
    scheduled_ts: int | None
    close_ts: int | None
    has_clob: bool

    @classmethod
    def from_row(cls, row: Row) -> ContractRef:
        token = _opt_str(row.get("pm_token_id"))
        return cls(
            contract_id=str(row.get("contract_id")),
            token_id=token,
            condition_id=_opt_str(row.get("pm_condition_id")),
            open_ts=_opt_int(row.get("open_ts")),
            scheduled_ts=_opt_int(row.get("scheduled_ts")),
            close_ts=_opt_int(row.get("close_ts")),
            has_clob=_opt_bool(row.get("pm_has_clob")) and token is not None,
        )


def split_window(start: int, end: int) -> list[tuple[int, int]]:
    """``[start, end]`` cut into pieces the CLOB accepts (at most 15 days each)."""
    out: list[tuple[int, int]] = []
    a = start
    while a < end:
        b = min(a + MAX_PRICE_WINDOW, end)
        out.append((a, b))
        a = b
    return out


def fine_windows(
    open_ts: int | None, scheduled_ts: int, close_ts: int | None
) -> list[tuple[int, int]]:
    """``[startTs, endTs]`` windows for the 10-minute series (each at most 15 days).

    Coverage needed: ``[scheduled - 8d, min(scheduled, close)]`` for the scheduled
    anchor and ``[close - 8d, close]`` for the close anchor. When the market closed
    at or before its scheduled time the second contains the first, so one window
    serves both; this replaces the separate early-close window. Windows that end at
    or before the market opened are dropped.
    """
    if close_ts is None:
        wins = [(scheduled_ts - FINE_LOOKBACK, scheduled_ts)]
    elif close_ts <= scheduled_ts:
        wins = [(close_ts - FINE_LOOKBACK, close_ts)]
    elif close_ts - (scheduled_ts - FINE_LOOKBACK) <= MAX_PRICE_WINDOW:
        wins = [(scheduled_ts - FINE_LOOKBACK, close_ts)]
    else:
        wins = [
            (scheduled_ts - FINE_LOOKBACK, scheduled_ts),
            (close_ts - FINE_LOOKBACK, close_ts),
        ]
    return [(a, b) for a, b in wins if b > a and (open_ts is None or b > open_ts)]


def _fine_params(token_id: str, start: int, end: int) -> Params:
    return {"market": token_id, "startTs": start, "endTs": end, "fidelity": FINE_FIDELITY_MIN}


def price_requests(
    token_id: str,
    open_ts: int | None,
    scheduled_ts: int,
    close_ts: int | None,
    extra_windows: Sequence[tuple[int, int]] = (),
) -> list[Params]:
    """The ``prices-history`` requests for one contract: the daily series, the planned
    fine windows, then any extra fine windows (split into 15-day pieces)."""
    reqs: list[Params] = [{"market": token_id, "interval": "max", "fidelity": 1440}]
    for a, b in fine_windows(open_ts, scheduled_ts, close_ts):
        reqs.append(_fine_params(token_id, a, b))
    for a, b in extra_windows:
        reqs.extend(_fine_params(token_id, x, y) for x, y in split_window(int(a), int(b)))
    return reqs


def history_to_path(data: Any) -> PricePath:
    """``{"history": [{"t": unix, "p": price}, ...]}`` as a sorted ``PricePath``.

    Points without a usable timestamp are dropped; prices that are not numbers become
    NaN (the snapshot logic treats them as unusable).
    """
    hist = data.get("history") if isinstance(data, Mapping) else None
    ts: list[int] = []
    ps: list[float] = []
    for pt in hist or []:
        if not isinstance(pt, Mapping):
            continue
        t = _opt_int(pt.get("t"))
        if t is None:
            continue
        try:
            p = float(pt.get("p"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            p = math.nan
        ts.append(t)
        ps.append(p)
    return PricePath.from_points(ts, ps)


# A source answers (host, path, params) with (present, body). The live source fetches
# through the cache; the cache source never touches the network.
Source = Callable[[str, str, Params], tuple[bool, Any]]


def _live_source(clients: Mapping[str, CachedClient]) -> Source:
    def src(host: str, path: str, params: Params) -> tuple[bool, Any]:
        return True, clients[host].get_json(path, params)

    return src


def _cache_source(clients: Mapping[str, CachedClient]) -> Source:
    def src(host: str, path: str, params: Params) -> tuple[bool, Any]:
        r = clients[host].cached(path, params)
        if r is None:
            return False, None
        return True, (r.data if 200 <= r.status < 300 else None)

    return src


def _load_path(
    ref: ContractRef, src: Source, extra_windows: Sequence[tuple[int, int]] = ()
) -> tuple[PricePath, int]:
    """The merged path (daily first, fine windows after so they win ties) and the
    number of requests the source did not have."""
    if not ref.has_clob or ref.token_id is None or ref.scheduled_ts is None:
        return PricePath.merge(), 0
    paths: list[PricePath] = []
    missing = 0
    reqs = price_requests(ref.token_id, ref.open_ts, ref.scheduled_ts, ref.close_ts, extra_windows)
    for params in reqs:
        present, data = src("clob", "prices-history", params)
        if not present:
            missing += 1
            continue
        paths.append(history_to_path(data))
    return PricePath.merge(*paths), missing


def load_price_path(
    row: Row, cfg: Config, extra_windows: Sequence[tuple[int, int]] = ()
) -> PricePath:
    """A contract's merged price path (daily series + fine windows), from the cache only.

    Works with ``PMCAL_OFFLINE=1``; requests that are not cached are skipped, so the
    path is empty if nothing is cached. ``extra_windows`` adds fine windows fetched
    with ``fetch_windows``.
    """
    with clob_client(cfg) as c:
        path, _ = _load_path(ContractRef.from_row(row), _cache_source({"clob": c}), extra_windows)
    return path


def fetch_windows(row: Row, windows: Sequence[tuple[int, int]], cfg: Config) -> None:
    """Download extra 10-minute windows ``[start, end]`` for one contract into the cache
    (longer windows are split into 15-day pieces). Read them back with
    ``load_price_path(row, cfg, extra_windows=windows)``."""
    ref = ContractRef.from_row(row)
    if not ref.has_clob or ref.token_id is None:
        return
    with clob_client(cfg) as c:
        for a, b in windows:
            for x, y in split_window(int(a), int(b)):
                c.get_json("prices-history", _fine_params(ref.token_id, x, y))


@dataclass(frozen=True)
class _Trades:
    ts: IntArray  # ascending
    price: FloatArray  # on the contract's side


def parse_trades(data: Any, token_id: str | None) -> _Trades:
    """data-api trades (both tokens of the market), sorted by time, with prices mapped
    to the contract's side (1 - price for the other token)."""
    ts: list[int] = []
    ps: list[float] = []
    for tr in data if isinstance(data, list) else []:
        if not isinstance(tr, Mapping):
            continue
        t = _opt_int(tr.get("timestamp"))
        if t is None:
            continue
        try:
            p = float(tr.get("price"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            p = math.nan
        if token_id is not None and str(tr.get("asset")) != token_id and math.isfinite(p):
            p = 1.0 - p
        ts.append(t)
        ps.append(p)
    t_arr = np.asarray(ts, dtype=np.int64)
    order = np.argsort(t_arr, kind="stable")
    return _Trades(t_arr[order], np.asarray(ps, dtype=np.float64)[order])


def _trade_params(cond: str, start: int | None, end: int, limit: int) -> Params:
    return {"market": cond, "start": start, "end": end, "limit": limit, "takerOnly": True}


@dataclass(frozen=True)
class TradeStats:
    n_24h: int | None
    last_age_s: float
    last_price: float


_NO_TRADES: Final = TradeStats(None, math.nan, math.nan)


def _stats_from(
    tr: _Trades, t: int, *, floor: int, cond: str, token_id: str | None, src: Source
) -> tuple[TradeStats, int]:
    """Trade statistics at ``t`` from trades known to be complete on ``[floor, t]``.

    When none of them is at or before ``t``, one ``limit=1`` request ending at
    ``floor`` finds the previous trade.
    """
    upto = int(np.searchsorted(tr.ts, t, side="right"))
    start = int(np.searchsorted(tr.ts, t - DAY, side="right"))
    n = min(upto - start, TRADES_PAGE)
    if upto > 0:
        return TradeStats(n, float(t - tr.ts[upto - 1]), float(tr.price[upto - 1])), 0
    present, data = src("data", "trades", _trade_params(cond, None, floor, 1))
    if not present:
        return TradeStats(n, math.nan, math.nan), 1
    prev = parse_trades(data, token_id)
    if prev.ts.size == 0:
        return TradeStats(n, math.nan, math.nan), 0
    return TradeStats(n, float(t - prev.ts[-1]), float(prev.price[-1])), 0


def _trade_stats_at(ref: ContractRef, t: int, src: Source) -> tuple[TradeStats, int]:
    """Trade statistics at ``t`` from one page over ``[t - 24h, t]``.

    ``n_24h`` is capped at ``TRADES_PAGE`` (the page holds the newest trades).
    """
    if not ref.condition_id:
        return _NO_TRADES, 0
    params = _trade_params(ref.condition_id, t - DAY, t, TRADES_PAGE)
    present, data = src("data", "trades", params)
    if not present:
        return _NO_TRADES, 1
    tr = parse_trades(data, ref.token_id)
    return _stats_from(tr, t, floor=t - DAY, cond=ref.condition_id, token_id=ref.token_id, src=src)


def add_trade_stats(
    snaps: list[Snapshot], ref: ContractRef, src: Source
) -> tuple[list[Snapshot], int]:
    """Fill ``n_trades_24h``, ``last_trade_age_s`` and ``last`` on the "ok" snapshots
    whose horizon is in ``TRADE_HORIZONS``.

    ``n_trades_24h`` counts taker trades (both tokens) in ``(t - 24h, t]``. One page
    over ``[min t - 24h, max t]`` serves all snapshots when it is not full; otherwise
    each snapshot gets its own page and its count is capped at ``TRADES_PAGE``.
    Returns the snapshots and the number of requests the source did not have.
    """
    todo = [
        i
        for i, s in enumerate(snaps)
        if s.status == "ok" and (TRADE_HORIZONS is None or s.horizon in TRADE_HORIZONS)
    ]
    cond = ref.condition_id
    if not todo or not cond:
        return snaps, 0
    lo = min(snaps[i].t_snap for i in todo) - DAY
    hi = max(snaps[i].t_snap for i in todo)
    present, data = src("data", "trades", _trade_params(cond, lo, hi, TRADES_PAGE))
    if not present:
        return snaps, 1
    complete = isinstance(data, list) and len(data) < TRADES_PAGE
    merged = parse_trades(data, ref.token_id)
    out = list(snaps)
    missing = 0
    for i in todo:
        t = snaps[i].t_snap
        if complete:
            st, miss = _stats_from(merged, t, floor=lo, cond=cond, token_id=ref.token_id, src=src)
        else:
            st, miss = _trade_stats_at(ref, t, src)
        missing += miss
        out[i] = replace(
            snaps[i], n_trades_24h=st.n_24h, last_trade_age_s=st.last_age_s, last=st.last_price
        )
    return out, missing


def load_trades_window(
    row: Row, t_end: int, cfg: Config, *, fetch: bool = False
) -> tuple[int | None, float]:
    """``(n_trades_24h, last_trade_age_s)`` at ``t_end`` for one contract.

    Reads one page over ``[t_end - 24h, t_end]`` (count capped at ``TRADES_PAGE``),
    plus a ``limit=1`` request for the previous trade when the window is empty. From
    the cache only unless ``fetch`` is True. ``(None, nan)`` when not available.
    """
    ref = ContractRef.from_row(row)
    with data_client(cfg) as c:
        src = _live_source({"data": c}) if fetch else _cache_source({"data": c})
        st, _ = _trade_stats_at(ref, int(t_end), src)
    return st.n_24h, st.last_age_s


@dataclass
class _ContractResult:
    rows: list[dict[str, Any]]
    path_n: int
    missing: int


def _snapshot_rows(
    contract_id: str,
    anchor: str,
    snaps: Sequence[Snapshot] | None,
    horizons: Mapping[str, int],
) -> list[dict[str, Any]]:
    if snaps is None:  # no anchor time (close anchor of a market without closedTime)
        return [
            {
                "venue": VENUE,
                "contract_id": contract_id,
                "anchor": anchor,
                "horizon": h,
                "horizon_s": s,
                "status": "no_data",
            }
            for h, s in horizons.items()
        ]
    return [
        {"venue": VENUE, "contract_id": contract_id, "anchor": anchor, **s.as_row()} for s in snaps
    ]


def contract_snapshots(row: Row, cfg: Config, src: Source) -> _ContractResult:
    """Plan, read and turn into snapshot rows everything for one contract.

    Used by both ``fetch_prices`` (live source: fills the cache) and
    ``build_snapshots`` (cache source), so both see the same requests.
    """
    horizons = cfg.snapshots.horizon_seconds()
    ref = ContractRef.from_row(row)
    path, missing = _load_path(ref, src)
    rows: list[dict[str, Any]] = []
    for anchor, anchor_ts in (("scheduled", ref.scheduled_ts), ("close", ref.close_ts)):
        snaps: list[Snapshot] | None = None
        if anchor_ts is not None:
            snaps = take_snapshots(
                path,
                anchor_ts=anchor_ts,
                open_ts=ref.open_ts,
                close_ts=ref.close_ts,
                horizons=horizons,
                max_spread=cfg.snapshots.max_spread,
            )
            if anchor == "scheduled" and ref.has_clob:
                snaps, miss = add_trade_stats(snaps, ref, src)
                missing += miss
        rows.extend(_snapshot_rows(ref.contract_id, anchor, snaps, horizons))
    for r in rows:
        r["pm_path_n"] = int(path.ts.size)
        r["pm_cache_missing"] = missing
    return _ContractResult(rows, int(path.ts.size), missing)


def _clients(cfg: Config) -> dict[str, CachedClient]:
    return {"clob": clob_client(cfg), "data": data_client(cfg)}


def _records(df: pd.DataFrame) -> list[Mapping[str, Any]]:
    return cast(list[Mapping[str, Any]], df.to_dict("records"))


def fetch_prices(contracts: pd.DataFrame, cfg: Config, *, workers: int = 8) -> None:
    """Download price history and trade prints for each contract into the cache, and
    the ``/live-volume`` of events whose volume Gamma did not report (read back with
    ``fill_missing_volume``).

    Idempotent: cached requests cost nothing on a re-run. Contracts without an order
    book (``pm_has_clob`` False) are skipped. Rates are capped per host (CLOB 20/s,
    data-api 10/s) and shared by all workers, so 16 workers are safe. A contract
    whose requests fail after the client's retries is logged and skipped; re-running
    retries it. Logs progress with rate and ETA (every 500 contracts on big runs).
    """
    recs = _records(contracts)
    todo = [r for r in recs if _opt_bool(r.get("pm_has_clob"))]
    log.info(
        "prices: %d contracts (%d without CLOB skipped), %d workers",
        len(todo),
        len(recs) - len(todo),
        workers,
    )
    if "volume_missing" in contracts.columns:
        need = contracts["volume_missing"].fillna(False).astype(bool)
        if bool((need & contracts["volume_usd"].isna()).any()):
            fill_missing_volume(contracts, cfg, fetch=True, workers=min(workers, 8))
    clients = _clients(cfg)
    src = _live_source(clients)
    every = 500 if len(todo) >= 5000 else max(25, len(todo) // 20 or 1)
    t0 = time.monotonic()
    failed = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = [pool.submit(contract_snapshots, r, cfg, src) for r in todo]
            for k, fut in enumerate(as_completed(futs), 1):
                try:
                    fut.result()
                except Exception as e:  # keep going; the contract is retried on re-run
                    failed += 1
                    log.warning("prices: contract failed: %s", e)
                if k % every == 0 or k == len(todo):
                    el = time.monotonic() - t0
                    rate = k / el if el > 0 else math.inf
                    eta = (len(todo) - k) / rate if rate > 0 else math.nan
                    log.info(
                        "prices: %d/%d contracts, %.2f/s, ETA %.1f min; clob %s; data %s",
                        k,
                        len(todo),
                        rate,
                        eta / 60,
                        _fmt_stats(clients["clob"]),
                        _fmt_stats(clients["data"]),
                    )
    finally:
        for c in clients.values():
            c.close()
    if failed:
        log.warning("prices: %d contracts failed; re-run fetch_prices to retry them", failed)


def _fmt_stats(c: CachedClient) -> str:
    s = c.stats
    return f"{s['fetches']} fetched/{s['hits']} cached/{s['retries']} retries/{s['throttled']} 429"


def build_snapshots(contracts: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Snapshot rows for every contract x anchor ("scheduled", "close") x horizon, from
    the cache only (works with ``PMCAL_OFFLINE=1``).

    The "close" anchor is the actual close: outcome-dependent for markets that
    resolved early, kept to show that artifact. A contract without price data gets
    "no_data" rows, except where the snapshot time is outside the market's life,
    which is "not_open" or "closed" as for any contract. The close anchor of a market
    without a closedTime is "no_data" with a null ``t_snap``. ``pm_cache_missing``
    counts planned requests that ``fetch_prices`` has not made.
    """
    clients = _clients(cfg)
    src = _cache_source(clients)
    rows: list[dict[str, Any]] = []
    missing = 0
    try:
        for rec in _records(contracts):
            res = contract_snapshots(rec, cfg, src)
            rows.extend(res.rows)
            missing += res.missing
    finally:
        for c in clients.values():
            c.close()
    if missing:
        log.warning("snapshots: %d planned requests not in the cache (run fetch_prices)", missing)
    df = pd.DataFrame.from_records(rows) if rows else pd.DataFrame()
    return coerce(df, {**SNAPSHOT_COLUMNS, **PM_SNAPSHOT_COLUMNS})
