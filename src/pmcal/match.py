"""Pairs of contracts, one on each venue, that are the same proposition with the same
resolution, so the two venues' prices for the same thing at the same moment can be
compared.

Outputs: ``interim/matches_all.parquet`` (every pair), ``interim/matches.parquet`` (the
core pairs: every non-sports pair and at most ``sample.max_events_per_series_month``
games per league and month, by seeded hash; ``pmcal sample`` forces their events into
the samples, and forcing every matched game would more than triple the downloads) and
``processed/matched_snapshots.parquet`` (the core pairs priced from the cache).

Why it is built this way
------------------------
A pair that is not really the same proposition is worse than no pair: it shows up as a
price gap between the venues and as a forecasting difference. So every family is
matched on structure first (league and start time, meeting, release month, station),
then on names, and anything ambiguous is dropped. Outcomes are never used to match;
they are only used afterwards to check (``outcome_agree``).

Candidates are blocked cheaply: by sport and start time for games, by meeting or
reference month for macro releases, by scheduled time and shared outcome names for
multi-outcome events. Only frame rows (``pmcal.sample.frame_mask``) are matched, and
each contract is in at most one pair (the higher score wins; ties, such as an event
Polymarket listed twice, go to the later listing, then a seeded hash). Nothing in the
matching uses volume: lifetime volume includes trading after the snapshot and selects
on the outcome.

Families
--------
``sports_moneyline``. Polymarket game markets are "A vs. B" binaries priced on one
side (chosen by hash, ``outcome_label``) or, for soccer and Test cricket, three Yes/No
markets (A, draw, B); Kalshi has one market per side in ``KX...GAME``/``MATCH``/
``FIGHT`` series, plus "Tie" in three-way sports. Games pair when the sport (and, for
team sports, women's vs men's) is the same, the starts are within ``GAME_TOL_S`` (12
hours for tennis, MMA and other individual sports, where Polymarket times the card or
session and Kalshi the bout), the structure is the same (a two-way market is never
paired with a three-way one: "not A" is not "B" when a draw is possible) and both
names match (``name_similarity``; US major leagues through ``US_TEAMS``, since Kalshi
writes "Denver" where Polymarket writes "Nuggets"). The orientation must be clear, so
"Eastern Michigan vs Western Michigan" cannot be read backwards. The Polymarket side is
paired with Kalshi's market for the same side; in a two-way game whose same-side market
is not in Kalshi's frame, the other side's market is used as the complement.

``sports_total``, ``sports_spread``. Full-game lines of matched games with exactly the
same line: Polymarket "O/U x" (Over or Under priced) with Kalshi "Over x", Polymarket
"Spread: T (-x)" with Kalshi "T wins by over x". Both venues count overtime and extra
innings (NHL shootouts count one goal on Kalshi, as in the official score) and soccer's
90 minutes plus stoppage time. Kalshi's partial-game series (1H, quarters, team
totals) carry other event keys and are never paired, and a line market whose own start
is not within ``GAME_TOL_S`` of Kalshi's (a rescheduled game listed twice) is skipped.

``fed_decision``. ``KXFEDDECISION`` and Polymarket's "Fed Decision in <Month>?" for the
same FOMC meeting: cut 25, cut >25 (Polymarket "50+ bps"), hold, hike 25, hike >25.
Both count moves in multiples of 25 bps (Polymarket rounds others up), so ">25" and
"50+" are the same set.

``cpi``, ``unemployment``. Polymarket's brackets are single values ("0.4%") with open
ends ("0.3% or less", "≥1.1%"); Kalshi's are thresholds ("Above 0.3%"). Both resolve
on the one-decimal BLS figure, so only the open ends have a Kalshi twin: "≥x" is
"above x-0.1" and "≤x" is the complement of "above x". Same statistic (headline or
core, MoM or YoY, U-3) and reference month.

``election``, ``awards``, ``golf_winner``, ``golf_top_n``, ``golf_make_cut``.
Multi-outcome events on both venues: scheduled within ``CATEGORICAL_TOL_S``, the same
category, the same title qualifiers (round, place, party primary, office, award
category, nominations vs win, "top 5" vs "top 10": ``title_qualifiers`` and the numbers
in the title), most content words of each title found in the other and no unexplained
words on both sides (``title_coverage``, ``titles_conflict``), the same mutual
exclusivity, and at least two outcomes, and 60% of the smaller event's, naming the same
people. Outcomes are then paired by name; a name matching two outcomes on the other
venue is dropped. Not paired: events whose outcomes are numeric brackets (margins,
turnout, seats), whose edges are written differently, and contests within a sub-field
("Top Past Champion"), where Kalshi splits ties.

Evaluated and excluded (``EXCLUDED_FAMILIES``):

- Weather. Polymarket resolves on Wunderground (NOAA's hourly series from late August
  2026), Kalshi on the NWS daily climate report. Most cities also use different
  stations; at the three shared airports (LAX, MIA, AUS) identical brackets agreed on
  only 82% of 1,193 resolved days, so these are different propositions.
- Other multi-outcome events (``other_categorical``): mostly generic outcome names
  ("Round of 32") shared by unrelated events.
- Not attempted after reading the rules: equity indices (Polymarket prices SPY on a Pyth
  one-minute close, Kalshi the S&P 500 index), WTI (Pyth close vs CME settlement, and
  "above $97" vs "above 96.99"), daily crypto (Kalshi's daily strikes live in hourly
  series outside its frame), payrolls and GDP (bracket edges differ: "at least 250k" vs
  "above 250k"; GDP ties go to the higher bracket on Polymarket), and "mentions"
  (word-variant rules differ).

Manual check
------------
Two rounds of reading pairs by hand, both venues' titles and rules text side by side:
169 pairs drawn across all families, then 116 more from the leagues and families the
first draw missed (esports, cricket, ITF/WTA/Challenger tennis, more soccer leagues,
unemployment, make-the-cut), plus every multi-outcome event pairing at event level
(about 180). Every error found was a class of error and was fixed by a rule, not by
dropping the pair: "Eastern Illinois" read as "Western Illinois" (distinguishing words
are now exact), "Kentucky" inside "Eastern Kentucky Colonels" (and inside "Texas A&M",
"Washington State"), an exact-score market "X to win 2-0?" taken for a moneyline,
Critics Choice "Best Stunt Design" vs "Best Cinematography", Oscars vs BAFTA and
nominations vs wins, F1 "fastest lap" vs race winner, "4th place" vs "first round
winner", golf "Top Past Champion" vs the tournament winner, Polymarket's "Which parties
enter parliament?" vs a winner market, election-margin brackets with different edges.
Of the 285 reviewed pairs one was wrong, in a class now excluded; none of the other 284
is, which puts the error rate below about 1% (95% upper bound) across the reviewed
families together, and no family had a wrong pair left.

Outcome agreement is the second check: a pair that is not the same proposition agrees
only by chance (about half the time for a two-way market). On resolved pairs it is
100% for awards, golf, Fed, CPI and unemployment, 99.99% for moneylines (3
disagreements in 67,000), 99.98% for totals and spreads, and 98% for elections. The
disagreements found are differences in resolution rather than in proposition: a tie
in seats in Baden-Württemberg broken differently, one Ligue 1 playoff and one club
friendly whose totals and spreads were settled on different scores, one ITF match.
The one structural risk left is esports: the same two teams can meet in two
tournaments on one day (one CS2 pair, Exort Fiesta vs CCT Europe, two hours apart).

Keys and directions
-------------------
``same_direction`` is True when Polymarket YES is Kalshi YES and False when one is the
complement of the other (Polymarket priced on "Under" or on the underdog of a spread,
a closed-low macro bracket, a two-way game paired through the other side). Kalshi
prices and outcomes are mapped to the Polymarket direction (``map_to_polymarket``).
``event_key`` is the real-world event used as the bootstrap cluster: the game for
sports (``MLB-26APR011215ATHATL``: Kalshi's event ticker without the market type, so a
game's moneyline, totals and spreads share it), the meeting or release for macro, the
Kalshi event for multi-outcome events.

Snapshots
---------
Both venues are priced at the same instants, ``t_snap = anchor - horizon`` with
``anchor = min(scheduled_ts)`` of the two contracts (they differ by minutes for games,
and by up to a few days for tournaments). Each venue's snapshot follows
``pmcal.snapshots.snapshot_at`` with its own open and close and is read from the cache
only. A snapshot is live if it is a tight quoted mid (Kalshi), the contract traded in the
24 hours before it, or its price moved in those 24 hours. Polymarket's price series is
the book midpoint, which in thin long-shot books can jump for a few hours (seen on golf
outrights), so the analysis should still look at spreads or trades there. Polymarket's
trade statistics come from the cache only where the common anchor is Polymarket's own
scheduled time (the windows its pipeline fetched); elsewhere liveness rests on moves.
"""

from __future__ import annotations

import difflib
import functools
import hashlib
import json
import logging
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from pmcal.config import Config
from pmcal.sample import frame_mask
from pmcal.snapshots import PricePath, Snapshot, snapshot_at

log = logging.getLogger(__name__)

HOUR: Final = 3600
DAY: Final = 86400
GAME_TOL_S: Final = 3 * HOUR  # game start times on the two venues
# Individual sports: Polymarket times a UFC fight at the start of the card and a tennis
# match at the start of the session, Kalshi at the expected start of the bout or match.
INDIVIDUAL_TOL_S: Final = 12 * HOUR
EVENT_TOL_S: Final = 36 * HOUR  # scheduled times of non-sports events
CATEGORICAL_TOL_S: Final = 7 * DAY
MATCHES_FILE: Final = "matches.parquet"  # the core pairs (``mark_core``)
ALL_MATCHES_FILE: Final = "matches_all.parquet"  # every pair, with the ``core`` flag
SNAPSHOTS_FILE: Final = "matched_snapshots.parquet"
# Families dropped after the manual check (see the module docstring).
EXCLUDED_FAMILIES: Final[frozenset[str]] = frozenset(
    {"other_categorical", "weather_high", "weather_low"}
)


# -- small helpers ---------------------------------------------------------------------


def _str(x: Any) -> str:
    return "" if x is None or x is pd.NA or (isinstance(x, float) and math.isnan(x)) else str(x)


def _opt_int(x: Any) -> int | None:
    if x is None or x is pd.NA or (isinstance(x, float) and math.isnan(x)):
        return None
    return int(x)


def _opt_bool(x: Any) -> bool | None:
    if x is None or x is pd.NA or (isinstance(x, float) and math.isnan(x)):
        return None
    return bool(x)


def _json_list(s: Any) -> list[str]:
    try:
        out = json.loads(str(s)) if s is not None and s is not pd.NA else []
    except ValueError:
        return []
    return [str(x) for x in out] if isinstance(out, list) else []


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], df.to_dict("records"))


def _at(df: pd.DataFrame, col: str, i: int) -> Any:
    """One cell by row position (index columns mix strings, numbers and NA)."""
    return df[col].iat[i]


def _col(df: pd.DataFrame, col: str) -> pd.Series:
    """A column as strings with nulls as "", so masks built on it have no NA."""
    return df[col].astype("string").fillna("")


def _rows_where(mask: Any) -> list[int]:
    """Row positions where a boolean mask is true."""
    return [int(i) for i in np.flatnonzero(np.asarray(mask, dtype=bool))]


def _groups(df: pd.DataFrame, key: str) -> dict[Any, list[dict[str, Any]]]:
    """Rows of ``df`` as dicts (with their position under "_pos"), grouped by ``key``."""
    out: dict[Any, list[dict[str, Any]]] = {}
    for pos, rec in zip(df.index, _records(df), strict=True):
        rec["_pos"] = int(pos)
        out.setdefault(rec[key], []).append(rec)
    return out


def within(a: float | int | None, b: float | int | None, tol_s: float) -> bool:
    """Whether two timestamps are both known and at most ``tol_s`` apart."""
    if a is None or b is None:
        return False
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if math.isnan(fa) or math.isnan(fb):
        return False
    return abs(fa - fb) <= tol_s


def map_to_polymarket(x: float, same_direction: bool) -> float:
    """A Kalshi price or outcome expressed in the Polymarket contract's direction."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return math.nan
    return float(x) if same_direction else 1.0 - float(x)


# -- pairs -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """A matched pair, by row position in each venue's frame."""

    pm_pos: int
    k_pos: int
    same_direction: bool
    family: str
    event_key: str
    score: float


def _tie_rank(p: Pair, pm_ids: Sequence[Any], k_ids: Sequence[Any]) -> int:
    """A seeded hash of the pair's contracts: breaks ties without looking at volume or
    anything else that is known only after the snapshot."""
    key = f"pmcal-match:{pm_ids[p.pm_pos]}|{k_ids[p.k_pos]}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def _best_unique(pairs: list[Pair], pm: pd.DataFrame, k: pd.DataFrame) -> list[Pair]:
    """At most one pair per contract on each venue: the higher score, then the later
    Polymarket listing, then a seeded hash.

    Polymarket re-lists an event when a listing goes wrong (the September 2026 Fed
    decision was listed twice and the first listing settled in May), so the later
    listing wins a tie. The listing time is fixed before any snapshot; volume is not
    used, since lifetime volume includes trading after the snapshot and selects on the
    outcome."""
    pm_ids = pm["contract_id"].tolist()
    k_ids = k["contract_id"].tolist()
    opened = pm["open_ts"].astype("float64").fillna(0.0).tolist()
    ranked = sorted(pairs, key=lambda p: (-p.score, -opened[p.pm_pos], _tie_rank(p, pm_ids, k_ids)))
    seen_p: set[int] = set()
    seen_k: set[int] = set()
    out: list[Pair] = []
    for p in ranked:
        if p.pm_pos in seen_p or p.k_pos in seen_k:
            continue
        seen_p.add(p.pm_pos)
        seen_k.add(p.k_pos)
        out.append(p)
    return out


def _within_pos(pm: pd.DataFrame, k: pd.DataFrame, i: int, j: int, tol: float) -> bool:
    return within(_at(pm, "scheduled_ts", i), _at(k, "scheduled_ts", j), tol)


# -- names -----------------------------------------------------------------------------

_TRANSLIT: Final = str.maketrans(
    {
        "ł": "l",
        "Ł": "l",
        "ø": "o",
        "Ø": "o",
        "æ": "ae",
        "Æ": "ae",
        "ß": "ss",
        "đ": "d",
        "Đ": "d",
        "\u0131": "i",
        "œ": "oe",
        "Œ": "oe",
        "þ": "th",
        "ð": "d",
    }
)


def fold(s: Any) -> str:
    """Lower-case ASCII with accents removed, apostrophes and full stops dropped and any
    other punctuation turned into spaces: "Atlético Nott'm St. (FL)" -> "atletico nottm st fl"."""
    if s is None or (isinstance(s, float) and math.isnan(s)) or s is pd.NA:
        return ""
    t = unicodedata.normalize("NFKD", str(s).translate(_TRANSLIT))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = t.replace("&", " and ")
    t = re.sub("['\u2019`\u00b4.]", "", t)  # apostrophes, backticks, acute accents, dots
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


# Words that carry no identity in team and player names.
NAME_STOPWORDS: Final = _words(
    """fc cf sc afc cd club the de del da do di la le el los las and of esports esport
    gaming team ac sk fk bk ik if ud sd cp ca vs"""
)
# Words that distinguish otherwise identical names (reserve, youth and women's sides).
# Two names only match if they carry the same set of these.
NAME_QUALIFIERS: Final = _words(
    """b ii iii u16 u17 u18 u19 u20 u21 u22 u23 jong reserves reserve academy youth women
    womens w wfc ladies castilla atletic"""
)
# Words that name a different school or club when added to a name ("Eastern Kentucky"
# is not "Kentucky", "Texas Tech" is not "Texas"). They never match fuzzily, and a
# name that has one the other lacks does not match it.
NAME_DISTINGUISHING: Final = _words(
    """north south east west northern southern eastern western central northeast northwest
    southeast southwest new state tech coastal atlantic pacific international christian
    baptist methodist wesleyan poly aandm aandt upstate"""
)
_NAME_ALIASES: Final[dict[str, str]] = {
    "se": "southeast",
    "sw": "southwest",
    "ne": "northeast",
    "nw": "northwest",
    "utd": "united",
    "intl": "international",
    "univ": "university",
    "mt": "mount",
    "ft": "fort",
    "nottm": "nottingham",
    "wolves": "wolverhampton",
    "spurs": "tottenham",
    "under": "u",  # "Under-19s" -> "u 19s" is joined below
}


def name_tokens(s: Any) -> tuple[str, ...]:
    """Tokens of a team or player name, normalised for comparison.

    "St." becomes "state" at the end of a name ("Ohio St.") and "saint" elsewhere
    ("St. Louis"); "Under-19s"/"U19" become "u19"; generic words (FC, Club, Esports...)
    are dropped. The order is kept (it matters for nothing but readability).
    """
    raw = re.sub(r"\b([a-z]) and ([a-z])\b", r"\1and\2", fold(s)).split()  # "A&M" -> "aandm"
    out: list[str] = []
    for i, t in enumerate(raw):
        t = _NAME_ALIASES.get(t, t)
        if t == "st":
            t = "state" if i == len(raw) - 1 and i > 0 else "saint"
        elif t == "ste":
            t = "sainte"
        if out and out[-1] == "u" and re.fullmatch(r"\d\ds?", t):
            out[-1] = "u" + t.rstrip("s")
            continue
        if re.fullmatch(r"u\d\ds", t):
            t = t[:-1]
        out.append(t)
    return tuple(t for t in out if t not in NAME_STOPWORDS)


def _token_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if a in NAME_DISTINGUISHING or b in NAME_DISTINGUISHING:
        return 0.0
    if len(a) >= 3 and len(b) >= 3 and (a.startswith(b) or b.startswith(a)):
        return 0.9
    if (len(a) == 1 and b.startswith(a)) or (len(b) == 1 and a.startswith(b)):
        return 0.6  # an initial
    if a.isdigit() or b.isdigit():
        return 0.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    return r if r >= 0.85 else 0.0


@functools.lru_cache(maxsize=200_000)
def name_similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] of two team or player names.

    1.0 when the normalised names are equal or one is the other's acronym ("PSG");
    otherwise each token of the shorter name is aligned with its best match in the
    longer one (exact 1, prefix 0.9, initial 0.6, near-identical spelling 0.85+), so a
    name contained in the other ("Arsenal" in "Arsenal FC") scores 1 less a small
    penalty for the unmatched tokens. Names whose qualifiers differ ("U21", "Women",
    "II", "Academy") score 0, and so do names where one has a distinguishing word the
    other lacks ("Kentucky" and "Eastern Kentucky Colonels", "Texas" and "Texas Tech").
    """
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb:
        return 0.0
    qa = {t for t in ta if t in NAME_QUALIFIERS}
    qb = {t for t in tb if t in NAME_QUALIFIERS}
    if qa != qb:
        return 0.0
    ta = tuple(t for t in ta if t not in NAME_QUALIFIERS) or ta
    tb = tuple(t for t in tb if t not in NAME_QUALIFIERS) or tb
    if ta == tb or "".join(ta) == "".join(tb):
        return 1.0
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    acronym = len(short) == 1 and len(short[0]) >= 2 and len(long_) >= 2
    if acronym and short[0] == "".join(t[0] for t in long_):
        return 0.95
    if {t for t in ta if t in NAME_DISTINGUISHING} != {t for t in tb if t in NAME_DISTINGUISHING}:
        return 0.0
    joined_s, joined_l = sorted(("".join(ta), "".join(tb)), key=len)
    if len(joined_s) >= 5 and joined_s in joined_l and len(joined_s) >= 0.6 * len(joined_l):
        # "The MongolZ" / "TheMongolz", "PortlandFire" / "Portland Fire"
        return 0.9
    used: set[int] = set()
    total = 0.0
    for t in short:
        best, best_j = 0.0, -1
        for j, u in enumerate(long_):
            if j in used:
                continue
            s = _token_sim(t, u)
            if s > best:
                best, best_j = s, j
        if best_j >= 0:
            used.add(best_j)
        total += best
    score = total / len(short)
    return score * (0.92 + 0.08 * len(short) / len(long_))


# -- US major leagues ------------------------------------------------------------------

_Team = tuple[tuple[str, ...], str, str, tuple[str, ...]]
US_TEAMS: Final[dict[str, tuple[_Team, ...]]] = {
    "nba": (
        (("ATL",), "atlanta", "hawks", ()),
        (("BKN", "BRK"), "brooklyn", "nets", ()),
        (("BOS",), "boston", "celtics", ()),
        (("CHA",), "charlotte", "hornets", ()),
        (("CHI",), "chicago", "bulls", ()),
        (("CLE",), "cleveland", "cavaliers", ("cavs",)),
        (("DAL",), "dallas", "mavericks", ("mavs",)),
        (("DEN",), "denver", "nuggets", ()),
        (("DET",), "detroit", "pistons", ()),
        (("GSW", "GS"), "golden state", "warriors", ()),
        (("HOU",), "houston", "rockets", ()),
        (("IND",), "indiana", "pacers", ()),
        (("LAC",), "los angeles c", "clippers", ("la clippers",)),
        (("LAL",), "los angeles l", "lakers", ("la lakers",)),
        (("MEM",), "memphis", "grizzlies", ()),
        (("MIA",), "miami", "heat", ()),
        (("MIL",), "milwaukee", "bucks", ()),
        (("MIN",), "minnesota", "timberwolves", ()),
        (("NOP", "NO"), "new orleans", "pelicans", ()),
        (("NYK", "NY"), "new york", "knicks", ()),
        (("OKC",), "oklahoma city", "thunder", ()),
        (("ORL",), "orlando", "magic", ()),
        (("PHI",), "philadelphia", "76ers", ("sixers",)),
        (("PHX", "PHO"), "phoenix", "suns", ()),
        (("POR",), "portland", "trail blazers", ("blazers",)),
        (("SAC",), "sacramento", "kings", ()),
        (("SAS", "SA"), "san antonio", "spurs", ()),
        (("TOR",), "toronto", "raptors", ()),
        (("UTA", "UTAH"), "utah", "jazz", ()),
        (("WAS", "WSH"), "washington", "wizards", ()),
    ),
    "wnba": (
        (("ATL",), "atlanta", "dream", ()),
        (("CHI",), "chicago", "sky", ()),
        (("CON", "CONN"), "connecticut", "sun", ()),
        (("DAL",), "dallas", "wings", ()),
        (("GSV", "GS"), "golden state", "valkyries", ()),
        (("IND",), "indiana", "fever", ()),
        (("LV", "LVA"), "las vegas", "aces", ()),
        (("LA", "LAS"), "los angeles", "sparks", ()),
        (("MIN",), "minnesota", "lynx", ()),
        (("NY", "NYL"), "new york", "liberty", ()),
        (("PHX", "PHO"), "phoenix", "mercury", ()),
        (("POR",), "portland", "fire", ()),
        (("SEA",), "seattle", "storm", ()),
        (("TOR",), "toronto", "tempo", ()),
        (("WAS", "WSH"), "washington", "mystics", ()),
    ),
    "nfl": (
        (("ARI",), "arizona", "cardinals", ()),
        (("ATL",), "atlanta", "falcons", ()),
        (("BAL",), "baltimore", "ravens", ()),
        (("BUF",), "buffalo", "bills", ()),
        (("CAR",), "carolina", "panthers", ()),
        (("CHI",), "chicago", "bears", ()),
        (("CIN",), "cincinnati", "bengals", ()),
        (("CLE",), "cleveland", "browns", ()),
        (("DAL",), "dallas", "cowboys", ()),
        (("DEN",), "denver", "broncos", ()),
        (("DET",), "detroit", "lions", ()),
        (("GB",), "green bay", "packers", ()),
        (("HOU",), "houston", "texans", ()),
        (("IND",), "indianapolis", "colts", ()),
        (("JAC", "JAX"), "jacksonville", "jaguars", ()),
        (("KC",), "kansas city", "chiefs", ()),
        (("LAC",), "los angeles c", "chargers", ("la chargers",)),
        (("LAR", "LA"), "los angeles r", "rams", ("la rams",)),
        (("LV",), "las vegas", "raiders", ()),
        (("MIA",), "miami", "dolphins", ()),
        (("MIN",), "minnesota", "vikings", ()),
        (("NE",), "new england", "patriots", ()),
        (("NO",), "new orleans", "saints", ()),
        (("NYG",), "new york g", "giants", ("ny giants",)),
        (("NYJ",), "new york j", "jets", ("ny jets",)),
        (("PHI",), "philadelphia", "eagles", ()),
        (("PIT",), "pittsburgh", "steelers", ()),
        (("SEA",), "seattle", "seahawks", ()),
        (("SF",), "san francisco", "49ers", ("niners",)),
        (("TB",), "tampa bay", "buccaneers", ("bucs",)),
        (("TEN",), "tennessee", "titans", ()),
        (("WAS", "WSH"), "washington", "commanders", ()),
    ),
    "nhl": (
        (("ANA",), "anaheim", "ducks", ()),
        (("BOS",), "boston", "bruins", ()),
        (("BUF",), "buffalo", "sabres", ()),
        (("CAR",), "carolina", "hurricanes", ()),
        (("CBJ",), "columbus", "blue jackets", ()),
        (("CGY",), "calgary", "flames", ()),
        (("CHI",), "chicago", "blackhawks", ()),
        (("COL",), "colorado", "avalanche", ()),
        (("DAL",), "dallas", "stars", ()),
        (("DET",), "detroit", "red wings", ()),
        (("EDM",), "edmonton", "oilers", ()),
        (("FLA",), "florida", "panthers", ()),
        (("LA", "LAK"), "los angeles", "kings", ()),
        (("MIN",), "minnesota", "wild", ()),
        (("MTL",), "montreal", "canadiens", ()),
        (("NJ", "NJD"), "new jersey", "devils", ()),
        (("NSH",), "nashville", "predators", ()),
        (("NYI",), "new york i", "islanders", ("ny islanders",)),
        (("NYR",), "new york r", "rangers", ("ny rangers",)),
        (("OTT",), "ottawa", "senators", ()),
        (("PHI",), "philadelphia", "flyers", ()),
        (("PIT",), "pittsburgh", "penguins", ()),
        (("SEA",), "seattle", "kraken", ()),
        (("SJ", "SJS"), "san jose", "sharks", ()),
        (("STL",), "saint louis", "blues", ()),
        (("TB", "TBL"), "tampa bay", "lightning", ()),
        (("TOR",), "toronto", "maple leafs", ()),
        (("UTA", "UTAH"), "utah", "mammoth", ("utah hockey club",)),
        (("VAN",), "vancouver", "canucks", ()),
        (("VGK",), "vegas", "golden knights", ()),
        (("WPG",), "winnipeg", "jets", ()),
        (("WSH", "WAS"), "washington", "capitals", ()),
    ),
    "mlb": (
        (("ATH", "OAK"), "athletics", "athletics", ("as", "oakland athletics")),
        (("ATL",), "atlanta", "braves", ()),
        (("AZ", "ARI"), "arizona", "diamondbacks", ("dbacks",)),
        (("BAL",), "baltimore", "orioles", ()),
        (("BOS",), "boston", "red sox", ()),
        (("CHC",), "chicago c", "cubs", ()),
        (("CIN",), "cincinnati", "reds", ()),
        (("CLE",), "cleveland", "guardians", ()),
        (("COL",), "colorado", "rockies", ()),
        (("CWS", "CHW"), "chicago ws", "white sox", ()),
        (("DET",), "detroit", "tigers", ()),
        (("HOU",), "houston", "astros", ()),
        (("KC",), "kansas city", "royals", ()),
        (("LAA",), "los angeles a", "angels", ()),
        (("LAD",), "los angeles d", "dodgers", ()),
        (("MIA",), "miami", "marlins", ()),
        (("MIL",), "milwaukee", "brewers", ()),
        (("MIN",), "minnesota", "twins", ()),
        (("NYM",), "new york m", "mets", ()),
        (("NYY",), "new york y", "yankees", ()),
        (("PHI",), "philadelphia", "phillies", ()),
        (("PIT",), "pittsburgh", "pirates", ()),
        (("SD",), "san diego", "padres", ()),
        (("SEA",), "seattle", "mariners", ()),
        (("SF",), "san francisco", "giants", ()),
        (("STL",), "saint louis", "cardinals", ()),
        (("TB",), "tampa bay", "rays", ()),
        (("TEX",), "texas", "rangers", ()),
        (("TOR",), "toronto", "blue jays", ()),
        (("WSH", "WAS"), "washington", "nationals", ()),
        (("AL",), "american league", "american league", ("al",)),
        (("NL",), "national league", "national league", ("nl",)),
    ),
}


def _phrase_in(phrase: str, text: str) -> bool:
    return f" {phrase} " in f" {text} "


def us_team(league: str, name: str, code: str | None = None) -> str | None:
    """The team (its nickname) a name or Kalshi code refers to in a US major league,
    or None when it is not exactly one team.

    The code is tried first. Then nicknames and aliases contained in the name, then
    the city: an exact city match, or a city followed by the initial(s) of the
    nickname as Kalshi writes shared cities ("Los Angeles C", "Chicago WS",
    "New York Y").
    """
    teams = US_TEAMS.get(league)
    if not teams:
        return None
    if code:
        hits = [t[2] for t in teams if code.upper() in t[0]]
        if len(hits) == 1:
            return hits[0]
    text = fold(name)
    text = re.sub(r"\bst\b", "saint", text)
    if not text:
        return None
    hits = [t[2] for t in teams if _phrase_in(t[2], text) or any(_phrase_in(a, text) for a in t[3])]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # "New York Rangers" in the NHL contains "rangers" only once, but MLB's
        # "Texas Rangers" vs NHL is never mixed; several hits mean an ambiguous name.
        longest = max(hits, key=len)
        return longest if sum(1 for h in hits if h in longest) == len(hits) else None
    hits = [t[2] for t in teams if text == t[1]]
    if len(hits) == 1:
        return hits[0]
    # "BOS Bruins" style: code + nickname (handled above), or city + initials
    for t in teams:
        city = t[1]
        if text.startswith(city + " "):
            rest = text[len(city) + 1 :].replace(" ", "")
            initials = "".join(w[0] for w in t[2].split())
            if rest and (initials.startswith(rest) or t[2].replace(" ", "").startswith(rest)):
                hits.append(t[2])
    return hits[0] if len(hits) == 1 else None


# -- sports: sport, league and matchup -------------------------------------------------

# Polymarket event tags -> sport, first match wins (esports before the generic tags).
_PM_SPORT_TAGS: Final[tuple[tuple[str, frozenset[str]], ...]] = (
    ("esports_cs2", frozenset({"counter-strike-2", "counter-strike", "cs2", "csgo"})),
    ("esports_lol", frozenset({"league-of-legends"})),
    ("esports_dota2", frozenset({"dota-2", "dota2"})),
    ("esports_valorant", frozenset({"valorant"})),
    ("esports_cod", frozenset({"call-of-duty"})),
    ("esports_r6", frozenset({"rainbow-six-siege"})),
    ("esports_ow", frozenset({"overwatch"})),
    ("esports_rl", frozenset({"rocket-league"})),
    ("table_tennis", frozenset({"table-tennis", "ping-pong"})),
    ("tennis", frozenset({"tennis", "atp", "wta", "itf"})),
    ("mma", frozenset({"ufc", "mma", "pfl"})),
    ("boxing", frozenset({"boxing"})),
    ("cricket", frozenset({"cricket"})),
    ("baseball", frozenset({"baseball", "mlb", "kbo", "npb"})),
    ("hockey", frozenset({"hockey", "nhl", "khl", "ahl"})),
    ("american_football", frozenset({"nfl", "cfb", "cfl", "ufl", "college-football"})),
    ("basketball", frozenset({"basketball", "nba", "wnba", "ncaa-basketball", "cwbb"})),
    ("rugby", frozenset({"rugby", "nrl"})),
    ("golf", frozenset({"golf", "pga"})),
    ("chess", frozenset({"chess"})),
    ("darts", frozenset({"darts"})),
    ("volleyball", frozenset({"volleyball"})),
    ("soccer", frozenset({"soccer"})),
)
# Kalshi series tags -> sport (esports are split by title below).
_K_SPORT_TAGS: Final[dict[str, str]] = {
    "Soccer": "soccer",
    "Basketball": "basketball",
    "Football": "american_football",
    "Baseball": "baseball",
    "Tennis": "tennis",
    "Golf": "golf",
    "Hockey": "hockey",
    "Cricket": "cricket",
    "MMA": "mma",
    "UFC": "mma",
    "Boxing": "boxing",
    "Rugby": "rugby",
    "Table Tennis": "table_tennis",
    "Chess": "chess",
    "Darts": "darts",
    "Volleyball": "volleyball",
    "Aussie Rules": "aussie_rules",
}
_ESPORTS_TITLE: Final[tuple[tuple[str, str], ...]] = (
    ("esports_cs2", r"counter.?strike|\bcs2\b|\bcsgo\b|KXCS"),
    ("esports_lol", r"league of legends|\blol\b|KXLOL"),
    ("esports_dota2", r"dota"),
    ("esports_valorant", r"valorant"),
    ("esports_cod", r"call of duty|KXCOD"),
    ("esports_r6", r"rainbow|\br6\b|KXR6"),
    ("esports_ow", r"overwatch|KXOW"),
    ("esports_rl", r"rocket league"),
)
# Fallback when the series list is not cached: sport from the series ticker.
_K_SPORT_ID: Final[tuple[tuple[str, str], ...]] = (
    ("esports_cs2", r"CS2|CSGO"),
    ("esports_lol", r"^KXLOL"),
    ("esports_dota2", r"DOTA"),
    ("esports_valorant", r"VALORANT"),
    ("esports_cod", r"^KXCOD"),
    ("esports_r6", r"^KXR6"),
    ("table_tennis", r"^KXTT|WTT"),
    ("tennis", r"ATP|WTA|ITF|DAVISCUP|UNITEDCUP|LAVERCUP"),
    ("mma", r"UFC|MMA|PFL"),
    ("boxing", r"BOX"),
    ("cricket", r"IPL|T20|ODI|TESTMATCH|CPL|HUNDRED|PSL|CRICKET"),
    ("baseball", r"MLB|NPB|KBO|WBC|NCAABB"),
    ("hockey", r"NHL|KHL|AHL|SHL|HOCKEY"),
    ("american_football", r"NFL|NCAAF|CFL|UFL"),
    ("basketball", r"NBA|NCAAMB|NCAAWB|EUROLEAGUE|EUROCUP|CBAGAME|NBL|JBLEAGUE|KBL|ACB|FIBA"),
)
# Sports whose games are contested by individuals: names are personal, so the
# women's/men's distinction (which Polymarket does not always tag) is not needed.
INDIVIDUAL_SPORTS: Final = frozenset({"tennis", "table_tennis", "mma", "boxing", "chess", "darts"})
_WOMEN_RE: Final = re.compile(
    r"women|womens|femenil|femminile|feminin|frauen|ladies|\bwnba\b|\bwta\b|\bcwbb\b|\bnwsl\b|"
    r"\bwsl\b|\bwwc\b|u20wwc|\(w\)|\bliga f\b|\bncaaw",
    re.IGNORECASE,
)


def pm_sport(tags: str | None) -> str | None:
    tagset = frozenset(str(tags or "").split("|"))
    for sport, wanted in _PM_SPORT_TAGS:
        if tagset & wanted:
            return sport
    return None


def kalshi_sport(series_id: str, meta: Mapping[str, Any] | None = None) -> str | None:
    """Sport of a Kalshi series from its tags and title (cached series list), else from
    the ticker."""
    if meta:
        tags = str(meta.get("tags") or "")
        title = f"{meta.get('series_title') or ''} {series_id}"
        if "Esports" in tags.split(","):
            for sport, pat in _ESPORTS_TITLE:
                if re.search(pat, title, re.IGNORECASE):
                    return sport
            return None
        for t in tags.split(","):
            if t in _K_SPORT_TAGS:
                return _K_SPORT_TAGS[t]
    for sport, pat in _K_SPORT_ID:
        if re.search(pat, series_id):
            return sport
    return "soccer" if meta is None else None


def pm_us_league(tags: str | None) -> str | None:
    tagset = frozenset(str(tags or "").split("|"))
    for lg in ("wnba", "nba", "nfl", "nhl", "mlb"):
        if lg in tagset:
            return lg
    if "nba-summer-league" in tagset:
        return "nba"
    return None


def kalshi_us_league(series_id: str) -> str | None:
    s = series_id.upper()
    if "WNBA" in s:
        return "wnba"
    for lg in ("NBA", "NFL", "NHL", "MLB"):
        if re.match(rf"^KX{lg}", s):
            return lg.lower()
    return None


_DRAW_RE: Final = re.compile(r"^(draw|tie)\b|\b(draw|tie)$", re.IGNORECASE)
_VS_RE: Final = re.compile(r"\s+(?:vs\.?|v\.?|at|@)\s+", re.IGNORECASE)


def split_matchup(title: str | None) -> tuple[str, str] | None:
    """("A", "B") from a matchup title such as "Counter-Strike: FaZe vs TheMongolz (BO3)
    - BLAST", "Belgium vs Senegal: Regulation Time Moneyline", "Miami (FL) at Ohio St.
    Winner?" or "Liverpool FC vs. Leeds United FC - More Markets"; None if there is no
    single "vs"/"at"."""
    if not title:
        return None
    parts = [p for p in re.split(r":\s+", str(title)) if _VS_RE.search(p)]
    if len(parts) != 1:
        return None
    t = parts[0]
    t = re.sub(r"\s+-\s+.*$", "", t)  # " - More Markets", " - BLAST Bounty Playoffs"
    t = re.sub(r"\s*\((?:BO\d|W)\)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*\((?:[A-Za-z' -]*weight[A-Za-z, ]*|Main Card|Prelims)[^)]*\)", "", t)
    t = re.sub(r"\s+Winner\??$|\s+winner\??$|\?$", "", t)
    sides = _VS_RE.split(t)
    if len(sides) != 2:
        return None
    a, b = (s.strip(" ,") for s in sides)
    return (a, b) if a and b else None


def _is_draw(label: Any) -> bool:
    s = fold(label)
    return bool(s) and (s in {"draw", "tie"} or bool(_DRAW_RE.search(s)))


def _yes_no(outcomes: list[str]) -> bool:
    return sorted(o.strip().lower() for o in outcomes) == ["no", "yes"]


def _side_of(label: str, names: tuple[str, str], min_sim: float = 0.8) -> str | None:
    """ "a" or "b": which of the two names a label refers to, if clearly one of them."""
    sa, sb = name_similarity(label, names[0]), name_similarity(label, names[1])
    if max(sa, sb) < min_sim or abs(sa - sb) < 0.15:
        return None
    return "a" if sa > sb else "b"


# -- sports: games on each venue -------------------------------------------------------


@dataclass
class Game:
    """One game on one venue: two sides, possibly a draw, and its moneyline contracts."""

    venue: str
    game_id: str
    sport: str
    women: bool
    league: str | None  # US major league, where names are mapped through US_TEAMS
    start: int
    names: tuple[str, str]
    three_way: bool
    contracts: dict[str, int] = field(default_factory=dict)  # side -> row position
    codes: tuple[str | None, str | None] = (None, None)  # Kalshi team codes
    aliases: frozenset[str] = frozenset()  # Polymarket game ids and title stems


# The only Yes/No game markets that are moneylines: "Will A win (on 2026-01-01)?" and
# "Will A vs. B end in a draw?". Not "A to win 2-0?" or "Draw (1-1)?" (series scores).
_PM_WIN_TITLE: Final = re.compile(r"^Will .+ win(?: on \d{4}-\d{2}-\d{2})?\?$", re.IGNORECASE)
_PM_DRAW_TITLE: Final = re.compile(
    r"^Will (?:.+ vs\.? .+|the match) end in a draw\?$", re.IGNORECASE
)


def polymarket_games(pm: pd.DataFrame) -> list[Game]:
    """Polymarket game-winner markets grouped into games.

    Two kinds: "A vs. B" markets whose outcomes are the two names (priced on one of
    them), and Yes/No markets "Will A win?" / "Will A vs. B end in a draw?" sharing an
    event (soccer, Test cricket), which make the game three-way.
    """
    sel = pm.loc[
        (_col(pm, "category") == "sports")
        & (_col(pm, "pm_sports_market_type") == "moneyline")
        & (_col(pm, "anchor_source") == "game_start"),
        [
            "event_id",
            "pm_tags",
            "pm_series_slug",
            "event_title",
            "scheduled_ts",
            "pm_outcomes",
            "outcome_label",
            "pm_group_item_title",
            "title",
            "pm_game_id",
        ],
    ]
    games: list[Game] = []
    for ev, g in _groups(sel, "event_id").items():
        first = g[0]
        sport = pm_sport(first["pm_tags"])
        if sport is None or sport == "golf":
            continue
        text = f"{_str(first['pm_tags']).replace('|', ' ')} {_str(first['pm_series_slug'])} "
        text += _str(first["event_title"])
        women = bool(_WOMEN_RE.search(text))
        league = pm_us_league(first["pm_tags"])
        start = int(first["scheduled_ts"])
        outs = [_json_list(r["pm_outcomes"]) for r in g]
        if all(len(o) == 2 and not _yes_no(o) for o in outs):
            if len({tuple(o) for o in outs}) != 1:
                continue
            names = (outs[0][0], outs[0][1])
            game = Game("polymarket", str(ev), sport, women, league, start, names, False)
            for r in g:
                label = _str(r["outcome_label"])
                two = "a" if label == names[0] else "b" if label == names[1] else None
                if two and two not in game.contracts:
                    game.contracts[two] = r["_pos"]
        elif all(_yes_no(o) for o in outs):
            mu = split_matchup(_str(first["event_title"]))
            if mu is None:
                continue
            game = Game("polymarket", str(ev), sport, women, league, start, mu, False)
            for r in g:
                if _str(r["outcome_label"]).lower() != "yes":
                    continue
                item, title = _str(r["pm_group_item_title"]), _str(r["title"]).strip()
                side: str | None = None
                if _PM_DRAW_TITLE.match(title):
                    side = "draw"
                elif _PM_WIN_TITLE.match(title):
                    side = _side_of(item, mu)
                if side and side not in game.contracts:
                    game.contracts[side] = r["_pos"]
            game.three_way = "draw" in game.contracts or sport == "soccer"
        else:
            continue
        game.aliases = _pm_game_aliases(str(ev), first["pm_game_id"], first["event_title"], start)
        if game.contracts:
            games.append(game)
    return games


def _pm_game_aliases(event_id: str, game_id: Any, event_title: Any, start: int) -> frozenset[str]:
    """Keys under which a Polymarket game's other events (totals and spreads, often in a
    separate "A vs. B - More Markets" event) can be found."""
    stem = fold(re.sub(r"\s+-\s+.*$", "", _str(event_title)))
    keys = {f"ev:{event_id}", f"stem:{stem}:{start}"}
    if _str(game_id):
        keys.add(f"gid:{_str(game_id)}")
    return frozenset(keys)


_K_GAME_SERIES: Final = re.compile(r"^KX.+(GAME|MATCH|FIGHT|BOUT)$")
_K_NOT_GAME: Final = re.compile(r"EXACT|MATCHUP|TEAMSIN|SERIES|TOTAL|SPREAD")
_K_WIN_TITLE: Final = re.compile(r"\bwin(?:s|ner)?\b", re.IGNORECASE)
_K_NOT_WIN_TITLE: Final = re.compile(
    r"\bby (?:over|more)\b|\bspread\b|\btotal\b|\bmap \d|\bset \d|\bgame \d|"
    r"\bfirst (?:half|set|map|quarter|period|inning|round)\b|"
    r"\b1st (?:half|set|map|quarter|period|inning)\b|\bhalf ?time\b|"
    r"\bseries\b|\bplayoff|\bchampionship\b|\btournament\b",
    re.IGNORECASE,
)
_LABEL_PREFIX: Final = re.compile(r"^(reg(ulation)?\.? ?time|reg)\s*:\s*", re.IGNORECASE)


def kalshi_games(k: pd.DataFrame, series_meta: Mapping[str, Mapping[str, Any]]) -> list[Game]:
    """Kalshi game-winner markets (``KX...GAME``/``MATCH``/``FIGHT`` series, one market per
    side and possibly a "Tie" market) grouped into games."""
    sid = _col(k, "series_id")
    sel = k[
        sid.str.match(_K_GAME_SERIES.pattern)
        & ~sid.str.contains(_K_NOT_GAME.pattern)
        & (_col(k, "anchor_source") == "game_start")
        & _col(k, "title").str.contains(_K_WIN_TITLE.pattern, case=False)
        & ~_col(k, "title").str.contains(_K_NOT_WIN_TITLE.pattern, case=False)
    ]
    games: list[Game] = []
    cols = [
        "event_id",
        "series_id",
        "outcome_label",
        "contract_id",
        "event_title",
        "title",
        "scheduled_ts",
    ]
    for ev, g in _groups(sel[cols], "event_id").items():
        first = g[0]
        series = str(first["series_id"])
        meta = series_meta.get(series)
        sport = kalshi_sport(series, meta)
        if sport is None or sport == "golf":
            continue
        title = f"{(meta or {}).get('series_title') or ''} {series}"
        women = bool(_WOMEN_RE.search(title))
        league = kalshi_us_league(series)
        teams = [
            (
                _LABEL_PREFIX.sub("", _str(r["outcome_label"])).strip(),
                str(r["contract_id"]).rsplit("-", 1)[-1],
                r["_pos"],
            )
            for r in g
        ]
        draws = [t for t in teams if _is_draw(t[0])]
        sides = [t for t in teams if not _is_draw(t[0])]
        if len(draws) > 1 or len(sides) > 2 or not sides:
            continue
        if len(sides) == 2:
            names = (sides[0][0], sides[1][0])
            game_codes: tuple[str | None, str | None] = (sides[0][1], sides[1][1])
            contracts = {"a": int(sides[0][2]), "b": int(sides[1][2])}
        else:
            mu = split_matchup(_str(first["event_title"])) or split_matchup(_str(first["title"]))
            if mu is None:
                continue
            side = _side_of(sides[0][0], mu, min_sim=0.6)
            if side is None:
                continue
            names = (sides[0][0], mu[1]) if side == "a" else (mu[0], sides[0][0])
            game_codes = (sides[0][1], None) if side == "a" else (None, sides[0][1])
            contracts = {side: int(sides[0][2])}
        if draws:
            contracts["draw"] = int(draws[0][2])
        start = int(first["scheduled_ts"])
        games.append(
            Game(
                "kalshi",
                str(ev),
                sport,
                women,
                league,
                start,
                names,
                bool(draws),
                contracts,
                game_codes,
            )
        )
    return games


# -- sports: matching games ------------------------------------------------------------

MIN_TEAM_SCORE: Final = 0.75  # each side
MIN_GAME_SCORE: Final = 0.85  # mean of the two sides
MIN_ORIENTATION_MARGIN: Final = 0.5  # the other orientation must fit clearly worse


def team_score(pg: Game, i: int, kg: Game, j: int) -> float:
    """Similarity of side ``i`` of a Polymarket game and side ``j`` of a Kalshi game."""
    if pg.league and pg.league == kg.league:
        tp = us_team(pg.league, pg.names[i])
        tk = us_team(kg.league, kg.names[j], kg.codes[j])
        if tp and tk:
            return 1.0 if tp == tk else 0.0
    return name_similarity(pg.names[i], kg.names[j])


def game_score(pg: Game, kg: Game) -> tuple[float, bool]:
    """(score, swapped): mean side similarity under the better orientation, 0 unless both
    sides reach ``MIN_TEAM_SCORE`` and the mean ``MIN_GAME_SCORE``. ``swapped`` means
    Polymarket's first name is Kalshi's second."""
    direct = (team_score(pg, 0, kg, 0), team_score(pg, 1, kg, 1))
    swap = (team_score(pg, 0, kg, 1), team_score(pg, 1, kg, 0))
    best, swapped = (direct, False) if sum(direct) >= sum(swap) else (swap, True)
    other = swap if not swapped else direct
    if min(best) < MIN_TEAM_SCORE or sum(best) / 2 < MIN_GAME_SCORE:
        return 0.0, swapped
    if sum(best) - sum(other) < MIN_ORIENTATION_MARGIN:
        return 0.0, swapped  # both orientations fit: which side is which is unclear
    return sum(best) / 2, swapped


def _name_keys(name: str, league: str | None, code: str | None = None) -> set[str]:
    """Blocking keys of a name: the first three letters of each token, of the name
    written as one word and its initials; the team for US major leagues. Two names
    that ``name_similarity`` can match share at least one key."""
    if league:
        team = us_team(league, name, code)
        if team:
            return {f"us:{team}"}
    toks = name_tokens(name)
    raw = fold(name).split()
    keys = {t[:3] for t in toks}
    keys.add("".join(raw)[:3])
    keys.add("".join(toks)[:3])
    if len(toks) >= 2:
        keys.add("".join(t[0] for t in toks)[:3])
    keys.discard("")
    return keys


def _game_keys(g: Game) -> tuple[set[str], set[str]]:
    return (
        _name_keys(g.names[0], g.league, g.codes[0]),
        _name_keys(g.names[1], g.league, g.codes[1]),
    )


def match_games(
    pgames: Sequence[Game], kgames: Sequence[Game]
) -> list[tuple[Game, Game, float, bool]]:
    """One-to-one game pairs: same sport (and women's/men's for team sports), same
    two-or-three-way structure, starts within ``GAME_TOL_S`` (``INDIVIDUAL_TOL_S`` for
    individual sports), both sides' names matching. Ties are broken towards the closer
    start time; each game is used once.

    Candidates must share a blocking key on each side (``_name_keys``), which keeps the
    comparison count manageable for tennis, where hundreds of matches start each day.
    """
    index: dict[tuple[str, str], list[int]] = {}
    kkeys = [_game_keys(kg) for kg in kgames]
    for ki, (ka, kb) in enumerate(kkeys):
        for key in ka | kb:
            index.setdefault((kgames[ki].sport, key), []).append(ki)
    cands: list[tuple[float, int, int, bool]] = []  # (score, pm idx, k idx, swapped)
    for pi, pg in enumerate(pgames):
        tol = INDIVIDUAL_TOL_S if pg.sport in INDIVIDUAL_SPORTS else GAME_TOL_S
        pa, pb = _game_keys(pg)
        hits_a = {ki for key in pa for ki in index.get((pg.sport, key), ())}
        hits_b = {ki for key in pb for ki in index.get((pg.sport, key), ())}
        for ki in hits_a & hits_b:
            kg = kgames[ki]
            if abs(pg.start - kg.start) > tol or pg.three_way != kg.three_way:
                continue
            if pg.sport not in INDIVIDUAL_SPORTS and pg.women != kg.women:
                continue
            sc, swapped = game_score(pg, kg)
            if sc <= 0:
                continue
            cands.append((sc - 0.01 * abs(pg.start - kg.start) / tol, pi, ki, swapped))
    cands.sort(key=lambda c: -c[0])
    used_p: set[int] = set()
    used_k: set[int] = set()
    out: list[tuple[Game, Game, float, bool]] = []
    for sc, pi, ki, swapped in cands:
        if pi in used_p or ki in used_k:
            continue
        used_p.add(pi)
        used_k.add(ki)
        out.append((pgames[pi], kgames[ki], round(sc, 4), swapped))
    return out


def _game_event_key(kg: Game) -> str:
    """The game, named after Kalshi's event ticker without the series' market type:
    ``KXMLBGAME-26APR011215ATHATL`` -> ``MLB-26APR011215ATHATL``, so moneyline, totals
    and spreads of one game share it."""
    series, _, rest = kg.game_id.partition("-")
    league = re.sub(r"^KX|(GAME|MATCH|FIGHT|BOUT)$", "", series)
    return f"{league}-{rest}"


def game_pairs(matched: Iterable[tuple[Game, Game, float, bool]]) -> list[Pair]:
    """Contract pairs within matched games.

    Each Polymarket contract (a side or the draw) is paired with Kalshi's market on the
    same side. In a two-way game whose same-side Kalshi market is not in the frame, the
    other side's market is used as the complement; never in a three-way game, where
    "not A" is not "B".
    """
    out: list[Pair] = []
    for pg, kg, score, swapped in matched:
        key = _game_event_key(kg)
        family = "sports_moneyline"
        flip = {"a": "b", "b": "a", "draw": "draw"}
        for side, ppos in pg.contracts.items():
            kside = flip[side] if swapped else side
            if kside in kg.contracts:
                out.append(Pair(ppos, kg.contracts[kside], True, family, key, score))
            elif not pg.three_way and side != "draw" and flip[kside] in kg.contracts:
                out.append(Pair(ppos, kg.contracts[flip[kside]], False, family, key, score))
    return out


# -- sports: totals and spreads of matched games ---------------------------------------

_K_TOTAL_LABEL: Final = re.compile(
    r"^over (\d+(?:\.\d+)?) (?:points|runs|goals)(?: scored)?$", re.IGNORECASE
)
_K_SPREAD_LABEL: Final = re.compile(
    r"^(.+?) wins (?:the game )?by (?:over|more than) (\d+(?:\.\d+)?) (?:points|runs|goals)$",
    re.IGNORECASE,
)
_PM_TOTAL_TITLE: Final = re.compile(r"(?:^|: )O/U (\d+(?:\.\d+)?)$")
_PM_SPREAD_TITLE: Final = re.compile(r"^Spread: (.+?) \(-(\d+(?:\.\d+)?)\)$")


def _line_key(series_event: str) -> str:
    series, _, rest = series_event.partition("-")
    league = re.sub(r"^KX|(TOTAL|SPREAD)$", "", series)
    return f"{league}-{rest}"


def line_pairs(
    matched: Iterable[tuple[Game, Game, float, bool]], pm: pd.DataFrame, k: pd.DataFrame
) -> list[Pair]:
    """Full-game totals and spreads of matched games with exactly the same line.

    Kalshi lists "Over x" totals and "T wins by over x" spreads; Polymarket an Over/Under
    or a T(-x)/other binary priced on one side, so Under and the other side are the
    complements of Kalshi's YES. Kalshi's partial-game series (1H, 1Q, team totals)
    have other event keys and are never paired. Both venues count overtime and extra
    innings, and soccer's 90 minutes plus stoppage time.
    """
    by_key: dict[str, tuple[Game, Game, bool]] = {}
    by_alias: dict[str, tuple[Game, Game, bool]] = {}
    for pg, kg, _, swapped in matched:
        by_key[_game_event_key(kg)] = (pg, kg, swapped)
        for a in pg.aliases:
            by_alias[a] = (pg, kg, swapped)
    # Kalshi line markets of matched games: (key, kind, kalshi side or "", line) -> position
    kl: dict[tuple[str, str, str, float], int] = {}
    sid = _col(k, "series_id")
    for j in _rows_where(sid.str.contains(r"(?:TOTAL|SPREAD)$", regex=True)):
        key = _line_key(str(_at(k, "event_id", j)))
        if key not in by_key:
            continue
        _, kg, _ = by_key[key]
        label = _str(_at(k, "outcome_label", j)).strip()
        if m := _K_TOTAL_LABEL.match(label):
            kl[(key, "total", "", float(m.group(1)))] = j
        elif m := _K_SPREAD_LABEL.match(label):
            code = re.sub(r"\d+$", "", str(_at(k, "contract_id", j)).rsplit("-", 1)[-1])
            side = (
                "a"
                if code and code == kg.codes[0]
                else "b"
                if code and code == kg.codes[1]
                else None
            )
            if side is None:
                side = _side_of(m.group(1), kg.names, min_sim=0.9)
            if side:
                kl[(key, "spread", side, float(m.group(2)))] = j
    out: list[Pair] = []
    smt = _col(pm, "pm_sports_market_type")
    for i in _rows_where(smt.isin(["totals", "spreads"])):
        aliases = _pm_game_aliases(
            str(_at(pm, "event_id", i)),
            _at(pm, "pm_game_id", i),
            _at(pm, "event_title", i),
            int(_at(pm, "scheduled_ts", i)),
        )
        hit = next((by_alias[a] for a in sorted(aliases) if a in by_alias), None)
        if hit is None:
            continue
        pg, kg, swapped = hit
        if not within(_at(pm, "scheduled_ts", i), kg.start, GAME_TOL_S):
            continue  # a rescheduled game whose line markets kept another start time
        key = _game_event_key(kg)
        title, label = _str(_at(pm, "title", i)).strip(), _str(_at(pm, "outcome_label", i))
        if smt.iat[i] == "totals":
            m = _PM_TOTAL_TITLE.search(title)
            if not m or label not in ("Over", "Under"):
                continue
            kj = kl.get((key, "total", "", float(m.group(1))))
            if kj is not None:
                out.append(Pair(i, kj, label == "Over", "sports_total", key, 1.0))
            continue
        m = _PM_SPREAD_TITLE.match(title)
        if not m:
            continue
        fav = _side_of(m.group(1), pg.names, min_sim=0.9)
        if fav is None and pg.league:
            t = us_team(pg.league, m.group(1))
            fav = (
                "a"
                if t and t == us_team(pg.league, pg.names[0])
                else ("b" if t and t == us_team(pg.league, pg.names[1]) else None)
            )
        if fav is None:
            continue
        kfav = {"a": "b", "b": "a"}[fav] if swapped else fav
        kj = kl.get((key, "spread", kfav, float(m.group(2))))
        if kj is None:
            continue
        # The Polymarket contract is priced on one of its two outcomes: the favourite
        # (YES = covers, as on Kalshi) or the other team (the complement).
        outs = _json_list(_at(pm, "pm_outcomes", i))
        fav_out = [o for o in outs if o == m.group(1)]
        if len(outs) != 2 or label not in outs or len(fav_out) != 1:
            continue
        out.append(Pair(i, kj, label == fav_out[0], "sports_spread", key, 1.0))
    return _best_unique(out, pm, k)


# -- macro: Fed decisions, CPI, unemployment -------------------------------------------

_MONTH_NAMES: Final = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_MONTH_ABBR: Final = {m[:3].upper(): i + 1 for i, m in enumerate(_MONTH_NAMES)}
_MONTH_RE: Final = re.compile(r"\b(" + "|".join(_MONTH_NAMES) + r")\b", re.IGNORECASE)


def _kalshi_period(event_id: str) -> tuple[int, int] | None:
    """(year, month) from an event ticker ending in ``-26APR`` (or ``-26APR30``)."""
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})?$", event_id)
    if not m or m.group(2) not in _MONTH_ABBR:
        return None
    return 2000 + int(m.group(1)), _MONTH_ABBR[m.group(2)]


def _pm_period(title: str, scheduled_ts: int, *, released_after: bool) -> tuple[int, int] | None:
    """(year, month) named in a Polymarket title. The year is the scheduled year, or the
    year before when the month is after the scheduled month and the market is about a
    statistic released after its reference month (December CPI is released in January)."""
    m = _MONTH_RE.search(title)
    if not m:
        return None
    month = _MONTH_NAMES.index(m.group(1).lower()) + 1
    y = re.search(r"\b(20\d\d)\b", title)
    if y:
        return int(y.group(1)), month
    d = pd.Timestamp(int(scheduled_ts), unit="s", tz="UTC")
    year = d.year - 1 if released_after and month > d.month else d.year
    return year, month


_FED_K: Final[dict[str, tuple[str, int]]] = {
    "C25": ("cut", 25),
    "C26": ("cut", 50),  # "Cut >25bps"
    "H0": ("hold", 0),
    "H25": ("hike", 25),
    "H26": ("hike", 50),  # "Hike >25bps"
}
_FED_PM_RE: Final = re.compile(
    r"fed (decrease|increase) interest rates by (\d+)(\+?) bps after the (\w+) (20\d\d) meeting|"
    r"no change in fed interest rates after the (\w+) (20\d\d) meeting",
    re.IGNORECASE,
)


def fed_bracket_pm(title: str) -> tuple[tuple[str, int], tuple[int, int]] | None:
    """((move, bps), (year, month)) of a Polymarket "Fed Decision in <Month>?" market.

    "50+ bps" is the open-ended bracket, the same set as Kalshi's ">25bps": both venues
    count moves in multiples of 25 bps (Polymarket rounds others up to the next 25)."""
    m = _FED_PM_RE.search(title)
    if not m:
        return None
    if m.group(1):
        move = "cut" if m.group(1).lower() == "decrease" else "hike"
        bps = int(m.group(2))
        if (bps == 25) == bool(m.group(3)) or bps not in (25, 50):
            return None  # only "25 bps" and "50+ bps" exist on both venues
        month, year = m.group(4), m.group(5)
        bracket = (move, bps)
    else:
        month, year = m.group(6), m.group(7)
        bracket = ("hold", 0)
    if month.lower() not in _MONTH_NAMES:
        return None
    return bracket, (int(year), _MONTH_NAMES.index(month.lower()) + 1)


def fed_pairs(pm: pd.DataFrame, k: pd.DataFrame) -> list[Pair]:
    """Kalshi ``KXFEDDECISION`` brackets and Polymarket "Fed Decision in <Month>?" markets
    for the same FOMC meeting and the same move."""
    kk: dict[tuple[tuple[str, int], tuple[int, int]], list[int]] = {}
    for j in _rows_where(_col(k, "series_id") == "KXFEDDECISION"):
        code = str(_at(k, "contract_id", j)).rsplit("-", 1)[-1]
        period = _kalshi_period(str(_at(k, "event_id", j)))
        if code in _FED_K and period:
            kk.setdefault((_FED_K[code], period), []).append(j)
    out: list[Pair] = []
    titles = _col(pm, "title")
    cand = titles.str.contains(r"Fed (?:de|in)crease interest rates|no change in Fed", case=False)
    for i in _rows_where(cand):
        parsed = fed_bracket_pm(str(titles.iat[i]))
        if parsed is None:
            continue
        for j in kk.get(parsed, []):
            if _within_pos(pm, k, i, j, EVENT_TOL_S):
                y, mth = parsed[1]
                out.append(Pair(i, j, True, "fed_decision", f"fed:{y}-{mth:02d}", 1.0))
    return _best_unique(out, pm, k)


# Statistics published to one decimal: an open-ended Polymarket bracket "≥X%" is the
# same set as Kalshi's "above X-0.1%", and "≤X%" is the complement of "above X%".
_MACRO_K: Final[dict[str, str]] = {
    "KXCPI": "cpi_mom",
    "KXCPIYOY": "cpi_yoy",
    "KXCPICORE": "core_cpi_mom",
    "KXCPICOREYOY": "core_cpi_yoy",
    "KXU3": "unemployment",
}
_MACRO_FAMILY: Final[dict[str, str]] = {
    "cpi_mom": "cpi",
    "cpi_yoy": "cpi",
    "core_cpi_mom": "cpi",
    "core_cpi_yoy": "cpi",
    "unemployment": "unemployment",
}


def macro_measure_pm(event_title: str) -> str | None:
    """Which statistic a Polymarket macro event is about (US only)."""
    t = event_title.strip()
    if re.fullmatch(r"\w+ Inflation US - Monthly", t):
        return "cpi_mom"
    if re.fullmatch(r"\w+ Inflation US - Annual(?: \((?:Higher|Lower) Brackets\))?", t):
        return "cpi_yoy"
    if re.fullmatch(r"Core CPI MoM - \w+ 20\d\d", t):
        return "core_cpi_mom"
    if re.fullmatch(r"Core CPI YoY - \w+ 20\d\d", t):
        return "core_cpi_yoy"
    if re.fullmatch(r"\w+ Unemployment Rate", t):
        return "unemployment"
    return None


def open_bracket(label: str) -> tuple[str, float] | None:
    """("ge", x) for "≥x%", "x%+", "x% or more"/"or higher"; ("le", x) for "≤x%", "x% or
    less"/"or lower"; None for a closed bracket such as "0.4%"."""
    s = label.replace(" ", "").replace("%", "").lower()
    m = re.fullmatch(r"(≥|>=)(-?\d+(?:\.\d+)?)|(-?\d+(?:\.\d+)?)(\+|ormore|orhigher|orabove)", s)
    if m:
        return "ge", float(m.group(2) or m.group(3))
    m = re.fullmatch(r"(≤|<=)(-?\d+(?:\.\d+)?)|(-?\d+(?:\.\d+)?)(orless|orlower|orbelow)", s)
    if m:
        return "le", float(m.group(2) or m.group(3))
    return None


def _kalshi_above(label: str) -> float | None:
    m = re.fullmatch(r"Above (-?\d+(?:\.\d+)?)%", label.strip())
    return float(m.group(1)) if m else None


def macro_pairs(pm: pd.DataFrame, k: pd.DataFrame) -> list[Pair]:
    """Open-ended Polymarket brackets of US CPI (headline/core, MoM/YoY) and unemployment
    releases paired with Kalshi's "Above x%" markets for the same reference month."""
    kk: dict[tuple[str, tuple[int, int], float], int] = {}
    for j in _rows_where(_col(k, "series_id").isin(list(_MACRO_K))):
        measure = _MACRO_K[str(_at(k, "series_id", j))]
        period = _kalshi_period(str(_at(k, "event_id", j)))
        x = _kalshi_above(str(_at(k, "outcome_label", j)))
        if period and x is not None:
            kk[(measure, period, round(x, 1))] = j
    out: list[Pair] = []
    ev_titles = _col(pm, "event_title")
    cand = ev_titles.str.contains(r"Inflation US|Core CPI|Unemployment Rate$", regex=True)
    for i in _rows_where(cand):
        pm_measure = macro_measure_pm(str(ev_titles.iat[i]))
        item = _at(pm, "pm_group_item_title", i)
        if pm_measure is None or item is None or item is pd.NA:
            continue
        br = open_bracket(str(item))
        period = _pm_period(
            f"{_at(pm, 'title', i)} {ev_titles.iat[i]}",
            int(_at(pm, "scheduled_ts", i)),
            released_after=True,
        )
        if br is None or period is None:
            continue
        side, x = br
        strike = round(x - 0.1, 1) if side == "ge" else round(x, 1)
        kj = kk.get((pm_measure, period, strike))
        if kj is None or not _within_pos(pm, k, i, kj, EVENT_TOL_S):
            continue
        key = f"{pm_measure}:{period[0]}-{period[1]:02d}"
        out.append(Pair(i, kj, side == "ge", _MACRO_FAMILY[pm_measure], key, 1.0))
    return _best_unique(out, pm, k)


# -- multi-outcome events: elections, awards, golf -------------------------------------

MIN_TITLE_COVERAGE: Final = 0.6
# Title features that change the proposition. Two events only match if they carry the
# same set ("Portugal Presidential Election" is not its "1st Round Winner", "Best
# Actor" is not "Best Supporting Actor", a party primary is not the general election).
_QUALIFIERS: Final[dict[str, str]] = {
    "round1": r"\b(?:1st|first) round\b",
    "round2": r"\b(?:2nd|second) round\b|\brun-?off\b",
    "place2": r"\b(?:2nd|second) place\b|\bsecond most\b",
    "place3": r"\b(?:3rd|third) place\b|\bthird most\b",
    "place4": r"\b(?:4th|fourth) place\b|\bfourth most\b",
    "place5": r"\b(?:5th|fifth) place\b|\bfifth most\b",
    "place_n": r"\b(?:[6-9]|\d\d)(?:st|nd|rd|th) place\b",
    "advance": r"\badvance|\bqualif|\btop two\b|\btop-two\b",
    "margin": r"\bmargin\b",
    "turnout": r"\bturnout\b",
    "seats": r"\bseats?\b",
    "popular_vote": r"\bpopular vote\b|\bvote share\b",
    "primary": r"\bprimar(?:y|ies)\b",
    "democratic": r"\bdemocrat(?:ic)?\b",
    "republican": r"\brepublican\b|\bgop\b",
    "nominee": r"\bnomin(?:ee|ation)s?\b",
    "leadership": r"\bleader(?:ship)?\b",
    "president": r"\bpresident(?:ial)?\b",
    "prime_minister": r"\bprime minister\b",
    "governor": r"\bgovernor|\bgubernatorial\b",
    "mayor": r"\bmayor",
    "senate": r"\bsenate\b",
    "house": r"\bhouse\b|\bcongressional\b|\b[A-Z]{2}-\d{1,2}\b",
    "supporting": r"\bsupporting\b",
    "actor": r"\bactor\b",
    "actress": r"\bactress\b",
    "director": r"\bdirector\b|\bdirecting\b",
    "picture": r"\bpicture\b|\bfilm\b|\bmovie\b",
    "screenplay": r"\bscreenplay\b",
    "original": r"\boriginal\b",
    "adapted": r"\badapted\b",
    "animated": r"\banimat",
    "international": r"\binternational\b|\bforeign\b",
    "documentary": r"\bdocumentar",
    "song": r"\bsong\b",
    "score": r"\bscore\b",
    "album": r"\balbum\b",
    "record": r"\brecord\b",
    "artist": r"\bartist\b",
    "drama": r"\bdrama\b",
    "comedy": r"\bcomedy\b|\bmusical\b",
    "series": r"\bseries\b",
    "limited": r"\blimited\b",
    "rookie": r"\brookie\b",
    "defensive": r"\bdefensive\b",
    "coach": r"\bcoach\b",
    "mvp": r"\bmvp\b|\bmost valuable\b",
    "women": r"\bwomen|\bwomen's\b|\bfemale\b",
    # a sub-field of a tournament ("Top Past Champion", "Top Debutant")
    "subfield": r"\bpast champion|\bdebut|\bamateur\b|\btop (?:american|european|asian|"
    r"international|rookie|lefty|left-handed|lefthanded)\b",
}
_QUALIFIER_RES: Final = {k: re.compile(v, re.IGNORECASE) for k, v in _QUALIFIERS.items()}
# Not paired: events whose outcomes are numeric brackets (margins, turnout, seat counts),
# whose edges are written differently on the two venues, and contests within a sub-field
# of a tournament ("Top Past Champion"), where Kalshi splits ties ($1/N).
_UNPAIRED_QUALIFIERS: Final = frozenset({"margin", "turnout", "seats", "popular_vote", "subfield"})
_ELECTION_RE: Final = re.compile(
    r"election|primary|runoff|run-off|mayor|governor|gubernatorial|senate|president|"
    r"prime minister|parliament|referendum|by-election|congress|nominee|chancellor",
    re.IGNORECASE,
)
_AWARD_RE: Final = re.compile(
    r"oscar|academy award|grammy|emmy|golden globe|bafta|tony award|\bmvp\b|award|trophy|"
    r"ballon d'or|eurovision|person of the year|rookie of the year|cy young|heisman",
    re.IGNORECASE,
)


def title_qualifiers(title: str) -> frozenset[str]:
    return frozenset(k for k, r in _QUALIFIER_RES.items() if r.search(title))


_TITLE_STOPWORDS: Final = _words(
    """will who what which win wins won winner winners the a an of in at for on be by to is
    next and or this that its it with from as"""
)
_ORDINALS: Final = {"1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth"}


def title_tokens(title: str) -> tuple[str, ...]:
    """Content words of an event title (years, filler words and punctuation removed)."""
    toks = [_ORDINALS.get(t, t) for t in fold(title).split()]
    return tuple(t for t in toks if t not in _TITLE_STOPWORDS and not re.fullmatch(r"20\d\d", t))


def _coverage_flags(ta: Sequence[str], tb: Sequence[str]) -> tuple[list[bool], list[bool]]:
    """Which words of each title are found in the other: equal words, words sharing a
    four-letter prefix, and acronyms with the run of words they abbreviate ("VMA" and
    "Video Music Awards")."""
    fa = [False] * len(ta)
    fb = [False] * len(tb)
    for i, t in enumerate(ta):
        for j, u in enumerate(tb):
            if t == u or (len(t) >= 4 and len(u) >= 4 and (t.startswith(u) or u.startswith(t))):
                fa[i] = fb[j] = True
    for x, y, fx, fy in ((ta, tb, fa, fb), (tb, ta, fb, fa)):
        for i, t in enumerate(x):
            if not 2 <= len(t) <= 4:
                continue
            for j in range(len(y) - len(t) + 1):
                if "".join(w[0] for w in y[j : j + len(t)]) == t:
                    fx[i] = True
                    for q in range(j, j + len(t)):
                        fy[q] = True
    return fa, fb


# Words that frame an event rather than define it; a title may have them unmatched.
_FRAMING_WORDS: Final = _words(
    """award awards election elections parliamentary general state national federal
    legislative regional game games playoffs annual mtv nba nfl nhl mlb wnba ncaa pga tour
    ceremony"""
)


def titles_conflict(a: str, b: str) -> bool:
    """Whether each title has a content word the other lacks ("Best Stunt Design" vs
    "Best Cinematography"), ignoring framing words. One title merely being more detailed
    than the other is not a conflict."""
    ta, tb = title_tokens(a), title_tokens(b)
    fa, fb = _coverage_flags(ta, tb)
    extra_a = {t for t, f in zip(ta, fa, strict=True) if not f} - _FRAMING_WORDS
    extra_b = {t for t, f in zip(tb, fb, strict=True) if not f} - _FRAMING_WORDS
    return bool(extra_a) and bool(extra_b)


def title_coverage(a: str, b: str) -> float:
    """The smaller of the two shares of each title's content words (framing words
    aside) found in the other."""
    ta, tb = title_tokens(a), title_tokens(b)
    if not ta or not tb:
        return 0.0
    fa, fb = _coverage_flags(ta, tb)

    def share(toks: Sequence[str], flags: Sequence[bool]) -> float:
        keep = [f for t, f in zip(toks, flags, strict=True) if t not in _FRAMING_WORDS]
        keep = keep or list(flags)
        return sum(keep) / len(keep)

    return min(share(ta, fa), share(tb, fb))


_DAY_RE: Final = re.compile(
    r"\b(" + "|".join(m[:3] for m in _MONTH_NAMES) + r")[a-z]*\.? (\d{1,2})\b", re.IGNORECASE
)


def title_days(title: str) -> frozenset[tuple[str, int]]:
    """(month, day) pairs named in a title: "on July 27" -> {("jul", 27)}."""
    return frozenset((m.group(1).lower(), int(m.group(2))) for m in _DAY_RE.finditer(title))


def _numbers(s: str) -> frozenset[str]:
    return frozenset(str(float(x)) for x in re.findall(r"\d+(?:\.\d+)?", s))


def _is_name_label(label: str) -> bool:
    """Whether an outcome label names someone or something (not a number or bracket)."""
    s = label.strip()
    if not s or len(re.sub(r"[^A-Za-z]", "", s)) < 3:
        return False
    if re.search(r"\d", s) and not re.search(r"[A-Za-z]{3,}", s):
        return False
    return fold(s) not in {"other", "others", "none", "nobody", "no one", "field", "tie", "draw"}


@dataclass
class CatEvent:
    venue: str
    event_id: str
    title: str
    category: str
    start: int
    qualifiers: frozenset[str]
    outcomes: dict[str, int]  # outcome name -> row position
    sport: str | None = None
    exclusive: bool | None = None  # at most one outcome can resolve YES


def _cat_events_pm(pm: pd.DataFrame) -> list[CatEvent]:
    sel = pm.loc[
        _col(pm, "pm_outcomes").str.contains('"Yes"', regex=False)
        & (_col(pm, "outcome_label") == "Yes")
        & (_col(pm, "pm_group_item_title") != "")
        & (_col(pm, "pm_sports_market_type") == ""),
        [
            "event_id",
            "event_title",
            "category",
            "scheduled_ts",
            "pm_group_item_title",
            "pm_tags",
            "mutually_exclusive",
        ],
    ]
    out: list[CatEvent] = []
    for ev, g in _groups(sel, "event_id").items():
        if len(g) < 2:
            continue
        first = g[0]
        names: dict[str, int] = {}
        for r in g:
            item = _str(r["pm_group_item_title"])
            if _is_name_label(item) and item not in names:
                names[item] = r["_pos"]
        if len(names) < 2:
            continue
        title = _str(first["event_title"])
        starts = sorted(int(r["scheduled_ts"]) for r in g)
        out.append(
            CatEvent(
                "polymarket",
                str(ev),
                title,
                str(first["category"]),
                starts[len(starts) // 2],
                title_qualifiers(title),
                names,
                pm_sport(first["pm_tags"]),
                _opt_bool(first["mutually_exclusive"]),
            )
        )
    return out


def _cat_events_kalshi(
    k: pd.DataFrame, series_meta: Mapping[str, Mapping[str, Any]], skip_series: Iterable[str]
) -> list[CatEvent]:
    skip = set(skip_series)
    sid = _col(k, "series_id")
    sel = k.loc[
        ~sid.isin(skip)
        & ~sid.str.match(_K_GAME_SERIES.pattern)
        & ~sid.str.contains(r"TOTAL|SPREAD|MENTION|^KXHIGH|^KXLOW", regex=True),
        [
            "event_id",
            "series_id",
            "event_title",
            "title",
            "category",
            "scheduled_ts",
            "outcome_label",
            "mutually_exclusive",
        ],
    ]
    out: list[CatEvent] = []
    for ev, g in _groups(sel, "event_id").items():
        if len(g) < 2:
            continue
        first = g[0]
        names: dict[str, int] = {}
        for r in g:
            lab = _str(r["outcome_label"])
            if _is_name_label(lab) and lab not in names:
                names[lab] = r["_pos"]
        if len(names) < 2:
            continue
        title = _str(first["event_title"])
        starts = sorted(int(r["scheduled_ts"]) for r in g)
        series = str(first["series_id"])
        out.append(
            CatEvent(
                "kalshi",
                str(ev),
                title,
                str(first["category"]),
                starts[len(starts) // 2],
                title_qualifiers(title),
                names,
                kalshi_sport(series, series_meta.get(series)) if series_meta else None,
                _opt_bool(first["mutually_exclusive"]),
            )
        )
    return out


def _outcome_pairs(
    pe: CatEvent, ke: CatEvent, min_sim: float = 0.9
) -> list[tuple[str, str, float]]:
    """Polymarket and Kalshi outcomes naming the same person or thing, one-to-one.

    An outcome that matches two outcomes on the other venue is dropped: Polymarket lists
    MTV's nominees as "Song (Artist)", Kalshi as the artist, and an artist with two
    nominated videos would make Kalshi's market a union of two of Polymarket's."""
    cands = []
    for a in pe.outcomes:
        for b in ke.outcomes:
            if _numbers(a) != _numbers(b):
                continue
            s = name_similarity(a, b)
            if s >= min_sim:
                cands.append((s, a, b))
    n_a: dict[str, int] = {}
    n_b: dict[str, int] = {}
    for _, a, b in cands:
        n_a[a] = n_a.get(a, 0) + 1
        n_b[b] = n_b.get(b, 0) + 1
    return [(a, b, s) for s, a, b in cands if n_a[a] == 1 and n_b[b] == 1]


_COMPATIBLE_CATEGORIES: Final = ({"politics", "world"},)


def _same_category(a: str, b: str) -> bool:
    return a == b or any(a in s and b in s for s in _COMPATIBLE_CATEGORIES)


def categorical_family(pe: CatEvent, ke: CatEvent) -> str:
    text = f"{pe.title} {ke.title}"
    if pe.sport == "golf" or ke.sport == "golf":
        if re.search(r"\btop \d+", text, re.IGNORECASE):
            return "golf_top_n"
        if re.search(r"\bmake the cut\b", text, re.IGNORECASE):
            return "golf_make_cut"
        return "golf_winner" if re.search(r"\bwinner\b", text, re.IGNORECASE) else "golf_other"
    if _AWARD_RE.search(text):
        return "awards"
    if _ELECTION_RE.search(text):
        return "election"
    return "other_categorical"


def match_categorical(
    pm: pd.DataFrame,
    k: pd.DataFrame,
    series_meta: Mapping[str, Mapping[str, Any]],
    skip_series: Iterable[str] = (),
) -> list[tuple[CatEvent, CatEvent, list[tuple[str, str, float]]]]:
    """Multi-outcome events listed on both venues: same category, scheduled within
    ``CATEGORICAL_TOL_S``, the same title qualifiers, and at least two outcomes, and at
    least 60% of the smaller event's outcomes, naming the same people or things."""
    pev = _cat_events_pm(pm)
    kev = _cat_events_kalshi(k, series_meta, skip_series)

    def keys(e: CatEvent) -> set[str]:
        return {t[:4] for n in e.outcomes for t in name_tokens(n)[-1:]}

    index: dict[str, list[int]] = {}
    for ki, ke in enumerate(kev):
        for key in keys(ke):
            index.setdefault(key, []).append(ki)
    cands = []
    for pe in pev:
        counts: dict[int, int] = {}
        for key in keys(pe):
            for ki in index.get(key, ()):
                counts[ki] = counts.get(ki, 0) + 1
        for ki, n in counts.items():
            ke = kev[ki]
            if n < 2 or abs(pe.start - ke.start) > CATEGORICAL_TOL_S:
                continue
            if not _same_category(pe.category, ke.category) or pe.qualifiers != ke.qualifiers:
                continue
            if pe.qualifiers & _UNPAIRED_QUALIFIERS:
                continue
            known = pe.exclusive is not None and ke.exclusive is not None
            if known and pe.exclusive != ke.exclusive:
                continue
            if _numbers(" ".join(title_tokens(pe.title))) != _numbers(
                " ".join(title_tokens(ke.title))
            ):
                continue
            if title_coverage(pe.title, ke.title) < MIN_TITLE_COVERAGE:
                continue
            if titles_conflict(pe.title, ke.title):
                continue
            dp, dk = title_days(pe.title), title_days(ke.title)
            if dp and dk and dp != dk:
                continue
            op = _outcome_pairs(pe, ke)
            n_min = min(len(pe.outcomes), len(ke.outcomes))
            if len(op) < 2 or len(op) < 0.6 * n_min:
                continue
            score = len(op) / n_min - 0.01 * abs(pe.start - ke.start) / CATEGORICAL_TOL_S
            cands.append((score, id(pe), id(ke), pe, ke, op))
    cands.sort(key=lambda c: -c[0])
    used_p: set[int] = set()
    used_k: set[int] = set()
    out = []
    for _, ip, ik, pe, ke, op in cands:
        if ip in used_p or ik in used_k:
            continue
        used_p.add(ip)
        used_k.add(ik)
        out.append((pe, ke, op))
    return out


def categorical_pairs(
    matched: Iterable[tuple[CatEvent, CatEvent, list[tuple[str, str, float]]]],
) -> list[Pair]:
    out: list[Pair] = []
    for pe, ke, op in matched:
        fam = categorical_family(pe, ke)
        for a, b, s in op:
            out.append(Pair(pe.outcomes[a], ke.outcomes[b], True, fam, f"event:{ke.event_id}", s))
    return out


# -- weather (evaluated, excluded) -----------------------------------------------------

# Polymarket series -> Kalshi series measuring the same station. Polymarket resolves on
# Wunderground (from late August 2026 on NOAA's hourly time series) at the station named
# in each market's rules: KLAX, KMIA and KAUS for these series, checked on every market
# of them. Kalshi resolves on the NWS daily climate report (CLI) of the same airports.
# Other cities use different stations on the two venues (New York: LaGuardia vs
# Central Park; Chicago: O'Hare vs Midway; Dallas: Love Field vs DFW; Denver: Buckley
# vs DEN) and are not paired.
WEATHER_SERIES: Final[dict[str, str]] = {
    "los-angeles-daily-weather": "KXHIGHLAX",
    "miami-daily-weather": "KXHIGHMIA",
    "austin-daily-weather": "KXHIGHAUS",
    "los-angeles-daily-lowest-temperature": "KXLOWTLAX",
    "miami-daily-lowest-temperature": "KXLOWTMIA",
    "austin-daily-lowest-temperature": "KXLOWTAUS",
}
_INF: Final = float("inf")


def temperature_bracket(label: str) -> tuple[float, float] | None:
    """Closed interval of whole degrees F: "76-77°F" and "96° to 97°" -> (76, 77);
    "23°F or below" -> (-inf, 23); "66° or above"/"90°F or higher" -> (66, inf)."""
    s = label.replace("°", "").replace("F", "").replace("ºF", "").strip().lower()
    if m := re.fullmatch(r"(-?\d+)\s*(?:-|to|\u2013)\s*(-?\d+)", s):
        return float(m.group(1)), float(m.group(2))
    if m := re.fullmatch(r"(-?\d+)\s*or (?:below|less|lower)", s):
        return -_INF, float(m.group(1))
    if m := re.fullmatch(r"(-?\d+)\s*or (?:above|higher|more)", s):
        return float(m.group(1)), _INF
    return None


def weather_pairs(pm: pd.DataFrame, k: pd.DataFrame) -> list[Pair]:
    """Daily high/low temperature brackets for the same station and day with identical
    bounds."""
    kk: dict[tuple[str, int, int, int, tuple[float, float]], int] = {}
    for j in _rows_where(_col(k, "series_id").isin(list(WEATHER_SERIES.values()))):
        ev = str(_at(k, "event_id", j))
        m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", ev)
        br = temperature_bracket(_str(_at(k, "outcome_label", j)))
        if not m or m.group(2) not in _MONTH_ABBR or br is None:
            continue
        day = (2000 + int(m.group(1)), _MONTH_ABBR[m.group(2)], int(m.group(3)))
        kk[(str(_at(k, "series_id", j)), *day, br)] = j
    out: list[Pair] = []
    slug = _col(pm, "pm_series_slug")
    for i in _rows_where(slug.isin(list(WEATHER_SERIES))):
        title = _str(_at(pm, "title", i))
        m = re.search(r"\bon (" + "|".join(_MONTH_NAMES) + r") (\d{1,2})\b", title, re.IGNORECASE)
        br = temperature_bracket(_str(_at(pm, "pm_group_item_title", i)))
        if not m or br is None or _str(_at(pm, "outcome_label", i)) != "Yes":
            continue
        year = pd.Timestamp(int(_at(pm, "scheduled_ts", i)), unit="s", tz="UTC").year
        day = (year, _MONTH_NAMES.index(m.group(1).lower()) + 1, int(m.group(2)))
        series = WEATHER_SERIES[str(slug.iat[i])]
        kj = kk.get((series, *day, br))
        if kj is not None and _within_pos(pm, k, i, kj, EVENT_TOL_S):
            key = f"{series}:{day[0]}-{day[1]:02d}-{day[2]:02d}"
            family = "weather_high" if "HIGH" in series else "weather_low"
            out.append(Pair(i, kj, True, family, key, 1.0))
    return _best_unique(out, pm, k)


# -- finding the matches ---------------------------------------------------------------


def load_frame(cfg: Config, venue: str) -> pd.DataFrame:
    """A venue's index restricted to its sampling frame (``pmcal.sample.frame_mask``)."""
    idx = pd.read_parquet(cfg.interim_dir() / f"{venue}_index.parquet")
    return idx.loc[frame_mask(idx, cfg)].reset_index(drop=True)


def kalshi_series_meta(cfg: Config) -> dict[str, dict[str, Any]]:
    """Series ticker -> {"series_title", "tags"} from the cached ``/series`` listing (no
    network; empty if it is not cached)."""
    from pmcal.venues import kalshi

    with kalshi.make_client(cfg) as c:
        r = c.cached("/series", {"include_volume": True})
    if r is None or not isinstance(r.data, Mapping):
        log.warning("Kalshi series list not cached; sports are inferred from tickers")
        return {}
    return {
        str(s.get("ticker")): {
            "series_title": s.get("title") or "",
            "tags": ",".join(s.get("tags") or []),
        }
        for s in r.data.get("series") or []
    }


MATCH_COLUMNS: Final[dict[str, str]] = {
    "pair_id": "string",
    "event_key": "string",  # the real-world event or game: the bootstrap cluster
    "family": "string",
    "category": "string",
    "polymarket_contract_id": "string",
    "kalshi_contract_id": "string",
    "polymarket_event_id": "string",
    "kalshi_event_id": "string",
    "same_direction": "boolean",  # Polymarket YES <=> Kalshi YES
    "match_score": "float64",
    "polymarket_title": "string",
    "kalshi_title": "string",
    "polymarket_label": "string",
    "kalshi_label": "string",
    "polymarket_scheduled_ts": "Int64",
    "kalshi_scheduled_ts": "Int64",
    "polymarket_outcome": "float64",
    "kalshi_outcome_mapped": "float64",  # Kalshi's outcome in the Polymarket direction
    "outcome_agree": "float64",  # 1/0 after direction mapping; NaN unless both resolved
    # A smaller set with the sample's cap on sports: at most
    # ``sample.max_events_per_series_month`` games per league and month, by seeded hash.
    "core": "boolean",
}


def pair_id(pm_contract: str, k_contract: str) -> str:
    """A stable id for a pair (16 hex digits of a hash of both contract ids)."""
    return hashlib.blake2b(f"{pm_contract}|{k_contract}".encode(), digest_size=8).hexdigest()


def pairs_frame(pairs: Sequence[Pair], pm: pd.DataFrame, k: pd.DataFrame) -> pd.DataFrame:
    """The matches table for pairs given by row positions."""
    if not pairs:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in MATCH_COLUMNS.items()})
    pi = np.array([p.pm_pos for p in pairs])
    ki = np.array([p.k_pos for p in pairs])
    P = pm.iloc[pi].reset_index(drop=True)
    K = k.iloc[ki].reset_index(drop=True)
    same = np.array([p.same_direction for p in pairs], dtype=bool)
    p_out = np.where(P["resolution_status"] == "resolved", P["outcome"].astype(float), np.nan)
    k_raw = np.where(K["resolution_status"] == "resolved", K["outcome"].astype(float), np.nan)
    k_out = np.where(same, k_raw, 1.0 - k_raw)
    agree = np.where(
        np.isfinite(p_out) & np.isfinite(k_out), (p_out == k_out).astype(float), np.nan
    )
    df = pd.DataFrame(
        {
            "pair_id": [
                pair_id(a, b) for a, b in zip(P["contract_id"], K["contract_id"], strict=True)
            ],
            "event_key": [p.event_key for p in pairs],
            "family": [p.family for p in pairs],
            "category": P["category"].to_numpy(),
            "polymarket_contract_id": P["contract_id"].to_numpy(),
            "kalshi_contract_id": K["contract_id"].to_numpy(),
            "polymarket_event_id": P["event_id"].to_numpy(),
            "kalshi_event_id": K["event_id"].to_numpy(),
            "same_direction": same,
            "match_score": [p.score for p in pairs],
            "polymarket_title": P["title"].to_numpy(),
            "kalshi_title": K["title"].to_numpy(),
            "polymarket_label": P["outcome_label"].to_numpy(),
            "kalshi_label": K["outcome_label"].to_numpy(),
            "polymarket_scheduled_ts": P["scheduled_ts"].to_numpy(),
            "kalshi_scheduled_ts": K["scheduled_ts"].to_numpy(),
            "polymarket_outcome": p_out,
            "kalshi_outcome_mapped": k_out,
            "outcome_agree": agree,
            "core": True,
        }
    )
    df = df.astype(MATCH_COLUMNS)
    return df.sort_values(["family", "event_key", "pair_id"], kind="stable").reset_index(drop=True)


SPORTS_FAMILIES: Final = frozenset({"sports_moneyline", "sports_total", "sports_spread"})


def mark_core(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Set ``core``: every non-sports pair, and the pairs of at most
    ``cfg.sample.max_events_per_series_month`` games per league and calendar month,
    chosen by a seeded hash of the game's ``event_key`` (as ``pmcal.sample`` caps a
    series). Matching all games would roughly double the price downloads; the core set
    keeps a few thousand pairs spread over every league and month."""
    out = df.copy()
    sports = out["family"].isin(list(SPORTS_FAMILIES))
    if not sports.any():
        out["core"] = True
        return out.astype({"core": "boolean"})
    g = out.loc[sports, ["event_key", "kalshi_scheduled_ts"]].drop_duplicates("event_key")
    month = pd.to_datetime(g["kalshi_scheduled_ts"].astype("int64"), unit="s", utc=True)
    g["stratum"] = g["event_key"].str.split("-").str[0] + "|" + month.dt.strftime("%Y-%m")
    seed = cfg.sample.seed
    g["rank"] = [
        int.from_bytes(hashlib.blake2b(f"{seed}:{e}".encode(), digest_size=8).digest(), "big")
        for e in g["event_key"]
    ]
    g["r"] = g.groupby("stratum")["rank"].rank(method="first")
    keep = set(g.loc[g["r"] <= cfg.sample.max_events_per_series_month, "event_key"])
    out["core"] = ~sports | out["event_key"].isin(keep)
    return out.astype({"core": "boolean"})


def find_pairs(
    pm: pd.DataFrame, k: pd.DataFrame, series_meta: Mapping[str, Mapping[str, Any]]
) -> list[Pair]:
    """Every family's pairs, by row position in the two frames, one pair per contract."""
    pairs: list[Pair] = []
    pgames = polymarket_games(pm)
    kgames = kalshi_games(k, series_meta)
    matched = match_games(pgames, kgames)
    log.info(
        "sports: %d Polymarket and %d Kalshi games, %d matched",
        len(pgames),
        len(kgames),
        len(matched),
    )
    pairs += game_pairs(matched)
    pairs += line_pairs(matched, pm, k)
    pairs += fed_pairs(pm, k)
    pairs += macro_pairs(pm, k)
    pairs += weather_pairs(pm, k)
    cat = match_categorical(pm, k, series_meta, skip_series=["KXFEDDECISION", *_MACRO_K])
    log.info("multi-outcome events: %d matched", len(cat))
    pairs += categorical_pairs(cat)
    pairs = [p for p in pairs if p.family not in EXCLUDED_FAMILIES]
    return _best_unique(pairs, pm, k)


def find_matches(cfg: Config) -> pd.DataFrame:
    """Pairs of contracts, one per venue, that are the same proposition, from the two
    venue indices (frame rows only).

    Every pair goes to ``interim/matches_all.parquet``; the core pairs (``mark_core``)
    to ``interim/matches.parquet``, which is what ``pmcal sample`` forces into the
    samples and what ``build_matched_snapshots`` prices. Forcing every matched game
    would more than triple the price downloads; the core set adds about a tenth.
    Returns the core pairs.
    """
    pm = load_frame(cfg, "polymarket")
    k = load_frame(cfg, "kalshi")
    log.info("frames: %d Polymarket and %d Kalshi contracts", len(pm), len(k))
    pairs = find_pairs(pm, k, kalshi_series_meta(cfg))
    df = mark_core(pairs_frame(pairs, pm, k), cfg)
    out_dir = cfg.interim_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / ALL_MATCHES_FILE, index=False)
    core = df[df["core"].fillna(False).astype(bool)].reset_index(drop=True)
    core.to_parquet(out_dir / MATCHES_FILE, index=False)
    log_match_summary(df)
    log.info(
        "wrote %s: %d pairs (%d events); %s: %d core pairs (%d events)",
        out_dir / ALL_MATCHES_FILE,
        len(df),
        df["event_key"].nunique(),
        out_dir / MATCHES_FILE,
        len(core),
        core["event_key"].nunique(),
    )
    return core


def log_match_summary(df: pd.DataFrame) -> None:
    if df.empty:
        log.info("no pairs")
        return
    g = df.groupby("family").agg(
        pairs=("pair_id", "size"),
        core=("core", "sum"),
        events=("event_key", "nunique"),
        resolved=("outcome_agree", "count"),
        agree=("outcome_agree", "mean"),
        same_direction=("same_direction", "mean"),
    )
    log.info("pairs by family:\n%s", g.to_string())


# -- matched snapshots -----------------------------------------------------------------

MATCHED_SNAPSHOT_COLUMNS: Final[dict[str, str]] = {
    "pair_id": "string",
    "event_key": "string",
    "family": "string",
    "category": "string",
    "horizon": "string",
    "horizon_s": "Int64",
    "anchor_ts": "Int64",  # min of the two venues' scheduled times
    "t_snap": "Int64",
    "p_polymarket": "float64",
    "p_kalshi": "float64",  # in the Polymarket contract's direction
    "status_polymarket": "string",
    "status_kalshi": "string",
    "live_polymarket": "boolean",
    "live_kalshi": "boolean",
    "price_source_polymarket": "string",
    "price_source_kalshi": "string",
    "last_trade_age_s_polymarket": "float64",
    "p_polymarket_mid": "float64",
    "bid_kalshi": "float64",
    "ask_kalshi": "float64",
    "last_kalshi": "float64",
    "last_polymarket": "float64",
    "last_trade_age_s_kalshi": "float64",
    "n_moves_24h_polymarket": "Int64",
    "n_moves_24h_kalshi": "Int64",
    "outcome": "float64",  # the Polymarket contract's outcome; NaN when the venues disagree
    "outcome_kalshi_mapped": "float64",
    "volume_usd_polymarket": "float64",
    "volume_native_kalshi": "float64",
}
LIVE_TRADE_AGE_S: Final = DAY


def is_live(price_source: str, last_trade_age_s: float, n_moves_24h: int | None = None) -> bool:
    """The same rule as ``pmcal.panel``: a quoted mid within the spread limit (Kalshi),
    or a trade in the last 24 hours. A move in the price series alone does not count,
    because Polymarket's series is the raw book midpoint, which moves whenever anyone
    posts a far-off quote in a thin book. ``n_moves_24h`` is accepted and ignored."""
    del n_moves_24h
    if price_source == "mid":
        return True
    return math.isfinite(last_trade_age_s) and last_trade_age_s <= LIVE_TRADE_AGE_S


def polymarket_price(mid: float, last: float, last_trade_age_s: float, max_spread: float) -> float:
    """Polymarket price under the panel's rule: the midpoint, unless a trade in the last
    24 hours sits more than ``max_spread`` away from it (a wide book), then that trade."""
    traded = math.isfinite(last_trade_age_s) and last_trade_age_s <= LIVE_TRADE_AGE_S
    if traded and math.isfinite(last) and abs(mid - last) > max_spread:
        return last
    return mid


def _traded_age(path: PricePath, t: int) -> float:
    """Seconds from the end of the last interval with volume at or before ``t`` (Kalshi
    candles: hour granularity), NaN if none."""
    if path.volume is None or path.ts.size == 0:
        return math.nan
    i = int(np.searchsorted(path.ts, t, side="right"))
    idx = np.flatnonzero(np.nan_to_num(path.volume[:i]) > 0)
    return float(t - path.ts[idx[-1]]) if idx.size else math.nan


def _index_rows(path: Any, ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    want = sorted(set(ids))
    if not want:
        return {}
    df = pd.read_parquet(path, filters=[("contract_id", "in", want)])
    df = df.drop_duplicates("contract_id")
    return {str(r["contract_id"]): r for r in _records(df)}


def _pair_snapshots(
    m: Mapping[str, Any],
    prow: Mapping[str, Any],
    krow: Mapping[str, Any],
    cfg: Config,
    clients: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from pmcal.venues import kalshi, polymarket

    def cache_src(host: str, path: str, params: Any) -> tuple[bool, Any]:
        r = clients[host].cached(path, params)
        if r is None:
            return False, None
        return True, (r.data if 200 <= r.status < 300 else None)

    horizons = cfg.snapshots.horizon_seconds()
    p_sched, k_sched = _opt_int(prow.get("scheduled_ts")), _opt_int(krow.get("scheduled_ts"))
    if p_sched is None or k_sched is None:
        return []
    anchor = min(p_sched, k_sched)
    ppath = polymarket.load_price_path(prow, cfg)
    kpath = kalshi.load_price_path(krow, cfg, client=clients["kalshi"])

    def snaps(path: PricePath, row: Mapping[str, Any]) -> list[Snapshot]:
        return [
            snapshot_at(
                path,
                anchor - h,
                horizon=label,
                horizon_s=h,
                open_ts=_opt_int(row.get("open_ts")),
                close_ts=_opt_int(row.get("close_ts")),
                max_spread=cfg.snapshots.max_spread,
            )
            for label, h in horizons.items()
        ]

    psn = snaps(ppath, prow)
    if ppath.ts.size:
        # Trade prints are only in the cache for the windows the venue pipeline asked
        # for, i.e. when the common anchor is Polymarket's own scheduled time.
        psn, _ = polymarket.add_trade_stats(psn, polymarket.ContractRef.from_row(prow), cache_src)
    ksn = snaps(kpath, krow)
    same = bool(m["same_direction"])
    agree = m["outcome_agree"]
    p_out = m["polymarket_outcome"]
    outcome = math.nan if (pd.notna(agree) and float(agree) == 0.0) else p_out
    rows = []
    for ps, ks in zip(psn, ksn, strict=True):
        k_age = (
            _traded_age(kpath, ks.t_snap)
            if ks.status in ("ok", "too_old", "bad_price")
            else math.nan
        )
        p_k = ks.price if ks.status == "ok" else math.nan
        rows.append(
            {
                "pair_id": m["pair_id"],
                "event_key": m["event_key"],
                "family": m["family"],
                "category": m["category"],
                "horizon": ps.horizon,
                "horizon_s": ps.horizon_s,
                "anchor_ts": anchor,
                "t_snap": ps.t_snap,
                "p_polymarket": (
                    polymarket_price(
                        ps.price, ps.last, ps.last_trade_age_s, cfg.snapshots.max_spread
                    )
                    if ps.status == "ok"
                    else math.nan
                ),
                "p_polymarket_mid": ps.price if ps.status == "ok" else math.nan,
                "last_polymarket": ps.last,
                "p_kalshi": map_to_polymarket(p_k, same),
                # Kalshi's quote and last trade in the Polymarket direction (bid and ask
                # swap under the complement), so the report can apply pmcal.panel's
                # price rule to both sides of the pair.
                "bid_kalshi": map_to_polymarket(ks.bid if same else ks.ask, same),
                "ask_kalshi": map_to_polymarket(ks.ask if same else ks.bid, same),
                "last_kalshi": map_to_polymarket(ks.last, same),
                "status_polymarket": ps.status,
                "status_kalshi": ks.status,
                "live_polymarket": ps.status == "ok"
                and is_live(ps.price_source, ps.last_trade_age_s, ps.n_moves_24h),
                "live_kalshi": ks.status == "ok"
                and is_live(ks.price_source, k_age, ks.n_moves_24h),
                "price_source_polymarket": ps.price_source,
                "price_source_kalshi": ks.price_source,
                "last_trade_age_s_polymarket": ps.last_trade_age_s,
                "last_trade_age_s_kalshi": k_age,
                "n_moves_24h_polymarket": ps.n_moves_24h,
                "n_moves_24h_kalshi": ks.n_moves_24h,
                "outcome": outcome,
                "outcome_kalshi_mapped": m["kalshi_outcome_mapped"],
                "volume_usd_polymarket": prow.get("volume_usd"),
                "volume_native_kalshi": krow.get("volume_native"),
            }
        )
    return rows


def build_matched_snapshots(cfg: Config, *, workers: int = 6) -> pd.DataFrame:
    """One row per (pair, horizon), both venues priced at the same instant
    ``t_snap = min(scheduled times) - horizon``, from the cache only (works with
    ``PMCAL_OFFLINE=1``). Written to ``processed/matched_snapshots.parquet``.

    Each venue's snapshot follows ``pmcal.snapshots.snapshot_at`` with that contract's
    own open and close, so a pair can be "ok" on one venue and "closed" on the other.
    """
    from pmcal.venues import kalshi, polymarket

    matches = pd.read_parquet(cfg.interim_dir() / MATCHES_FILE)
    prows = _index_rows(
        cfg.interim_dir() / "polymarket_index.parquet", matches["polymarket_contract_id"]
    )
    krows = _index_rows(cfg.interim_dir() / "kalshi_index.parquet", matches["kalshi_contract_id"])
    recs = [
        r
        for r in matches.to_dict("records")
        if r["polymarket_contract_id"] in prows and r["kalshi_contract_id"] in krows
    ]
    clients = {
        "kalshi": kalshi.make_client(cfg),
        "clob": polymarket.clob_client(cfg),
        "data": polymarket.data_client(cfg),
    }
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            parts = list(
                pool.map(
                    lambda r: _pair_snapshots(
                        r,
                        prows[r["polymarket_contract_id"]],
                        krows[r["kalshi_contract_id"]],
                        cfg,
                        clients,
                    ),
                    recs,
                )
            )
    finally:
        for c in clients.values():
            c.close()
    rows = [row for part in parts for row in part]
    df = pd.DataFrame(rows, columns=list(MATCHED_SNAPSHOT_COLUMNS))
    df = df.astype(MATCHED_SNAPSHOT_COLUMNS)
    out = cfg.processed_dir() / SNAPSHOTS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    both = (df["status_polymarket"] == "ok") & (df["status_kalshi"] == "ok")
    log.info(
        "wrote %s: %d rows for %d pairs; both venues priced in %d rows (%d live on both)",
        out,
        len(df),
        df["pair_id"].nunique(),
        int(both.sum()),
        int((both & df["live_polymarket"] & df["live_kalshi"]).sum()),
    )
    return df


def _price_cache_exists(cfg: Config) -> bool:
    clob = cfg.raw_dir("polymarket") / "clob"
    return clob.is_dir() and any(clob.iterdir())


def run(cfg: Config, workers: int = 6) -> None:
    """``pmcal match``: find the pairs, then build their snapshots if any price data is
    cached (the downloads themselves are left to ``pmcal prices``, which includes the
    matched events once ``pmcal sample`` has been re-run)."""
    find_matches(cfg)
    if _price_cache_exists(cfg):
        build_matched_snapshots(cfg, workers=workers)
    else:
        log.info("no price data cached yet; run `pmcal sample` and `pmcal prices` first")
