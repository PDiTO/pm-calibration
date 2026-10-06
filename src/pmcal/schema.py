"""The two tables every venue pipeline produces, and the canonical categories.

``contracts``: one row per binary contract (a YES/NO claim that pays 1 or 0).
A Polymarket market contributes one contract (its YES token, or for markets
whose outcomes are not Yes/No, one side chosen by a seeded hash). A Kalshi market
is one contract, priced on its YES side. Multi-outcome events contribute one
contract per outcome, all sharing ``event_id`` so the bootstrap can resample
them together.

``snapshots``: one row per (contract, anchor, horizon). The price of the
contract at ``t_snap = anchor_ts - horizon_s``, plus what is needed to judge
whether that price was live or stale.

Timestamps are integer Unix seconds (UTC). Prices are probabilities in [0, 1].
"""

from __future__ import annotations

from typing import Final

CATEGORIES: Final = (
    "politics",  # elections, government, policy, courts
    "sports",
    "crypto",  # crypto prices and crypto-native events
    "economics",  # macro data, central banks, rates
    "finance",  # equity indices, commodities, FX, company events
    "culture",  # entertainment, awards, media, celebrities, "mentions"
    "science_tech",  # AI, tech companies' products, space, science, health
    "weather",  # weather and climate
    "world",  # geopolitics and international affairs not tied to elections
    "other",
)

VENUES: Final = ("polymarket", "kalshi")

# column -> pandas dtype. "Int64" is pandas' nullable integer.
CONTRACT_COLUMNS: Final[dict[str, str]] = {
    "venue": "string",
    "contract_id": "string",  # unique within venue: Polymarket token id, Kalshi ticker
    "market_id": "string",  # Polymarket conditionId, Kalshi ticker
    "event_id": "string",  # sampling unit: Polymarket event id, Kalshi event ticker
    # Cluster for the bootstrap. The game when a venue splits one game over several
    # events (moneyline, spread, totals, props), otherwise the event.
    "cluster_id": "string",
    "series_id": "string",  # recurring series if any (nullable)
    "title": "string",  # market question
    "event_title": "string",
    "outcome_label": "string",  # which side the price refers to, e.g. "Yes" or a team name
    "category": "string",  # one of CATEGORIES
    "venue_category": "string",  # the venue's own label(s), for audit
    "n_in_event": "Int64",  # contracts (markets) in the event, after filters
    "mutually_exclusive": "boolean",  # exactly one outcome in the event can resolve YES
    "open_ts": "Int64",  # trading opened
    "close_ts": "Int64",  # trading actually stopped (early for markets that resolved early)
    "scheduled_ts": "Int64",  # ex-ante time the outcome is due; the horizon anchor
    "anchor_source": "string",  # where scheduled_ts came from, e.g. "game_start", "end_date"
    "event_start_ts": "Int64",  # start of the underlying game or event, if known
    "resolved_ts": "Int64",  # settlement time
    "outcome": "float64",  # 1.0 or 0.0; NaN unless resolution_status == "resolved"
    "resolution_status": "string",  # "resolved", "void", "split", "unresolved"
    "volume_usd": "float64",  # lifetime traded notional in USD (Kalshi: contracts x price)
    "volume_native": "float64",  # the venue's own volume figure (Kalshi: contracts)
    "liquidity_usd": "float64",  # venue's liquidity figure at listing time if any (nullable)
    "is_short_crypto": "boolean",  # 5m/15m/hourly crypto up-down style markets
}

SNAPSHOT_COLUMNS: Final[dict[str, str]] = {
    "venue": "string",
    "contract_id": "string",
    "anchor": "string",  # "scheduled" (ex-ante, the main analysis) or "close" (leaky, for contrast)
    "horizon": "string",  # label such as "7d"
    "horizon_s": "Int64",
    "t_snap": "Int64",
    "status": "string",  # see snapshots.STATUSES
    "price": "float64",  # the price used; NaN unless status == "ok"
    "price_source": "string",  # "mid", "last" or "series"
    "obs_ts": "Int64",  # timestamp of the observation the price came from
    "bid": "float64",
    "ask": "float64",
    "last": "float64",  # last trade price, where the venue gives one
    "volume_24h": "float64",  # native volume traded in the 24h up to t_snap, NaN if unknown
    "n_moves_24h": "Int64",  # number of price changes in the 24h up to t_snap
    "last_move_age_s": "float64",  # seconds since the price last changed
    # From trade prints where fetched (Polymarket data-api trades; Kalshi candles give
    # the last-trade age at candle granularity and no trade count).
    "n_trades_24h": "Int64",  # trades in the 24h up to t_snap (capped at the page size)
    "last_trade_age_s": "float64",  # seconds from the last trade at or before t_snap to t_snap
}

HORIZONS: Final[dict[str, int]] = {
    "30d": 30 * 86400,
    "14d": 14 * 86400,
    "7d": 7 * 86400,
    "3d": 3 * 86400,
    "1d": 86400,
    "6h": 6 * 3600,
    "1h": 3600,
}
