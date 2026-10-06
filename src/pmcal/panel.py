"""Join contracts and snapshots into the analysis panel, and count what each
selection rule removes.

The panel has one row per (contract, anchor, horizon). The filters applied
for the main analysis, in order, each with its reason:

1. ``resolution_status == "resolved"``. Void and 50-50 markets have no binary
   outcome. They are rare and are counted in the selection table.
2. ``status == "ok"``: the market was open at the snapshot time and a recent
   enough price existed (see ``pmcal.snapshots``).
3. ``confirmed``: the price at the snapshot was a tight two-sided market, judged
   on the log-odds scale so that "tight" means the same thing at 3c as at 50c.
   ``max_logit_spread`` = 0.402 is a 10c spread at even odds, about 2c at 5c and
   0.4c at 1c.
   - Kalshi: logit(ask) - logit(bid) <= 0.402. The price is the mid.
   - Polymarket publishes only the midpoint (no spread), and in a thin book it
     reads 0.50 at 0.01/0.99 and moves with any far-off quote. So a trade in the
     previous ``stale_s`` is required, and the midpoint must lie within half that
     width of the trade: |logit(mid) - logit(trade)| <= 0.201. The price is the mid.
   Rows that fail this but had a trade in the previous ``stale_s`` are priced at
   that trade (``price_rule == "last trade"``) and analysed separately: in thin
   markets the last print is often a buyer lifting a wide ask, which leans the
   sample towards an overpriced YES side. Rows with neither are stale.
   ``live`` = confirmed or last trade. ``price_abs`` / ``live_abs`` keep the first
   definition (an absolute 10c rule, Polymarket substituting the trade when the
   midpoint was 10c away), for the robustness table.

Short-duration crypto markets are outside the frame altogether.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pmcal.config import Config
from pmcal.schema import VENUES

DAY = 86400


@dataclass(frozen=True)
class PanelRules:
    stale_s: int = DAY  # a trade must have happened within this window before t_snap
    # Tight book: a 10c spread at even odds (45/55 is 0.4013 wide in log-odds), about
    # 2c at 5c and 0.4c at 1c.
    max_logit_spread: float = 0.402
    max_spread: float = 0.10  # the first, absolute definition (robustness only)
    near_certain: float = 0.02  # for the robustness cut that drops p < x or p > 1 - x


DEFAULT_RULES = PanelRules()


def processed_paths(cfg: Config, venue: str) -> tuple[str, str]:
    d = cfg.processed_dir()
    return str(d / f"{venue}_contracts.parquet"), str(d / f"{venue}_snapshots.parquet")


def load_venue(cfg: Config, venue: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    c_path, s_path = processed_paths(cfg, venue)
    return pd.read_parquet(c_path), pd.read_parquet(s_path)


def build_panel(
    contracts: pd.DataFrame, snapshots: pd.DataFrame, rules: PanelRules = DEFAULT_RULES
) -> pd.DataFrame:
    """All snapshots joined to their contract, with derived flags. No rows dropped."""
    keep_c = [
        c
        for c in (
            "venue",
            "contract_id",
            "event_id",
            "cluster_id",
            "series_id",
            "title",
            "event_title",
            "outcome_label",
            "category",
            "n_in_event",
            "mutually_exclusive",
            "open_ts",
            "close_ts",
            "scheduled_ts",
            "anchor_source",
            "resolved_ts",
            "outcome",
            "resolution_status",
            "volume_usd",
            "volume_native",
            "sample_weight",
            "forced",
        )
        if c in contracts.columns
    ]
    df = snapshots.merge(contracts[keep_c], on=["venue", "contract_id"], how="inner")
    for col in ("price", "outcome", "last_trade_age_s", "volume_usd"):
        df[col] = df[col].astype("float64")
    if "cluster_id" not in df.columns:
        df["cluster_id"] = df["event_id"]
    df["cluster_id"] = df["cluster_id"].fillna(df["event_id"]).astype("string")
    df["early_close"] = (df["close_ts"].astype("float64") < df["scheduled_ts"] - DAY).fillna(False)
    age = df["last_trade_age_s"].fillna(np.inf)
    last = df["last"].astype("float64")
    traded = (age <= rules.stale_s) & last.between(0.0, 1.0)
    pm = df["venue"] == "polymarket"
    snap_price = df["price"].copy()
    df["price_mid"] = snap_price

    # The first, absolute definition, kept for the robustness table.
    wide_abs = pm & traded & ((snap_price - last).abs() > rules.max_spread).fillna(False)
    df["price_abs"] = snap_price.where(~wide_abs, last)
    df["live_abs"] = (df["price_source"] == "mid") | traded

    price, rule = classify_prices(
        df["venue"], snap_price, df["bid"], df["ask"], last, df["last_trade_age_s"], rules
    )
    df["price"] = price
    df["price_rule"] = rule
    df["confirmed"] = rule == "quote"
    df["live"] = rule != "stale"
    df["resolved"] = df["resolution_status"] == "resolved"
    df["usable"] = df["resolved"] & (df["status"] == "ok") & df["price"].notna()
    if "sample_weight" not in df.columns:
        df["sample_weight"] = 1.0
    if "forced" not in df.columns:
        df["forced"] = False
    df["forced"] = df["forced"].fillna(False).astype(bool)
    return df


def classify_prices(
    venue: pd.Series,
    snap_price: pd.Series,
    bid: pd.Series,
    ask: pd.Series,
    last: pd.Series,
    last_trade_age_s: pd.Series,
    rules: PanelRules = DEFAULT_RULES,
) -> tuple[pd.Series, pd.Series]:
    """(price, rule) per row under the rule in the module docstring.

    ``rule`` is "quote" (a tight two-sided market; price = the mid), "last trade" (no
    tight quote but a trade in the last ``stale_s``; price = that trade) or "stale"
    (price = the snapshot price, unused in the main analysis). Also used for the
    cross-venue pairs, so both comparisons apply the same rule.
    """
    snap_price = snap_price.astype("float64")
    bid = bid.astype("float64")
    ask = ask.astype("float64")
    last = last.astype("float64")
    age = last_trade_age_s.astype("float64").fillna(np.inf)
    traded = (age <= rules.stale_s) & last.between(0.0, 1.0)
    km = venue == "kalshi"
    pm = venue == "polymarket"
    two_sided = km & (bid > 0) & (ask < 1) & (ask >= bid)
    k_quote = two_sided & ((_logit(ask) - _logit(bid)) <= rules.max_logit_spread).fillna(False)
    gap = (_logit(snap_price) - _logit(last)).abs()
    pm_quote = pm & traded & (gap <= rules.max_logit_spread / 2).fillna(False)
    quote = k_quote | pm_quote
    rule = pd.Series(
        np.select([quote, traded], ["quote", "last trade"], "stale"), index=venue.index
    )
    price = np.where(traded, last, snap_price)  # last trade, or the stale snapshot price
    price = np.where(pm_quote, snap_price, price)  # Polymarket: the confirmed midpoint
    price = np.where(k_quote, 0.5 * (bid + ask), price)  # Kalshi: the quoted mid
    return pd.Series(price, index=venue.index), rule


def _logit(p: pd.Series) -> pd.Series:
    q = p.astype("float64").clip(1e-4, 1 - 1e-4)
    return pd.Series(np.log(q) - np.log1p(-q), index=p.index)


def main_mask(df: pd.DataFrame, anchor: str = "scheduled") -> pd.Series:
    """Rows in the main per-venue analysis: confirmed quotes (see the module
    docstring). Events forced into the sample because they matched across venues
    are not a random draw, so they are left out here and used only in the
    cross-venue comparison."""
    return df["usable"] & df["confirmed"] & (df["anchor"] == anchor) & ~df["forced"]


def load_panel(cfg: Config, rules: PanelRules = DEFAULT_RULES) -> pd.DataFrame:
    parts = []
    for v in VENUES:
        try:
            c, s = load_venue(cfg, v)
        except FileNotFoundError:
            continue
        parts.append(build_panel(c, s, rules))
    if not parts:
        raise FileNotFoundError(f"no processed data under {cfg.processed_dir()}")
    return pd.concat(parts, ignore_index=True)


def selection_table(
    contracts: pd.DataFrame, panel: pd.DataFrame, horizons: tuple[str, ...]
) -> pd.DataFrame:
    """Contracts surviving each rule, per venue (scheduled anchor)."""
    rows = []
    for venue, c in contracts.groupby("venue"):
        p = panel[(panel["venue"] == venue) & (panel["anchor"] == "scheduled")]
        row: dict[str, object] = {
            "venue": venue,
            "sampled contracts": len(c),
            "sampled events": c["event_id"].nunique(),
        }
        for st in ("resolved", "void", "split", "unresolved"):
            row[st] = int((c["resolution_status"] == st).sum())
        row["early close (>1d before schedule)"] = int(
            (c["close_ts"].astype("float64") < c["scheduled_ts"].astype("float64") - DAY).sum()
        )
        for h in horizons:
            ph = p[(p["horizon"] == h) & p["resolved"]]
            row[f"{h}: open & priced"] = int((ph["status"] == "ok").sum())
            row[f"{h}: live (quote or trade)"] = int(((ph["status"] == "ok") & ph["live"]).sum())
            row[f"{h}: confirmed quote"] = int(((ph["status"] == "ok") & ph["confirmed"]).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("venue").T


def snapshot_status_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Snapshot status counts by venue and horizon, scheduled anchor, resolved contracts."""
    p = panel[(panel["anchor"] == "scheduled") & panel["resolved"]]
    t = p.groupby(["venue", "horizon", "status"]).size().unstack("status", fill_value=0)
    return t
