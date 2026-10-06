"""Event-level sample drawn from a venue's sampling frame.

The unit of sampling is the event, so a multi-outcome event is either in the
sample with all of its outcomes or not at all. Recurring series would otherwise
swamp the sample (a single tennis or weather series can list hundreds of events
a month), so each (series, calendar month) is capped at ``k`` events. Events
are ranked by a seeded hash of their id and the lowest ``k`` are kept. Using a
hash rather than a shuffle makes the draw independent of row order and stable
when the frame grows: adding events to a month never evicts an event that would
have been kept otherwise, unless it pushes it past the cap.

Events outside any series are all kept. Within a kept event with more than
``max_contracts_per_event`` contracts, that many are kept, again by a seeded
hash. The inclusion probability of each contract is stored (``sample_weight``
is its inverse), so population-weighted estimates are available alongside the
per-sample ones.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

from pmcal.config import Config


def _hash_rank(seed: int, key: str) -> int:
    h = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")


def frame_mask(index: pd.DataFrame, cfg: Config) -> pd.Series:
    """Rows of a venue index that belong to the sampling frame."""
    s = index["scheduled_ts"].astype("float64")
    m = (s >= cfg.study.start_ts) & (s < cfg.study.end_ts)
    # The floor is on dollar notional for Polymarket and on contracts traded for
    # Kalshi. Kalshi's dollar notional is only estimated later from candles, and
    # notional on the YES side grows with the YES price, so a dollar floor would
    # favour thin contracts that drifted towards YES. A contract count treats
    # both sides alike (one contract is one YES and one NO).
    usd = index["volume_usd"].astype("float64")
    native = index["volume_native"].astype("float64") if "volume_native" in index else usd
    vol = pd.Series(np.where(index["venue"] == "kalshi", native, usd), index=index.index)
    ok_vol = (vol > 0) & (vol >= cfg.frame.min_volume_usd)
    if "volume_missing" in index.columns:
        # The venue could not report volume for these (a Polymarket API gap); they
        # stay in the frame and the volume floor is applied after the price fetch.
        ok_vol |= index["volume_missing"].fillna(False).astype(bool)
    m &= ok_vol.fillna(False)
    # A market that opened at or after its scheduled time (in-play markets) can
    # never have a snapshot before it, so it only costs requests.
    m &= (index["open_ts"].astype("float64") < s - 3600).fillna(True)
    if cfg.frame.exclude_short_crypto:
        m &= ~index["is_short_crypto"].fillna(False).astype(bool)
    return m.fillna(False).astype(bool)


def forced_from_matches(
    matches: pd.DataFrame, venue: str, *, seed: int, max_pairs_per_key: int = 12
) -> list[str]:
    """Contract ids on ``venue`` to force into the sample for the cross-venue study.

    At most ``max_pairs_per_key`` pairs per matched real-world event (``event_key``),
    chosen by a seeded hash of the pair id, so that a golf tournament with hundreds
    of top-N pairs does not cost hundreds of downloads for one cluster. The same
    pairs are chosen for both venues.
    """
    m = matches.copy()
    m["_r"] = [_hash_rank(seed + 2, str(p)) for p in m["pair_id"]]
    m["_k"] = m.groupby("event_key")["_r"].rank(method="first")
    m = m[m["_k"] <= max_pairs_per_key]
    return m[f"{venue}_contract_id"].dropna().astype(str).unique().tolist()


def draw_sample(
    index: pd.DataFrame,
    cfg: Config,
    *,
    force_events: Iterable[str] = (),
    force_contracts: Iterable[str] = (),
) -> pd.DataFrame:
    """Contracts of the sampled events, with ``sample_weight`` = 1 / P(event sampled).

    ``force_contracts`` (contracts matched across venues) are added on top of the
    design sample, exempt from both caps, with ``forced = True`` unless the design
    drew them anyway. ``force_events`` does the same for whole events. Forced rows
    get weight 1 and are left out of the per-venue analysis (``pmcal.panel``),
    because matched propositions are not a random draw.
    """
    frame = index.loc[frame_mask(index, cfg)].copy()
    venues: list[str] = (
        [str(v) for v in frame["venue"].unique()] if "venue" in frame.columns else []
    )
    if len(venues) > 1:
        raise ValueError("draw_sample works on one venue's index at a time")
    scfg = cfg.sample.for_venue(str(venues[0])) if len(venues) else cfg.sample
    k = scfg.max_events_per_series_month
    seed = scfg.seed
    events = (
        frame.groupby("event_id", sort=False)
        .agg(series_id=("series_id", "first"), scheduled_ts=("scheduled_ts", "min"))
        .reset_index()
    )
    month = pd.to_datetime(events["scheduled_ts"].astype("int64"), unit="s", utc=True)
    events["month"] = month.dt.strftime("%Y-%m")
    events["rank"] = [_hash_rank(seed, str(e)) for e in events["event_id"]]
    has_series = events["series_id"].notna() & (events["series_id"].astype("string") != "")
    events["stratum"] = np.where(
        has_series, events["series_id"].astype("string") + "|" + events["month"], "<none>"
    )
    events["stratum_size"] = events.groupby("stratum")["event_id"].transform("size")
    events["rank_in_stratum"] = events.groupby("stratum")["rank"].rank(method="first")
    keep = ~has_series | (events["rank_in_stratum"] <= k)
    forced = events["event_id"].isin(set(force_events))
    events["p_sample"] = np.where(
        has_series, np.minimum(1.0, k / events["stratum_size"].astype(float)), 1.0
    )
    events["forced"] = forced & ~keep
    chosen = events.loc[keep | forced, ["event_id", "stratum", "p_sample", "forced"]]
    out = frame.merge(chosen, on="event_id", how="inner")
    # contract cap within large events
    m_cap = scfg.max_contracts_per_event
    out["n_event_frame"] = out.groupby("event_id")["contract_id"].transform("size")
    out["_rank"] = [_hash_rank(seed + 1, str(c)) for c in out["contract_id"]]
    out["_r"] = out.groupby("event_id")["_rank"].rank(method="first")
    out = out[out["_r"] <= m_cap].drop(columns=["_rank", "_r"])
    out["p_contract"] = np.minimum(1.0, m_cap / out["n_event_frame"].astype(float))
    out["sample_weight"] = 1.0 / (out["p_sample"] * out["p_contract"])
    fc = set(map(str, force_contracts)) - set(out["contract_id"].astype(str))
    if fc:
        extra = frame[frame["contract_id"].astype(str).isin(fc)].copy()
        extra["stratum"] = "<forced>"
        extra["p_sample"] = 1.0
        extra["p_contract"] = 1.0
        extra["forced"] = True
        extra["sample_weight"] = 1.0
        extra["n_event_frame"] = extra.groupby("event_id")["contract_id"].transform("size")
        out = pd.concat([out, extra[out.columns]], ignore_index=True)
    return out.reset_index(drop=True)


def summarise_sample(index: pd.DataFrame, sample: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Frame vs sample counts by category, for the data section of the report."""
    frame = index.loc[frame_mask(index, cfg)]
    f = frame.groupby("category").agg(
        frame_contracts=("contract_id", "size"), frame_events=("event_id", "nunique")
    )
    s = sample.groupby("category").agg(
        sample_contracts=("contract_id", "size"), sample_events=("event_id", "nunique")
    )
    return f.join(s, how="outer").fillna(0).astype(int).sort_values("frame_events", ascending=False)
