"""Synthetic contracts and snapshots in the processed-table schema, with a known
favourite-longshot bias, for testing the analysis and report end to end."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special, stats

from pmcal.schema import CONTRACT_COLUMNS, HORIZONS, SNAPSHOT_COLUMNS

DAY = 86400
T0 = 1_767_225_600  # 2026-01-01


def _shift(p: float, d: float) -> float:
    """Move a probability by ``d`` on the log-odds scale."""
    return float(special.expit(special.logit(np.clip(p, 1e-4, 1 - 1e-4)) + d))


def make_tables(
    venue: str, n_events: int = 600, seed: int = 0, flb_slope: float = 1.3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prices satisfy logit(true prob) = flb_slope * logit(price): with a slope
    above 1, cheap contracts are overpriced and expensive ones underpriced."""
    rng = np.random.default_rng(seed)
    cats = ["politics", "sports", "crypto", "economics", "culture"]
    c_rows = []
    s_rows = []
    for e in range(n_events):
        k = int(rng.choice([1, 1, 1, 2, 4]))
        sched = T0 + int(rng.integers(5, 260)) * DAY
        cat = cats[e % len(cats)]
        # Single contracts: the outcome is 1{S > 0} with S = mu + W(total), and the
        # true probability at horizon h is P(S > 0 | W up to h), an exact martingale.
        # Multi-outcome events: fixed Dirichlet probabilities, winner drawn from them.
        q = rng.dirichlet(np.ones(k) * 0.8) if k > 1 else None
        winner = int(rng.choice(k, p=q)) if q is not None else -1
        mu = rng.normal(0, 1.0)
        incs = rng.normal(0, 1.0, len(HORIZONS) + 1)  # one per horizon gap, coarse
        cum = np.cumsum(incs)  # info revealed by horizon index (30d first ... 1h last)
        total = cum[-1]
        y_single = float(mu + total > 0)
        for j in range(k):
            cid = f"{venue[:1]}{e}-{j}"
            y = float(j == winner) if q is not None else y_single
            open_ts = sched - int(rng.integers(2, 60)) * DAY
            early = rng.uniform() < 0.1
            close_ts = sched - 3 * DAY if early else sched + 3600
            c_rows.append(
                {
                    "venue": venue,
                    "contract_id": cid,
                    "market_id": cid,
                    "event_id": f"{venue[:1]}E{e}",
                    "cluster_id": f"{venue[:1]}E{e}",
                    "series_id": None,
                    "title": f"Synthetic {e} outcome {j}",
                    "event_title": f"Synthetic event {e}",
                    "outcome_label": "Yes",
                    "category": cat,
                    "venue_category": cat,
                    "n_in_event": k,
                    "mutually_exclusive": k > 1,
                    "open_ts": open_ts,
                    "close_ts": close_ts,
                    "scheduled_ts": sched,
                    "anchor_source": "end_date",
                    "event_start_ts": None,
                    "resolved_ts": close_ts + 3600,
                    "outcome": y,
                    "resolution_status": "resolved",
                    "volume_usd": float(10 ** rng.uniform(3, 7)),
                    "volume_native": 1.0,
                    "liquidity_usd": np.nan,
                    "is_short_crypto": False,
                }
            )
            for anchor in ("scheduled", "close"):
                a_ts = sched if anchor == "scheduled" else close_ts
                for h, hs in HORIZONS.items():
                    t = a_ts - hs
                    ok = open_ts <= t < close_ts
                    if q is not None:
                        true = float(q[j])
                    else:
                        hi = list(HORIZONS).index(h)  # 0 = 30d (least info)
                        known = mu + cum[hi]
                        remaining = len(incs) - hi - 1
                        true = float(stats.norm.cdf(known / np.sqrt(remaining)))
                    price = float(
                        special.expit(special.logit(np.clip(true, 1e-4, 1 - 1e-4)) / flb_slope)
                    )
                    s_rows.append(
                        {
                            "venue": venue,
                            "contract_id": cid,
                            "anchor": anchor,
                            "horizon": h,
                            "horizon_s": hs,
                            "t_snap": t,
                            "status": "ok" if ok else "closed",
                            "price": price if ok else np.nan,
                            "price_source": "mid" if venue == "kalshi" else "series",
                            "obs_ts": t if ok else None,
                            # Kalshi: a two-sided book 0.2 wide in log-odds around the
                            # price; Polymarket: a last trade at the price.
                            "bid": _shift(price, -0.1) if ok and venue == "kalshi" else np.nan,
                            "ask": _shift(price, 0.1) if ok and venue == "kalshi" else np.nan,
                            "last": price if ok else np.nan,
                            "volume_24h": np.nan,
                            "n_moves_24h": int(rng.integers(0, 5)),
                            "last_move_age_s": 600.0,
                            "n_trades_24h": int(rng.integers(0, 80)),
                            "last_trade_age_s": float(rng.exponential(20_000)),
                        }
                    )
    c = pd.DataFrame(c_rows).astype(CONTRACT_COLUMNS)
    c["sample_weight"] = 1.0
    c["forced"] = False
    s = pd.DataFrame(s_rows).astype(SNAPSHOT_COLUMNS)
    return c, s
