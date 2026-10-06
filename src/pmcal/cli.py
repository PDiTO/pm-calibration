"""Command-line interface. Every stage reads the previous stage's files, and every
download goes through the on-disk cache, so any stage can be re-run cheaply.

pmcal index    --venue polymarket|kalshi|all   list the sampling frame (metadata only)
pmcal sample   --venue ...                     draw the event-level sample
pmcal prices   --venue ...                     download price histories for the sample
pmcal build    --venue ...                     contracts + snapshots tables, from cache only
pmcal match                                    pair the same proposition across venues
pmcal report   [--quick]                       tables, figures and report.md into results/
pmcal run      --venue ...                     index, sample, prices, build in one go

Set PMCAL_OFFLINE=1 to forbid network access (cache misses then raise).
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pandas as pd

from pmcal.config import Config, load_config
from pmcal.sample import draw_sample, forced_from_matches, summarise_sample
from pmcal.schema import VENUES

log = logging.getLogger("pmcal")


def _venue_module(venue: str) -> ModuleType:
    return importlib.import_module(f"pmcal.venues.{venue}")


def _venues(arg: str) -> list[str]:
    return list(VENUES) if arg == "all" else [arg]


def _forced_contracts(cfg: Config, venue: str) -> list[str]:
    """Contracts matched across venues (``pmcal match``), forced into the sample."""
    p = cfg.interim_dir() / "matches.parquet"
    if not p.exists():
        return []
    return forced_from_matches(pd.read_parquet(p), venue, seed=cfg.sample.seed)


def cmd_index(cfg: Config, venue: str, workers: int) -> None:
    mod = _venue_module(venue)
    t0 = time.time()
    idx = mod.build_index(cfg, workers=workers)
    log.info(
        "%s index: %d contracts, %d events in %.0fs",
        venue,
        len(idx),
        idx["event_id"].nunique(),
        time.time() - t0,
    )


def cmd_sample(cfg: Config, venue: str) -> pd.DataFrame:
    idx = pd.read_parquet(cfg.interim_dir() / f"{venue}_index.parquet")
    s = draw_sample(idx, cfg, force_contracts=_forced_contracts(cfg, venue))
    out = cfg.interim_dir() / f"{venue}_sample.parquet"
    s.to_parquet(out)
    log.info(
        "%s sample: %d contracts, %d events -> %s", venue, len(s), s["event_id"].nunique(), out
    )
    print(summarise_sample(idx, s, cfg).to_string())
    return s


def cmd_prices(cfg: Config, venue: str, workers: int, limit: int | None) -> None:
    s = pd.read_parquet(cfg.interim_dir() / f"{venue}_sample.parquet")
    if limit:
        s = s.sample(n=min(limit, len(s)), random_state=0)
    _venue_module(venue).fetch_prices(s, cfg, workers=workers)


def cmd_build(cfg: Config, venue: str) -> None:
    from pmcal.schema import CONTRACT_COLUMNS

    mod = _venue_module(venue)
    s = pd.read_parquet(cfg.interim_dir() / f"{venue}_sample.parquet")
    if "volume_missing" in s.columns and hasattr(mod, "fill_missing_volume"):
        # Contracts whose volume the venue did not report were kept in the frame;
        # their volume is now known from trade prints, so apply the floor.
        s = mod.fill_missing_volume(s, cfg)
        low = s["volume_usd"].astype(float).fillna(0) < cfg.frame.min_volume_usd
        log.info(
            "%s: dropping %d contracts under the volume floor after filling volume",
            venue,
            int(low.sum()),
        )
        s = s[~low].reset_index(drop=True)
    if hasattr(mod, "estimate_volume_usd"):
        # Kalshi reports contracts traded; dollar notional comes from the candles.
        s = mod.estimate_volume_usd(s, cfg)
    snaps = mod.build_snapshots(s, cfg)
    proc = cfg.processed_dir()
    proc.mkdir(parents=True, exist_ok=True)
    keep = [*CONTRACT_COLUMNS, "sample_weight", "forced", "stratum", "volume_missing"]
    s[[c for c in keep if c in s.columns]].to_parquet(proc / f"{venue}_contracts.parquet")
    snaps.to_parquet(proc / f"{venue}_snapshots.parquet")
    st = snaps[snaps["anchor"] == "scheduled"].groupby(["horizon", "status"]).size()
    log.info("%s snapshots: %d rows\n%s", venue, len(snaps), st.unstack(fill_value=0))


def cmd_match(cfg: Config, workers: int) -> None:
    from pmcal import match

    match.run(cfg, workers=workers)


def cmd_report(cfg: Config, out: Path, quick: bool) -> None:
    from pmcal.report import make_report

    path = make_report(cfg, out, quick=quick)
    log.info("wrote %s", path)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pmcal", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/main.toml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("index", "sample", "prices", "build", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--venue", choices=[*VENUES, "all"], default="all")
        sp.add_argument("--workers", type=int, default=6)
        sp.add_argument("--limit", type=int, default=None, help="prices: only N random contracts")
    sp = sub.add_parser("match")
    sp.add_argument("--workers", type=int, default=6)
    sp = sub.add_parser("report")
    sp.add_argument("--out", default="results")
    sp.add_argument("--quick", action="store_true", help="100 bootstrap draws instead of 1000")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_config(args.config)

    if args.cmd == "match":
        cmd_match(cfg, args.workers)
        return 0
    if args.cmd == "report":
        cmd_report(cfg, Path(args.out), args.quick)
        return 0
    for venue in _venues(args.venue):
        if args.cmd in ("index", "run"):
            cmd_index(cfg, venue, args.workers)
        if args.cmd in ("sample", "run"):
            cmd_sample(cfg, venue)
        if args.cmd in ("prices", "run"):
            cmd_prices(cfg, venue, args.workers, args.limit)
        if args.cmd in ("build", "run"):
            cmd_build(cfg, venue)
    return 0


if __name__ == "__main__":
    sys.exit(main())
