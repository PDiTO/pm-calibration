"""TOML run configuration.

Unknown keys are errors, so a typo cannot quietly fall back to a default. See
``configs/main.toml`` for the configuration behind the results in the README.
"""

from __future__ import annotations

import datetime as dt
import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from pmcal.schema import HORIZONS


def _utc_ts(day: str, end_of_day: bool = False) -> int:
    d = dt.datetime.fromisoformat(day).replace(tzinfo=dt.UTC)
    if end_of_day:
        d += dt.timedelta(days=1)
    return int(d.timestamp())


@dataclass(frozen=True)
class StudyConfig:
    # Markets enter the study if their *scheduled* resolution time falls in
    # [start, end]. Filtering on the scheduled time rather than the actual close
    # keeps long-dated markets that resolved early out of the sample unless their
    # not-yet-resolved peers are in it too (see README, "Selection").
    start: str = "2026-01-01"
    end: str = "2026-09-30"
    data_dir: str = "data"

    @property
    def start_ts(self) -> int:
        return _utc_ts(self.start)

    @property
    def end_ts(self) -> int:
        """Exclusive upper bound: midnight after ``end``."""
        return _utc_ts(self.end, end_of_day=True)

    @property
    def data_path(self) -> Path:
        env = os.environ.get("PMCAL_DATA_DIR")
        return Path(env).expanduser() if env else Path(self.data_dir)


@dataclass(frozen=True)
class FrameConfig:
    """Which markets are in the sampling frame."""

    # Contracts must have traded at least once (volume > 0) and have at least this
    # lifetime volume. The default of 0 means "traded at all"; see configs/main.toml
    # for why a higher lifetime floor biases calibration.
    min_volume_usd: float = 0.0
    exclude_short_crypto: bool = True


@dataclass(frozen=True)
class SampleConfig:
    """Event-level sample drawn from the frame.

    Recurring series (daily sports leagues, daily weather, daily index ranges) are
    capped at ``max_events_per_series_month`` events per series per calendar month,
    chosen at random with ``seed``. Events outside any series are all kept. Large
    events are subsampled to ``max_contracts_per_event`` contracts.
    """

    seed: int = 20261006
    max_events_per_series_month: int = 3
    # Within a sampled event keep at most this many contracts, chosen at random.
    # Multi-outcome events (candidate fields, strike ladders, "mentions") can have
    # dozens of contracts, each costing several requests. A random subset is
    # unbiased for per-contract calibration.
    max_contracts_per_event: int = 12
    # Per-venue overrides of the two caps, e.g. {"kalshi": {"max_events_per_series_month": 1}}.
    # Kalshi lists thousands of small series (one per team, player or city), so the
    # same caps would give it twice Polymarket's sample at a quarter of the request rate.
    venue: dict[str, dict[str, int]] = field(default_factory=dict)

    def for_venue(self, venue: str) -> SampleConfig:
        over = self.venue.get(venue, {})
        unknown = set(over) - {"max_events_per_series_month", "max_contracts_per_event"}
        if unknown:
            raise ValueError(f"unknown keys in [sample.venue.{venue}]: {sorted(unknown)}")
        kw: dict[str, Any] = dict(over)
        return replace(self, **kw) if kw else self


@dataclass(frozen=True)
class SnapshotConfig:
    horizons: tuple[str, ...] = tuple(HORIZONS)
    max_spread: float = 0.10

    def horizon_seconds(self) -> dict[str, int]:
        return {h: HORIZONS[h] for h in self.horizons}


@dataclass(frozen=True)
class Config:
    name: str = "main"
    study: StudyConfig = field(default_factory=StudyConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    snapshots: SnapshotConfig = field(default_factory=SnapshotConfig)
    analysis: dict[str, Any] = field(default_factory=dict)

    @property
    def data_path(self) -> Path:
        return self.study.data_path

    def raw_dir(self, venue: str) -> Path:
        return self.data_path / "raw" / venue

    def interim_dir(self) -> Path:
        return self.data_path / "interim"

    def processed_dir(self) -> Path:
        return self.data_path / "processed"


def _build(cls: type[Any], raw: dict[str, Any], where: str) -> Any:
    names = {f.name for f in fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown keys in [{where}]: {sorted(unknown)}")
    kwargs = dict(raw)
    for k, v in kwargs.items():
        if isinstance(v, list):
            kwargs[k] = tuple(v)
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    allowed = {"name", "study", "frame", "sample", "snapshots", "analysis"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown top-level keys: {sorted(unknown)}")
    snaps = raw.get("snapshots", {})
    for h in snaps.get("horizons", []):
        if h not in HORIZONS:
            raise ValueError(f"unknown horizon {h!r}; choose from {list(HORIZONS)}")
    return Config(
        name=raw.get("name", Path(path).stem),
        study=_build(StudyConfig, raw.get("study", {}), "study"),
        frame=_build(FrameConfig, raw.get("frame", {}), "frame"),
        sample=_build(SampleConfig, raw.get("sample", {}), "sample"),
        snapshots=_build(SnapshotConfig, snaps, "snapshots"),
        analysis=dict(raw.get("analysis", {})),
    )
