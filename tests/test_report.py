from __future__ import annotations

import json
from pathlib import Path

import pytest

from pmcal.config import Config, StudyConfig
from pmcal.report import make_report
from tests.synthetic import make_tables


def test_report_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PMCAL_DATA_DIR", str(tmp_path / "data"))
    cfg = Config(
        study=StudyConfig(),
        analysis={"n_boot": 30, "min_train": 300, "min_train_events": 60, "min_events": 20},
    )
    proc = cfg.processed_dir()
    proc.mkdir(parents=True)
    for i, v in enumerate(("polymarket", "kalshi")):
        c, s = make_tables(v, n_events=400, seed=10 + i)
        c.to_parquet(proc / f"{v}_contracts.parquet")
        s.to_parquet(proc / f"{v}_snapshots.parquet")
    out = make_report(cfg, tmp_path / "results")
    text = out.read_text()
    for section in (
        "1. Calibration",
        "2. Favourite-longshot",
        "3. Calibration and time",
        "5. Out-of-sample",
        "6. Robustness",
    ):
        assert section in text
    figs = {p.name for p in (tmp_path / "results" / "figures").glob("*.png")}
    assert {"reliability_1d.png", "longshot_returns_1d.png", "horizon_panel.png"} <= figs
    summary = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert "polymarket_1d" in summary["longshot_slope"]
    # the longshot slope is the as-listed fit, reported with its intercept
    assert "intercept" in summary["longshot_slope"]["polymarket_1d"]
    assert "polymarket_1d" in summary["longshot_slope_symmetrised"]
    tables = {p.name for p in (tmp_path / "results" / "tables").glob("*.csv")}
    assert "clustering_sensitivity_1d.csv" in tables
    assert "reliability (debiased)" in text
