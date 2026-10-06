"""Tables, figures and ``report.md`` from the processed panel.

``make_report`` is the only entry point. It reads ``data/processed`` (and
``data/interim`` for the frame counts), runs every analysis in
``pmcal.analysis``, and writes:

- ``results/tables/*.csv``: every number behind every figure;
- ``results/figures/*.png``;
- ``results/report.md``: the tables, formatted, in the order of the questions;
- ``results/summary.json``: the headline numbers quoted in the README.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pmcal import plots
from pmcal.analysis import (
    AnalysisConfig,
    activity_bucket,
    clustering_sensitivity,
    longshot_returns,
    nested_panel,
    paired_differences,
    recalibration,
    recluster,
    reliability_table,
    summarise,
    summarise_by,
    symmetrise,
    volume_bucket,
)
from pmcal.config import Config
from pmcal.panel import (
    DEFAULT_RULES,
    build_panel,
    classify_prices,
    load_venue,
    main_mask,
    selection_table,
    snapshot_status_table,
)
from pmcal.sample import frame_mask
from pmcal.schema import HORIZONS as HORIZON_S
from pmcal.schema import VENUES
from pmcal.stats import bin_edges

log = logging.getLogger(__name__)

VENUE_NAME = {"polymarket": "Polymarket", "kalshi": "Kalshi"}
DAY = 86400
HORIZON_ORDER = ("30d", "14d", "7d", "3d", "1d", "6h", "1h")


def ci(row: pd.Series | dict[str, Any], m: str, digits: int = 3, scale: float = 1.0) -> str:
    v, lo, hi = (row[m] * scale, row[f"{m}_lo"] * scale, row[f"{m}_hi"] * scale)
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |"]
    out.append("| " + " | ".join(":---" if i == 0 else "---:" for i in range(len(cols))) + " |")
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float | np.floating):
                cells.append("" if not np.isfinite(v) else floatfmt.format(v))
            elif isinstance(v, int | np.integer):
                cells.append(f"{int(v):,}")
            else:
                cells.append("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


class Report:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.tables = out / "tables"
        self.figures = out / "figures"
        self.tables.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.lines: list[str] = []
        self.summary: dict[str, Any] = {}

    def h(self, level: int, text: str) -> None:
        self.lines += ["", "#" * level + " " + text, ""]

    def p(self, text: str) -> None:
        self.lines += [text, ""]

    def table(self, name: str, df: pd.DataFrame, shown: pd.DataFrame | None = None) -> None:
        df.to_csv(self.tables / f"{name}.csv", index=False)
        self.lines += [md_table(df if shown is None else shown), ""]

    def figure(self, path: Path, alt: str) -> None:
        rel = path.relative_to(self.out)
        self.lines += [f"![{alt}]({rel.as_posix()})", ""]

    def write(self, title: str) -> None:
        text = "# " + title + "\n" + "\n".join(self.lines).rstrip() + "\n"
        (self.out / "report.md").write_text(text)
        (self.out / "summary.json").write_text(json.dumps(self.summary, indent=2, default=_json))


def _json(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o))


def _metric_table(t: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for _, r in t.iterrows():
        row: dict[str, Any] = {k: r[k] for k in keys}
        row["contracts"] = int(r["n"])
        row["clusters"] = int(r["clusters"])
        row["base rate"] = f"{r['base_rate']:.3f}"
        row["mean price"] = f"{r['mean_price']:.3f}"
        row["Brier"] = ci(r, "brier", 4)
        if "reliability_db" in r:
            row["reliability (debiased)"] = ci(r, "reliability_db", 4)
        else:
            row["reliability"] = ci(r, "reliability", 4)
        row["resolution"] = ci(r, "resolution", 4)
        if "ece_bc" in r:
            row["ECE plug-in"] = f"{r['ece']:.4f}"
            row["ECE bias-corr."] = f"{r['ece_bc']:.4f} [{r['ece_lo']:.4f}, {r['ece_hi']:.4f}]"
        else:
            row["ECE"] = ci(r, "ece", 4)
        if "ece_null" in r:
            row["ECE if calibrated"] = f"{r['ece_null']:.4f}"
        row["log loss"] = ci(r, "log_loss", 4)
        row["slope b"] = _slope_ci(r, "slope", 2)
        row["intercept a"] = _slope_ci(r, "intercept", 2)
        rows.append(row)
    return pd.DataFrame(rows)


SEPARATION_MAX = 0.05  # largest share of separated bootstrap fits for which b is shown


def _slope_ci(row: pd.Series | dict[str, Any], m: str, digits: int) -> str:
    """The slope or intercept with its interval, or a note when too many bootstrap
    replicates separated (the interval would then be conditional on not separating)."""
    fail = row.get("slope_boot_fail", 0.0)
    if fail is not None and np.isfinite(fail) and fail > SEPARATION_MAX:
        return f"unstable ({100 * fail:.0f}% of fits separate)"
    return ci(row, m, digits)


def _event_weight(df: pd.DataFrame) -> pd.Series:
    """1 / contracts of the same event in ``df``: every event counts once in ``df``
    (computed on the subset analysed, so partly covered events are not down-weighted)."""
    n = df.groupby(["venue", "event_id", "anchor", "horizon"])["contract_id"].transform("size")
    return 1.0 / n.astype(float)


def _order_h(hs: Any) -> list[str]:
    s = set(hs)
    return [h for h in HORIZON_ORDER if h in s]


def make_report(cfg: Config, out: Path, *, quick: bool = False) -> Path:
    acfg = AnalysisConfig.from_dict(cfg.analysis)
    if quick:
        acfg = AnalysisConfig(**{**acfg.__dict__, "n_boot": 100})
    rules = DEFAULT_RULES
    rep = Report(out)

    contracts_l, panels = [], []
    for v in VENUES:
        try:
            c, s = load_venue(cfg, v)
        except FileNotFoundError:
            log.warning("no processed data for %s", v)
            continue
        contracts_l.append(c)
        panels.append(build_panel(c, s, rules))
    contracts = pd.concat(contracts_l, ignore_index=True)
    panel = pd.concat(panels, ignore_index=True)
    panel["game_cluster_id"] = panel["cluster_id"]
    if acfg.cluster_by:
        panel["cluster_id"] = recluster(panel, acfg.cluster_by)
    panel["event_weight"] = _event_weight(panel)
    horizons = _order_h(panel["horizon"].unique())
    H = acfg.headline_horizon
    main = panel[main_mask(panel)].copy()
    edges10 = bin_edges(acfg.n_bins)
    venues = [v for v in VENUES if v in set(main["venue"])]

    # ------------------------------------------------------------------ data
    rep.h(2, "Data and selection")
    frame_rows = []
    for v in venues:
        ip = cfg.interim_dir() / f"{v}_index.parquet"
        if ip.exists():
            idx = pd.read_parquet(ip)
            fr = idx[frame_mask(idx, cfg)]
            frame_rows.append(
                {
                    "venue": VENUE_NAME[v],
                    "frame contracts": len(fr),
                    "frame events": fr["event_id"].nunique(),
                    "sampled contracts": int((contracts["venue"] == v).sum()),
                    "sampled events": contracts.loc[contracts["venue"] == v, "event_id"].nunique(),
                }
            )
    if frame_rows:
        rep.table("frame_vs_sample", pd.DataFrame(frame_rows))
    by_cat = (
        contracts.groupby(["category", "venue"])["event_id"]
        .nunique()
        .unstack("venue", fill_value=0)
        .reset_index()
    )
    rep.p("Sampled events by category:")
    rep.table("sample_by_category", by_cat)
    sel = selection_table(contracts, panel, tuple(horizons)).reset_index(names="step")
    rep.p("Contracts surviving each rule (scheduled anchor):")
    rep.table("selection", sel)
    st = snapshot_status_table(panel).reset_index()
    rep.p("Snapshot status by horizon, resolved contracts, scheduled anchor:")
    rep.table("snapshot_status", st)

    # ------------------------------------------------------------- 1. calibration
    rep.h(2, f"1. Calibration ({H} before scheduled resolution unless stated)")
    t_all = summarise_by(main, ["venue", "horizon"], acfg)
    t_all["horizon"] = pd.Categorical(t_all["horizon"], HORIZON_ORDER, ordered=True)
    t_all = t_all.sort_values(["venue", "horizon"])
    t_all.to_csv(rep.tables / "calibration_by_venue_horizon.csv", index=False)
    head = t_all[t_all["horizon"] == H]
    rep.table(f"calibration_{H}", head, _metric_table(head, ["venue", "horizon"]))
    rel_tabs = {
        v: reliability_table(main[(main["venue"] == v) & (main["horizon"] == H)], edges10)
        for v in venues
    }
    for v, t in rel_tabs.items():
        t.to_csv(rep.tables / f"reliability_{v}_{H}.csv", index=False)
    fig = plots.reliability(
        rel_tabs,
        f"Reliability, {H} before scheduled resolution",
        rep.figures / f"reliability_{H}.png",
        caption="Bars: 95% intervals with cluster-adjusted effective sample size. "
        "Diagonal: perfect calibration. Lower panels: contracts per price bin (log scale).",
    )
    rep.figure(fig, "Reliability diagrams")
    rep.p(
        "Reliability (debiased) subtracts each bin's cluster-robust sampling variance, so "
        "it is about zero for calibrated prices and its interval is the one to read for "
        "'is the miscalibration larger than noise'. The plug-in ECE is biased upwards; its "
        "bias-corrected value and basic interval are shown, but near zero that interval is "
        "too narrow (in simulation it covered a true ECE of 0 about half the time), so it "
        "is descriptive. 'ECE if calibrated' is the mean ECE of outcomes simulated as "
        "independent draws at the quoted prices."
    )
    rep.p("All horizons (each horizon on its own set of contracts):")
    rep.table("calibration_all_horizons", t_all, _metric_table(t_all, ["venue", "horizon"]))
    rep.summary["calibration"] = {
        f"{r['venue']}_{r['horizon']}": {k: r[k] for k in r.index if k not in ("venue", "horizon")}
        for _, r in t_all.iterrows()
    }

    # ------------------------------------------------------------- 2. longshot
    rep.h(2, "2. Favourite-longshot bias")
    rep.p(
        "Slope b and intercept a from logit P(YES) = a + b logit(price), fitted on the "
        "contracts as listed. b > 1: prices too close to 0.5 (cheap contracts overpriced, "
        "expensive ones underpriced, around the point where the fitted curve crosses the "
        "diagonal); a < 0: the listed (YES) side overpriced at all prices. Fitting a and b "
        "together is what separates the two. The symmetrised slope (every contract and its "
        "complement) forces a = 0, and with listed prices skewed towards cheap YES "
        "contracts it absorbs a YES lean as an apparent longshot bias, so it is shown only "
        "for comparison."
    )
    sym = symmetrise(main)
    t_sym = summarise_by(sym, ["venue", "horizon"], acfg)
    t_sym["horizon"] = pd.Categorical(t_sym["horizon"], HORIZON_ORDER, ordered=True)
    t_sym = t_sym.sort_values(["venue", "horizon"])
    flb = pd.DataFrame(
        {
            "venue": t_all["venue"],
            "horizon": t_all["horizon"].astype(str),
            "contracts": t_all["n"].astype(int),
            "slope b": [_slope_ci(r, "slope", 3) for _, r in t_all.iterrows()],
            "slope cluster-robust se": t_all["slope_se_cluster"].round(3),
            "intercept a": [_slope_ci(r, "intercept", 3) for _, r in t_all.iterrows()],
        }
    ).merge(
        pd.DataFrame(
            {
                "venue": t_sym["venue"],
                "horizon": t_sym["horizon"].astype(str),
                "symmetrised slope (a forced to 0)": [
                    _slope_ci(r, "slope", 3) for _, r in t_sym.iterrows()
                ],
            }
        ),
        on=["venue", "horizon"],
    )
    t_sym.to_csv(rep.tables / "longshot_slope_symmetrised.csv", index=False)
    rep.table("longshot_slopes", flb)
    rep.summary["longshot_slope"] = {
        f"{r['venue']}_{r['horizon']}": {
            "slope": r["slope"],
            "lo": r["slope_lo"],
            "hi": r["slope_hi"],
            "intercept": r["intercept"],
            "intercept_lo": r["intercept_lo"],
            "intercept_hi": r["intercept_hi"],
        }
        for _, r in t_all.iterrows()
    }
    # The same fit by how the price was formed, at the headline horizon. Liquid
    # markets (a tight quote) and thin ones (only a last trade) can lean in opposite
    # directions, so the pooled numbers are a mix.
    pr_rows = panel[
        panel["usable"]
        & panel["live"]
        & (panel["anchor"] == "scheduled")
        & ~panel["forced"]
        & (panel["horizon"] == H)
    ]
    by_rule = summarise_by(pr_rows, ["venue", "price_rule"], acfg)
    if not by_rule.empty:
        rep.p(
            f"By how the price was formed, {H} out. 'quote': a tight two-sided market "
            "(the main analysis). 'last trade': no tight quote, priced at a trade in the "
            "previous 24h."
        )
        rep.table(f"by_price_rule_{H}", by_rule, _metric_table(by_rule, ["venue", "price_rule"]))
        rep.summary["by_price_rule"] = by_rule.to_dict(orient="records")
    rep.summary["longshot_slope_symmetrised"] = {
        f"{r['venue']}_{r['horizon']}": {
            "slope": r["slope"],
            "lo": r["slope_lo"],
            "hi": r["slope_hi"],
        }
        for _, r in t_sym.iterrows()
    }
    fine = bin_edges(scheme="fine_tails")
    ls_tabs = {}
    for v in venues:
        sv = sym[(sym["venue"] == v) & (sym["horizon"] == H)]
        ls = longshot_returns(sv, fine, acfg)
        ls_tabs[v] = ls
        ls.assign(venue=v).to_csv(rep.tables / f"longshot_returns_{v}_{H}.csv", index=False)
        shown = pd.DataFrame(
            {
                "price bin": [
                    f"{a:.2f}-{b:.2f}" for a, b in zip(ls["lo_edge"], ls["hi_edge"], strict=True)
                ],
                "contracts": ls["n"],
                "clusters": ls["clusters"],
                "mean price": ls["mean_price"],
                "return per $1 at the price": [
                    f"{100 * r:+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]" if np.isfinite(r) else ""
                    for r, lo, hi in zip(
                        ls["return"], ls["return_lo"], ls["return_hi"], strict=True
                    )
                ],
            }
        )
        rep.p(
            f"{VENUE_NAME[v]}, {H} out: total payout over total cost of buying one contract "
            "of each, at the recorded price (the mid, or the last trade when the book is "
            "wide), before spread and fees. Each bin mirrors the bin at 1 - price "
            "(the same contracts bought on the other side), so the two halves of the "
            "table are not independent evidence."
        )
        rep.table(f"longshot_returns_{v}", ls, shown)
    fig = plots.longshot(
        ls_tabs,
        f"Return per $1 by price paid, {H} before resolution",
        rep.figures / f"longshot_returns_{H}.png",
        caption="Both sides of every contract (buying YES at p and NO at 1 - p), at the "
        "recorded price. Bars: 95% cluster bootstrap intervals. No spread or fees.",
    )
    rep.figure(fig, "Favourite-longshot returns")

    # ------------------------------------------------------------- 3. horizons
    rep.h(2, "3. Calibration and time to resolution")
    ph = tuple(h for h in acfg.panel_horizons if h in horizons)
    # The population is fixed by a confirmed quote at the longest horizon. Later
    # horizons are scored with any live price (a confirmed quote or a trade in the
    # previous 24h): requiring a tight quote throughout would drop the contracts whose
    # prices head towards 0 or 1, where a 1c tick cannot make a tight book.
    live_rows = panel[
        panel["usable"] & panel["live"] & (panel["anchor"] == "scheduled") & ~panel["forced"]
    ]
    horizon_df = pd.concat(
        [main[main["horizon"] == ph[0]], live_rows[live_rows["horizon"].isin(ph[1:])]],
        ignore_index=True,
    )
    bal = nested_panel(horizon_df, ph)
    rep.p(
        f"Contracts with a confirmed quote {ph[0]} before scheduled resolution, scored "
        f"again at {', '.join(ph[1:])} with any live price (a confirmed quote or a trade in "
        "the previous 24h) while they were still open. The population is fixed "
        f"by what was known {ph[0]} out and shrinks only by what was known at each later "
        "horizon. (Requiring a live price at every horizon would keep only markets that "
        "did not resolve early, which for 'by date' markets selects on the outcome and "
        "makes the long-horizon prices look worse than they were.) Differences are against "
        "the longest horizon, with paired cluster intervals."
    )
    diffs = []
    long_rows = []
    for v in venues:
        b = bal[bal["venue"] == v]
        if b["cluster_id"].nunique() < acfg.min_events:
            continue
        d = paired_differences(
            b,
            "horizon",
            list(ph[1:]),
            ph[0],
            acfg,
            metrics=("brier", "ece", "log_loss", "slope", "intercept"),
            balanced=False,
        )
        d.insert(0, "venue", v)
        d["contracts"] = d["n"]
        d["clusters"] = d["level_clusters"]
        diffs.append(d)
        for h in ph:
            r = {"venue": v, "horizon": h}
            for met in ("brier", "ece", "slope", "log_loss"):
                x = d[(d["horizon"] == h) & (d["metric"] == met)].iloc[0]
                r[met], r[f"{met}_lo"], r[f"{met}_hi"] = x["value"], x["lo"], x["hi"]
            long_rows.append(r)
    if diffs:
        dd = pd.concat(diffs, ignore_index=True)
        dd = dd.reindex(
            columns=[
                *dd.columns.drop(["diff_vs_base", "diff_lo", "diff_hi"], errors="ignore"),
                "diff_vs_base",
                "diff_lo",
                "diff_hi",
            ]
        )
        shown = dd.assign(
            value=[
                f"{a:.4f} [{b:.4f}, {c:.4f}]"
                for a, b, c in zip(dd["value"], dd["lo"], dd["hi"], strict=True)
            ],
            diff=[
                f"{a:+.4f} [{b:+.4f}, {c:+.4f}]" if np.isfinite(a) else ""
                for a, b, c in zip(
                    dd["diff_vs_base"],
                    dd["diff_lo"],
                    dd["diff_hi"],
                    strict=True,
                )
            ],
        )[["venue", "horizon", "metric", "contracts", "clusters", "value", "diff"]]
        rep.table("horizon_panel", dd, shown)
        lr = pd.DataFrame(long_rows)
        fig = plots.metric_by_level(
            lr,
            "horizon",
            ph,
            [("brier", "Brier score"), ("ece", "ECE"), ("slope", "calibration slope b")],
            "Same contracts, priced at different times before resolution",
            rep.figures / "horizon_panel.png",
            ref_lines={"slope": 1.0},
            caption=f"Contracts priced {ph[0]} out, at each later horizon while open. "
            "Intervals: 95% cluster bootstrap.",
        )
        rep.figure(fig, "Metrics by horizon")
        rep.summary["horizon_panel"] = lr.to_dict(orient="records")
        rbh = {
            v: {
                h: reliability_table(bal[(bal["venue"] == v) & (bal["horizon"] == h)], edges10)
                for h in ph
            }
            for v in venues
            if (bal["venue"] == v).any()
        }
        fig = plots.reliability_by_horizon(
            rbh,
            f"Reliability by time before resolution (contracts priced {ph[0]} out)",
            rep.figures / "reliability_by_horizon.png",
        )
        rep.figure(fig, "Reliability by horizon")

    # ------------------------------------------------------------- 4. breakdowns
    rep.h(2, f"4. Category, liquidity and venue ({H})")
    mh = main[main["horizon"] == H].copy()
    cat = summarise_by(mh, ["venue", "category"], acfg)
    cat.to_csv(rep.tables / f"by_category_{H}.csv", index=False)
    rep.table(f"by_category_{H}", cat, _metric_table(cat, ["venue", "category"]))
    rep.p(
        "Slope b and intercept a are fitted together on the contracts as listed. With "
        "this many categories, horizons and venues some intervals will exclude the "
        "reference value by chance (about 1 in 20 when nothing differs)."
    )
    order = (
        mh.groupby("category")["cluster_id"].nunique().sort_values(ascending=False).index.tolist()
    )
    order = [c for c in order if c in set(cat["category"])]
    if order:
        fig = plots.dot_rows(
            cat,
            "category",
            order,
            [("brier", "Brier score"), ("ece", "ECE"), ("slope", "slope b")],
            f"By category, {H} before resolution",
            rep.figures / f"by_category_{H}.png",
            ref_lines={"slope": 1.0},
            caption=f"Categories with at least {acfg.min_events} events per venue. "
            "Intervals: 95% cluster bootstrap.",
        )
        rep.figure(fig, "By category")

    mh["volume bucket"] = volume_bucket(mh["volume_usd"], acfg.volume_buckets)
    mh["activity"] = activity_bucket(mh)
    for col, name, note in (
        (
            "activity",
            "by_activity",
            "Trading in the 24h before the snapshot (known at the "
            "time). busy: at least 50 trades (Polymarket) or 1,000 contracts traded (Kalshi, "
            "which reports no trade count), so 'busy' is not the same bar on the two venues; "
            "traded: at least one trade; "
            "no trade: price confirmed only by a two-sided quote within 10c (Kalshi only; "
            "Polymarket requires a trade).",
        ),
        (
            "volume bucket",
            "by_volume",
            "Lifetime volume. This includes trading after the "
            "snapshot, so it is descriptive only: markets where something surprising happened "
            "attract volume, so high-volume buckets over-represent upsets (b pushed below 1) "
            "and low-volume buckets under-represent them. Kalshi's dollar volume is contracts "
            "times the YES-side average trade price, which a YES winner's late trades near $1 "
            "inflate, so its buckets also sort on the outcome directly.",
        ),
    ):
        t = summarise_by(mh.dropna(subset=[col]), ["venue", col], acfg)
        rep.p(note)
        rep.table(f"{name}_{H}", t, _metric_table(t, ["venue", col]))

    mpath = cfg.processed_dir() / "matched_snapshots.parquet"
    if mpath.exists():
        rep.h(3, "Same event on both venues")
        m = pd.read_parquet(mpath)
        _cross_venue(rep, m, acfg)

    # ------------------------------------------------------------- 5. recalibration
    rep.h(2, "5. Out-of-sample recalibration")
    rep.p(
        f"Walk-forward, refitted every {acfg.refit_every_days} days on contracts already "
        "resolved at the refit time, applied only to snapshots taken after it. Fitted "
        "separately per venue and horizon. Gain = raw score minus recalibrated score "
        "(positive means recalibration helped), with paired cluster intervals."
    )
    rc_df = main[main["horizon"].isin(horizons)]
    scores, refits = recalibration(rc_df, acfg)
    if not scores.empty:
        refits.to_csv(rep.tables / "recalibration_refits.csv", index=False)
        scores["horizon"] = pd.Categorical(scores["horizon"], HORIZON_ORDER, ordered=True)
        scores = scores.sort_values(["venue", "score", "horizon", "method"])
        shown = scores.assign(
            gain=[
                f"{g:+.5f} [{lo:+.5f}, {hi:+.5f}]"
                for g, lo, hi in zip(
                    scores["gain"], scores["gain_lo"], scores["gain_hi"], strict=True
                )
            ],
            rel=[f"{x:+.2f}%" for x in scores["rel_gain_pct"]],
        )[
            [
                "venue",
                "horizon",
                "method",
                "score",
                "n_test",
                "events_test",
                "raw",
                "recalibrated",
                "gain",
                "rel",
            ]
        ]
        rep.table("recalibration", scores, shown)
        fig = plots.recalibration_gains(
            scores,
            "Out-of-sample Brier improvement from recalibration",
            rep.figures / "recalibration.png",
            caption="Walk-forward by information time. Above zero: the recalibrated "
            "forecast beat the raw price. Intervals: 95% paired cluster bootstrap.",
        )
        rep.figure(fig, "Recalibration gains")
        rep.summary["recalibration"] = scores.assign(horizon=scores["horizon"].astype(str)).to_dict(
            orient="records"
        )

    # ------------------------------------------------------------- 6. robustness
    rep.h(2, f"6. Robustness ({H})")
    rob = []

    def add(label: str, df: pd.DataFrame, weight: str | None = None) -> None:
        for v in venues:
            g = df[(df["venue"] == v) & (df["horizon"] == H)]
            if g["cluster_id"].nunique() < acfg.min_events:
                continue
            if weight == "event_weight":
                g = g.assign(event_weight=_event_weight(g))
            s = summarise(g, acfg, weight=weight)
            ss = summarise(symmetrise(g), acfg, weight=weight)
            rob.append(
                {
                    "variant": label,
                    "venue": v,
                    **s,
                    "slope_sym": ss["slope"],
                    "slope_sym_lo": ss["slope_lo"],
                    "slope_sym_hi": ss["slope_hi"],
                }
            )

    add("main", main)
    forced_rows = panel["usable"] & panel["confirmed"] & (panel["anchor"] == "scheduled")
    if (forced_rows & panel["forced"]).any():
        # Events forced into the sample because they matched across venues are not a
        # random draw (high-profile propositions listed on both venues), so the main
        # analysis leaves them out. This row adds them back.
        add("including events forced in by cross-venue matching", panel[forced_rows])
    usable = panel[panel["usable"] & (panel["anchor"] == "scheduled") & ~panel["forced"]]
    # How the price is defined. Each of these is decided with what was known at the
    # snapshot; they change which markets are scored and at what price.
    last_rows = usable[usable["price_rule"] == "last trade"]
    add("last-trade prices only (thin markets, no tight quote)", last_rows)
    add("quotes and last trades together (all live prices)", usable[usable["live"]])
    add(
        "first definition: 10c absolute spread, Polymarket trade swapped in at 10c",
        usable[usable["live_abs"]].assign(price=usable["price_abs"]),
    )
    pm_traded = usable[(usable["venue"] == "polymarket") & usable["live"]]
    add("Polymarket midpoint on every traded row", pm_traded.assign(price=pm_traded["price_mid"]))
    add("stale prices included (midpoint or last print)", usable.assign(price=usable["price_mid"]))
    add("each event weighted once", main, weight="event_weight")
    add(
        "population-weighted (inverse sampling prob.)",
        main[~main["forced"]],
        weight="sample_weight",
    )
    # Split on the event's structure, known when it was listed. ``n_in_event`` counts
    # the event's markets that passed the lifetime-volume floor, which depends on
    # trading after the snapshot.
    me = main["mutually_exclusive"].fillna(False).astype(bool)
    add("mutually exclusive (one-of-k) events only", main[me])
    add("other markets only", main[~me])
    nc = rules.near_certain
    add(f"prices in [{nc}, {1 - nc}] only", main[main["price"].between(nc, 1 - nc)])
    floor_hi = 10_000.0
    add(
        f"lifetime volume >= ${floor_hi:,.0f} (selects on later trading)",
        main[main["volume_usd"] >= floor_hi],
    )
    traded = main[main["last"].notna() & (main["last_trade_age_s"] <= 3600)]
    add("traded in the hour before: recorded price", traded)
    add(
        "traded in the hour before: last trade price",
        traded.assign(price=traded["last"].clip(0.0, 1.0)),
    )
    add("markets that closed early only", main[main["early_close"]])
    # Polymarket trade prints were fetched around the scheduled-anchor snapshots only,
    # so the close-anchored rows are compared without the liveness filter, against
    # scheduled-anchor rows also without it.
    close = panel[panel["usable"] & (panel["anchor"] == "close") & ~panel["forced"]]
    add("anchored on actual close (leaky; no liveness filter)", close)
    add(
        "anchored on actual close, early closers (leaky; no liveness filter)",
        close[close["early_close"]],
    )
    if rob:
        rb = pd.DataFrame(rob)
        shown = pd.DataFrame(
            {
                "variant": rb["variant"],
                "venue": rb["venue"],
                "contracts": rb["n"].astype(int),
                "clusters": rb["clusters"].astype(int),
                "base rate": rb["base_rate"].round(3),
                "mean price": rb["mean_price"].round(3),
                "Brier": [ci(r, "brier", 4) for _, r in rb.iterrows()],
                "ECE bias-corr.": [
                    f"{r['ece_bc']:.4f} [{r['ece_lo']:.4f}, {r['ece_hi']:.4f}]"
                    for _, r in rb.iterrows()
                ],
                "slope b": [_slope_ci(r, "slope", 2) for _, r in rb.iterrows()],
                "intercept a": [_slope_ci(r, "intercept", 2) for _, r in rb.iterrows()],
            }
        )
        rep.table("robustness", rb, shown)
        rep.summary["robustness"] = rb.to_dict(orient="records")
    if HORIZON_S.get(H, 0) <= DAY:
        rep.p(
            f"Markets that closed more than a day before their scheduled time have no {H} "
            "snapshot on the scheduled anchor (they were already closed); the anchor "
            "contrast below compares them at a longer horizon."
        )

    # How the intervals depend on what is resampled together.
    sens = []
    for v in venues:
        g = main[(main["venue"] == v) & (main["horizon"] == H)]
        if g["cluster_id"].nunique() < acfg.min_events:
            continue
        t = clustering_sensitivity(g.assign(cluster_id=g["game_cluster_id"]), acfg)
        sens.append(t.assign(venue=v))
    if sens:
        cs = pd.concat(sens, ignore_index=True)
        rep.p(
            "Bootstrap intervals under coarser clusters. '+ category-day' also resamples "
            "together every contract in one category scheduled on the same UTC day "
            "(common shocks: one crypto price path, one news day, one deadline); '+ day' "
            "everything scheduled that day. Ratios are interval widths relative to game/event "
            "clusters."
        )
        shown = pd.DataFrame(
            {
                "venue": cs["venue"],
                "clustering": cs["clustering"],
                "clusters": cs["clusters"],
                "Brier": [ci(r, "brier", 4) for _, r in cs.iterrows()],
                "slope b": [ci(r, "slope", 2) for _, r in cs.iterrows()],
                "intercept a": [ci(r, "intercept", 2) for _, r in cs.iterrows()],
                "Brier width x": cs["brier_width_ratio"].round(2),
                "slope width x": cs["slope_width_ratio"].round(2),
                "intercept width x": cs["intercept_width_ratio"].round(2),
            }
        )
        rep.table(f"clustering_sensitivity_{H}", cs, shown)
        rep.summary["clustering_sensitivity"] = cs.to_dict(orient="records")

    # anchor artifact figure: early closers priced both ways, at the shortest horizon
    # at which enough of them were still open on the scheduled anchor
    ec_ids = set(panel.loc[panel["early_close"], "contract_id"])
    h_ec = next(
        (
            h
            for h in reversed(horizons)
            if usable.loc[
                usable["contract_id"].isin(ec_ids) & (usable["horizon"] == h), "cluster_id"
            ].nunique()
            >= acfg.min_events
        ),
        None,
    )
    if h_ec is not None:
        sched = usable[usable["contract_id"].isin(ec_ids) & (usable["horizon"] == h_ec)]
        clo = close[close["contract_id"].isin(ec_ids) & (close["horizon"] == h_ec)]
        if clo["cluster_id"].nunique() >= 10:
            fig = plots.anchor_contrast(
                {
                    f"{h_ec} before scheduled time": reliability_table(sched, edges10),
                    f"{h_ec} before actual close (leaky)": reliability_table(clo, edges10),
                },
                "Markets that closed early, two anchors",
                rep.figures / "anchor_contrast.png",
                caption="Both venues pooled, no liveness filter on either anchor. Anchoring "
                "on the actual close picks the snapshot time using the outcome.",
            )
            rep.figure(fig, "Anchor contrast")

    rep.write(f"pm-calibration report: {cfg.name}")
    return out / "report.md"


SPORTS_LINES = ("sports_moneyline", "sports_spread", "sports_total")


def _cross_venue(rep: Report, m: pd.DataFrame, acfg: AnalysisConfig) -> None:
    """``m``: one row per (pair, horizon) with p_polymarket, p_kalshi, outcome, event_key.

    Only pairs with a confirmed quote on both venues at the snapshot are compared,
    under the same rule as the per-venue analysis.
    Sports lines are reported apart from the rest: their two venues schedule the
    same kick-off, so the common anchor is the same moment on both, while golf,
    awards, elections and macro releases can differ by hours or days.
    """
    # The same price rule as the per-venue analysis (pmcal.panel.classify_prices),
    # applied to each side; both sides must be a confirmed quote.
    n = len(m)
    p_pm, r_pm = classify_prices(
        pd.Series(["polymarket"] * n, index=m.index),
        m["p_polymarket_mid"],
        pd.Series(np.nan, index=m.index),
        pd.Series(np.nan, index=m.index),
        m["last_polymarket"],
        m["last_trade_age_s_polymarket"],
    )
    p_k, r_k = classify_prices(
        pd.Series(["kalshi"] * n, index=m.index),
        m["p_kalshi"],
        m["bid_kalshi"],
        m["ask_kalshi"],
        m["last_kalshi"],
        m["last_trade_age_s_kalshi"],
    )
    ok = (m["status_polymarket"] == "ok") & (m["status_kalshi"] == "ok")
    m = m.assign(p_polymarket=p_pm, p_kalshi=p_k)[ok & (r_pm == "quote") & (r_k == "quote")]
    m = m.dropna(subset=["p_polymarket", "p_kalshi", "outcome"]).copy()
    m["group"] = np.where(m["family"].isin(SPORTS_LINES), "sports lines", "other families")
    rows = []
    shown_rows = []
    for (grp, h), g in m.groupby(["group", "horizon"], sort=True):
        if g["event_key"].nunique() < 10:
            continue
        long = pd.concat(
            [
                pd.DataFrame(
                    {
                        "venue": v,
                        "contract_id": g["pair_id"],
                        "event_id": g["event_key"],
                        "cluster_id": g["event_key"],
                        "price": g[f"p_{v}"],
                        "outcome": g["outcome"],
                    }
                )
                for v in ("polymarket", "kalshi")
            ],
            ignore_index=True,
        )
        d = paired_differences(
            long, "venue", ["kalshi"], "polymarket", acfg, metrics=("brier", "log_loss", "ece")
        )
        for _, r in d.iterrows():
            rows.append(
                {
                    "group": grp,
                    "horizon": h,
                    "pairs": len(g),
                    "clusters": g["event_key"].nunique(),
                    **r.to_dict(),
                }
            )
        gap = (g["p_polymarket"] - g["p_kalshi"]).abs()
        # When the venues disagree by 5c or more, which one was closer to the outcome?
        dis = g[gap >= 0.05]
        pm_closer = (dis["p_polymarket"] - dis["outcome"]).abs() < (
            dis["p_kalshi"] - dis["outcome"]
        ).abs()
        bd = d[d["metric"] == "brier"].set_index("venue")
        k = bd.loc["kalshi"]
        pmb = bd.loc["polymarket"]
        shown_rows.append(
            {
                "group": grp,
                "horizon": h,
                "pairs": len(g),
                "clusters": g["event_key"].nunique(),
                "Brier Polymarket": f"{pmb['value']:.4f} [{pmb['lo']:.4f}, {pmb['hi']:.4f}]",
                "Brier Kalshi": f"{k['value']:.4f} [{k['lo']:.4f}, {k['hi']:.4f}]",
                "Kalshi - Polymarket": (
                    f"{k['diff_vs_base']:+.4f} [{k['diff_lo']:+.4f}, {k['diff_hi']:+.4f}]"
                ),
                "median |gap|": f"{gap.median():.3f}",
                "gap >= 5c": f"{(gap >= 0.05).mean():.1%}",
                "Polymarket closer when gap >= 5c": (
                    f"{pm_closer.mean():.0%} of {len(dis)}" if len(dis) else ""
                ),
            }
        )
    if not rows:
        rep.p("Too few matched pairs with prices at a common time.")
        return
    t = pd.DataFrame(rows)
    rep.p(
        "Pairs of contracts on the same proposition, priced at the same moment on both "
        "venues, with a confirmed quote on both. Kalshi minus Polymarket Brier "
        "differences with paired intervals, clustered on the real-world event."
    )
    rep.table("cross_venue", t, pd.DataFrame(shown_rows))
    fam = (
        m[m["horizon"] == acfg.headline_horizon]
        .groupby("family")
        .agg(pairs=("pair_id", "size"), clusters=("event_key", "nunique"))
        .reset_index()
        .sort_values("pairs", ascending=False)
    )
    rep.p(f"Pairs with confirmed quotes on both venues, by family, at {acfg.headline_horizon}:")
    rep.table("cross_venue_families", fam)
    rep.summary["cross_venue"] = t.to_dict(orient="records")
    rep.summary["cross_venue_shown"] = shown_rows
    H = acfg.headline_horizon
    gh = m[(m["horizon"] == H) & (m["group"] == "sports lines")]
    if len(gh):
        fig = plots.paired_scatter(
            gh.assign(resolved=np.where(gh["outcome"] > 0.5, "YES", "NO")),
            "p_polymarket",
            "p_kalshi",
            "resolved",
            f"Same sports line, both venues, {H} before start",
            rep.figures / f"cross_venue_{H}.png",
        )
        rep.figure(fig, "Cross-venue scatter")
