"""Data quality checks. Any look-ahead violation is a hard failure."""
from __future__ import annotations

import logging

import pandas as pd

from .panel import STALE_LIMIT_DAYS, value_columns

log = logging.getLogger(__name__)

# Loose sanity ranges for level columns: (min, max)
RANGES = {
    "DFF": (-1, 25), "DGS3MO": (-1, 25), "DGS2": (-1, 25), "DGS10": (-1, 25), "DGS30": (-1, 25),
    "T10Y2Y": (-5, 5), "T10Y3M": (-6, 6), "VIXCLS": (5, 100), "DCOILWTICO": (-50, 250),
    "DTWEXBGS": (50, 200), "BAMLH0A0HYM2": (0, 30), "BAA10Y": (0, 10), "T5YIE": (-3, 6),
    "UNRATE": (0, 30), "NFCI": (-3, 10), "STLFSI4": (-5, 15), "MORTGAGE30US": (1, 20),
    "UMCSENT": (20, 120), "CFNAI": (-25, 10),
}


def run_checks(raw: dict[str, pd.DataFrame], specs: list[dict], panel: pd.DataFrame,
               eval_set: pd.DataFrame | None, end, eval_start) -> dict:
    end = pd.Timestamp(end)
    report: dict = {"errors": [], "warnings": [], "series": {}, "eval_set": {}}

    # 1. Raw pulls: release on/after observation date, coverage, staleness
    for spec in specs:
        sid, freq = spec["id"], spec["freq"]
        df = raw.get(sid)
        info = {"rows": 0}
        if df is None or df.empty:
            report["errors"].append(f"{sid}: no data pulled")
            report["series"][sid] = info
            continue
        bad = int((df["release_date"] < df["date"]).sum())
        if bad:
            report["errors"].append(f"{sid}: {bad} rows released before their observation date")
        last_release = pd.Timestamp(df["release_date"].max())
        days_since = int((end - last_release).days)
        if days_since > STALE_LIMIT_DAYS[freq]:
            report["warnings"].append(f"{sid}: last release {last_release.date()} is {days_since} days before end")
        info.update({
            "rows": int(len(df)),
            "first_obs": str(df["date"].min().date()),
            "last_obs": str(df["date"].max().date()),
            "last_release": str(last_release.date()),
            "release_source": {k: int(v) for k, v in df["release_source"].value_counts().items()},
        })
        if sid in RANGES:
            lo, hi = RANGES[sid]
            out = int(((df["value"] < lo) | (df["value"] > hi)).sum())
            if out:
                report["warnings"].append(f"{sid}: {out} values outside [{lo}, {hi}]")
        report["series"][sid] = info

    # 2. No look-ahead in the panel: every value used was released on/before the as-of date
    window = panel.loc[pd.Timestamp(eval_start):]
    for spec in specs:
        rel = f"{spec['id']}__release_date"
        if rel in panel.columns:
            violations = int((panel[rel] > panel.index.to_series()).sum())
            if violations:
                report["errors"].append(f"{spec['id']}: {violations} panel rows use unreleased data")
        for col in value_columns(spec):
            miss = float(window[col].isna().mean()) if len(window) else 1.0
            report["series"].setdefault(spec["id"], {}).setdefault("missing_rate", {})[col] = round(miss, 4)
            if miss > 0.2:
                report["warnings"].append(f"{col}: {miss:.0%} missing since {pd.Timestamp(eval_start).date()}")

    # 3. Evaluation set
    if eval_set is not None:
        report["eval_set"] = {
            "rows": int(len(eval_set)),
            "sectors": int(eval_set["etf"].nunique()),
            "first_as_of": str(eval_set["as_of_date"].min().date()),
            "last_as_of": str(eval_set["as_of_date"].max().date()),
            "rows_by_split": {k: int(v) for k, v in eval_set["split"].value_counts().items()},
            "labeled_by_split": {k: int(v) for k, v in
                                 eval_set[eval_set["has_label"]]["split"].value_counts().items()},
            "label_counts_by_split": {
                s: {k: int(v) for k, v in g["risk_label"].value_counts().items()}
                for s, g in eval_set[eval_set["has_label"]].groupby("split")
            },
        }
        if eval_set.duplicated(["as_of_date", "etf"]).any():
            report["errors"].append("eval_set: duplicate (as_of_date, etf) rows")

    report["passed"] = not report["errors"]
    return report
