"""Forward-looking risk labels and the sector x month-end evaluation set."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .etf_pull import SECTOR_ETFS
from .panel import feature_columns

log = logging.getLogger(__name__)

LABELS = ["LOW", "MODERATE", "HIGH"]


def forward_risk(closes: pd.DataFrame, as_of_dates, sectors: list[str], horizon: int = 60) -> pd.DataFrame:
    """Realized risk over the `horizon` trading days AFTER each as-of date.

    fwd_vol_60d: annualized std of daily log returns on days t+1..t+h
    fwd_mdd_60d: max drawdown of the price path t..t+h (negative number)
    fwd_ret_60d: total return from t to t+h
    Rows without a full forward window get NaN (the most recent ~3 months).
    """
    logret = np.log(closes.where(closes > 0)).diff()
    rows = []
    n = len(closes)
    for d in pd.DatetimeIndex(as_of_dates):
        i = closes.index.searchsorted(d, side="right") - 1
        if i < 0:
            continue
        for etf in sectors:
            if etf not in closes.columns or pd.isna(closes[etf].iloc[i]):
                continue
            vol = mdd = ret = np.nan
            if i + horizon <= n - 1:
                p = closes[etf].iloc[i:i + horizon + 1]
                r = logret[etf].iloc[i + 1:i + horizon + 1]
                if not p.isna().any():
                    vol = float(r.std(ddof=1) * np.sqrt(252))
                    mdd = float((p / p.cummax() - 1.0).min())
                    ret = float(p.iloc[-1] / p.iloc[0] - 1.0)
            rows.append({"as_of_date": d, "etf": etf, "fwd_vol_60d": vol,
                         "fwd_mdd_60d": mdd, "fwd_ret_60d": ret})
    return pd.DataFrame(rows)


def assign_split(dates: pd.Series, splits: dict) -> pd.Series:
    d = pd.to_datetime(dates)
    return pd.Series(np.select(
        [d <= pd.Timestamp(splits["train_end"]),
         d <= pd.Timestamp(splits["val_end"]),
         d <= pd.Timestamp(splits["test_end"])],
        ["train", "val", "test"], default="post_cutoff"), index=dates.index)


def tercile_thresholds(values: pd.Series) -> tuple[float, float]:
    v = values.dropna()
    if v.empty:
        raise ValueError("no labeled training rows; cannot set label thresholds")
    return float(v.quantile(1 / 3)), float(v.quantile(2 / 3))


def to_label(values: pd.Series, lo: float, hi: float) -> pd.Series:
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    out[values.notna() & (values <= lo)] = "LOW"
    out[values.notna() & (values > lo) & (values <= hi)] = "MODERATE"
    out[values.notna() & (values > hi)] = "HIGH"
    return out


def build_eval_set(panel: pd.DataFrame, betas: pd.DataFrame, fwd: pd.DataFrame,
                   splits: dict) -> tuple[pd.DataFrame, dict]:
    """One row per (as-of month-end, sector) with macro features, sensitivities and labels."""
    feats = panel[feature_columns(panel)]
    keys = betas.merge(fwd, on=["as_of_date", "etf"], how="left")
    keys = keys[keys["beta_mkt"].notna()].copy()  # needs enough price history

    snap = feats.reindex(pd.DatetimeIndex(keys["as_of_date"].unique()).sort_values(), method="ffill")
    snap.index.name = "as_of_date"
    df = keys.merge(snap.reset_index(), on="as_of_date", how="left")

    df.insert(2, "sector", df["etf"].map(SECTOR_ETFS))
    df.insert(3, "split", assign_split(df["as_of_date"], splits))
    lo, hi = tercile_thresholds(df.loc[df["split"] == "train", "fwd_vol_60d"])
    df.insert(4, "risk_label", to_label(df["fwd_vol_60d"], lo, hi))
    df.insert(5, "has_label", df["risk_label"].notna())

    meta = {
        "label_target": "fwd_vol_60d (annualized realized volatility over next 60 trading days)",
        "thresholds_from": "train split terciles, pooled across sectors",
        "low_max": lo,
        "moderate_max": hi,
    }
    return df.sort_values(["as_of_date", "etf"]).reset_index(drop=True), meta


def build_latest_snapshot(panel: pd.DataFrame, betas: pd.DataFrame, as_of) -> pd.DataFrame:
    """Unlabeled rows for the most recent date, used by the live macro agent."""
    as_of = pd.Timestamp(as_of)
    feats = panel[feature_columns(panel)]
    row = feats.loc[:as_of].iloc[[-1]].reset_index(drop=True)
    b = betas[betas["as_of_date"] == as_of].reset_index(drop=True)
    df = b.merge(row.assign(_k=1), how="cross") if len(b) else b
    if "_k" in df.columns:
        df = df.drop(columns="_k")
    if len(df):
        df.insert(2, "sector", df["etf"].map(SECTOR_ETFS))
    return df
