"""Measured macro sensitivities for each sector ETF.

For each as-of date and sector, regress daily sector returns on SPY returns plus daily
changes in five macro factors over the previous `window` trading days. Because SPY is
in the regression, each macro beta is the sector's sensitivity net of the overall
market. The window ends the trading day BEFORE the as-of date, because FRED's daily
values for day t are only published on t+1.

Factor units: d_dgs10, d_vix, d_credit = daily change in level (percentage points / VIX
points); r_oil, r_usd = daily simple return. Betas are per unit of factor move.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FACTOR_SPECS = [
    ("d_dgs10", "DGS10", "diff"),
    ("r_oil", "DCOILWTICO", "pct"),
    ("r_usd", "DTWEXBGS", "pct"),
    ("d_vix", "VIXCLS", "diff"),
    ("d_credit", "BAMLH0A0HYM2", "diff"),
]
CREDIT_FALLBACK = "BAA10Y"


def build_factor_changes(raw: dict[str, pd.DataFrame], trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    out = {}
    for name, sid, kind in FACTOR_SPECS:
        df = raw.get(sid)
        if name == "d_credit":
            too_short = (df is None or df.empty
                         or df["date"].min() > trading_index[0] + pd.Timedelta(days=365))
            if too_short:
                log.info("credit factor: %s history too short, using %s", sid, CREDIT_FALLBACK)
                df = raw.get(CREDIT_FALLBACK)
        if df is None or df.empty:
            log.warning("factor %s: no data", name)
            out[name] = pd.Series(np.nan, index=trading_index)
            continue
        level = (df.set_index("date")["value"].sort_index()
                   .reindex(trading_index).ffill(limit=3))
        if kind == "diff":
            out[name] = level.diff()
        else:
            # clip handles the April 2020 negative oil print
            out[name] = level.pct_change(fill_method=None).clip(-0.5, 0.5)
    return pd.DataFrame(out, index=trading_index)


def month_end_trading_days(index: pd.DatetimeIndex, start) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    s = s[s >= pd.Timestamp(start)]
    return pd.DatetimeIndex(s.groupby([s.index.year, s.index.month]).max().to_numpy(), name="as_of_date")


def compute_betas(closes: pd.DataFrame, factors: pd.DataFrame, as_of_dates, sectors: list[str],
                  benchmark: str = "SPY", window: int = 252, min_obs: int = 200) -> pd.DataFrame:
    rets = closes.pct_change(fill_method=None)
    X_all = pd.concat([rets[benchmark].rename("mkt"), factors], axis=1)
    xcols = list(X_all.columns)
    rows = []
    for d in pd.DatetimeIndex(as_of_dates):
        end_i = closes.index.searchsorted(d, side="right") - 1  # last trading day <= d
        if end_i < 1:
            continue
        lo = max(0, end_i - window)
        Xw = X_all.iloc[lo:end_i]
        for etf in sectors:
            if etf not in rets.columns:
                continue
            row = {"as_of_date": d, "etf": etf, "n_obs": 0, "r2": np.nan}
            row.update({f"beta_{c}": np.nan for c in xcols})
            y = rets[etf].iloc[lo:end_i].rename("y")
            usable = [c for c in xcols if pd.concat([y, Xw[c]], axis=1).dropna().shape[0] >= min_obs]
            if "mkt" in usable:
                m = pd.concat([y, Xw[usable]], axis=1).dropna()
                if len(m) >= min_obs:
                    A = np.column_stack([np.ones(len(m)), m[usable].to_numpy()])
                    yv = m["y"].to_numpy()
                    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
                    resid = yv - A @ coef
                    ss_tot = float(((yv - yv.mean()) ** 2).sum())
                    row["n_obs"] = len(m)
                    row["r2"] = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
                    for k, c in enumerate(usable):
                        row[f"beta_{c}"] = coef[k + 1]
            rows.append(row)
    return pd.DataFrame(rows)
