"""Pull FRED series together with the date each value was first published.

Daily market series (yields, VIX, oil, spreads) are not revised, so a value for day t
is treated as usable from the next business day.

Weekly, monthly and quarterly series are revised and published with a lag, so we use
ALFRED vintages: for each observation we keep the FIRST published value and the date
it was published. FRED stamps observations at the START of the period (March CPI is
dated 2024-03-01 but published mid-April), so without this step the model would see
numbers before they existed.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

log = logging.getLogger(__name__)

# Conservative publication lags (calendar days after FRED's observation date) used when
# ALFRED has no usable vintage for an observation.
FALLBACK_LAG_DAYS = {"W": 8, "M": 50, "Q": 125}
# A first vintage later than this means ALFRED only started tracking the series then;
# the value was not actually published that late, so the fallback lag is used instead.
MAX_PLAUSIBLE_LAG_DAYS = {"W": 45, "M": 120, "Q": 250}
REQUEST_PAUSE_S = 0.6  # FRED allows roughly 120 requests per minute

COLUMNS = ["series_id", "date", "value", "release_date", "release_source"]


def make_fred(api_key: str | None = None):
    from fredapi import Fred

    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set (put it in .env or the environment)")
    return Fred(api_key=key)


def _pull_daily(fred, sid: str, start, end) -> pd.DataFrame:
    s = fred.get_series(sid, observation_start=start, observation_end=end)
    s = pd.to_numeric(s, errors="coerce").dropna()
    df = pd.DataFrame({"date": pd.to_datetime(s.index), "value": s.to_numpy(dtype=float)})
    df["release_date"] = df["date"] + BDay(1)
    df["release_source"] = "market_close_t+1"
    return df


def _pull_with_vintages(fred, sid: str, freq: str, start, end) -> pd.DataFrame:
    raw = None
    try:
        raw = fred.get_series_all_releases(sid)
    except Exception as exc:  # some series have no ALFRED history
        log.warning("%s: ALFRED vintages unavailable (%s); using fallback lag", sid, exc)

    if raw is not None and len(raw) > 0:
        raw = raw.copy()
        raw["date"] = pd.to_datetime(raw["date"])
        raw["realtime_start"] = pd.to_datetime(raw["realtime_start"])
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        raw = raw.dropna(subset=["value"])
        first = (raw.sort_values(["date", "realtime_start"])
                    .groupby("date", as_index=False).first())
        lag_days = (first["realtime_start"] - first["date"]).dt.days
        implausible = lag_days > MAX_PLAUSIBLE_LAG_DAYS[freq]
        fallback = first["date"] + pd.Timedelta(days=FALLBACK_LAG_DAYS[freq])
        first["release_date"] = first["realtime_start"].where(~implausible, fallback)
        first["release_source"] = np.where(implausible, "fallback_lag", "alfred_first_release")
        df = first[["date", "value", "release_date", "release_source"]].copy()
    else:
        s = pd.to_numeric(fred.get_series(sid), errors="coerce").dropna()
        df = pd.DataFrame({"date": pd.to_datetime(s.index), "value": s.to_numpy(dtype=float)})
        df["release_date"] = df["date"] + pd.Timedelta(days=FALLBACK_LAG_DAYS[freq])
        df["release_source"] = "fallback_lag_latest_vintage"

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    return df[(df["date"] >= start) & (df["release_date"] <= end)]


def pull_series(fred, spec: dict, start, end) -> pd.DataFrame:
    sid, freq = spec["id"], spec["freq"]
    if freq == "D":
        df = _pull_daily(fred, sid, start, end)
    elif freq in FALLBACK_LAG_DAYS:
        df = _pull_with_vintages(fred, sid, freq, start, end)
    else:
        raise ValueError(f"{sid}: unknown frequency {freq!r}")
    df = df.assign(series_id=sid)[COLUMNS].sort_values("date").reset_index(drop=True)
    return df


def pull_all(fred, specs: list[dict], start, end, pause: float = REQUEST_PAUSE_S) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for spec in specs:
        sid = spec["id"]
        try:
            df = pull_series(fred, spec, start, end)
            out[sid] = df
            log.info("%-14s %6d obs  %s -> %s", sid, len(df),
                     df["date"].min().date() if len(df) else "-",
                     df["date"].max().date() if len(df) else "-")
        except Exception as exc:
            log.error("%s: pull failed (%s)", sid, exc)
            out[sid] = pd.DataFrame(columns=COLUMNS)
        if pause:
            time.sleep(pause)
    return out
