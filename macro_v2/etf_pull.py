"""Daily adjusted closes for the 11 SPDR sector ETFs plus SPY."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

BENCHMARK = "SPY"
SECTOR_ETFS = {
    "XLK": "Information Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",          # starts Oct 2015
    "XLC": "Communication Services",  # starts Jun 2018
}

# Map sector names used elsewhere in the project (GICS and Yahoo Finance spellings)
# to the matching ETF, so companies can be joined to sector rows.
SECTOR_TO_ETF = {
    "information technology": "XLK", "technology": "XLK",
    "financials": "XLF", "financial services": "XLF",
    "energy": "XLE",
    "health care": "XLV", "healthcare": "XLV",
    "consumer discretionary": "XLY", "consumer cyclical": "XLY",
    "consumer staples": "XLP", "consumer defensive": "XLP",
    "industrials": "XLI",
    "materials": "XLB", "basic materials": "XLB",
    "utilities": "XLU",
    "real estate": "XLRE",
    "communication services": "XLC", "communications": "XLC",
}


def pull_etfs(start, end) -> pd.DataFrame:
    """Wide frame: index = trading date, one column of adjusted close per ticker."""
    import yfinance as yf

    tickers = list(SECTOR_ETFS) + [BENCHMARK]
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=pd.Timestamp(start).strftime("%Y-%m-%d"), end=end_exclusive,
                       auto_adjust=True, progress=False, group_by="column", threads=True)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    close = close.dropna(how="all").sort_index()
    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    close.index.name = "date"

    missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    if BENCHMARK in missing:
        raise RuntimeError("SPY download failed; cannot compute sector sensitivities")
    if missing:
        log.warning("no data for: %s", ", ".join(missing))
    for t in tickers:
        if t in close.columns and not close[t].dropna().empty:
            log.info("%-5s %5d days  %s -> %s", t, close[t].notna().sum(),
                     close[t].first_valid_index().date(), close[t].last_valid_index().date())
    return close[[t for t in tickers if t in close.columns]]
