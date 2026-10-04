"""Build the point-in-time daily macro panel.

For every business day the panel holds the latest value that had been PUBLISHED on or
before that day. Transforms (yoy, 3-month change, ...) are computed on first-release
values in observation order, then attached to the release date of the newest
observation. Z-scores use trailing windows only.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PERIODS = {
    "D": {"yoy": 252, "3m": 63},
    "W": {"yoy": 52, "3m": 13},
    "M": {"yoy": 12, "3m": 3},
    "Q": {"yoy": 4, "3m": 1},
}
# Blank a value if its latest release is older than this (series stopped updating).
STALE_LIMIT_DAYS = {"D": 10, "W": 21, "M": 80, "Q": 200}


def column_name(sid: str, transform: str) -> str:
    return sid if transform == "level" else f"{sid}_{transform}"


def value_columns(spec: dict) -> list[str]:
    return [column_name(spec["id"], t) for t in spec["transforms"]]


def zscore_columns(spec: dict) -> list[str]:
    return [f"{column_name(spec['id'], t)}_z3y" for t in spec.get("zscore", [])]


def transform(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    sid, freq = spec["id"], spec["freq"]
    s = df.set_index("date")["value"].sort_index().astype(float)
    p = PERIODS[freq]
    cols: dict[str, pd.Series] = {}
    for t in spec["transforms"]:
        name = column_name(sid, t)
        if t == "level":
            cols[name] = s
        elif t == "yoy":
            cols[name] = (s / s.shift(p["yoy"]) - 1.0) * 100.0
        elif t == "chg_3m":
            cols[name] = s - s.shift(p["3m"])
        elif t == "ret_3m":
            cols[name] = (s / s.shift(p["3m"]) - 1.0) * 100.0
        elif t in ("chg_4w", "pct_4w"):
            if freq != "W":
                raise ValueError(f"{sid}: {t} only applies to weekly series")
            cols[name] = s - s.shift(4) if t == "chg_4w" else (s / s.shift(4) - 1.0) * 100.0
        elif t == "diff_1":
            cols[name] = s - s.shift(1)
        else:
            raise ValueError(f"{sid}: unknown transform {t!r}")
    out = pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan)
    out[f"{sid}__obs_date"] = out.index
    out[f"{sid}__release_date"] = df.set_index("date")["release_date"].reindex(out.index)
    return out.reset_index(drop=True)


def build_panel(raw: dict[str, pd.DataFrame], specs: list[dict], start, end,
                z_window: int = 756, z_min: int = 252) -> pd.DataFrame:
    """Return a business-day panel indexed by as_of_date (from `start` to `end`).

    Each series contributes its value columns plus `<ID>__obs_date` and
    `<ID>__release_date` provenance columns, used for evidence tagging and the
    no-look-ahead check.
    """
    non_empty = [d["date"].min() for d in raw.values() if d is not None and len(d)]
    first = min(non_empty) if non_empty else pd.Timestamp(start)
    idx = pd.bdate_range(min(first, pd.Timestamp(start)), pd.Timestamp(end), name="as_of_date")
    base = pd.DataFrame({"as_of_date": idx})

    parts = []
    for spec in specs:
        sid = spec["id"]
        vcols = value_columns(spec)
        obs, rel = f"{sid}__obs_date", f"{sid}__release_date"
        df = raw.get(sid)
        if df is None or df.empty:
            log.warning("%s: no data, columns left empty", sid)
            empty = pd.DataFrame(index=idx, columns=vcols, dtype=float)
            empty[obs] = pd.NaT
            empty[rel] = pd.NaT
            parts.append(empty)
            continue
        tf = transform(df, spec).sort_values([rel, obs])
        merged = pd.merge_asof(base, tf, left_on="as_of_date", right_on=rel, direction="backward")
        stale = (merged["as_of_date"] - merged[rel]).dt.days > STALE_LIMIT_DAYS[spec["freq"]]
        merged.loc[stale, vcols] = np.nan
        merged.loc[stale, [obs, rel]] = pd.NaT
        parts.append(merged.set_index("as_of_date")[vcols + [obs, rel]])

    panel = pd.concat(parts, axis=1)
    panel.index.name = "as_of_date"

    zcols = {}
    for spec in specs:
        for t in spec.get("zscore", []):
            col = column_name(spec["id"], t)
            x = panel[col].astype(float)
            mean = x.rolling(z_window, min_periods=z_min).mean()
            std = x.rolling(z_window, min_periods=z_min).std()
            zcols[f"{col}_z3y"] = (x - mean) / std.replace(0.0, np.nan)
    if zcols:
        panel = pd.concat([panel, pd.DataFrame(zcols, index=panel.index)], axis=1)

    return panel.loc[pd.Timestamp(start):]


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Model-facing columns (drops the provenance columns)."""
    return [c for c in panel.columns if "__" not in c]
