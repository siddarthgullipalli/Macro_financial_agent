"""Pipeline stages. Each stage reads/writes the local Bronze/Silver/Gold folders, so the
same functions run from the command line or as separate Airflow tasks."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from . import betas as betas_mod
from . import etf_pull, fred_pull, labels
from .checks import run_checks
from .panel import build_panel, value_columns, zscore_columns
from .store import BRONZE, GOLD, SILVER, Store, upload_to_s3

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).with_name("series.yaml")

P_FRED_ALL = f"{BRONZE}/fred/fred_all_series.parquet"
P_ETF = f"{BRONZE}/etf/sector_etf_close.parquet"
P_MANIFEST = f"{BRONZE}/manifest.json"
P_PANEL = f"{SILVER}/panel.parquet"
P_BETAS = f"{SILVER}/sector_betas.parquet"
P_EVAL = f"{GOLD}/eval_set.parquet"
P_LATEST = f"{GOLD}/latest_snapshot.parquet"
P_LABEL_META = f"{GOLD}/label_thresholds.json"
P_DICT = f"{GOLD}/data_dictionary.csv"
P_REPORT = f"{GOLD}/quality_report.json"

STAGES = ["pull_fred", "pull_etfs", "panel", "betas", "eval_set", "checks"]


@dataclass
class Context:
    out_dir: str
    end: str
    start: str | None = None
    config_path: str | Path = CONFIG_PATH
    fred: object | None = None          # injectable for tests
    etf_puller: object | None = None    # injectable for tests
    cfg: dict = field(init=False)
    store: Store = field(init=False)

    def __post_init__(self):
        self.cfg = yaml.safe_load(Path(self.config_path).read_text())
        if self.start is None:
            self.start = self.cfg["pipeline"]["start"]
        self.store = Store(self.out_dir)

    @property
    def p(self) -> dict:
        return self.cfg["pipeline"]

    @property
    def specs(self) -> list[dict]:
        return self.cfg["series"]

    @property
    def pull_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.start) - pd.DateOffset(years=int(self.p["warmup_years"]))


def _load_raw(ctx: Context) -> dict[str, pd.DataFrame]:
    df = ctx.store.read_parquet(P_FRED_ALL)
    df["date"] = pd.to_datetime(df["date"])
    df["release_date"] = pd.to_datetime(df["release_date"])
    return {sid: g.reset_index(drop=True) for sid, g in df.groupby("series_id")}


def _load_closes(ctx: Context) -> pd.DataFrame:
    closes = ctx.store.read_parquet(P_ETF)
    closes = closes.set_index("date") if "date" in closes.columns else closes
    closes.index = pd.to_datetime(closes.index)
    return closes.sort_index()


def _update_manifest(ctx: Context, section: str, payload: dict) -> None:
    manifest = ctx.store.read_json(P_MANIFEST) if ctx.store.exists(P_MANIFEST) else {}
    manifest.update({"start": ctx.start, "end": ctx.end, "pull_start": str(ctx.pull_start.date())})
    manifest[section] = payload
    ctx.store.write_json(manifest, P_MANIFEST)


def _versions() -> dict:
    out = {"pandas": pd.__version__}
    for mod in ("fredapi", "yfinance", "numpy", "pyarrow"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


# ---------------------------------------------------------------- stages
def stage_pull_fred(ctx: Context) -> None:
    fred = ctx.fred or fred_pull.make_fred()
    raw = fred_pull.pull_all(fred, ctx.specs, ctx.pull_start, ctx.end,
                             pause=0 if ctx.fred is not None else fred_pull.REQUEST_PAUSE_S)
    frames = [df for df in raw.values() if len(df)]
    if not frames:
        raise RuntimeError("no FRED series were pulled")
    combined = pd.concat(frames, ignore_index=True)
    ctx.store.write_parquet(combined, P_FRED_ALL)
    for sid, df in raw.items():
        if len(df):
            ctx.store.write_parquet(df, f"{BRONZE}/fred/series/{sid}.parquet")
    _update_manifest(ctx, "fred", {
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": _versions(),
        "series": {sid: {"rows": int(len(df)),
                         "first_obs": str(df["date"].min().date()) if len(df) else None,
                         "last_obs": str(df["date"].max().date()) if len(df) else None}
                   for sid, df in raw.items()},
        "failed": [sid for sid, df in raw.items() if df.empty],
    })


def stage_pull_etfs(ctx: Context) -> None:
    puller = ctx.etf_puller or etf_pull.pull_etfs
    closes = puller(ctx.pull_start, ctx.end)
    ctx.store.write_parquet(closes.reset_index(), P_ETF)
    _update_manifest(ctx, "etf", {
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tickers": {t: {"first": str(closes[t].first_valid_index().date()),
                        "last": str(closes[t].last_valid_index().date())}
                    for t in closes.columns if closes[t].notna().any()},
    })


def stage_panel(ctx: Context) -> None:
    raw = _load_raw(ctx)
    panel = build_panel(raw, ctx.specs, ctx.start, ctx.end,
                        z_window=int(ctx.p["zscore_window"]), z_min=int(ctx.p["zscore_min_obs"]))
    ctx.store.write_parquet(panel, P_PANEL, index=True)


def stage_betas(ctx: Context) -> None:
    raw = _load_raw(ctx)
    closes = _load_closes(ctx)
    closes = closes.loc[:pd.Timestamp(ctx.end)]
    factors = betas_mod.build_factor_changes(raw, closes.index)
    dates = betas_mod.month_end_trading_days(closes.index, ctx.p["eval_start"])
    dates = dates.union(pd.DatetimeIndex([closes.index[-1]]))
    sectors = [t for t in etf_pull.SECTOR_ETFS if t in closes.columns]
    b = betas_mod.compute_betas(closes, factors, dates, sectors, etf_pull.BENCHMARK,
                                window=int(ctx.p["beta_window"]), min_obs=int(ctx.p["beta_min_obs"]))
    ctx.store.write_parquet(b, P_BETAS)


def stage_eval_set(ctx: Context) -> None:
    panel = ctx.store.read_parquet(P_PANEL)
    panel = panel.set_index("as_of_date") if "as_of_date" in panel.columns else panel
    panel.index = pd.to_datetime(panel.index)
    b = ctx.store.read_parquet(P_BETAS)
    b["as_of_date"] = pd.to_datetime(b["as_of_date"])
    closes = _load_closes(ctx).loc[:pd.Timestamp(ctx.end)]

    month_ends = betas_mod.month_end_trading_days(closes.index, ctx.p["eval_start"])
    sectors = [t for t in etf_pull.SECTOR_ETFS if t in closes.columns]
    fwd = labels.forward_risk(closes, month_ends, sectors, horizon=int(ctx.p["horizon_days"]))
    eval_set, meta = labels.build_eval_set(panel, b[b["as_of_date"].isin(month_ends)], fwd, ctx.p["splits"])
    latest = labels.build_latest_snapshot(panel, b, closes.index[-1])

    ctx.store.write_parquet(eval_set, P_EVAL)
    ctx.store.write_parquet(latest, P_LATEST)
    ctx.store.write_json({**meta, "splits": ctx.p["splits"], "horizon_days": ctx.p["horizon_days"]}, P_LABEL_META)
    ctx.store.write_csv(data_dictionary(ctx), P_DICT)


def stage_checks(ctx: Context) -> dict:
    raw = _load_raw(ctx)
    panel = ctx.store.read_parquet(P_PANEL)
    panel = panel.set_index("as_of_date") if "as_of_date" in panel.columns else panel
    panel.index = pd.to_datetime(panel.index)
    eval_set = ctx.store.read_parquet(P_EVAL) if ctx.store.exists(P_EVAL) else None
    if eval_set is not None:
        eval_set["as_of_date"] = pd.to_datetime(eval_set["as_of_date"])
    report = run_checks(raw, ctx.specs, panel, eval_set, ctx.end, ctx.p["eval_start"])
    ctx.store.write_json(report, P_REPORT)
    for w in report["warnings"]:
        log.warning("CHECK: %s", w)
    if not report["passed"]:
        for e in report["errors"]:
            log.error("CHECK FAILED: %s", e)
        raise RuntimeError(f"quality checks failed: {len(report['errors'])} error(s), see {P_REPORT}")
    log.info("quality checks passed (%d warnings)", len(report["warnings"]))
    return report


def stage_upload(ctx: Context, bucket: str, region: str = "us-east-2") -> int:
    n = upload_to_s3(ctx.store.root, bucket, region)
    log.info("uploaded %d files to s3://%s", n, bucket)
    return n


STAGE_FUNCS = {
    "pull_fred": stage_pull_fred,
    "pull_etfs": stage_pull_etfs,
    "panel": stage_panel,
    "betas": stage_betas,
    "eval_set": stage_eval_set,
    "checks": stage_checks,
}


def data_dictionary(ctx: Context) -> pd.DataFrame:
    lag_text = {
        "D": "market data; usable from next business day",
        "W": "first-release value from ALFRED, available from its publication date",
        "M": "first-release value from ALFRED, available from its publication date",
        "Q": "first-release value from ALFRED, available from its publication date",
    }
    rows = []
    for s in ctx.specs:
        for t, col in zip(s["transforms"], value_columns(s)):
            rows.append({"column": col, "source": "FRED", "series_id": s["id"], "name": s["name"],
                         "category": s["category"], "frequency": s["freq"], "units": s["units"],
                         "transform": t, "point_in_time": lag_text[s["freq"]]})
        for col in zscore_columns(s):
            rows.append({"column": col, "source": "derived", "series_id": s["id"], "name": s["name"],
                         "category": s["category"], "frequency": "business day",
                         "units": "standard deviations", "transform": "trailing ~3y z-score",
                         "point_in_time": "trailing window only"})
    derived = [
        ("etf", "SPDR sector ETF ticker", "", ""),
        ("sector", "GICS sector name", "", ""),
        ("split", "train / val / test / post_cutoff by as-of date", "", ""),
        ("risk_label", "LOW / MODERATE / HIGH from fwd_vol_60d train terciles", "", "uses future prices (label only)"),
        ("beta_mkt", "sensitivity to SPY daily return", "per unit return", "trailing 252 trading days"),
        ("beta_d_dgs10", "sensitivity to daily change in 10Y yield, net of market", "return per 1 pp", "trailing 252 trading days"),
        ("beta_r_oil", "sensitivity to daily WTI oil return, net of market", "per unit return", "trailing 252 trading days"),
        ("beta_r_usd", "sensitivity to daily dollar-index return, net of market", "per unit return", "trailing 252 trading days"),
        ("beta_d_vix", "sensitivity to daily VIX change, net of market", "return per VIX point", "trailing 252 trading days"),
        ("beta_d_credit", "sensitivity to daily high-yield (or Baa) spread change, net of market", "return per 1 pp", "trailing 252 trading days"),
        ("r2", "R-squared of the sensitivity regression", "", ""),
        ("fwd_vol_60d", "annualized realized volatility over next 60 trading days", "decimal", "future data (label only)"),
        ("fwd_mdd_60d", "max drawdown over next 60 trading days", "decimal (negative)", "future data (label only)"),
        ("fwd_ret_60d", "total return over next 60 trading days", "decimal", "future data (label only)"),
    ]
    for col, desc, units, pit in derived:
        rows.append({"column": col, "source": "derived", "series_id": "", "name": desc, "category": "sector",
                     "frequency": "month-end", "units": units, "transform": "", "point_in_time": pit})
    return pd.DataFrame(rows)


def run(ctx: Context, stages: list[str] | None = None, bucket: str | None = None) -> None:
    for name in stages or STAGES:
        log.info("=== stage: %s ===", name)
        STAGE_FUNCS[name](ctx)
    if bucket:
        stage_upload(ctx, bucket)
