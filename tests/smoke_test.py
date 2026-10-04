"""Offline end-to-end test with synthetic data shaped like fredapi / yfinance output.

Run from the repo root:  python tests/smoke_test.py
"""
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from macro_v2 import labels  # noqa: E402
from macro_v2.etf_pull import BENCHMARK, SECTOR_ETFS  # noqa: E402
from macro_v2.pipeline import P_EVAL, P_LATEST, P_PANEL, Context, run  # noqa: E402

rng = np.random.default_rng(7)
FREQ_RULE = {"W": "W-SAT", "M": "MS", "Q": "QS"}
REAL_LAG = {"W": 5, "M": 14, "Q": 28}  # days after period END


class FakeFred:
    """Mimics fredapi.Fred.get_series / get_series_all_releases."""

    def __init__(self, cfg_specs):
        self.freq = {s["id"]: s["freq"] for s in cfg_specs}

    def get_series(self, sid, observation_start=None, observation_end=None):
        idx = pd.bdate_range("1998-01-01", "2026-09-30")
        s = pd.Series(3 + np.cumsum(rng.normal(0, 0.02, len(idx))), index=idx)
        if sid == "BAMLH0A0HYM2":           # simulate truncated licensed history
            s = s.loc["2023-10-01":]
        if observation_start is not None:
            s = s.loc[pd.Timestamp(observation_start):]
        if observation_end is not None:
            s = s.loc[:pd.Timestamp(observation_end)]
        s.iloc[::50] = np.nan               # holidays / missing prints
        return s

    def get_series_all_releases(self, sid):
        f = self.freq[sid]
        if sid == "CFNAI":
            raise ValueError("no vintages for this series")    # exercise fallback path
        dates = pd.date_range("1998-01-01", "2026-09-01", freq=FREQ_RULE[f])
        r = np.random.default_rng(sum(map(ord, sid)))   # deterministic per series
        level = 100 * np.exp(np.cumsum(r.normal(0.002, 0.004, len(dates))))
        period_end = {"W": dates + pd.Timedelta(days=6), "M": dates + pd.offsets.MonthEnd(0),
                      "Q": dates + pd.offsets.QuarterEnd(0)}[f]
        first_rel = period_end + pd.Timedelta(days=REAL_LAG[f])
        if sid == "UMCSENT":                 # ALFRED tracking only started in 2012
            first_rel = first_rel.where(dates >= "2012-01-01", pd.Timestamp("2012-06-15"))
        rows = [pd.DataFrame({"realtime_start": first_rel, "date": dates, "value": level}),
                pd.DataFrame({"realtime_start": first_rel + pd.Timedelta(days=30), "date": dates,
                              "value": level * 1.01})]   # later revision must NOT be used
        return pd.concat(rows, ignore_index=True)


def fake_etfs(start, end):
    idx = pd.bdate_range(start, end, name="date")
    data = {}
    for t in list(SECTOR_ETFS) + [BENCHMARK]:
        p = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(idx))))
        s = pd.Series(p, index=idx)
        if t == "XLRE":
            s[s.index < "2015-10-08"] = np.nan
        if t == "XLC":
            s[s.index < "2018-06-19"] = np.nan
        data[t] = s
    return pd.DataFrame(data)


def test_forward_vol_alignment():
    idx = pd.bdate_range("2020-01-01", periods=100)
    prices = pd.DataFrame({"A": np.exp(np.r_[0, np.cumsum(np.tile([0.01, -0.01], 50))[:-1]])}, index=idx)
    out = labels.forward_risk(prices, [idx[10]], ["A"], horizon=60)
    lr = np.log(prices["A"]).diff().iloc[11:71]
    assert abs(out["fwd_vol_60d"].iloc[0] - lr.std(ddof=1) * np.sqrt(252)) < 1e-12
    print("forward-vol alignment ok")


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    test_forward_vol_alignment()
    out = Path(tempfile.mkdtemp())
    try:
        ctx = Context(out_dir=str(out), end="2026-09-30")
        ctx.fred = FakeFred(ctx.specs)
        ctx.etf_puller = fake_etfs
        run(ctx)

        panel = pd.read_parquet(out / P_PANEL)
        panel = panel.set_index("as_of_date") if "as_of_date" in panel.columns else panel
        ev = pd.read_parquet(out / P_EVAL)
        latest = pd.read_parquet(out / P_LATEST)

        # first-release value used, not the revision; never before its release date
        cpi_rel = panel["CPIAUCSL__release_date"]
        assert (cpi_rel.dropna() <= cpi_rel.dropna().index).all()
        raw = ctx.fred.get_series_all_releases("CPIAUCSL")
        first = raw.sort_values("realtime_start").groupby("date").first()["value"]
        d = pd.Timestamp("2015-06-30")
        obs = panel.loc[d, "CPIAUCSL__obs_date"]
        expected_yoy = (first[obs] / first[obs - pd.DateOffset(years=1)] - 1) * 100
        assert abs(panel.loc[d, "CPIAUCSL_yoy"] - expected_yoy) < 1e-9
        assert obs == pd.Timestamp("2015-05-01"), obs   # May CPI (released mid-June) is newest on Jun 30

        assert set(ev["split"]) == {"train", "val", "test", "post_cutoff"}, set(ev["split"])
        assert ev["etf"].nunique() == 11
        assert ev.loc[ev["etf"] == "XLC", "as_of_date"].min() >= pd.Timestamp("2019-03-29")
        assert ev["has_label"].sum() > 0 and (~ev["has_label"]).sum() > 0
        assert len(latest) == 11
        print(f"eval_set rows={len(ev)}  cols={ev.shape[1]}  latest rows={len(latest)}")
        print(ev.groupby("split")["risk_label"].value_counts().unstack().fillna(0).astype(int))
        print("ALL SMOKE TESTS PASSED")
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
