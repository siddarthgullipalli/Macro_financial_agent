# Macro v2 data pipeline

Builds the macro agent's dataset: a point-in-time FRED panel (29 series), measured
sector sensitivities for the 11 SPDR sector ETFs, forward 60-day risk labels, and a
sector x month-end evaluation set with train / val / test / post_cutoff splits.

## Outputs (local `./data/` mirrors the S3 keys)

| Path | Contents |
|---|---|
| `bronze/macro_v2/fred/fred_all_series.parquet` | Raw pulls: value, observation date, publication date, source |
| `bronze/macro_v2/etf/sector_etf_close.parquet` | Adjusted closes, 11 sector ETFs + SPY |
| `bronze/macro_v2/manifest.json` | Pull time, versions, coverage per series |
| `silver/macro_v2/panel.parquet` | Business-day panel; values only after publication; `__obs_date` / `__release_date` provenance columns |
| `silver/macro_v2/sector_betas.parquet` | Rolling 252-day sensitivities to market, 10Y yield, oil, dollar, VIX, credit spread |
| `gold/macro_v2/eval_set.parquet` | One row per (month-end, sector): features, sensitivities, labels, split |
| `gold/macro_v2/latest_snapshot.parquet` | Most recent date, 11 sector rows, no labels (for the live agent) |
| `gold/macro_v2/label_thresholds.json` | LOW / MODERATE / HIGH cut-offs (train terciles of fwd_vol_60d) |
| `gold/macro_v2/data_dictionary.csv` | Every column: source, units, transform, timing |
| `gold/macro_v2/quality_report.json` | Coverage, staleness, ranges, look-ahead check (hard fail) |

## Run on EC2

```bash
cd Multi-Agent-Debate-Financial-Risk-Analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r macro_v2/requirements.txt
echo "FRED_API_KEY=your_key" > .env          # keep .env out of git

python tests/smoke_test.py                    # offline test, ~1 minute
tmux new -s macro
python run_macro_v2.py --end 2026-09-30       # full pull + build, local only
python run_macro_v2.py --end 2026-09-30 --s3  # same, then upload to S3
```

Rebuild without re-pulling: `python run_macro_v2.py --stages panel,betas,eval_set,checks`

## Airflow (after the script works)

1. Copy `macro_v2/` and `dags/macro_v2_dag.py` into the Airflow `dags/` folder.
2. In docker-compose, add `fredapi yfinance pyarrow pyyaml` to `_PIP_ADDITIONAL_REQUIREMENTS`
   and mount `./data:/opt/airflow/data` in the shared volumes; restart.
3. Admin -> Variables: add `FRED_API_KEY`.
4. For S3 from inside Docker, set the EC2 metadata hop limit to 2 (instance IAM role).
5. Trigger `macro_v2_pipeline` in the UI.

## Design notes

* **No look-ahead.** Revised series use ALFRED first-release values from their publication
  date. Daily market series are usable the next business day. When ALFRED has no usable
  vintage, a conservative lag is used (weekly 8, monthly 50, quarterly 125 days after the
  observation date) and the row is tagged `fallback_lag`.
* **Stale data** is blanked rather than carried forward forever (daily 10, weekly 21,
  monthly 80, quarterly 200 days).
* **Sensitivities** are net of the market (SPY in the regression) and use data through the
  trading day before the as-of date. If the high-yield spread history on FRED is short,
  the credit factor switches to Moody's Baa spread (BAA10Y).
* **Labels** use future prices and must never be shown to a model as inputs.
* **Prompts for LLM evaluation** should not reveal the as-of date or the provenance
  columns, or models can recall what happened next.
