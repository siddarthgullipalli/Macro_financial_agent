"""Airflow DAG for the macro v2 data pipeline.

Setup (see README):
  * copy the `macro_v2/` package folder into the Airflow dags folder, next to this file
  * add `fredapi yfinance pyarrow pyyaml` to the Airflow image
  * Airflow Variable FRED_API_KEY
  * mount a shared folder at /opt/airflow/data so all tasks see the same files
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

OUT_DIR = "/opt/airflow/data/macro_v2"
BUCKET = "fin-risk-pipeline-siddarth"
END = "{{ macros.ds_add(ds, -1) }}"   # data through the day before the run


def _run_stage(stage: str, end: str, **_):
    os.environ.setdefault("FRED_API_KEY", Variable.get("FRED_API_KEY", default_var=""))
    from macro_v2.pipeline import STAGE_FUNCS, Context, stage_upload

    ctx = Context(out_dir=OUT_DIR, end=end)
    if stage == "upload_s3":
        stage_upload(ctx, BUCKET)
    else:
        STAGE_FUNCS[stage](ctx)


default_args = {"owner": "macro-agent", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="macro_v2_pipeline",
    description="Point-in-time FRED macro panel, sector sensitivities and forward risk labels",
    start_date=datetime(2026, 10, 1),
    schedule=None,          # trigger manually; switch to "@weekly" once stable
    catchup=False,
    default_args=default_args,
    tags=["macro", "fin-risk"],
) as dag:
    tasks = [
        PythonOperator(task_id=name, python_callable=_run_stage,
                       op_kwargs={"stage": name, "end": END})
        for name in ["pull_fred", "pull_etfs", "panel", "betas", "eval_set", "checks", "upload_s3"]
    ]
    pull_fred, pull_etfs, panel, betas, eval_set, checks, upload = tasks
    [pull_fred, pull_etfs] >> panel >> betas >> eval_set >> checks >> upload
