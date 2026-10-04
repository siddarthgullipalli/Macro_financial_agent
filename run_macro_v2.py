#!/usr/bin/env python3
"""Run the macro v2 data pipeline.

Examples
  python run_macro_v2.py                                  # full run, local output only
  python run_macro_v2.py --s3                             # full run + upload to S3
  python run_macro_v2.py --stages panel,betas,eval_set,checks   # rebuild without re-pulling
"""
import argparse
import logging
import sys
from datetime import date, timedelta

from macro_v2.pipeline import STAGES, Context, run

DEFAULT_BUCKET = "fin-risk-pipeline-siddarth"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=None, help="first panel date (default from series.yaml)")
    ap.add_argument("--end", default=str(date.today() - timedelta(days=1)), help="last date (default yesterday)")
    ap.add_argument("--out", default="./data", help="local output root")
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma list from: {', '.join(STAGES)}")
    ap.add_argument("--s3", action="store_true", help="upload outputs to S3 at the end")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        ap.error(f"unknown stage(s): {unknown}")
    ctx = Context(out_dir=args.out, start=args.start, end=args.end)
    run(ctx, stages, bucket=args.bucket if args.s3 else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
