"""Recompute selected metrics and merge them into an existing evaluation_results.csv,
without re-running the others.

Useful when a metric was deferred during the main pass (e.g. UTMOS via EVAL_SKIP_UTMOS=1,
the slow long pole) or when its isolated environment only became available afterwards.
Columns are computed with run_eval.metric_columns, so a backfilled column is identical to
what a full `run_eval.py` pass would have produced. Summaries and figures are regenerated
unless --no-figures is given.

    python backfill_metrics.py                       # backfill UTMOS (the common case)
    python backfill_metrics.py --metrics utmos aid   # recompute several columns
    python backfill_metrics.py --csv path/to.csv --no-figures

Run with the same environment as run_eval.py (see run_pipeline.sh).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_config as cfg
import run_eval
import visualize_results


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", nargs="+", default=["utmos"], choices=run_eval.METRIC_ORDER,
                    help="metric(s) to recompute and overwrite in the CSV (default: utmos)")
    ap.add_argument("--csv", default=str(cfg.SOTA / "evaluation_results.csv"),
                    help="results CSV to update in place")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip regenerating summary tables + figures")
    args = ap.parse_args()

    csv = Path(args.csv)
    if not csv.exists():
        raise SystemExit(f"no results CSV at {csv} -- run run_eval.py first")
    df = pd.read_csv(csv)

    for metric in args.metrics:
        cols = run_eval.metric_columns(df, metric)
        for name, vals in cols.items():
            df[name] = vals
        filled = ", ".join(f"{name}={int(pd.Series(vals).notna().sum())}/{len(df)}"
                           for name, vals in cols.items())
        print(f"[{metric}] {filled}")

    df.to_csv(csv, index=False)
    print(f"wrote {csv}")

    if not args.no_figures:
        visualize_results.main()
        print("regenerated summaries + figures")


if __name__ == "__main__":
    main()
