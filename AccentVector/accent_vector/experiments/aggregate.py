"""[A1 aggregation] Pool per-speaker metric CSVs for an accent into a
cross-speaker summary -- the "consistent across speakers?" check.

Each speaker's sweep (``results/<accent>/<speaker>/``) is scored on its own with the
existing rq* modules using that speaker's own L1 reference + natural clips, writing
e.g. ``results/<accent>/<speaker>/rq1.csv``. This collates those per-speaker CSVs
(matched on the ``alpha`` column) into:

    <out-dir>/by_speaker.csv   speaker, alpha, <metric>...          (long form)
    <out-dir>/aggregate.csv    alpha, <metric>_mean, _std, _n       (across speakers)

so you see both the per-speaker curves and the accent-level mean +/- spread (a small
spread = the vector behaves consistently across speakers). It is metric-agnostic: any
per-speaker CSVs sharing an ``alpha`` column work (rq1, rq3, ...).

    # score each speaker (per-speaker refs), then pool:
    for s in results/indian/*/; do sp=$(basename "$s")
      python -m accent_vector.experiments.rq1_reproduction --sweep-dir "$s" \
        --transcripts transcripts/eval_transcripts.txt --ref-wav refs/indian/$sp.wav \
        --accent-ref natural/indian/$sp --target-accent Indian --out-csv "$s/rq1.csv"
    done
    python -m accent_vector.experiments.aggregate \
        --accent-dir results/indian --csv-name rq1.csv --out-dir results/indian
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def read_metric_csv(path):
    """[(alpha, {col: float})] from a metric CSV, skipping '#' summary footer lines
    and any non-numeric cells."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            a = row.get("alpha")
            if a in (None, "", "None"):
                continue
            vals = {}
            for k, v in row.items():
                if k == "alpha":
                    continue
                try:
                    vals[k] = float(v)
                except (TypeError, ValueError):
                    pass
            rows.append((float(a), vals))
    return rows


def collate(speaker_csvs, out_dir):
    """speaker_csvs: dict speaker -> per-speaker metric CSV path."""
    per, metrics, alphas = {}, set(), set()
    for spk, path in speaker_csvs.items():
        d = {}
        for a, vals in read_metric_csv(path):
            d[a] = vals
            alphas.add(a)
            metrics.update(vals)
        per[spk] = d
    metrics, alphas = sorted(metrics), sorted(alphas)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- long form: one row per (speaker, alpha) ---
    with open(out_dir / "by_speaker.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "alpha"] + metrics)
        for spk in sorted(per):
            for a in alphas:
                vals = per[spk].get(a, {})
                w.writerow([spk, a] + [f"{vals[m]:.6g}" if m in vals else "" for m in metrics])

    # --- aggregate: mean / std / n across speakers per alpha ---
    with open(out_dir / "aggregate.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std", "n")])
        for a in alphas:
            row = [a]
            for m in metrics:
                xs = [per[spk][a][m] for spk in per
                      if a in per[spk] and m in per[spk][a] and not np.isnan(per[spk][a][m])]
                row += ([f"{np.mean(xs):.6g}", f"{np.std(xs):.6g}", len(xs)] if xs else ["", "", 0])
            w.writerow(row)

    print(f"[aggregate] {len(per)} speakers ({', '.join(sorted(per))}); alphas={alphas}")
    print(f"[aggregate] wrote {out_dir}/by_speaker.csv + aggregate.csv")


def _discover(accent_dir, csv_name):
    """{speaker: <accent_dir>/<speaker>/<csv_name>} for speaker subdirs that have it."""
    out = {}
    for d in sorted(Path(accent_dir).iterdir() if Path(accent_dir).is_dir() else []):
        if d.is_dir() and (d / csv_name).exists():
            out[d.name] = str(d / csv_name)
    return out


def main():
    ap = argparse.ArgumentParser(description="Pool per-speaker metric CSVs across speakers")
    ap.add_argument("--accent-dir", help="results/<accent> (auto-discovers <speaker>/<csv-name>)")
    ap.add_argument("--csv-name", default="rq1.csv", help="per-speaker CSV filename to collate")
    ap.add_argument("--csv", action="append", default=[],
                    help="speaker=path (repeatable; use instead of --accent-dir)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    if a.csv:
        speaker_csvs = dict(s.split("=", 1) for s in a.csv)
    elif a.accent_dir:
        speaker_csvs = _discover(a.accent_dir, a.csv_name)
        if not speaker_csvs:
            raise SystemExit(f"no <speaker>/{a.csv_name} found under {a.accent_dir}")
    else:
        raise SystemExit("need --accent-dir (auto-discover) or --csv speaker=path ...")
    collate(speaker_csvs, a.out_dir)


if __name__ == "__main__":
    main()
