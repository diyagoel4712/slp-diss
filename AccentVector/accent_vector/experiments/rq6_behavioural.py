"""[RQ1 x RQ6 -- behavioural trajectory analysis, CPU] Compare the model's OUTPUT
metrics across training checkpoints at MATCHED alpha -- the "does the accent arrive
before the language?" question answered in output space.

``rq6_temporal`` tracks the vector in WEIGHT space (||tau_t||, direction).
``checkpoint_grid`` renders each checkpoint's alpha sweep and ``rq1_reproduction``
scores each into ``<by-step-dir>/step_<step>/rq1.csv``. This module collates those
per-checkpoint CSVs (matched on the ``alpha`` column) into:

    trajectory_long.csv          step, alpha, <metric>...        (tidy long form)
    <metric>_by_step_alpha.csv   step x alpha grid per metric    (read a COLUMN to
                                                                  compare 25k vs 45k
                                                                  at a fixed alpha)
    matched_alpha_trends.csv     alpha, spearman_step_<metric>   (at fixed alpha,
                                                                  does the metric
                                                                  rise/fall with
                                                                  training?)
    by_step_summary.csv          step, wer_leak_onset, <metric>@max-alpha

How to read it for the core question:
  * accent (accent_cs) rising with STEP at low-mid alpha, then flat
    => accent is basically learned; more steps add little accent.
  * wer rising with STEP at fixed alpha, and wer_leak_onset FALLING with step
    => more training makes English *less* fluent / leaks sooner (RQ1b). If accent
    saturates early while leakage keeps worsening, the accent needs far fewer steps
    than the language -- and an earlier checkpoint at moderate alpha is the sweet
    spot. (Metric-agnostic: point --csv-name at rq3.csv to trend the segmental/
    suprasegmental columns across training instead.)

    python -m accent_vector.experiments.rq6_behavioural \
        --by-step-dir results/dutch/native/by_step --csv-name rq1.csv \
        --out-dir results/dutch/native/trajectory
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from accent_vector.experiments import shared
from accent_vector.experiments.aggregate import read_metric_csv

DROP_COLS = {"n"}  # bookkeeping columns that aren't metrics


def _spearman(xs, ys):
    """Rank correlation of ys against xs; nan if <3 usable pairs."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = ~(np.isnan(xs) | np.isnan(ys))
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(xs[ok]))
    ry = np.argsort(np.argsort(ys[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def discover_steps(by_step_dir, csv_name):
    """{step: <by_step_dir>/step_<step>/<csv_name>} for step subdirs that have it."""
    out = {}
    for d in sorted(Path(by_step_dir).iterdir() if Path(by_step_dir).is_dir() else []):
        m = re.fullmatch(r"step_(\d+)", d.name)
        if m and d.is_dir() and (d / csv_name).exists():
            out[int(m.group(1))] = str(d / csv_name)
    return out


def load(step_csvs):
    """(data, metrics, steps, alphas): data[step][alpha] = {metric: float}."""
    data, metrics, alphas = {}, set(), set()
    for step, path in step_csvs.items():
        per_alpha = {}
        for a, vals in read_metric_csv(path):
            keep = {k: v for k, v in vals.items() if k not in DROP_COLS}
            per_alpha[a] = keep
            alphas.add(a)
            metrics.update(keep)
        data[step] = per_alpha
    return data, sorted(metrics), sorted(step_csvs), sorted(alphas)


def _val(data, step, alpha, metric):
    return data.get(step, {}).get(alpha, {}).get(metric, float("nan"))


def write_long(data, metrics, steps, alphas, out_dir):
    with open(out_dir / "trajectory_long.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "alpha"] + metrics)
        for step in steps:
            for a in alphas:
                w.writerow([step, a] + [f"{_val(data, step, a, m):.6g}" for m in metrics])


def write_pivots(data, metrics, steps, alphas, out_dir):
    """One step x alpha grid per metric -- read a column to compare checkpoints at
    a fixed alpha, a row to see a checkpoint's whole sweep."""
    for m in metrics:
        with open(out_dir / f"{m}_by_step_alpha.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step"] + [f"alpha_{a}" for a in alphas])
            for step in steps:
                w.writerow([step] + [f"{_val(data, step, a, m):.6g}" for a in alphas])


def write_matched_alpha_trends(data, metrics, steps, alphas, out_dir):
    """At each fixed alpha, Spearman(step, metric): does the metric climb or fall
    as training proceeds *at that strength*? This is the RQ6 x RQ1 crossing --
    e.g. accent flat but wer rising = language learned later than accent."""
    with open(out_dir / "matched_alpha_trends.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha"] + [f"spearman_step_{m}" for m in metrics])
        for a in alphas:
            row = [a]
            for m in metrics:
                ys = [_val(data, s, a, m) for s in steps]
                row.append(f"{_spearman(steps, ys):.4f}")
            w.writerow(row)


def write_step_summary(data, metrics, steps, alphas, out_dir, wer_thr):
    """Per checkpoint: the WER-based leakage onset (how far alpha scales before
    English breaks) and each metric at full strength -- so you see the usable range
    shrink (or not) with training."""
    max_a = alphas[-1] if alphas else None
    has_wer = "wer" in metrics
    with open(out_dir / "by_step_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = (["wer_leak_onset"] if has_wer else []) + [f"{m}@alpha_{max_a}" for m in metrics]
        w.writerow(["step"] + cols)
        for step in steps:
            row = [step]
            if has_wer:
                onset = shared.leakage_onset(
                    alphas, [_val(data, step, a, "wer") for a in alphas], wer_thr, rising=True)
                row.append(f"{onset:.4f}")
            row += [f"{_val(data, step, max_a, m):.6g}" for m in metrics]
            w.writerow(row)


def run(by_step_dir, csv_name, out_dir, step_csvs=None, wer_thr=0.5):
    step_csvs = step_csvs or discover_steps(by_step_dir, csv_name)
    if not step_csvs:
        raise SystemExit(f"no step_<step>/{csv_name} found under {by_step_dir}")
    data, metrics, steps, alphas = load(step_csvs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_long(data, metrics, steps, alphas, out_dir)
    write_pivots(data, metrics, steps, alphas, out_dir)
    write_matched_alpha_trends(data, metrics, steps, alphas, out_dir)
    write_step_summary(data, metrics, steps, alphas, out_dir, wer_thr)

    print(f"[rq6-behav] {len(steps)} checkpoints (steps {steps}); "
          f"alphas={alphas}; metrics={metrics}")
    print(f"[rq6-behav] wrote trajectory_long.csv, {len(metrics)} *_by_step_alpha.csv, "
          f"matched_alpha_trends.csv, by_step_summary.csv -> {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Compare checkpoints at matched alpha (RQ1 x RQ6)")
    p.add_argument("--by-step-dir", help="dir of step_<step>/ subdirs (auto-discovers <csv-name>)")
    p.add_argument("--csv-name", default="rq1.csv", help="per-checkpoint metric CSV to collate")
    p.add_argument("--csv", action="append", default=[],
                   help="step=path override (repeatable; use instead of --by-step-dir)")
    p.add_argument("--wer-leak-threshold", type=float, default=0.5,
                   help="WER above which content is treated as leaked (per-step onset)")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    step_csvs = None
    if a.csv:
        step_csvs = {int(k): v for k, v in (s.split("=", 1) for s in a.csv)}
    run(a.by_step_dir, a.csv_name, a.out_dir, step_csvs=step_csvs, wer_thr=a.wer_leak_threshold)


if __name__ == "__main__":
    main()
