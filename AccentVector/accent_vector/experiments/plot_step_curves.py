"""One figure per (prompt condition, metric): training step on X, metric on Y,
one line per alpha.

Reuses ``plot_results.load_tree`` to read
    results/<accent>/<tag>/<ref_kind>/<speaker>/metrics/step_<N>/{rq1,rq3,utmos}.csv
and pools the m/f speakers by mean.

    python -m accent_vector.experiments.plot_step_curves --root results/hindi/lr3e5_r16
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from accent_vector.experiments.plot_results import load_tree

REF_LABEL = {"l1": "L1 prompt", "native": "GAE prompt"}
SKIP = {"alpha", "step", "speaker", "ref_kind", "n"}


def metric_cols(df):
    """Numeric metric columns (everything that isn't bookkeeping or a merge dup)."""
    return [c for c in df.columns
            if c not in SKIP and not c.endswith("_dup")
            and df[c].dtype.kind in "fi" and df[c].notna().any()]


def plot_metric(df, ref, col, out_dir):
    """step-vs-metric, one line per alpha, for a single ref_kind."""
    d = df[df.ref_kind == ref].dropna(subset=[col])
    if d.empty:
        return
    g = d.groupby(["alpha", "step"])[col].mean().reset_index()   # pool speakers
    fig, ax = plt.subplots(figsize=(6, 4))
    for alpha, sub in g.groupby("alpha"):
        sub = sub.sort_values("step")
        ax.plot(sub.step, sub[col], marker="o", ms=3, lw=1.2, label=f"α={alpha:g}")
    ax.set_xlabel("training step")
    ax.set_ylabel(col)
    ax.set_title(f"{col} — {REF_LABEL.get(ref, ref)}")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = Path(out_dir) / f"{ref}_{col}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="results/<accent>/<tag> (contains l1/ native/)")
    p.add_argument("--out-dir", help="default <root>/figures/step_curves")
    a = p.parse_args()

    long_df, _ = load_tree(a.root)
    out_dir = a.out_dir or Path(a.root) / "figures" / "step_curves"
    for ref in sorted(long_df.ref_kind.unique()):
        for col in metric_cols(long_df):
            plot_metric(long_df, ref, col, out_dir)


if __name__ == "__main__":
    main()
