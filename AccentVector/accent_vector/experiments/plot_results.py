"""Analysis + figures over the RQ metric CSVs (the missing plotting layer).

Reads the per-sweep CSVs written by the eval array --
    results/<accent>/<tag>/<ref_kind>/<speaker>/metrics/step_<N>/{rq1,rq3,utmos}.csv
-- pools the m/f speakers, and renders the core dissertation figures per accent into
    results/<accent>/<tag>/figures/
plus a printed summary table (the headline numbers).

    python -m accent_vector.experiments.plot_results --root results/hindi/lr3e5_r16
    python -m accent_vector.experiments.plot_results --root results/hindi/lr3e5_r16 --traj-alpha 0.3

Figures (small multiples, never dual-axis; l1 vs native as the two series):
  fig_rq1_alpha.pdf   final-checkpoint alpha-curves: accent_cs, spk_sim, P(English), WER, UTMOS
  fig_rq2_trajectory  accent_cs (at matched alpha) + leakage-onset vs training step
  fig_rq3_decomp.pdf  segmental vs suprasegmental gap-closure vs alpha

Design: Okabe-Ito CVD-safe palette + redundant line-style/marker (colour is never the
only cue), recessive grid/axes, one legend, thin marks.
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- CVD-safe styling: colour + a redundant non-colour cue per series ------------
REF_STYLE = {
    "l1":     {"c": "#0072B2", "ls": "-",  "m": "o", "label": "L1 reference"},
    "native": {"c": "#D55E00", "ls": "--", "m": "s", "label": "Neutral (GAE) reference"},
}
INK, MUTED, GRID = "#222222", "#666666", "#cccccc"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.autolayout": False,
})

RQ1_PANELS = [   # (column, pretty label, better-direction arrow)
    ("accent_cs", "Accent similarity (cos)", "↑"),
    ("spk_sim",   "Speaker similarity",      "↑"),
    ("eng_lid",   "P(English)",              "↑"),
    ("wer",       "WER",                     "↓"),
    ("utmos",     "UTMOS (MOS)",             "↑"),
]


# --- loading ---------------------------------------------------------------------
def _read_footer(path):
    """Parse the trailing '# key=val, ...' summary line of an rq CSV into a dict."""
    out = {}
    last = ""
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            last = line
    for m in re.finditer(r"(\w+)=([-\d.eE]+)", last):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return out


def _read_csv(path):
    return pd.read_csv(path, comment="#") if Path(path).exists() else None


def load_tree(root):
    """Return (long_df, footer_df) over root=results/<accent>/<tag> (has l1/ native/)."""
    root = Path(root)
    rows, footers = [], []
    for ref in ("l1", "native"):
        for spk in ("m", "f"):
            mdir = root / ref / spk / "metrics"
            if not mdir.is_dir():
                continue
            for step_dir in sorted(mdir.glob("step_*")):
                step = int(step_dir.name.split("_")[1])
                rq1 = _read_csv(step_dir / "rq1.csv")
                if rq1 is None:
                    continue
                df = rq1.copy()
                for extra in ("rq3.csv", "utmos.csv"):
                    e = _read_csv(step_dir / extra)
                    if e is not None:
                        df = df.merge(e, on="alpha", how="left", suffixes=("", "_dup"))
                df["ref_kind"], df["speaker"], df["step"] = ref, spk, step
                rows.append(df)
                fdict = _read_footer(step_dir / "rq1.csv")
                fdict.update(ref_kind=ref, speaker=spk, step=step)
                footers.append(fdict)
    if not rows:
        raise SystemExit(f"no rq1.csv found under {root}/<ref>/<spk>/metrics/step_*/")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(footers)


def _pool(df, value, by=("ref_kind", "step", "alpha")):
    """Mean over speakers + min/max band for `value`. Drops all-NaN groups."""
    g = df.dropna(subset=[value]).groupby(list(by))[value]
    return g.agg(mean="mean", lo="min", hi="max").reset_index()


# --- figures ---------------------------------------------------------------------
def fig_rq1_alpha(long_df, final_step, out):
    """Final-checkpoint alpha curves, one panel per metric, l1 vs native (pooled)."""
    d = long_df[long_df.step == final_step]
    n = len(RQ1_PANELS)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2))
    for ax, (col, lab, arrow) in zip(axes, RQ1_PANELS):
        if col not in d.columns:
            ax.set_visible(False); continue
        for ref, st in REF_STYLE.items():
            p = _pool(d[d.ref_kind == ref], col)
            if p.empty:
                continue
            p = p.sort_values("alpha")
            ax.fill_between(p.alpha, p.lo, p.hi, color=st["c"], alpha=0.12, linewidth=0)
            ax.plot(p.alpha, p["mean"], st["ls"], color=st["c"], marker=st["m"],
                    ms=5, lw=2, label=st["label"])
        ax.set_title(f"{lab}  {arrow}", fontsize=10)
        ax.set_xlabel("α")
        ax.margins(x=0.03)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"RQ1 — accent strength, identity & leakage vs α  (step {final_step:,})",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, out)


def fig_rq2_trajectory(long_df, footer_df, traj_alpha, out):
    """accent_cs at matched α + WER/LID leakage-onset, vs training step."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    # nearest available alpha to the requested matched value
    alphas = sorted(long_df.alpha.unique())
    a = min(alphas, key=lambda x: abs(x - traj_alpha))
    # panel 1: accent_cs vs step at matched alpha
    for ref, st in REF_STYLE.items():
        p = _pool(long_df[(long_df.ref_kind == ref) & (long_df.alpha == a)], "accent_cs",
                  by=("ref_kind", "step")).sort_values("step")
        if not p.empty:
            axes[0].fill_between(p.step, p.lo, p.hi, color=st["c"], alpha=0.12, lw=0)
            axes[0].plot(p.step, p["mean"], st["ls"], color=st["c"], marker=st["m"],
                         ms=4, lw=2, label=st["label"])
    axes[0].set_title(f"Accent similarity @ α={a:g}  ↑", fontsize=10)
    # panels 2-3: leakage onset vs step (higher = tolerates more α before leaking)
    for ax, key, lab in ((axes[1], "wer_leak_onset", "WER leakage-onset α  ↑"),
                         (axes[2], "lid_leak_onset", "LID leakage-onset α  ↑")):
        for ref, st in REF_STYLE.items():
            f = footer_df[footer_df.ref_kind == ref]
            if key not in f.columns:
                continue
            g = f.groupby("step")[key].mean().reset_index().sort_values("step")
            ax.plot(g.step, g[key], st["ls"], color=st["c"], marker=st["m"], ms=4, lw=2,
                    label=st["label"])
        ax.set_title(lab, fontsize=10)
    for ax in axes:
        ax.set_xlabel("training step")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("RQ2 — does the accent arrive before the language leaks?", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, out)


def fig_rq3_decomp(long_df, final_step, out):
    """Segmental vs suprasegmental gap-closure vs α (final checkpoint, pooled)."""
    if "seg_closure" not in long_df.columns:
        print("[plot] no rq3 columns; skipping RQ3 figure")
        return
    d = long_df[long_df.step == final_step]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    series = [("seg_closure", "Segmental (PPG-KL)", "#0072B2", "-", "o"),
              ("supra_closure_mean", "Suprasegmental (F0/rhythm)", "#D55E00", "--", "s")]
    for ax, ref in zip(axes, ("l1", "native")):
        for col, lab, c, ls, mk in series:
            p = _pool(d[d.ref_kind == ref], col).sort_values("alpha")
            if not p.empty:
                ax.plot(p.alpha, p["mean"], ls, color=c, marker=mk, ms=5, lw=2, label=lab)
        ax.axhline(0, color=MUTED, lw=0.8, ls=":")
        ax.set_title(REF_STYLE[ref]["label"], fontsize=10)
        ax.set_xlabel("α")
    axes[0].set_ylabel("gap-closure toward natural  (1 = fully closed)")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"RQ3 — segmental vs suprasegmental transfer  (step {final_step:,})",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, out)


def _save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out.with_suffix('.pdf')} (+ .png)")


# --- summary table ---------------------------------------------------------------
def print_summary(footer_df, long_df, final_step, traj_alpha):
    print(f"\n=== summary @ final step {final_step:,} ===")
    hdr = ["ref_kind", "spearman_accent_cs", "spearman_spk_sim", "wer_leak_onset", "lid_leak_onset"]
    print("  " + " | ".join(f"{h:>18}" for h in hdr))
    for ref in ("l1", "native"):
        f = footer_df[(footer_df.ref_kind == ref) & (footer_df.step == final_step)]
        if f.empty:
            continue
        vals = [ref] + [f"{f[c].mean():.3f}" if c in f.columns else "—" for c in hdr[1:]]
        print("  " + " | ".join(f"{v:>18}" for v in vals))
    # RQ3 headline: seg vs supra closure at the peak-accent alpha
    if "seg_closure" in long_df.columns:
        d = long_df[long_df.step == final_step]
        peak_a = (d.groupby("alpha")["accent_cs"].mean().idxmax()
                  if "accent_cs" in d else traj_alpha)
        print(f"\n=== RQ3 gap-closure @ peak-accent α={peak_a:g} (final step) ===")
        for ref in ("l1", "native"):
            dd = d[(d.ref_kind == ref) & (d.alpha == peak_a)]
            seg = dd["seg_closure"].mean(); sup = dd["supra_closure_mean"].mean()
            print(f"  {ref:>7}:  seg_closure={seg:+.3f}   supra_closure_mean={sup:+.3f}"
                  f"   ({'seg-dominated' if seg > sup else 'supra≥seg'})")


def main():
    p = argparse.ArgumentParser(description="RQ analysis + figures over the metric CSVs")
    p.add_argument("--root", required=True, help="results/<accent>/<tag> (contains l1/ native/)")
    p.add_argument("--out-dir", help="figure dir (default <root>/figures)")
    p.add_argument("--traj-alpha", type=float, default=0.3,
                   help="matched α for the RQ2 trajectory panel (snapped to nearest run α)")
    a = p.parse_args()
    root = Path(a.root)
    out_dir = Path(a.out_dir) if a.out_dir else root / "figures"

    long_df, footer_df = load_tree(root)
    final_step = int(long_df.step.max())
    print(f"[plot] {root}: {long_df.ref_kind.nunique()} ref_kinds, "
          f"{sorted(long_df.step.unique())} steps, α={sorted(long_df.alpha.unique())}")

    fig_rq1_alpha(long_df, final_step, out_dir / "fig_rq1_alpha")
    fig_rq2_trajectory(long_df, footer_df, a.traj_alpha, out_dir / "fig_rq2_trajectory")
    fig_rq3_decomp(long_df, final_step, out_dir / "fig_rq3_decomp")
    print_summary(footer_df, long_df, final_step, a.traj_alpha)


if __name__ == "__main__":
    main()
