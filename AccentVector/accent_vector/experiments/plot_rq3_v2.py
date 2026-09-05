"""RQ3 figures on the corrected scale -- everything derivable from the CSVs we already have.

The headline RQ3 figure puts a *distance ratio* (seg_closure) and a *signed scalar
projection* (supra_closure_mean) on one axis as though a value of 1 meant the same
thing in both. It does not, and the composite is an unbounded mean of ratios that one
near-degenerate denominator can dominate (Dutch f0_mean_closure = 27.7).

This module rebuilds RQ3 as **two distances to the natural target, each divided by its
own alpha=0 baseline**, so both channels mean the same thing:

    r(alpha) = d(alpha) / d(0)      1 = no movement, <1 = toward natural, >1 = away

  segmental    d = mean symmetric PPG-KL to natural           (already in rq3.csv)
  prosodic     d = RMS over the six scaled descriptors of
                   (x_alpha - x_natural) / s_feature          (built here)

`s_feature` is the SD of the alpha=0 baseline for that descriptor pooled over every
(accent, ref_kind, speaker, step) -- a between-condition SD standing in for the
within-natural-speech SD, which rq3.csv does not store. Documented, not hidden.
The contour metrics (f0_rmse in cents, the PPG-KL warping path) need a re-run of the
eval and are deliberately absent here.

    python -m accent_vector.experiments.plot_rq3_v2
    python -m accent_vector.experiments.plot_rq3_v2 --accents dutch hindi --alpha 0.5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from accent_vector.experiments.plot_results import load_tree

SUPRA = ["pct_voiced", "npvi_voiced", "artic_rate", "f0_mean", "f0_std", "f0_range"]
SUPRA_LABEL = {
    "pct_voiced": "Voiced fraction (%V proxy)", "npvi_voiced": "nPVI over voiced runs",
    "artic_rate": "Articulation rate", "f0_mean": "F0 mean", "f0_std": "F0 std",
    "f0_range": "F0 range",
}
SUPRA_UNIT = {
    "pct_voiced": "fraction of frames", "npvi_voiced": "nPVI", "artic_rate": "voiced runs / s",
    "f0_mean": "Hz", "f0_std": "Hz", "f0_range": "Hz",
}

INK, MUTED, GRID = "#222222", "#666666", "#cccccc"
SEG_C, SUP_C = "#0072B2", "#D55E00"
ACCENT_STYLE = {
    "dutch":   {"c": "#0072B2", "ls": "-",  "m": "o", "label": "Dutch"},
    "bengali": {"c": "#D55E00", "ls": "--", "m": "s", "label": "Bengali"},
    "arabic":  {"c": "#009E73", "ls": "-.", "m": "^", "label": "Arabic"},
    "hindi":   {"c": "#CC79A7", "ls": ":",  "m": "D", "label": "Hindi"},
    "mandarin": {"c": "#56B4E9", "ls": (0, (5, 1)), "m": "P", "label": "Mandarin"},
}
REF_LABEL = {"l1": "L1 prompt", "native": "GAE prompt"}
REF_SLUG = {"l1": "l1", "native": "gae"}
WER_UNUSABLE = 0.20     # above this the output is not usable English any more

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 13,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.autolayout": False,
})


# --- recovering what rq3.csv did not store --------------------------------------
def recover_natural(df):
    """{(ref,spk,step,feat): x_natural}. rq3.csv keeps x_alpha and the closure
    c = (x_alpha - x_0)/(x_nat - x_0) but not x_nat; invert it. Every non-degenerate
    alpha gives the same answer, so take the median for numerical safety."""
    out = {}
    for (ref, spk, step), g in df.groupby(["ref_kind", "speaker", "step"]):
        g = g.sort_values("alpha")
        for feat in SUPRA:
            cc = f"{feat}_closure"
            if feat not in g or cc not in g:
                continue
            base = g[feat].iloc[0]
            ests = [base + (x - base) / c
                    for x, c in zip(g[feat].iloc[1:], g[cc].iloc[1:])
                    if pd.notna(c) and abs(c) > 1e-9 and pd.notna(x)]
            if ests:
                out[(ref, spk, step, feat)] = float(np.median(ests))
    return out


def feature_scales(data):
    """SD of each descriptor's alpha=0 baseline pooled over every condition."""
    base = {f: [] for f in SUPRA}
    for df in data.values():
        for _, g in df.groupby(["ref_kind", "speaker", "step"]):
            g = g.sort_values("alpha")
            for f in SUPRA:
                if f in g and pd.notna(g[f].iloc[0]):
                    base[f].append(float(g[f].iloc[0]))
    return {f: (float(np.std(v)) if len(v) > 1 and np.std(v) > 0 else np.nan)
            for f, v in base.items()}


def relative_distances(df, scales):
    """Add seg_rel and sup_rel: distance-to-natural at alpha, over the same at alpha=0."""
    nat = recover_natural(df)
    df = df.copy()
    df["seg_rel"] = np.nan
    df["sup_rel"] = np.nan
    for (ref, spk, step), g in df.groupby(["ref_kind", "speaker", "step"]):
        g = g.sort_values("alpha")
        idx = g.index
        kl = g["seg_ppg_kl_to_natural"].to_numpy(dtype=float)
        if np.isfinite(kl).all() and kl[0] > 0:
            df.loc[idx, "seg_rel"] = kl / kl[0]
        d = []
        for _, row in g.iterrows():
            terms = []
            for f in SUPRA:
                xn = nat.get((ref, spk, step, f))
                s = scales.get(f)
                if xn is None or s is None or not np.isfinite(s) or pd.isna(row.get(f)):
                    continue
                terms.append(((row[f] - xn) / s) ** 2)
            d.append(np.sqrt(np.mean(terms)) if terms else np.nan)
        d = np.asarray(d, dtype=float)
        if np.isfinite(d).all() and d[0] > 0:
            df.loc[idx, "sup_rel"] = d / d[0]
    return df, nat


def noise_sd(values):
    """Robust SD of checkpoint-to-checkpoint jitter (successive-difference estimator)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 4:
        return np.nan
    dd = np.diff(v)
    return float(1.4826 * np.median(np.abs(dd - np.median(dd))) / np.sqrt(2))


def pooled(df, col, ref, step):
    g = df[(df.ref_kind == ref) & (df.step == step)].dropna(subset=[col])
    if g.empty:
        return pd.DataFrame(columns=["alpha", "mean", "lo", "hi"])
    return (g.groupby("alpha")[col].agg(mean="mean", lo="min", hi="max")
             .reset_index().sort_values("alpha"))


def leak_alpha(df, ref, step):
    """Lowest alpha whose pooled WER exceeds WER_UNUSABLE (nan if it never does)."""
    g = df[(df.ref_kind == ref) & (df.step == step)]
    if "wer" not in g or g.empty:
        return np.nan
    p = g.groupby("alpha")["wer"].mean().sort_index()
    over = p[p > WER_UNUSABLE]
    return float(over.index[0]) if len(over) else np.nan


def ratio_axis(ax, which="y", ticks=(0.5, 1, 2, 4), lim=None):
    """Log axis in ratio units with clean labels -- matplotlib's default log minor
    ticks render as '6 x 10^-1' and collide with the decade labels."""
    for w in which:
        axis = ax.yaxis if w == "y" else ax.xaxis
        (ax.set_yscale if w == "y" else ax.set_xscale)("log")
        (ax.set_yticks if w == "y" else ax.set_xticks)(list(ticks))
        (ax.set_yticklabels if w == "y" else ax.set_xticklabels)([f"{t:g}" for t in ticks])
        axis.set_minor_locator(plt.NullLocator())
        axis.set_minor_formatter(plt.NullFormatter())
        if lim:
            (ax.set_ylim if w == "y" else ax.set_xlim)(*lim)


def _save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[rq3v2] wrote {out.with_suffix('.pdf')} (+ .png)")


# --- figure 1: both channels as relative distance, log axis ----------------------
def fig_relative(data, finals, ref, out):
    """2x2 accent small multiples. y = d(alpha)/d(0), log. 1.0 = the vector moved nothing."""
    accs = [a for a in data if not pooled(data[a], "seg_rel", ref, finals[a]).empty]
    if not accs:
        print(f"[rq3v2] no seg_rel for ref={ref}; skipping")
        return
    ncol = 2
    nrow = int(np.ceil(len(accs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.15 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    flat = [ax for r in axes for ax in r]
    for ax, acc in zip(flat, accs):
        df, step = data[acc], finals[acc]
        la = leak_alpha(df, ref, step)
        if np.isfinite(la):
            ax.axvspan(la, 1.02, color=INK, alpha=.055, lw=0)
            ax.text(min(la + .02, .97), 3.4, "WER > 20%", fontsize=11, color=MUTED,
                    ha="left", va="top")
        ax.axhline(1.0, color=INK, lw=1.0, ls=(0, (1, 2)))
        for col, c, ls, mk, lab in ((("sup_rel"), SUP_C, "--", "s", "prosodic"),
                                    (("seg_rel"), SEG_C, "-", "o", "segmental")):
            p = pooled(df, col, ref, step)
            if p.empty:
                continue
            # +/- 2 sigma from checkpoint-to-checkpoint jitter at each alpha
            errs = []
            for a in p.alpha:
                s = df[(df.ref_kind == ref) & (df.alpha == a)].groupby("step")[col].mean()
                errs.append(2 * (noise_sd(s.sort_index().to_numpy()) or np.nan))
            ax.errorbar(p.alpha, p["mean"], yerr=errs, fmt=ls, color=c, marker=mk,
                        ms=4.5, lw=1.9, elinewidth=1.0, capsize=2.4, label=lab)
        ratio_axis(ax, "y", (0.5, 1, 2, 4), (0.42, 4.6))
        ax.set_title(ACCENT_STYLE.get(acc, {}).get("label", acc), fontsize=14)
        ax.margins(x=.04)
    n = len(accs)
    # spare slots: blanked but NOT set_visible(False), so one can host the legend
    for ax in flat[n:]:
        ax.axis("off")
    # the last DATA axis in each column carries the x-axis (flat index = r*ncol + c);
    # with an odd accent count the bottom row is short, so axes[-1] alone leaves a
    # column mute. Keyed on the data count, not visibility, since the legend slot
    # stays "visible".
    for c in range(ncol):
        idxs = [r * ncol + c for r in range(nrow) if r * ncol + c < n]
        if idxs:
            flat[idxs[-1]].set_xlabel("accent strength α")
            flat[idxs[-1]].tick_params(labelbottom=True)
    for r in axes:
        r[0].set_ylabel("relative distance")
    handles = [Line2D([], [], color=SEG_C, ls="-", marker="o", ms=4.5, lw=1.9,
                      label="Segmental — PPG-KL to natural"),
               Line2D([], [], color=SUP_C, ls="--", marker="s", ms=4.5, lw=1.9,
                      label="Prosodic — scaled descriptor distance"),
               Line2D([], [], color=INK, ls=(0, (1, 2)), lw=1.0,
                      label="1.0 = the vector moved nothing")]
    if n < len(flat):
        # an odd accent count leaves a grid slot free: put the legend in it rather
        # than adding a strip of height below the figure.
        flat[n].legend(handles=handles, loc="center", fontsize=12,
                       frameon=False, handlelength=2.8, labelspacing=1.0)
        fig.tight_layout()
    else:
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=12,
                   bbox_to_anchor=(0.5, -0.07))
        fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, out)


# --- figure 2: the movement plane -------------------------------------------------
def fig_plane(data, finals, out):
    """x = segmental relative distance, y = prosodic. Baseline sits at (1,1);
    natural is the origin. The direction of travel IS the answer to RQ3."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5), sharex=True, sharey=True)
    for ax, ref in zip(axes, ("l1", "native")):
        ax.axhline(1, color=GRID, lw=1.0)
        ax.axvline(1, color=GRID, lw=1.0)
        ax.plot([1], [1], marker="+", ms=11, color=INK, mew=1.4, ls="none", zorder=4)
        ax.annotate("α = 0\n(no movement)", (1, 1), textcoords="offset points",
                    xytext=(-8, -22), fontsize=7.5, color=MUTED, ha="right")
        for acc, df in data.items():
            st = ACCENT_STYLE.get(acc, {"c": INK, "ls": "-", "m": "o", "label": acc})
            step = finals[acc]
            xs = pooled(df, "seg_rel", ref, step)
            ys = pooled(df, "sup_rel", ref, step)
            if xs.empty or ys.empty:
                continue
            m = xs.merge(ys, on="alpha", suffixes=("_s", "_p")).sort_values("alpha")
            ax.plot(m["mean_s"], m["mean_p"], color=st["c"], linestyle=st["ls"],
                    marker=st["m"], ms=4.5, lw=1.6, label=st["label"], zorder=3)
            last = m.iloc[-1]
            ax.annotate("1.0", (last["mean_s"], last["mean_p"]), fontsize=7,
                        color=st["c"], textcoords="offset points", xytext=(4, -3))
        ratio_axis(ax, "x", (0.5, 1, 2, 4), (0.45, 5.0))
        ratio_axis(ax, "y", (0.5, 1, 2, 4), (0.45, 5.0))
        ax.set_xlabel("segmental  ←  toward natural")
        ax.set_title(REF_LABEL[ref], fontsize=10.5)
    axes[0].set_ylabel("prosodic  ↓  toward natural")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("RQ3 — the movement plane: which way does the vector travel?  "
                 "(origin = natural speech; markers are α)", fontsize=11.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out)


# --- figure 3: raw units, with the gap the vector was meant to close --------------
def fig_gap(df, nat, acc, step, out):
    """Every descriptor in its own units, with the alpha=0 baseline, the natural
    target, and the band between them shaded -- the distance the vector must travel."""
    cols = [("seg_ppg_kl_to_natural", "Segmental PPG-KL to natural", "sym. KL (nats)")] + \
           [(f, SUPRA_LABEL[f], SUPRA_UNIT[f]) for f in SUPRA]
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.6), squeeze=False)
    flat = [ax for r in axes for ax in r]
    for ax, (col, lab, unit) in zip(flat, cols):
        for ref, c, ls, mk in (("l1", "#0072B2", "-", "o"), ("native", "#D55E00", "--", "s")):
            p = pooled(df, col, ref, step)
            if p.empty:
                continue
            ax.plot(p.alpha, p["mean"], ls, color=c, marker=mk, ms=3.8, lw=1.7)
            if col in SUPRA:
                base = float(p["mean"].iloc[0])
                tgt = np.nanmean([v for (r_, s_, st_, f_), v in nat.items()
                                  if r_ == ref and st_ == step and f_ == col])
                if np.isfinite(tgt):
                    lo, hi = sorted((base, tgt))
                    ax.axhspan(lo, hi, color=c, alpha=.10, lw=0)
                    ax.axhline(tgt, color=c, lw=1.0, ls=":")
        ax.set_title(lab, fontsize=8.5)
        ax.set_xlabel("α"); ax.set_ylabel(unit, fontsize=7.5)
        ax.margins(x=.04)
    for ax in flat[len(cols):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color="#0072B2", ls="-", marker="o", ms=3.8, label="L1 prompt"),
               Line2D([], [], color="#D55E00", ls="--", marker="s", ms=3.8, label="GAE prompt"),
               Line2D([], [], color=MUTED, ls=":", lw=1.0, label="natural target (shaded = the gap)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"RQ3 — {ACCENT_STYLE.get(acc, {}).get('label', acc)}: every descriptor in its "
                 f"own units, with the gap to close  ({step:,} steps)", fontsize=11.5, y=1.01)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    _save(fig, out)


# --- figure 4: is the movement a training trend or checkpoint noise? --------------
def fig_heat(df, acc, out):
    """step x alpha heatmap of each channel's relative distance. A real effect is a
    smooth field; noise is speckle."""
    steps = sorted(df.step.unique())
    alphas = sorted(df.alpha.unique())
    if len(steps) < 3:
        print(f"[rq3v2] {acc}: <3 checkpoints; skipping heatmap")
        return
    rows = [("seg_rel", "Segmental"), ("sup_rel", "Prosodic")]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.4), squeeze=False, layout="constrained")
    for r, (col, lab) in enumerate(rows):
        vals = df[col].dropna()
        if vals.empty:
            continue
        # ratios are multiplicative: make the colour scale symmetric in LOG space,
        # else a channel that moves *toward* natural saturates at the blue end.
        hi = float(np.nanpercentile(vals, 98))
        lo = float(np.nanpercentile(vals, 2))
        vmax = max(hi, 1.0 / lo if lo > 0 else hi)
        for c, ref in enumerate(("l1", "native")):
            sub = df[df.ref_kind == ref]
            M = np.full((len(steps), len(alphas)), np.nan)
            piv = sub.groupby(["step", "alpha"])[col].mean()
            for i, s in enumerate(steps):
                for j, a in enumerate(alphas):
                    if (s, a) in piv.index:
                        M[i, j] = piv.loc[(s, a)]
            ax = axes[r][c]
            im = ax.imshow(np.log10(M), aspect="auto", origin="lower", cmap="RdBu_r",
                           vmin=-np.log10(vmax), vmax=np.log10(vmax))
            ax.set_xticks(range(len(alphas)),
                          [f"{a:g}" for a in alphas] if r == 1 else [], fontsize=7)
            ax.set_yticks(range(len(steps)),
                          [f"{s // 1000}k" for s in steps] if c == 0 else [], fontsize=7)
            ax.grid(False)
            ax.set_title(f"{lab} — {REF_LABEL[ref]}", fontsize=8.5)
            if r == 1:
                ax.set_xlabel("α")
            if c == 0:
                ax.set_ylabel("training step")
        cb = fig.colorbar(im, ax=list(axes[r]), fraction=.035, pad=.015)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_visible(False)
        ticks = [1 / vmax, 1 / np.sqrt(vmax), 1.0, np.sqrt(vmax), vmax]
        cb.set_ticks(np.log10(ticks))
        cb.set_ticklabels([f"{t:.2f}" for t in ticks])
        cb.set_label("d(α)/d(0)", fontsize=7.5)
    fig.suptitle(f"RQ3 — {ACCENT_STYLE.get(acc, {}).get('label', acc)}: relative distance "
                 f"across every checkpoint × α  (white = 1.0 = no movement)", fontsize=11)
    _save(fig, out)


# --- figure 5: one row per accent, both channels, at a matched alpha ---------------
def fig_summary(data, finals, alpha, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4), sharex=True)
    accs = list(data)
    for ax, ref in zip(axes, ("l1", "native")):
        ax.axvline(1.0, color=INK, lw=1.0, ls=(0, (1, 2)))
        for i, acc in enumerate(accs):
            df, step = data[acc], finals[acc]
            a = min(sorted(df.alpha.unique()), key=lambda v: abs(v - alpha))
            sub = df[(df.ref_kind == ref) & (df.step == step) & (df.alpha == a)]
            if sub.empty:
                continue
            s_, p_ = sub["seg_rel"].mean(), sub["sup_rel"].mean()
            if np.isfinite(s_) and np.isfinite(p_):
                ax.plot([min(s_, p_), max(s_, p_)], [i, i], color=GRID, lw=2.4, zorder=1)
            ax.plot([s_], [i], marker="o", ms=8, color=SEG_C, zorder=3)
            ax.plot([p_], [i], marker="s", ms=8, color=SUP_C, zorder=3)
        ax.set_yticks(range(len(accs)),
                      [ACCENT_STYLE.get(a_, {}).get("label", a_) for a_ in accs], fontsize=9)
        ax.set_ylim(len(accs) - .5, -.6)
        ratio_axis(ax, "x", (0.5, 1, 2), (0.45, 2.3))
        ax.set_xlabel("distance to natural ÷ baseline")
        ax.set_title(REF_LABEL[ref], fontsize=10.5)
    handles = [Line2D([], [], color=SEG_C, marker="o", ms=8, ls="none", label="Segmental"),
               Line2D([], [], color=SUP_C, marker="s", ms=8, ls="none", label="Prosodic"),
               Line2D([], [], color=INK, ls=(0, (1, 2)), lw=1.0, label="no movement")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(f"RQ3 — both channels at α ≈ {alpha:g}, final checkpoint per accent",
                 fontsize=11.5, y=1.02)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _save(fig, out)


# --- driver -----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="results")
    p.add_argument("--tag", default="lr3e5_r16")
    p.add_argument("--accents", nargs="+", default=["dutch", "hindi", "bengali", "arabic", "mandarin"])
    p.add_argument("--out-dir", default="results/figures/rq3_v2")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="matched α for the cross-accent summary figure")
    a = p.parse_args()
    out = Path(a.out_dir)

    raw = {}
    for acc in a.accents:
        root = Path(a.results_root) / acc / a.tag
        if not root.is_dir():
            print(f"[rq3v2] skip {acc}: no {root}")
            continue
        try:
            long_df, _ = load_tree(root)
        except SystemExit as e:
            print(f"[rq3v2] skip {acc}: {e}")
            continue
        raw[acc] = long_df
    if not raw:
        raise SystemExit("no accent trees loaded")

    scales = feature_scales(raw)
    print("[rq3v2] per-feature scales (SD of the α=0 baseline, pooled over conditions):")
    for f, s in scales.items():
        print(f"          {f:<14} {s:.4g}  ({SUPRA_UNIT[f]})")

    data, nats, finals = {}, {}, {}
    for acc, df in raw.items():
        d, nat = relative_distances(df, scales)
        data[acc], nats[acc] = d, nat
        finals[acc] = int(d.step.max())

    for ref in ("l1", "native"):
        fig_relative(data, finals, ref, out / f"fig_rq3_relative_{REF_SLUG[ref]}")
    fig_plane(data, finals, out / "fig_rq3_plane")
    fig_summary(data, finals, a.alpha, out / "fig_rq3_summary")
    for acc in data:
        fig_gap(data[acc], nats[acc], acc, finals[acc], out / f"fig_rq3_gap_{acc}")
        fig_heat(data[acc], acc, out / f"fig_rq3_heat_{acc}")

    # printed table + a tidy CSV of the corrected scale
    print(f"\n=== RQ3 relative distance d(α)/d(0)  —  1.0 = the vector moved nothing ===")
    print(f"  {'accent':>8} {'ref':>7} {'α':>5} {'segmental':>10} {'prosodic':>9}  {'WER%':>6}")
    tidy = []
    for acc, df in data.items():
        for ref in ("l1", "native"):
            step = finals[acc]
            sub = df[(df.ref_kind == ref) & (df.step == step)]
            for al in sorted(sub.alpha.unique()):
                s = sub[sub.alpha == al]
                sr, pr = s["seg_rel"].mean(), s["sup_rel"].mean()
                wer = s["wer"].mean() * 100 if "wer" in s else np.nan
                print(f"  {acc:>8} {ref:>7} {al:>5.2f} {sr:>10.3f} {pr:>9.3f}  {wer:>6.1f}")
                tidy.append(dict(accent=acc, ref_kind=ref, step=step, alpha=al,
                                 seg_rel=sr, sup_rel=pr, wer=wer / 100 if np.isfinite(wer) else np.nan))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tidy).to_csv(out / "rq3_relative.csv", index=False)
    print(f"\n[rq3v2] wrote {out / 'rq3_relative.csv'}")


if __name__ == "__main__":
    main()


# --- figure 1b: both prompts in one panel set -------------------------------------
def fig_relative_both(data, finals, out):
    """Both reference conditions in one figure. Colour = channel (blue segmental,
    orange prosodic); line style + marker fill = prompt (solid/filled L1, dashed/open
    GAE). Each prompt's leakage onset is a vertical rule in its own style, replacing
    the shaded band that two overlapping conditions would make unreadable."""
    accs = [a for a in data
            if not (pooled(data[a], "seg_rel", "l1", finals[a]).empty
                    and pooled(data[a], "seg_rel", "native", finals[a]).empty)]
    if not accs:
        print("[rq3v2] nothing to plot")
        return
    ncol = 2
    nrow = int(np.ceil((len(accs) + 1) / ncol))      # +1 reserves a legend slot
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.15 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    flat = [ax for r in axes for ax in r]
    REFS = (("l1", "-", True), ("native", "--", False))
    for ax, acc in zip(flat, accs):
        df, step = data[acc], finals[acc]
        ax.axhline(1.0, color=INK, lw=1.0, ls=(0, (1, 2)), zorder=1)
        for ref, ls, filled in REFS:
            la = leak_alpha(df, ref, step)
            if np.isfinite(la):
                ax.axvline(la, color=MUTED, ls=ls, lw=1.1, alpha=.7, zorder=1)
            for col, c, mk in (("sup_rel", SUP_C, "s"), ("seg_rel", SEG_C, "o")):
                p = pooled(df, col, ref, step)
                if p.empty:
                    continue
                errs = []
                for a in p.alpha:
                    s = df[(df.ref_kind == ref) & (df.alpha == a)].groupby("step")[col].mean()
                    errs.append(2 * (noise_sd(s.sort_index().to_numpy()) or np.nan))
                ax.errorbar(p.alpha, p["mean"], yerr=errs, color=c, ls=ls, marker=mk,
                            ms=4.2, lw=1.7, elinewidth=0.8, capsize=2.0, zorder=3,
                            markerfacecolor=(c if filled else "white"),
                            markeredgecolor=c, markeredgewidth=1.1)
        ratio_axis(ax, "y", (0.5, 1, 2, 4), (0.42, 4.6))
        ax.set_title(ACCENT_STYLE.get(acc, {}).get("label", acc), fontsize=14)
        ax.margins(x=.04)
    n = len(accs)
    for ax in flat[n:]:
        ax.axis("off")
    for c in range(ncol):
        idxs = [r * ncol + c for r in range(nrow) if r * ncol + c < n]
        if idxs:
            flat[idxs[-1]].set_xlabel("accent strength α")
            flat[idxs[-1]].tick_params(labelbottom=True)
    for r in axes:
        r[0].set_ylabel("relative distance")

    def _h(c, ls, mk, filled, lab):
        return Line2D([], [], color=c, ls=ls, marker=mk, ms=4.2, lw=1.7, label=lab,
                      markerfacecolor=(c if filled else "white"),
                      markeredgecolor=c, markeredgewidth=1.1)
    handles = [_h(SEG_C, "-", "o", True, "Segmental — L1 prompt"),
               _h(SEG_C, "--", "o", False, "Segmental — GAE prompt"),
               _h(SUP_C, "-", "s", True, "Prosodic — L1 prompt"),
               _h(SUP_C, "--", "s", False, "Prosodic — GAE prompt"),
               Line2D([], [], color=MUTED, ls="-", lw=1.1, label="leakage onset (WER > 20%)"),
               Line2D([], [], color=INK, ls=(0, (1, 2)), lw=1.0, label="1.0 = no movement")]
    if n < len(flat):
        flat[n].legend(handles=handles, loc="center", fontsize=11, frameon=False,
                       handlelength=2.8, labelspacing=.9)
        fig.tight_layout()
    else:
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
                   bbox_to_anchor=(0.5, -0.07))
        fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, out)
