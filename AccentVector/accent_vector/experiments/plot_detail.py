"""Full-detail figures over the RQ metric CSVs -- nothing pooled, nothing abstracted.

``plot_results`` renders the three headline figures (one line per ref_kind, speakers
pooled, composite indices only). This module is its opposite: EVERY metric column that
exists in the CSVs gets its own panel, every speaker keeps its own line, every training
checkpoint is shown, and every composite (``supra_closure_mean``) is decomposed into the
per-feature quantities it averages.

    python -m accent_vector.experiments.plot_detail --root results/hindi/lr3e5_r16

Figures written to <root>/figures/detail/:

  fig_rq1_detail        final-checkpoint alpha curves, one panel per rq1/utmos metric,
                        one line per (ref_kind x speaker) -- 4 lines, never averaged
  fig_rq1_by_step       step x alpha heatmap per metric per ref_kind (speakers averaged
                        only here, where a cell must be a single number) -- every
                        checkpoint visible at once
  fig_rq2_detail        rows = metric, cols = alpha; x = training step. The whole
                        checkpoint x alpha x metric cube as small multiples
  fig_rq2_onsets        WER / LID leakage onset vs step, per ref_kind x speaker, with
                        CENSORED markers where the signal never crossed threshold
                        (plot_results silently drops those -- see _read_footer)
  fig_rq3_raw           every rq3 feature in ITS OWN UNITS (Hz, %, runs/s, KL) vs alpha,
                        with the natural-target value drawn as a rule
  fig_rq3_closure       every per-feature gap-closure vs alpha on a symlog axis, plus
                        the supra_closure_mean it feeds -- shows which feature's blown-up
                        ratio is driving the composite
  fig_rq3_seg_by_step   segmental KL + closure across every checkpoint

Also writes detail_long.csv: the tidy (ref_kind, speaker, step, alpha, metric, value)
table behind every panel, so any number in any figure can be traced back.

Design: two-colour categorical palette (blue = L1 ref, orange = neutral GAE ref) --
validated all-pairs for CVD (worst ΔE 24.7 protan, 33.6 normal vision) -- with speaker
carried redundantly by line style and marker, so colour is never the only cue. Alpha is
faceted rather than colour-mapped (a 7-step ordinal ramp fails the adjacent-lightness
gate). Heatmaps use a single-hue blue ramp for unsigned magnitudes and a blue/red
diverging ramp with a neutral-grey midpoint for signed gap-closure, always with a
colourbar. Recessive grid and axes; thin marks.
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

from accent_vector.experiments.plot_results import load_tree

# --- palette -------------------------------------------------------------------
REF = {
    "l1":     {"c": "#2a78d6", "label": "L1 reference"},
    "native": {"c": "#eb6834", "label": "Neutral (GAE) reference"},
}
SPK = {"m": {"ls": "-", "m": "o"}, "f": {"ls": "--", "m": "s"}}

INK, SECOND, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list(
    "seqblue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("divbr", ["#d03b3b", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": SECOND, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.7,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.titlesize": 8.5, "legend.frameon": False, "figure.autolayout": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

# --- metric registry: column -> (pretty label, unit, better-direction) ----------
RQ1_METRICS = [
    ("accent_cs", "Accent similarity", "GenAID cos", "↑"),
    ("spk_sim",   "Speaker similarity", "cos to reference", "↑"),
    ("eng_lid",   "P(English)", "VoxLingua107 posterior", "↑"),
    ("wer",       "WER", "Whisper base.en", "↓"),
    ("utmos",     "UTMOS", "predicted MOS", "↑"),
]
# the six suprasegmental descriptors, in their own units, + the segmental distance
RQ3_RAW = [
    ("seg_ppg_kl_to_natural", "Segmental PPG-KL to natural", "sym. KL (nats)", "↓"),
    ("pct_voiced",  "Voiced fraction (%V proxy)", "fraction of frames", "→ natural"),
    ("npvi_voiced", "nPVI over voiced runs", "nPVI", "→ natural"),
    ("artic_rate",  "Articulation rate", "voiced runs / s", "→ natural"),
    ("f0_mean",     "F0 mean", "Hz", "→ natural"),
    ("f0_std",      "F0 std", "Hz", "→ natural"),
    ("f0_range",    "F0 range", "Hz", "→ natural"),
]
SUPRA = ["pct_voiced", "npvi_voiced", "artic_rate", "f0_mean", "f0_std", "f0_range"]
RQ3_CLOSURE = ([("seg_closure", "Segmental closure (PPG-KL)")]
               + [(f"{k}_closure", RQ3_RAW[i + 1][1]) for i, k in enumerate(SUPRA)]
               + [("supra_closure_mean", "supra_closure_mean (the composite)")])


# --- natural-target recovery ----------------------------------------------------
def recover_natural(df):
    """{(ref_kind, speaker, step, feature): x_natural} for the six supra features.

    rq3.csv stores the synthesised value x_a and the closure
    c_a = (x_a - x_0)/(x_nat - x_0) but NOT x_nat itself. It is exactly recoverable:
        x_nat = x_0 + (x_a - x_0)/c_a
    Every alpha with a non-degenerate closure gives the same answer (verified to
    machine precision), so we take the median over alphas for numerical safety.
    """
    out = {}
    keys = ["ref_kind", "speaker", "step"]
    for (ref, spk, step), g in df.groupby(keys):
        g = g.sort_values("alpha")
        for feat in SUPRA:
            cc = f"{feat}_closure"
            if feat not in g or cc not in g:
                continue
            base = g[feat].iloc[0]
            ests = []
            for x, c in zip(g[feat].iloc[1:], g[cc].iloc[1:]):
                if pd.notna(c) and abs(c) > 1e-9 and pd.notna(x):
                    ests.append(base + (x - base) / c)
            if ests:
                out[(ref, spk, step, feat)] = float(np.median(ests))
    return out


# --- shared drawing helpers ------------------------------------------------------
def _series(ax, df, col, x="alpha"):
    """One line per (ref_kind, speaker). Colour = ref_kind, style+marker = speaker."""
    drawn = False
    for ref, rs in REF.items():
        for spk, ss in SPK.items():
            d = df[(df.ref_kind == ref) & (df.speaker == spk)].dropna(subset=[col])
            if d.empty:
                continue
            d = d.sort_values(x)
            ax.plot(d[x], d[col], ss["ls"], color=rs["c"], marker=ss["m"], ms=3.4,
                    lw=1.4, mew=0, alpha=0.95)
            drawn = True
    return drawn


def _legend_handles():
    h = [Line2D([], [], color=r["c"], lw=2.2, label=r["label"]) for r in REF.values()]
    h += [Line2D([], [], color=MUTED, lw=1.4, ls=s["ls"], marker=s["m"], ms=3.4,
                 label=f"speaker {k}") for k, s in SPK.items()]
    return h


def _figure_legend(fig, ncol=4, y=0.0):
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=ncol,
               fontsize=8, bbox_to_anchor=(0.5, y))


def _save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[detail] wrote {out.with_suffix('.pdf')} (+ .png)")


def _grid(nrow, ncol, w=2.35, h=1.95, **kw):
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), squeeze=False, **kw)
    return fig, axes


def _heat(ax, M, alphas, steps, title, bottom, left, title_size=8, **im_kw):
    """One step x alpha cell grid. Tick labels only on the outer edges -- with a
    5-row stack the inner labels collide with the panel titles above them."""
    im = ax.imshow(M, aspect="auto", origin="lower", **im_kw)
    ax.set_xticks(range(len(alphas)), [f"{a:g}" for a in alphas] if bottom else [],
                  fontsize=7)
    ax.set_yticks(range(len(steps)), [f"{s//1000}k" for s in steps] if left else [],
                  fontsize=7)
    ax.grid(False)
    ax.set_title(title, color=INK, fontsize=title_size)
    if bottom:
        ax.set_xlabel("α")
    if left:
        ax.set_ylabel("training step")
    return im


def _cell_matrix(sub, col, steps, alphas):
    """step x alpha matrix of `col`; speakers averaged, since a cell is one number."""
    M = np.full((len(steps), len(alphas)), np.nan)
    piv = sub.groupby(["step", "alpha"])[col].mean()
    for i, s in enumerate(steps):
        for j, a in enumerate(alphas):
            if (s, a) in piv.index:
                M[i, j] = piv.loc[(s, a)]
    return M


# --- figure 1: RQ1, every metric, every speaker, final checkpoint -----------------
def fig_rq1_detail(df, final_step, out):
    d = df[df.step == final_step]
    cols = [m for m in RQ1_METRICS if m[0] in d.columns]
    fig, axes = _grid(1, len(cols), w=2.5, h=2.35)
    for ax, (col, lab, unit, arrow) in zip(axes[0], cols):
        _series(ax, d, col)
        ax.set_title(f"{lab}  {arrow}", color=INK)
        ax.set_xlabel("α")
        ax.set_ylabel(unit, fontsize=7.5)
        ax.margins(x=0.04)
    fig.suptitle(f"RQ1 — every metric, every speaker, no pooling  (step {final_step:,})",
                 fontsize=10.5, y=1.04)
    _figure_legend(fig, y=-0.10)
    fig.tight_layout()
    _save(fig, out)


# --- figure 2: RQ1 across every checkpoint, as heatmaps ---------------------------
def fig_rq1_by_step(df, out):
    cols = [m for m in RQ1_METRICS if m[0] in df.columns]
    steps = sorted(df.step.unique())
    alphas = sorted(df.alpha.unique())
    fig, axes = _grid(len(cols), 2, w=3.4, h=1.9, layout="constrained")
    for r, (col, lab, unit, arrow) in enumerate(cols):
        # one shared colour scale per metric row so the two panels are comparable
        vals = df[col].dropna()
        vmin, vmax = (float(vals.min()), float(vals.max())) if len(vals) else (0, 1)
        bottom = r == len(cols) - 1
        for c, ref in enumerate(("l1", "native")):
            im = _heat(axes[r][c], _cell_matrix(df[df.ref_kind == ref], col, steps, alphas),
                       alphas, steps, f"{lab} {arrow} — {REF[ref]['label']}",
                       bottom, c == 0, cmap=SEQ, vmin=vmin, vmax=vmax)
        cb = fig.colorbar(im, ax=list(axes[r]), fraction=0.035, pad=0.015)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_visible(False)
    fig.suptitle("RQ1 — every metric at every checkpoint × α  (speakers averaged per cell)",
                 fontsize=10.5)
    _save(fig, out)


# --- figure 3: RQ2, the full checkpoint x alpha x metric cube ---------------------
def fig_rq2_detail(df, out):
    cols = [m for m in RQ1_METRICS if m[0] in df.columns]
    alphas = sorted(df.alpha.unique())
    fig, axes = _grid(len(cols), len(alphas), w=1.95, h=1.7, sharex=True)
    for r, (col, lab, unit, arrow) in enumerate(cols):
        lo = float(df[col].min()) if df[col].notna().any() else 0.0
        hi = float(df[col].max()) if df[col].notna().any() else 1.0
        pad = 0.06 * (hi - lo or 1.0)
        for c, a in enumerate(alphas):
            ax = axes[r][c]
            _series(ax, df[df.alpha == a], col, x="step")
            ax.set_ylim(lo - pad, hi + pad)          # shared y across the row
            ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
            if r == 0:
                ax.set_title(f"α = {a:g}", color=INK)
            if c == 0:
                ax.set_ylabel(f"{lab} {arrow}", fontsize=8, color=INK)
            else:
                ax.tick_params(labelleft=False)
            if r == len(cols) - 1:
                ax.set_xlabel("step")
    fig.suptitle("RQ2 — every metric vs training step, at every α, per speaker "
                 "(row-shared y-scale)", fontsize=10.5, y=1.005)
    _figure_legend(fig, y=-0.03)
    fig.tight_layout()
    _save(fig, out)


# --- figure 4: leakage onsets, with censoring made explicit -----------------------
def fig_rq2_onsets(footer, df, out):
    """Onset vs step. Where an onset is absent the signal NEVER crossed threshold in
    the swept range -- that is a censored observation, not missing data, and is drawn
    at the top of the axis with an open marker rather than dropped."""
    keys = [("wer_leak_onset", "WER leakage-onset α  ↑", "WER never reached 0.5"),
            ("lid_leak_onset", "LID leakage-onset α  ↑", "P(English) never fell below 0.5")]
    max_a = float(df.alpha.max())
    fig, axes = _grid(1, 2, w=4.4, h=3.0)
    for ax, (key, lab, censor_note) in zip(axes[0], keys):
        all_steps = sorted(footer.step.unique())
        for ref, rs in REF.items():
            for spk, ss in SPK.items():
                f = footer[(footer.ref_kind == ref) & (footer.speaker == spk)]
                if f.empty:
                    continue
                f = f.set_index("step").reindex(all_steps)
                y = f[key] if key in f.columns else pd.Series(np.nan, index=all_steps)
                obs = y.dropna()
                if len(obs):
                    ax.plot(obs.index, obs.values, ss["ls"], color=rs["c"],
                            marker=ss["m"], ms=3.6, lw=1.4, mew=0)
                cens = [s for s in all_steps if pd.isna(y.get(s, np.nan))]
                if cens:                                    # censored: never leaked
                    ax.plot(cens, [max_a * 1.06] * len(cens), ss["m"], mfc="none",
                            mec=rs["c"], mew=1.1, ms=5, ls="none")
        ax.axhline(max_a, color=AXIS, lw=0.8, ls=":")
        ax.text(0.01, max_a * 1.09, f"censored — {censor_note}", fontsize=7,
                color=SECOND, transform=ax.get_yaxis_transform(), va="bottom")
        ax.set_title(lab, color=INK)
        ax.set_xlabel("training step")
        ax.set_ylabel("α at which content is treated as leaked")
        ax.set_ylim(-0.03, max_a * 1.22)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    fig.suptitle("RQ2 — leakage onset vs training step (open markers = never leaked)",
                 fontsize=10.5, y=1.02)
    _figure_legend(fig, y=-0.06)
    fig.tight_layout()
    _save(fig, out)


# --- figure 5: RQ3 raw features in their own units --------------------------------
def fig_rq3_raw(df, final_step, natural, out):
    d = df[df.step == final_step]
    cols = [m for m in RQ3_RAW if m[0] in d.columns]
    fig, axes = _grid(2, 4, w=2.6, h=2.25)
    flat = [ax for row in axes for ax in row]
    for ax, (col, lab, unit, arrow) in zip(flat, cols):
        _series(ax, d, col)
        # the natural-target value this feature is supposed to move toward
        if col in SUPRA:
            for ref, rs in REF.items():
                for spk in SPK:
                    v = natural.get((ref, spk, final_step, col))
                    if v is not None:
                        ax.axhline(v, color=rs["c"], lw=0.9, ls=":", alpha=0.75)
            ax.text(0.02, 0.04, "dotted = natural target", transform=ax.transAxes,
                    fontsize=6.8, color=SECOND)
        ax.set_title(f"{lab}  {arrow}", color=INK, fontsize=8)
        ax.set_xlabel("α")
        ax.set_ylabel(unit, fontsize=7.5)
        ax.margins(x=0.04)
    for ax in flat[len(cols):]:
        ax.set_visible(False)
    fig.suptitle(f"RQ3 — every feature in its own units, with the natural target  "
                 f"(step {final_step:,})", fontsize=10.5, y=1.01)
    _figure_legend(fig, y=-0.02)
    fig.tight_layout()
    _save(fig, out)


# --- figure 6: RQ3 per-feature closures, decomposing the composite -----------------
def fig_rq3_closure(df, final_step, out):
    d = df[df.step == final_step]
    cols = [c for c in RQ3_CLOSURE if c[0] in d.columns]
    fig, axes = _grid(2, 4, w=2.6, h=2.25)
    flat = [ax for row in axes for ax in row]
    for ax, (col, lab) in zip(flat, cols):
        _series(ax, d, col)
        ax.axhline(0, color=AXIS, lw=0.8, ls=":")       # no movement
        ax.axhline(1, color="#0ca30c", lw=0.8, ls="--", alpha=0.7)  # fully closed
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.9)
        ax.set_title(lab, color=INK, fontsize=8)
        ax.set_xlabel("α")
        ax.margins(x=0.04)
        is_composite = col == "supra_closure_mean"
        if is_composite:
            for s in ax.spines.values():
                s.set_edgecolor(INK)
                s.set_linewidth(1.2)
    axes[0][0].set_ylabel("gap closure  (0 = no move, 1 = reached natural)", fontsize=7.5)
    for ax in flat[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("RQ3 — per-feature gap closure (symlog; green dash = 1.0 = fully closed).  "
                 "The boxed panel is the mean of the six before it.",
                 fontsize=10.5, y=1.01)
    _figure_legend(fig, y=-0.02)
    fig.tight_layout()
    _save(fig, out)


# --- figure 7: segmental across every checkpoint -----------------------------------
def fig_rq3_seg_by_step(df, out):
    steps = sorted(df.step.unique())
    alphas = sorted(df.alpha.unique())
    pairs = [("seg_ppg_kl_to_natural", "Segmental PPG-KL to natural (raw)", SEQ, False),
             ("seg_closure", "Segmental gap closure (signed)", DIV, True)]
    pairs = [p for p in pairs if p[0] in df.columns]
    if not pairs:
        return
    fig, axes = _grid(len(pairs), 2, w=3.4, h=2.1, layout="constrained")
    for r, (col, lab, cmap, signed) in enumerate(pairs):
        vals = df[col].dropna()
        vmin, vmax = float(vals.min()), float(vals.max())
        # signed closure: diverging ramp pinned so the neutral grey sits exactly on 0
        kw = ({"cmap": cmap, "norm": TwoSlopeNorm(vmin=min(vmin, -1e-6), vcenter=0.0,
                                                  vmax=max(vmax, 1e-6))} if signed else
              {"cmap": cmap, "vmin": vmin, "vmax": vmax})
        bottom = r == len(pairs) - 1
        for c, ref in enumerate(("l1", "native")):
            im = _heat(axes[r][c], _cell_matrix(df[df.ref_kind == ref], col, steps, alphas),
                       alphas, steps, f"{lab}\n{REF[ref]['label']}", bottom, c == 0,
                       title_size=7.5, **kw)
        cb = fig.colorbar(im, ax=list(axes[r]), fraction=0.035, pad=0.015)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_visible(False)
    fig.suptitle("RQ3 — segmental transfer across every checkpoint × α", fontsize=10.5)
    _save(fig, out)


# --- tidy dump -------------------------------------------------------------------
def write_long(df, natural, out_csv):
    id_cols = ["ref_kind", "speaker", "step", "alpha"]
    metrics = [c for c in df.columns if c not in id_cols + ["n"]]
    long = df.melt(id_vars=id_cols, value_vars=metrics,
                   var_name="metric", value_name="value").dropna(subset=["value"])
    long = long.sort_values(id_cols + ["metric"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_csv, index=False)
    nat_csv = Path(out_csv).with_name("natural_targets.csv")
    with open(nat_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ref_kind", "speaker", "step", "feature", "x_natural"])
        for (ref, spk, step, feat), v in sorted(natural.items()):
            w.writerow([ref, spk, step, feat, f"{v:.6g}"])
    print(f"[detail] wrote {out_csv} ({len(long):,} rows) and {nat_csv}")


def main():
    p = argparse.ArgumentParser(description="Full-detail RQ figures (nothing pooled)")
    p.add_argument("--root", required=True, help="results/<accent>/<tag> (contains l1/ native/)")
    p.add_argument("--out-dir", help="default <root>/figures/detail")
    a = p.parse_args()
    root = Path(a.root)
    out = Path(a.out_dir) if a.out_dir else root / "figures" / "detail"

    df, footer = load_tree(root)
    final_step = int(df.step.max())
    natural = recover_natural(df)
    print(f"[detail] {root}: {sorted(df.ref_kind.unique())} × {sorted(df.speaker.unique())} × "
          f"{len(df.step.unique())} steps × {len(df.alpha.unique())} α = {len(df)} rows; "
          f"final step {final_step:,}")

    fig_rq1_detail(df, final_step, out / "fig_rq1_detail")
    fig_rq1_by_step(df, out / "fig_rq1_by_step")
    fig_rq2_detail(df, out / "fig_rq2_detail")
    fig_rq2_onsets(footer, df, out / "fig_rq2_onsets")
    fig_rq3_raw(df, final_step, natural, out / "fig_rq3_raw")
    fig_rq3_closure(df, final_step, out / "fig_rq3_closure")
    fig_rq3_seg_by_step(df, out / "fig_rq3_seg_by_step")
    write_long(df, natural, out / "detail_long.csv")


if __name__ == "__main__":
    main()
