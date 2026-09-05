"""Dissertation chapter figures (Ch.2 accented TTS, Ch.3 accent conversion, Ch.4 trajectory).

Reads the per-sweep CSVs written by the eval array --
    results/<accent>/<tag>/<ref_kind>/<speaker>/metrics/step_<N>/{rq1,rq3,utmos}.csv
-- pools the m/f speakers, and renders the figures called for in the structure doc:

  Ch.2 "Accent vector on F5-TTS"  (L1 prompt  -> ref_kind=l1)
        ch2_<accent>_l1        one figure per accent at its final checkpoint: 2x3 grid
                               of WER, accent CS, LID, KL-PPG, SS, UTMOS vs alpha
                               (Figures 2-5 in the doc)
        ch2_all_accents_l1     2x2 overview, one line per accent -- WER, accent CS,
                               LID and speaker sim only (see OVERVIEW_PANELS)
  Ch.3 "Accent conversion"       (GAE prompt -> ref_kind=native)
        ch3_<accent>_gae, ch3_all_accents_gae   -- identical layouts, GAE prompt
  Ch.4 "Trajectory mapping"      per accent, x = training step, y = slope of the metric
        ch4_slopes_<accent>_<ref>   2x3 grid, one line per alpha, zero-slope rule and a
                                    marker at the step where the metric stops moving
        ch4_stabilisation_<ref>     step at which each metric stabilises, per accent

    python -m accent_vector.experiments.plot_dissertation
    python -m accent_vector.experiments.plot_dissertation --accents dutch hindi --chapters 2 4

Design: Okabe-Ito CVD-safe palette with a redundant marker/line-style per series (colour
is never the only cue), recessive grid/axes, one legend per figure, thin marks.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from accent_vector.experiments.plot_results import load_tree

# --- style ----------------------------------------------------------------------
INK, MUTED, GRID = "#222222", "#666666", "#cccccc"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.autolayout": False,
})

# Okabe-Ito, one entry per accent: colour + a redundant marker/line-style cue.
ACCENT_STYLE = {
    "dutch":   {"c": "#0072B2", "ls": "-",   "m": "o", "label": "Dutch"},
    "bengali": {"c": "#D55E00", "ls": "--",  "m": "s", "label": "Bengali"},
    "arabic":  {"c": "#009E73", "ls": "-.",  "m": "^", "label": "Arabic"},
    "hindi":   {"c": "#CC79A7", "ls": ":",   "m": "D", "label": "Hindi"},
    "british": {"c": "#56B4E9", "ls": (0, (3, 1, 1, 1)), "m": "v", "label": "British"},
    "mandarin": {"c": "#E69F00", "ls": (0, (5, 1)), "m": "P", "label": "Mandarin"},
}
SPEAKER_STYLE = {"m": {"ls": "--", "m": "v", "label": "male prompt"},
                 "f": {"ls": ":",  "m": "^", "label": "female prompt"}}
# alpha lines on the Ch.4 slope panels (redundant style, not a colour ramp)
ALPHA_STYLE = [("#0072B2", "-", "o"), ("#D55E00", "--", "s"), ("#009E73", "-.", "^"),
               ("#CC79A7", ":", "D"), ("#E69F00", (0, (3, 1, 1, 1)), "v")]

REF_LABEL = {"l1": "L1 prompt", "native": "GAE prompt"}
REF_SLUG = {"l1": "l1", "native": "gae"}

# (column, panel title, y-axis label, better-direction, scale factor)
PANELS = [
    ("wer",                    "Word error rate",     "WER (%)",        "↓", 100.0),
    ("accent_cs",              "Accent similarity",   "cosine sim.",    "↑", 1.0),
    ("eng_lid",                "Language ID",         "P(English)",     "↑", 1.0),
    ("seg_ppg_kl_to_natural",  "Phonetic posteriorgram KL", "sym. KL (nats)", "↓", 1.0),
    ("spk_sim",                "Speaker similarity",  "cosine sim.",    "↑", 1.0),
    ("utmos",                  "UTMOS",               "predicted MOS",  "↑", 1.0),
]

# The cross-accent overview drops KL-PPG and UTMOS: neither separates the accents
# on this figure. KL-PPG is dominated by a per-accent constant -- the ground-truth
# speaker is not the prompt speaker for four of five accents -- so its lines are five
# flat offsets, and UTMOS carries an unquantified bias against non-native accents.
# Both stay on the per-accent figures, where the alpha-trend is what is read.
OVERVIEW_DROP = ("seg_ppg_kl_to_natural", "utmos")
OVERVIEW_PANELS = [p for p in PANELS if p[0] not in OVERVIEW_DROP]


# --- loading --------------------------------------------------------------------
def load_accents(results_root, tag, accents):
    """{accent: long_df} for every accent whose <results_root>/<accent>/<tag> tree loads."""
    out = {}
    for acc in accents:
        root = Path(results_root) / acc / tag
        if not root.is_dir():
            print(f"[plot] skip {acc}: no {root}")
            continue
        try:
            long_df, _ = load_tree(root)
        except SystemExit as e:
            print(f"[plot] skip {acc}: {e}")
            continue
        out[acc] = long_df
        print(f"[plot] {acc}: refs={sorted(long_df.ref_kind.unique())} "
              f"steps={sorted(long_df.step.unique())} α={sorted(long_df.alpha.unique())}")
    return out


def _pool_alpha(df, col):
    """Speaker-pooled mean + min/max band over alpha, for one (accent, ref, step) slice."""
    d = df.dropna(subset=[col])
    if d.empty:
        return pd.DataFrame(columns=["alpha", "mean", "lo", "hi"])
    return (d.groupby("alpha")[col].agg(mean="mean", lo="min", hi="max")
             .reset_index().sort_values("alpha"))


def _panel_axes(nrows=2, ncols=3, w=3.5, h=3.0):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows))
    return fig, axes.ravel()


def _axes_for(panels, w=3.5, h=3.0):
    """Grid shaped to the panel count: 2 columns up to four panels, else 3."""
    n = len(panels)
    ncols = 2 if n <= 4 else 3
    nrows = -(-n // ncols)
    return _panel_axes(nrows, ncols, w, h)


def _finish_panel(ax, title, ylabel, arrow):
    ax.set_title(f"{title}  {arrow}", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.margins(x=0.04)


def _save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out.with_suffix('.pdf')} (+ .png)")


# --- Ch.2 / Ch.3: metric vs alpha at the final checkpoint ------------------------
def _legend_rows(fig, handles, per_row=3, y0=-0.035, dy=0.042, fontsize=9):
    """Centred multi-row figure legend.

    matplotlib packs a single `ncol` legend column-major, so a final row with fewer
    entries than columns hangs off to the left. Drawing one centred legend per row
    keeps every row centred under the panels.
    """
    rows = [handles[i:i + per_row] for i in range(0, len(handles), per_row)]
    for i, row in enumerate(rows):                    # first row on top, each next lower
        fig.add_artist(fig.legend(handles=row, loc="lower center", ncol=len(row),
                                  fontsize=fontsize,
                                  bbox_to_anchor=(0.5, y0 - dy * i)))


# The L1-prompt ceiling drawn onto the GAE figures. RQ2 reads accent gain under the GAE
# prompt as a fraction of the interval between the GAE-prompt floor (alpha=0 on the same
# panel) and this ceiling, so without it the accent-similarity panel gives no sense of
# scale -- a rise of 0.05 looks the same whether the interval is 0.03 or 0.5 wide.
CEILING_PANEL = "accent_cs"


def _ceiling(df, step, speaker=None):
    """Speaker-pooled AccentCS of the L1 prompt at alpha=0: the RQ2 ceiling."""
    d = df[(df.ref_kind == "l1") & (df.step == step) & (df.alpha == 0.0)]
    if speaker is not None:
        d = d[d.speaker == speaker]
    d = d.dropna(subset=["accent_cs"])
    return float(d.accent_cs.mean()) if not d.empty else None


def fig_alpha_single_accent(long_df, accent, ref, out, step=None):
    """2x3 metric grid vs alpha for one accent at one checkpoint (Figures 2-5 / Ch.3)."""
    d = long_df[long_df.ref_kind == ref]
    if d.empty:
        print(f"[plot] {accent}/{ref}: no rows; skipping")
        return None
    step = int(d.step.max()) if step is None else int(step)
    d = d[d.step == step]
    st = ACCENT_STYLE.get(accent, {"c": "#0072B2", "ls": "-", "m": "o", "label": accent})

    fig, axes = _panel_axes()
    drew_ceiling = False
    for ax, (col, title, ylab, arrow, scale) in zip(axes, PANELS):
        if col not in d.columns or d[col].notna().sum() == 0:
            ax.set_visible(False)
            continue
        for spk, sst in SPEAKER_STYLE.items():         # each speaker kept visible
            s = d[d.speaker == spk].dropna(subset=[col]).sort_values("alpha")
            if not s.empty:
                ax.plot(s.alpha, s[col] * scale, sst["ls"], color=MUTED, marker=sst["m"],
                        ms=3, lw=0.9, alpha=0.8)
        p = _pool_alpha(d, col)                        # pooled mean on top
        if not p.empty:
            ax.fill_between(p.alpha, p.lo * scale, p.hi * scale, color=st["c"],
                            alpha=0.12, linewidth=0)
            ax.plot(p.alpha, p["mean"] * scale, "-", color=st["c"], marker=st["m"],
                    ms=5, lw=2)
        if ref == "native" and col == CEILING_PANEL:
            ceil = _ceiling(long_df, step)
            if ceil is not None:
                ax.axhline(ceil * scale, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=1)
                drew_ceiling = True
        ax.set_xlabel("accent strength α")
        _finish_panel(ax, title, ylab, arrow)

    handles = [Line2D([], [], color=st["c"], lw=2, marker=st["m"], ms=5,
                      label=f"{st['label']} (speakers pooled)")]
    handles += [Line2D([], [], color=MUTED, lw=0.9, ls=s["ls"], marker=s["m"], ms=3,
                       label=s["label"]) for s in SPEAKER_STYLE.values()]
    if drew_ceiling:
        handles.append(Line2D([], [], color=INK, lw=1.2, ls=(0, (4, 3)),
                              label="L1-prompt ceiling (α = 0)"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 3), fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{st['label']} — {REF_LABEL[ref]}, after {step:,} steps",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    _save(fig, out)
    return step


def fig_alpha_all_accents(data, ref, out, steps=None, panels=None):
    """Metric grid vs alpha, one line per accent (each at its final checkpoint).

    `panels` defaults to OVERVIEW_PANELS (WER / accent CS / LID / speaker sim); pass
    PANELS for the full six-metric version. The grid shape follows the panel count.
    """
    panels = OVERVIEW_PANELS if panels is None else panels
    fig, axes = _axes_for(panels)
    used, drew_ceiling = {}, False
    for ax, (col, title, ylab, arrow, scale) in zip(axes, panels):
        drawn = False
        for acc, long_df in data.items():
            d = long_df[long_df.ref_kind == ref]
            if d.empty:
                continue
            step = int(d.step.max()) if not steps else int(steps.get(acc, d.step.max()))
            used[acc] = step
            d = d[d.step == step]
            if col not in d.columns:
                continue
            p = _pool_alpha(d, col)
            if p.empty:
                continue
            st = ACCENT_STYLE.get(acc, {"c": INK, "ls": "-", "m": "o", "label": acc})
            ax.plot(p.alpha, p["mean"] * scale, color=st["c"], ls=st["ls"],
                    marker=st["m"], ms=4, lw=1.8, label=st["label"])
            if ref == "native" and col == CEILING_PANEL:
                ceil = _ceiling(long_df, step)
                if ceil is not None:
                    ax.axhline(ceil * scale, color=st["c"], lw=1.0, ls=(0, (4, 3)),
                               alpha=0.8, zorder=1)
                    drew_ceiling = True
            drawn = True
        if not drawn:
            ax.set_visible(False)
            continue
        ax.set_xlabel("accent strength α")
        _finish_panel(ax, title, ylab, arrow)

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    handles = [Line2D([], [], color=ACCENT_STYLE[a]["c"], ls=ACCENT_STYLE[a]["ls"],
                      marker=ACCENT_STYLE[a]["m"], ms=4, lw=1.8,
                      label=f"{ACCENT_STYLE[a]['label']} ({used[a]:,} steps)")
               for a in used if a in ACCENT_STYLE]
    if drew_ceiling:
        handles.append(Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 3)),
                              label="L1-prompt ceiling (α = 0, per accent)"))
    # Wrap the legend onto rows of at most three: a single row of five entries is wider
    # than the panel grid, so the tight bbox pads the figure sideways and shrinks the
    # panels. Each row is a legend of its own so that a short final row stays CENTRED --
    # one multi-column legend would left-align it into the first columns instead.
    _legend_rows(fig, handles, per_row=3)
    # No title: the figure is captioned in the text, and the legend already names each
    # accent's final checkpoint.
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, out)


# --- Ch.4: slope of each metric against training step ----------------------------
def _rolling_slope(steps, values, window=5):
    """Local OLS slope (metric units per 1,000 steps) in a centred window.

    Point-to-point differences are dominated by evaluation noise (5 utterances per
    speaker per checkpoint), so each point is the slope of a least-squares line through
    the `window` checkpoints around it rather than a raw finite difference.
    """
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    half = window // 2
    out = np.full(len(steps), np.nan)
    for i in range(len(steps)):
        lo, hi = max(0, i - half), min(len(steps), i + half + 1)
        if hi - lo < 3:
            continue
        out[i] = np.polyfit(steps[lo:hi], values[lo:hi], 1)[0] * 1000.0
    return steps, out


def _noise_sd(values):
    """Robust SD of the checkpoint-to-checkpoint jitter (successive-difference estimator)."""
    d = np.diff(np.asarray(values, dtype=float))
    if len(d) < 2:
        return np.nan
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)


def _settling_step(steps, values, k=2.0, min_tail=3):
    """Earliest step from which the metric stays within the evaluation-noise band.

    The band is +/- k * sigma around the final value, where sigma is estimated from the
    checkpoint-to-checkpoint jitter of the metric itself -- so "settled" means later
    training moves the metric by no more than the eval noise, rather than by some
    arbitrary fixed fraction. `min_tail` checkpoints must remain, so the final point
    can never trivially settle.

    Returns (step, band, band_frac_of_range); step is None if it never settles.
    """
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    sigma = _noise_sd(values)
    rng = np.nanmax(values) - np.nanmin(values)
    if not np.isfinite(sigma) or sigma <= 0:
        return None, np.nan, np.nan
    band = k * sigma
    frac = band / rng if rng > 0 else np.nan
    ok = np.abs(values - values[-1]) <= band
    for i in range(len(ok) - min_tail + 1):
        if ok[i:].all():
            return steps[i], band, frac
    return None, band, frac


def fig_slopes(long_df, accent, ref, out, alphas, k_noise=2.0, window=5):
    """2x3 grid: local slope of each metric vs training step, one line per alpha."""
    d = long_df[long_df.ref_kind == ref]
    if d.empty or d.step.nunique() < 4:
        print(f"[plot] {accent}/{ref}: <4 checkpoints; skipping slope figure")
        return []
    avail = sorted(d.alpha.unique())
    picks = sorted(dict.fromkeys(min(avail, key=lambda x: abs(x - a)) for a in alphas))

    rows = []
    fig, axes = _panel_axes()
    for ax, (col, title, ylab, arrow, scale) in zip(axes, PANELS):
        if col not in d.columns or d[col].notna().sum() == 0:
            ax.set_visible(False)
            continue
        for (c, ls, mk), a in zip(ALPHA_STYLE, picks):
            g = (d[d.alpha == a].dropna(subset=[col])
                 .groupby("step")[col].mean().reset_index().sort_values("step"))
            if len(g) < 4:
                continue
            vals = g[col].to_numpy() * scale
            steps, sl = _rolling_slope(g.step.to_numpy(), vals, window)
            ax.plot(steps, sl, color=c, ls=ls, marker=mk, ms=3.5, lw=1.5, label=f"α={a:g}")
            settled, band, frac = _settling_step(g.step.to_numpy(), vals, k_noise)
            if settled is not None:
                j = int(np.where(steps == settled)[0][0])
                ax.plot([settled], [sl[j]], marker="*", ms=12, color=c,
                        markeredgecolor="white", markeredgewidth=0.6, zorder=5, ls="none")
            rows.append({"accent": accent, "ref_kind": ref, "metric": col, "alpha": a,
                         "settles_at": settled, "noise_band": band,
                         "band_frac_of_range": frac, "final_step": int(steps[-1])})
        ax.axhline(0, color=INK, lw=0.9, ls=(0, (1, 2)))
        ax.set_xlabel("training step")
        _finish_panel(ax, title, f"Δ {ylab} / 1k steps", arrow)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))

    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=3.5, lw=1.5, label=f"α={a:g}")
               for (c, ls, mk), a in zip(ALPHA_STYLE, picks)]
    handles.append(Line2D([], [], color=MUTED, marker="*", ms=12, ls="none",
                          label=f"metric settles (stays within ±{k_noise:g}σ of its "
                                f"final value thereafter; σ = eval noise)"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    st = ACCENT_STYLE.get(accent, {"label": accent})
    fig.suptitle(f"{st['label']} — rate of change per metric, {REF_LABEL[ref]} "
                 f"({window}-checkpoint local slope)", fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    _save(fig, out)
    return rows


def fig_stabilisation(rows, ref, out, alpha):
    """Dot plot: the step at which each metric settles, one row per metric, per accent."""
    df = pd.DataFrame(rows)
    df = df[(df.ref_kind == ref) & (df.alpha == alpha)]
    if df.empty:
        print(f"[plot] no stabilisation rows for {ref} @ α={alpha}; skipping")
        return
    metrics = [p[0] for p in PANELS if p[0] in set(df.metric)]
    labels = {p[0]: p[1] for p in PANELS}
    fig, ax = plt.subplots(figsize=(7.5, 0.62 * len(metrics) + 2.4))
    accs = sorted(df.accent.unique())
    # small vertical offset per accent so co-located settling steps stay countable
    dy = {a: (i - (len(accs) - 1) / 2) * 0.16 for i, a in enumerate(accs)}
    for acc, sub in df.groupby("accent"):
        st = ACCENT_STYLE.get(acc, {"c": INK, "m": "o", "label": acc})
        xs, ys, cens = [], [], []
        for m, s_, fin in zip(sub.metric, sub.settles_at, sub.final_step):
            if m not in metrics:
                continue
            y = metrics.index(m) + dy[acc]
            if pd.isna(s_):
                cens.append((fin, y))          # never settled: mark at the last checkpoint
            else:
                xs.append(s_); ys.append(y)
        ax.plot(xs, ys, ls="none", marker=st["m"], ms=8, color=st["c"], label=st["label"])
        if cens:
            cx, cy = zip(*cens)
            ax.plot(cx, cy, ls="none", marker=">", ms=8, mfc="none", color=st["c"])
    # No title: the figure is captioned in the text, and the caption carries the
    # ref_kind and alpha this panel is drawn at.
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([labels[m] for m in metrics], fontsize=13)
    ax.set_ylim(len(metrics) - 0.5, -0.7)
    ax.set_xlabel("training step at which the metric stops moving", fontsize=13)
    # k-suffixed ticks rather than a shared 1e3 exponent: at this type size the
    # offset text collides with the axis label.
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
    ax.tick_params(axis="x", labelsize=12)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], ls="none", marker=">", ms=8, mfc="none", color=MUTED,
                          label="still moving at the final checkpoint"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 3), fontsize=12,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, out)


# --- driver ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="results")
    p.add_argument("--tag", default="lr3e5_r16")
    p.add_argument("--accents", nargs="+",
                   default=["dutch", "bengali", "arabic", "hindi", "mandarin"],
                   help="order follows Figures 2-5 in the structure doc")
    p.add_argument("--out-dir", default="results/figures/dissertation")
    p.add_argument("--chapters", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--slope-alphas", nargs="+", type=float, default=[0.3, 0.5, 1.0],
                   help="alphas drawn on the Ch.4 slope panels (snapped to the run grid)")
    p.add_argument("--summary-alpha", type=float, default=0.5,
                   help="alpha used for the Ch.4 stabilisation summary")
    p.add_argument("--settle-k", type=float, default=2.0,
                   help="width of the 'settled' band, in units of the estimated eval-noise σ")
    p.add_argument("--slope-window", type=int, default=5,
                   help="checkpoints in the centred window used for the local slope")
    a = p.parse_args()

    out_dir = Path(a.out_dir)
    data = load_accents(a.results_root, a.tag, a.accents)
    if not data:
        raise SystemExit("no accent trees loaded")

    if 2 in a.chapters:
        print("\n--- Chapter 2: accented TTS (L1 prompt) ---")
        for acc, df in data.items():
            fig_alpha_single_accent(df, acc, "l1", out_dir / f"ch2_{acc}_l1")
        fig_alpha_all_accents(data, "l1", out_dir / "ch2_all_accents_l1")

    if 3 in a.chapters:
        print("\n--- Chapter 3: accent conversion (GAE prompt) ---")
        for acc, df in data.items():
            fig_alpha_single_accent(df, acc, "native", out_dir / f"ch3_{acc}_gae")
        fig_alpha_all_accents(data, "native", out_dir / "ch3_all_accents_gae")

    if 4 in a.chapters:
        print("\n--- Chapter 4: trajectory mapping ---")
        rows = []
        for ref in ("l1", "native"):
            for acc, df in data.items():
                rows += fig_slopes(df, acc, ref,
                                   out_dir / f"ch4_slopes_{acc}_{REF_SLUG[ref]}",
                                   a.slope_alphas, a.settle_k, a.slope_window)
        if rows:
            summary = pd.DataFrame(rows)
            csv_path = out_dir / "ch4_stabilisation.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(csv_path, index=False)
            print(f"[plot] wrote {csv_path}")
            alphas = sorted(summary.alpha.unique())
            alpha = min(alphas, key=lambda x: abs(x - a.summary_alpha))
            for ref in ("l1", "native"):
                fig_stabilisation(rows, ref,
                                  out_dir / f"ch4_stabilisation_{REF_SLUG[ref]}", alpha)


if __name__ == "__main__":
    main()
