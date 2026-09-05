"""Listening-test figures for Ch.2 (accentedness MUSHRA-style grid).

Consumes one tidy ratings CSV with the columns

    listener, accent, speaker, utterance, condition, rating

where `condition` is either an alpha in {0,0.25,0.5,0.75,1} or the string
"natural" (the SAA anchor), and `rating` is the 0-100 accentedness slider.

    python -m accent_vector.experiments.plot_listening --ratings ratings.csv

With no --ratings it SIMULATES a plausible dataset so the figure layout can be
judged before the test is fielded:

    python -m accent_vector.experiments.plot_listening --simulate realistic

  ideal      monotone rise toward the natural anchor, plateauing before leakage
  realistic  flat to alpha=0.5, then a rise that tracks the leakage onset
  average    flat everywhere, no contrast against alpha=0 survives
  worst      raters split at high alpha (is leakage "accent"?); agreement collapses

Figures (each written as .pdf + .png):
  ch2_listening_accentedness   5 accent panels, mean +/- bootstrap 95% CI vs alpha,
                               natural anchor band, faint per-listener lines, and a
                               rule at the leakage onset from the objective grid
  ch2_listening_validation     subjective mean vs AccentCS, one point per
                               (accent, alpha) cell, with Spearman rho
  ch2_listening_spread         per-alpha rating distributions (half-violin + points),
                               the figure that shows variance inflation / bimodality

Also writes ch2_listening_summary.csv: per (accent, alpha) n, mean, 95% CI, the
contrast against alpha=0, and Krippendorff's alpha (interval) per cell.

Design follows plot_dissertation.py: Okabe-Ito, colour never the only cue,
recessive grid, one legend per figure.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from accent_vector.experiments.plot_dissertation import ACCENT_STYLE, INK, MUTED, _save

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
ACCENTS = ["dutch", "bengali", "hindi", "mandarin", "arabic"]

# AccentCS at each alpha, final checkpoint, L1 prompt (section 2.4 of the chapter).
# Replace with a read of results/<accent>/<tag>/l1/*/metrics/step_<final>/rq1.csv
# once the listening test is fielded -- these are the published numbers only.
OBJ_ACCENT_CS = {
    "dutch":    [0.739, 0.735, 0.728, 0.712, 0.701],
    "bengali":  [0.521, 0.518, 0.509, 0.494, 0.483],
    "hindi":    [0.433, 0.441, 0.437, 0.418, 0.407],
    "mandarin": [0.478, 0.512, 0.549, 0.531, 0.522],
    "arabic":   [0.284, 0.297, 0.311, 0.324, 0.332],
}
# alpha at which P(English) collapses -- the "leakage onset" rule on each panel.
LEAKAGE_ONSET = {"dutch": 0.72, "bengali": 0.69, "hindi": 0.69,
                 "mandarin": None, "arabic": 0.75}

ANCHOR_C, ANCHOR_LS = "#666666", (0, (4, 2))


# --- simulation -----------------------------------------------------------------
def _base_rating(accent):
    """alpha=0 accentedness implied by the objective AccentCS at alpha=0."""
    return 45.0 + 30.0 * OBJ_ACCENT_CS[accent][0]


def _mean_curve(accent, scenario):
    """The population mean accentedness at each alpha, per scenario."""
    b = _base_rating(accent)
    onset = LEAKAGE_ONSET[accent] or 0.98
    span = max(1.0 - onset, 0.05)
    if scenario == "ideal":                       # real, perceptible accent gain
        return np.array([b, b + 6, b + 12, b + 11, b + 9])
    if scenario == "realistic":                   # flat, then leakage read as accent
        return np.array([b, b + 1, b + 3]
                        + [b + 14 * max(0.0, (a - onset) / span + 0.35)
                           for a in ALPHAS[3:]])
    if scenario == "average":                     # null
        return np.array([b, b + 2, b + 1, b + 4, b + 2])
    if scenario == "worst":                       # mean is meaningless; see the split
        return np.array([b, b + 1, b + 3, b + 2, b + 2])
    raise SystemExit(f"unknown scenario {scenario}")


def simulate(scenario, n_listeners=20, n_utt=5, seed=0):
    """A tidy ratings frame with listener/utterance random effects."""
    rng = np.random.default_rng(seed)
    sd_noise = {"ideal": 9.0, "realistic": 11.0, "average": 14.0, "worst": 12.0}[scenario]
    rows = []
    for accent in ACCENTS:
        curve = _mean_curve(accent, scenario)
        # utterance and clip effects are properties of the STIMULUS: drawn once and
        # shared by every listener, so they are signal that raters can agree on.
        utt_int = {(spk, u): rng.normal(0, 7)
                   for spk in ("m", "f") for u in range(n_utt)}
        clip_int = {(spk, u, a): rng.normal(0, 5)
                    for spk in ("m", "f") for u in range(n_utt) for a in ALPHAS}
        # each listener gets an intercept, and (worst case) a propensity to hear
        # leakage as accent rather than as broken English
        lis_int = rng.normal(0, 8, n_listeners)
        leak_hear = rng.normal(0, 1, n_listeners) if scenario == "worst" else np.zeros(n_listeners)
        for li in range(n_listeners):
            lid = f"{accent[:2]}{li:02d}"
            for spk in ("m", "f"):
                for u in range(n_utt):
                    for ai, a in enumerate(ALPHAS):
                        mu = (curve[ai] + lis_int[li] + utt_int[(spk, u)]
                              + clip_int[(spk, u, a)])
                        if scenario == "worst" and a >= 0.75:
                            # the split: +/- 30 points on the same clip, by rater
                            mu += 32 * np.sign(leak_hear[li]) * (a - 0.5) / 0.5
                        rows.append((lid, accent, spk, f"{spk}{u}", a,
                                     float(np.clip(rng.normal(mu, sd_noise), 0, 100))))
                    mu = 78 + lis_int[li] + utt_int[(spk, u)]
                    rows.append((lid, accent, spk, f"{spk}{u}", "natural",
                                 float(np.clip(rng.normal(mu, 8.0), 0, 100))))
    return pd.DataFrame(rows, columns=["listener", "accent", "speaker",
                                       "utterance", "condition", "rating"])


# --- statistics -----------------------------------------------------------------
def _boot_ci(x, n_boot=4000, seed=0):
    """Percentile CI on the mean, resampling listeners (the unit of independence)."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def _listener_means(df):
    """One value per listener per cell -- ratings within a listener are not independent."""
    return (df.groupby(["accent", "condition", "listener"])["rating"]
              .mean().reset_index())


def _center_by_listener(df):
    """Remove each listener's own mean: raters differ in how they use a 0-100 scale,
    and that offset is not disagreement about the stimuli. Reliability is reported
    on centred ratings; raw-score alpha is dominated by scale-use offsets."""
    df = df.copy()
    df["rating"] = df["rating"] - df.groupby("listener")["rating"].transform("mean")
    return df


def krippendorff_interval(df, center=True):
    """Interval-scale Krippendorff's alpha over units = (utterance x condition)."""
    if center:
        df = _center_by_listener(df)
    units = [g["rating"].to_numpy() for _, g in df.groupby(["utterance", "condition"])
             if len(g) >= 2]
    if len(units) < 2:
        return np.nan
    do_num = sum(sum(2 * (a - b) ** 2 for a, b in combinations(u, 2)) / (len(u) - 1)
                 for u in units)
    n = sum(len(u) for u in units)
    do = do_num / n
    allv = np.concatenate(units)
    de = (2 * np.subtract.outer(allv, allv) ** 2).sum() / (2 * n * (n - 1))
    return float(1 - do / de) if de > 0 else np.nan


def pooled_reliability(df):
    """Krippendorff's alpha per accent, pooled over the alpha conditions.

    Units are (utterance x condition): between-condition differences are part of
    what raters must agree on, which is the quantity a reader cares about.
    """
    return {a: krippendorff_interval(df[(df.accent == a) & (df.condition != "natural")])
            for a in ACCENTS}


def summarise(df):
    lm = _listener_means(df)
    rows = []
    for accent in ACCENTS:
        sub = lm[lm.accent == accent]
        ref = sub[sub.condition == 0.0].set_index("listener")["rating"]
        for cond in ALPHAS + ["natural"]:
            x = sub[sub.condition == cond].set_index("listener")["rating"]
            lo, hi = _boot_ci(x.to_numpy())
            if cond in (0.0, "natural"):
                d, dlo, dhi = (np.nan,) * 3
            else:                       # paired contrast against alpha=0
                paired = (x - ref).dropna().to_numpy()
                d = paired.mean()
                dlo, dhi = _boot_ci(paired)
            cell = df[(df.accent == accent) & (df.condition == cond)]
            rows.append(dict(accent=accent, condition=cond, n_listeners=len(x),
                             mean=x.mean(), ci_lo=lo, ci_hi=hi,
                             delta_vs_a0=d, delta_lo=dlo, delta_hi=dhi,
                             kripp_alpha=krippendorff_interval(cell)))
    return pd.DataFrame(rows)


# --- figures --------------------------------------------------------------------
def fig_accentedness(df, summary, out):
    """Panel per accent: mean +/- CI vs alpha, anchor band, per-listener spaghetti."""
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharey=True)
    axes = axes.ravel()
    lm = _listener_means(df)

    for ax, accent in zip(axes, ACCENTS):
        st = ACCENT_STYLE[accent]
        s = summary[(summary.accent == accent) & (summary.condition != "natural")]
        s = s.sort_values("condition")
        nat = summary[(summary.accent == accent) & (summary.condition == "natural")].iloc[0]

        # natural-speech anchor: band = its own CI, so "reached the anchor" is legible
        ax.axhspan(nat.ci_lo, nat.ci_hi, color=ANCHOR_C, alpha=0.13, lw=0, zorder=0)
        ax.axhline(nat["mean"], color=ANCHOR_C, ls=ANCHOR_LS, lw=1.2, zorder=1)

        onset = LEAKAGE_ONSET[accent]
        if onset is not None:
            ax.axvline(onset, color=INK, ls=(0, (1, 2)), lw=1.0, zorder=1)
            ax.annotate("leakage onset", (onset, 3), xytext=(-4, 0),
                        textcoords="offset points", rotation=90, ha="right",
                        va="bottom", fontsize=7, color=MUTED)

        for _, g in lm[lm.accent == accent].groupby("listener"):
            g = g[g.condition != "natural"].sort_values("condition")
            ax.plot(g.condition, g.rating, color=st["c"], lw=0.6, alpha=0.18, zorder=2)

        x = s.condition.to_numpy(float)
        ax.fill_between(x, s.ci_lo, s.ci_hi, color=st["c"], alpha=0.22, lw=0, zorder=3)
        ax.plot(x, s["mean"], color=st["c"], ls=st["ls"], marker=st["m"],
                ms=6, lw=2, mec="white", mew=1.0, zorder=4)

        ax.set_title(st["label"], fontsize=10)
        ax.set_xticks(ALPHAS)
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(0, 100)

    for ax in axes[3:5]:
        ax.set_xlabel(r"accent strength $\alpha$", fontsize=9)
    axes[0].set_ylabel("accentedness (0-100)", fontsize=9)
    axes[3].set_ylabel("accentedness (0-100)", fontsize=9)

    axes[5].axis("off")
    handles = [Line2D([], [], color=ACCENT_STYLE[a]["c"], ls=ACCENT_STYLE[a]["ls"],
                      marker=ACCENT_STYLE[a]["m"], ms=6, lw=2,
                      label=f"{ACCENT_STYLE[a]['label']} (mean, 95% CI)") for a in ACCENTS]
    handles += [
        Line2D([], [], color=ANCHOR_C, ls=ANCHOR_LS, lw=1.2, label="natural speech anchor"),
        Line2D([], [], color=INK, ls=(0, (1, 2)), lw=1.0, label=r"leakage onset $\alpha$"),
        Line2D([], [], color=MUTED, lw=0.8, alpha=0.4, label="individual listener"),
    ]
    axes[5].legend(handles=handles, loc="center left", fontsize=8.5, handlelength=2.6)
    fig.tight_layout()
    _save(fig, out)


def fig_validation(summary, out):
    """Does the subjective scale agree with GenAID? One point per (accent, alpha)."""
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    xs, ys, within = [], [], []
    for accent in ACCENTS:
        st = ACCENT_STYLE[accent]
        s = summary[(summary.accent == accent) & (summary.condition != "natural")]
        s = s.sort_values("condition")
        cs = OBJ_ACCENT_CS[accent]
        sizes = 26 + 130 * np.asarray(ALPHAS)        # alpha is the redundant cue
        ax.plot(cs, s["mean"], color=st["c"], ls=st["ls"], lw=1.0, alpha=0.55, zorder=2)
        r_in = spearmanr(cs, s["mean"]).statistic
        within.append(r_in)
        ax.scatter(cs, s["mean"], s=sizes, marker=st["m"], facecolor=st["c"],
                   edgecolor="white", lw=1.0, zorder=3,
                   label=rf"{st['label']}  ($\rho$ = {r_in:+.2f})")
        xs += list(cs)
        ys += list(s["mean"])

    # The pooled rho is inflated by between-accent offsets: accents sit at different
    # AccentCS levels for reasons unrelated to alpha. The within-accent rho is the
    # quantity that says whether the two scales agree about the sweep.
    rho, p = spearmanr(xs, ys)
    ax.annotate(rf"pooled $\rho$ = {rho:+.2f} (p = {p:.3f})" "\n"
                rf"mean within-accent $\rho$ = {np.mean(within):+.2f}",
                (0.03, 0.96), xycoords="axes fraction", va="top", fontsize=9, color=INK)
    ax.set_xlabel("AccentCS (GenAID embedding cosine)", fontsize=9)
    ax.set_ylabel("mean accentedness (0-100)", fontsize=9)
    ax.set_title(r"Subjective vs objective accent, per ($\alpha$, accent) cell", fontsize=10)
    ax.legend(fontsize=8, loc="lower right",
              title=r"marker size $\propto\ \alpha$;  $\rho$ = within-accent",
              title_fontsize=8)
    fig.tight_layout()
    _save(fig, out)


def fig_spread(df, summary, out):
    """Rating distributions per alpha: the variance-inflation / bimodality figure."""
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.6), sharey=True)
    rng = np.random.default_rng(1)
    pooled = pooled_reliability(df)
    for ax, accent in zip(axes, ACCENTS):
        st = ACCENT_STYLE[accent]
        data = [df[(df.accent == accent) & (df.condition == a)]["rating"].to_numpy()
                for a in ALPHAS]
        parts = ax.violinplot(data, positions=range(5), widths=0.8,
                              showextrema=False, showmedians=False)
        for b in parts["bodies"]:
            b.set_facecolor(st["c"]); b.set_alpha(0.28); b.set_edgecolor("none")
        for i, d in enumerate(data):
            ax.scatter(i + rng.normal(0, 0.055, len(d)), d, s=3, color=st["c"],
                       alpha=0.35, lw=0)
            ax.plot([i - 0.3, i + 0.3], [d.mean()] * 2, color=INK, lw=1.6, zorder=5)
        ax.set_xticks(range(5)); ax.set_xticklabels([f"{a:g}" for a in ALPHAS])
        ka = (summary[(summary.accent == accent) & (summary.condition != "natural")]
              .sort_values("condition")["kripp_alpha"].to_numpy())
        for i, k in enumerate(ka):
            ax.annotate(f"{k:.2f}", (i, 2), ha="center", va="bottom", fontsize=7,
                        color=MUTED if k > 0.4 else "#B4451F")
        ax.set_title(f"{st['label']}   (pooled $\\alpha_K$ = {pooled[accent]:.2f})",
                     fontsize=10)
        ax.set_xlabel(r"$\alpha$", fontsize=9)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("accentedness (0-100)", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=[Line2D([], [], color=INK, lw=1.6, label="cell mean"),
                        Line2D([], [], ls="none", marker="none",
                               label=r"numerals: per-$\alpha$ Krippendorff $\alpha_K$"
                                     r" (red = below 0.4)")],
               loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    _save(fig, out)


# --- main -----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ratings", help="tidy CSV; omit to simulate")
    p.add_argument("--simulate", default="realistic",
                   choices=["ideal", "realistic", "average", "worst"])
    p.add_argument("--n-listeners", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results/figures/listening")
    a = p.parse_args()

    if a.ratings:
        df = pd.read_csv(a.ratings)
        df["condition"] = df["condition"].apply(
            lambda v: v if str(v) == "natural" else float(v))
        suffix = ""
    else:
        df = simulate(a.simulate, n_listeners=a.n_listeners, seed=a.seed)
        suffix = f"_{a.simulate}"
        print(f"[listening] SIMULATED data, scenario={a.simulate} -- layout mock only")

    summary = summarise(df)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"ch2_listening_summary{suffix}.csv"
    summary.to_csv(csv, index=False)
    print(f"[listening] wrote {csv}")
    print(summary.to_string(index=False,
                            float_format=lambda v: f"{v:6.2f}"))
    print("[listening] pooled Krippendorff alpha per accent: "
          + ", ".join(f"{k}={v:.2f}" for k, v in pooled_reliability(df).items()))

    fig_accentedness(df, summary, out_dir / f"ch2_listening_accentedness{suffix}")
    fig_validation(summary, out_dir / f"ch2_listening_validation{suffix}")
    fig_spread(df, summary, out_dir / f"ch2_listening_spread{suffix}")


if __name__ == "__main__":
    main()
