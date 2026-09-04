#!/usr/bin/env python3
"""Analyse the accentedness listening test from Qualtrics exports structured as: 
One CSV per accent, columns QID<q>_<k>: q = utterance, k = the alpha slider.

To be called:
    python analyse_listening.py --csv "export/*.csv" --out-dir out

Outputs to --out-dir:
    listening_summary.csv   accent x alpha: n, mean, sd, se, 95% CI
    listening_long.csv      one row per (participant, utterance, alpha) rating
    fig_listening.pdf/.png  per-accent means with 95% CI against alpha
  and prints, per accent: retained/excluded counts, the peak alpha, a Friedman
  test across the five conditions, and pairwise Wilcoxon tests against alpha=0
  with Holm correction.

IMPORTANT -- slider k = 1..5 maps to --alphas in the order given.
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#222222", "#666666", "#cccccc"
ACCENT_ORDER = ["dutch", "hindi", "bengali", "arabic", "mandarin"]
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 13,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})

# Qualtrics exports the two rating blocks under different prefixes: the first as
# QID<n>_<k>, the second as Q<n>_<k> (its ImportIds are the long numeric QIDs).
# Matching only the first silently halves the data, so capture both and keep the
# block identity -- what the blocks MEAN (speaker? prompt condition?) is a survey
# design fact the caller supplies via --blocks.
RATING = re.compile(r"^(QID|Q)(\d+)_(\d+)$")
NON_RATING = {"Q1", "Q2"}                     # consent / free-text items


def read_export(path, alphas, female_qs, blocks, only_block=None):
    """Qualtrics CSV -> long format. Rows 2-3 of the file are header cruft.

    The survey exports TWO rating blocks: QID<n>_<k> (block A) and Q<n>_<k>
    (block B, whose ImportIds are the long numeric QIDs). They are different
    prompt conditions, not different speakers -- pass --only-block to analyse one.
    Within a block, <n> is the utterance and <k> the alpha slider.
    """
    df = pd.read_csv(path, skiprows=[1, 2], low_memory=False)
    if "Finished" in df:                       # present in some exports, not all
        df = df[df["Finished"].astype(str).str.lower().isin(["true", "1"])]
    if "Q2" in df:                             # consent item, when present
        df = df[df["Q2"].astype(str).str.contains("consent", case=False, na=False)]
    pid = "ResponseId" if "ResponseId" in df else df.columns[0]

    matched = [(c, m) for c, m in ((c, RATING.match(c)) for c in df.columns)
               if m and c not in NON_RATING]
    if not matched:
        return None
    rows = []
    for col, m in matched:
        block = "A" if m.group(1) == "QID" else "B"
        if only_block and block != only_block:
            continue
        q, k = int(m.group(2)), int(m.group(3))
        if k > len(alphas):
            raise SystemExit(f"{path}: slider index {k} exceeds --alphas ({len(alphas)})")
        rows.append(pd.DataFrame({
            "participant": df[pid].values, "block": block,
            "condition": blocks.get(block, block), "utterance": q,
            "speaker": "F" if q in female_qs else "M", "alpha": alphas[k - 1],
            "rating": pd.to_numeric(df[col], errors="coerce").values,
        }))
    if not rows:
        return None
    # dropna also removes the all-empty rows Qualtrics pads exports with.
    return pd.concat(rows, ignore_index=True).dropna(subset=["rating"])


def screen(per_pa, threshold):
    """MUSHRA-style post-screening: drop raters whose 5-condition profile does not
    track the group's. Correlation is leave-one-out so a rater cannot inflate the
    mean they are compared against. Returns (kept_ids, dropped_ids)."""
    wide = per_pa.pivot(index="participant", columns="alpha", values="rating")
    kept, dropped = [], []
    for pid, row in wide.iterrows():
        others = wide.drop(index=pid).mean(axis=0)
        ok = row.notna() & others.notna()
        if ok.sum() < 3 or row[ok].std() == 0:
            dropped.append(pid)
            continue
        r = stats.spearmanr(row[ok], others[ok]).statistic
        (kept if np.isfinite(r) and r > threshold else dropped).append(pid)
    return kept, dropped


def ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return np.nan
    return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)


def analyse(accent, long_df, threshold):
    # Aggregate to one value per participant per alpha FIRST: ratings within a
    # participant are dependent, so the participant is the unit for CIs and tests.
    per_pa = (long_df.groupby(["participant", "alpha"], as_index=False)["rating"].mean())
    n_before = per_pa.participant.nunique()
    if threshold is not None:
        kept, dropped = screen(per_pa, threshold)
        per_pa = per_pa[per_pa.participant.isin(kept)]
    else:
        dropped = []
    n_after = per_pa.participant.nunique()

    wide = per_pa.pivot(index="participant", columns="alpha", values="rating").dropna()
    alphas = list(wide.columns)
    summary = []
    for a in alphas:
        v = wide[a].to_numpy()
        summary.append(dict(accent=accent, alpha=a, n=len(v), mean=v.mean(),
                            sd=v.std(ddof=1), se=v.std(ddof=1) / np.sqrt(len(v)),
                            ci95=ci95(v)))
    s = pd.DataFrame(summary)
    s["ci_lo"], s["ci_hi"] = s["mean"] - s["ci95"], s["mean"] + s["ci95"]

    print(f"\n=== {accent} ===")
    print(f"  participants: {n_before} finished -> {n_after} retained "
          f"({len(dropped)} excluded by post-screening)")
    peak = s.loc[s["mean"].idxmax()]
    print(f"  peak rating at alpha = {peak.alpha:g}  ({peak['mean']:.1f})")

    # Omnibus: Friedman (non-parametric repeated measures over the 5 conditions)
    chi2, p = stats.friedmanchisquare(*[wide[a].to_numpy() for a in alphas])
    print(f"  Friedman: chi2({len(alphas)-1}) = {chi2:.2f}, p = {p:.2g}")

    # Post-hoc: each alpha vs alpha = 0, Wilcoxon signed-rank, Holm-corrected
    ref = alphas[0]
    raw = [(a, stats.wilcoxon(wide[a], wide[ref]).pvalue) for a in alphas[1:]]
    order = np.argsort([q for _, q in raw])
    holm, m = {}, len(raw)
    running = 0.0
    for rank, idx in enumerate(order):
        a, q = raw[idx]
        running = max(running, min(1.0, q * (m - rank)))
        holm[a] = running
    for a, q in raw:
        d = wide[a].mean() - wide[ref].mean()
        print(f"    alpha {a:g} vs {ref:g}: delta = {d:+.1f}, "
              f"p_raw = {q:.3g}, p_holm = {holm[a]:.3g}"
              f"{'  *' if holm[a] < 0.05 else ''}")

    # Rater agreement: mean pairwise Spearman between 5-condition profiles.
    prof = wide.to_numpy()
    rs = [stats.spearmanr(prof[i], prof[j]).statistic
          for i in range(len(prof)) for j in range(i + 1, len(prof))]
    rs = [r for r in rs if np.isfinite(r)]
    print(f"  mean pairwise inter-rater Spearman: {np.mean(rs):.2f}" if rs else "")
    return s, n_before, n_after, wide


def figure(summary, wides, out):
    """Left: mean similarity per accent. Right: the PAIRED difference from alpha=0,
    which is the quantity the Wilcoxon tests actually evaluate. Independent CIs on
    the left panel overlap even where a within-participant contrast is significant,
    so the right panel is what a reader should judge differences from."""
    accs = [a for a in ACCENT_ORDER if a in wides] + \
           [a for a in wides if a not in ACCENT_ORDER]
    cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]
    marks = ["o", "s", "^", "D", "P"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))

    for i, acc in enumerate(accs):
        c, mk = cols[i % len(cols)], marks[i % len(marks)]
        d = summary[summary.accent == acc].sort_values("alpha")
        n = int(d["n"].iloc[0])
        axes[0].errorbar(d.alpha, d["mean"], yerr=d["ci95"], color=c, marker=mk,
                         ms=5, lw=1.8, capsize=3, elinewidth=1.0,
                         label=f"{acc.capitalize()} (n={n})")
        # paired differences from alpha = 0, per participant
        w = wides[acc]
        ref = w.columns[0]
        diff = w.sub(w[ref], axis=0)
        m, e = diff.mean(axis=0), diff.apply(ci95, axis=0)
        axes[1].errorbar(diff.columns, m, yerr=e, color=c, marker=mk, ms=5,
                         lw=1.8, capsize=3, elinewidth=1.0)

    lo = min(summary.ci_lo.min(), 0) - 4
    hi = summary.ci_hi.max() + 4
    axes[0].set_ylim(max(0, lo), min(100, hi))
    axes[0].set_ylabel("similarity to reference accent")
    axes[0].set_title("Mean rating", fontsize=12)
    axes[0].legend(fontsize=10)

    axes[1].axhline(0, color=INK, lw=1.0, ls=(0, (1, 2)))
    axes[1].set_ylabel("change from α = 0 (paired)")
    axes[1].set_title("Paired difference from α = 0", fontsize=12)
    for ax in axes:
        ax.set_xlabel("accent strength α")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(Path(out).with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {Path(out).with_suffix('.pdf')} (+ .png)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="glob for the per-accent exports")
    p.add_argument("--out-dir", default="out")
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1",
                   help="alpha for slider k = 1..n, in order")
    p.add_argument("--blocks", default="A:L1,B:GAE",
                   help="label per rating block. Block A = the QID<n>_<k> columns "
                        "(L1 prompt, RQ1); block B = the Q<n>_<k> columns (GAE, RQ2).")
    p.add_argument("--only-block", default="A", choices=["A", "B", "both"],
                   help="which block to analyse; 'A' for the RQ1 (L1 prompt) chapter")
    p.add_argument("--female-questions", default="1-5",
                   help="utterance numbers from the female speaker within a block")
    p.add_argument("--screen-threshold", default="0.0",
                   help="drop raters whose profile correlates at or below this with "
                        "the group mean; 'none' disables post-screening")
    a = p.parse_args()

    alphas = [float(x) for x in a.alphas.split(",")]
    blocks = dict(kv.split(":", 1) for kv in a.blocks.split(","))
    lo, hi = (int(x) for x in a.female_questions.split("-"))
    female_qs = set(range(lo, hi + 1))
    only = None if a.only_block == "both" else a.only_block
    print(f"[config] block(s): {a.only_block}  ->  {blocks}   alphas: {alphas}")
    thr = None if a.screen_threshold.lower() == "none" else float(a.screen_threshold)

    paths = sorted(glob.glob(a.csv))
    if not paths:
        raise SystemExit(f"no files matched {a.csv}")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summaries, longs, wides = [], [], {}
    for path in paths:
        accent = re.sub(r"^mock_|\.csv$", "", Path(path).name).lower()
        long_df = read_export(path, alphas, female_qs, blocks, only)
        if long_df is None:
            print(f"[skip] {path}: no rating columns")
            continue
        long_df.insert(0, "accent", accent)
        longs.append(long_df)
        s, n0, n1, wide = analyse(accent, long_df, thr)
        summaries.append(s)
        wides[accent] = wide

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out / "listening_summary.csv", index=False)
    pd.concat(longs, ignore_index=True).to_csv(out / "listening_long.csv", index=False)
    print(f"\nwrote {out / 'listening_summary.csv'} and {out / 'listening_long.csv'}")
    figure(summary, wides, out / "fig_listening")
    ns = ", ".join(f"{a.capitalize()} {int(summary[summary.accent==a].n.iloc[0])}"
                   for a in summary.accent.unique())
    print(f"\n[caption] retained participants per accent: {ns}.")
    print("[caption] The natural-speech anchor was played as a reference but not "
          "rated, so no ceiling value is available.")


if __name__ == "__main__":
    main()
