"""RQ2 recovery ratio: how much of the L1-prompt accent the vector restores under GAE.

Chapter 3 defines, per accent at one checkpoint,

    R(alpha) = (CS_gae(alpha) - CS_gae(0)) / (CS_l1(0) - CS_gae(0))

with CS = AccentCS. R = 0 is the GAE-prompt floor (base model, no vector), R = 1 the
L1-prompt ceiling of Chapter 2. This module is the only place R is computed; the
chapter figure, the operating-point table and the two per-prompt-speaker appendix
tables all come out of one pass so they can never disagree.

    python -m accent_vector.experiments.rq2_recovery
    python -m accent_vector.experiments.rq2_recovery --accents dutch hindi --min-gap 0.05

Written:
  <out-dir>/ch3_recovery_gae.{pdf,png}          Ch.3 figure (R vs alpha + the intervals)
  <table-dir>/ch3_recovery_operating_point.{tex,csv}   main-text table
  <appendix-dir>/rq2_recovery_prompt_{m,f}.{tex,csv}   appendix tables

A narrow denominator is the failure mode of a ratio like this: when the L1 prompt buys
almost no accent over the GAE prompt, R divides a real difference by noise and reports
a confident-looking multiple. So the gap is carried beside R everywhere and R is
suppressed (printed as --, drawn faint) wherever the gap falls below --min-gap.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from accent_vector.experiments.plot_dissertation import (
    ACCENT_STYLE, INK, MUTED, load_accents, _save)

# Below this AccentCS gap the ceiling and the floor are not meaningfully apart and R is
# not reported. Not a significance test -- rq1.csv stores only per-alpha means over the
# n utterances, so no within-condition SD is available to build one from -- it is a
# stated legibility convention, and the gap itself is printed so a reader can re-judge.
MIN_GAP = 0.10

# Above this WER the output is not usable English any more (the threshold plot_rq3_v2
# already uses). Recovery peaks at or near alpha=1 for most accents, where WER is 24-66%
# -- a peak that cannot be operated at -- so every artefact here reports both the
# unconstrained peak and the best recovery that stays intelligible.
WER_USABLE = 0.20


# --- the ratio -------------------------------------------------------------------
def _cs_at_final(long_df, speaker=None):
    """(step, cs_l1, cs_gae, wer_gae) vs alpha at the accent's final checkpoint.

    `speaker` None pools the m/f prompts by averaging AccentCS first and taking the
    ratio of the pooled curves -- rather than averaging two per-speaker ratios, which
    would let one near-degenerate denominator dominate the mean.
    """
    d = long_df.dropna(subset=["accent_cs"])
    if speaker is not None:
        d = d[d.speaker == speaker]
    if d.empty:
        return None, None, None, None
    step = int(d.step.max())
    d = d[d.step == step]
    cs = {ref: d[d.ref_kind == ref].groupby("alpha").accent_cs.mean().sort_index()
          for ref in ("l1", "native")}
    if cs["l1"].empty or cs["native"].empty or 0.0 not in cs["native"].index:
        return None, None, None, None
    gae = d[d.ref_kind == "native"]
    wer = (gae.groupby("alpha").wer.mean().sort_index() if "wer" in gae.columns
           else pd.Series(dtype=float))
    return step, cs["l1"], cs["native"], wer


def recovery(long_df, speaker=None, min_gap=MIN_GAP):
    """Per-alpha recovery for one accent, or None if the sweep is missing.

    Returns {step, ceiling, floor, gap, valid, alpha_star, alpha_int, cs, wer, R}. R is
    a Series over alpha (all-NaN when the gap is too narrow to divide by); alpha_star is
    the best-recovery alpha and alpha_int the best that also keeps WER at or below
    WER_USABLE. Both are argmax of CS_gae over the alphas in question, which is argmax of
    R whenever R is defined -- the two differ by a positive affine map -- so they stay
    well-defined for the accents whose R is suppressed.
    """
    step, cs_l1, cs_gae, wer = _cs_at_final(long_df, speaker)
    if step is None or 0.0 not in cs_l1.index:
        return None
    ceiling, floor = float(cs_l1.loc[0.0]), float(cs_gae.loc[0.0])
    gap = ceiling - floor
    valid = gap >= min_gap
    R = (cs_gae - floor) / gap if valid else pd.Series(np.nan, index=cs_gae.index)
    pos = cs_gae[cs_gae.index > 0]
    ok = pos[[a in wer.index and wer.loc[a] <= WER_USABLE for a in pos.index]]
    return {"step": step, "ceiling": ceiling, "floor": floor, "gap": gap, "valid": valid,
            "alpha_star": float(pos.idxmax()) if not pos.empty else float("nan"),
            "alpha_int": float(ok.idxmax()) if not ok.empty else float("nan"),
            "cs": cs_gae, "wer": wer, "R": R}


def recovery_long(data, min_gap=MIN_GAP):
    """Tidy per-(accent, prompt, alpha) frame -- the CSV backing every artefact here."""
    rows = []
    for acc, df in data.items():
        for prompt in ("pooled", "m", "f"):
            r = recovery(df, None if prompt == "pooled" else prompt, min_gap)
            if r is None:
                continue
            for a in r["cs"].index:
                rows.append({"accent": acc, "prompt": prompt, "step": r["step"],
                             "alpha": float(a), "cs_gae": float(r["cs"].loc[a]),
                             "recovery": float(r["R"].loc[a]),
                             "ceiling": r["ceiling"], "floor": r["floor"],
                             "gap": r["gap"], "gap_ok": r["valid"],
                             "alpha_star": r["alpha_star"], "alpha_int": r["alpha_int"]})
    return pd.DataFrame(rows)


# --- figure ----------------------------------------------------------------------
def fig_recovery(data, out, min_gap=MIN_GAP, intervals=False):
    """R vs alpha, one line per accent.

    R is a fraction of an interval that differs fivefold across accents, and an accent
    whose interval is a hair wide can post a large R that means very little -- but the
    interval is the first three columns of the operating-point table and the dashed
    ceiling of the all-accents figure, so drawing it a third time here only asks the
    reader to reconcile three views of the same three numbers per accent. `intervals`
    puts that panel back for a version of the figure that has to stand alone.
    """
    if intervals:
        fig, (ax, axb) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                                      gridspec_kw={"width_ratios": [1.35, 1]})
    else:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        axb = None
    accs = [a for a in data if recovery(data[a], None, min_gap)]
    recs = {a: recovery(data[a], None, min_gap) for a in accs}

    # (a) the ratio
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.axhline(0.0, color=MUTED, lw=1.0)
    for acc in accs:
        r, st = recs[acc], ACCENT_STYLE.get(acc, {"c": INK, "ls": "-", "m": "o", "label": acc})
        if not r["valid"]:
            continue
        ax.plot(r["R"].index, r["R"].values, color=st["c"], ls=st["ls"], marker=st["m"],
                ms=4, lw=1.8, label=st["label"])
        a = r["alpha_star"]
        ax.plot([a], [r["R"].loc[a]], marker="o", ms=9, mfc="none", mew=1.4, color=st["c"])
        ai = r["alpha_int"]
        if np.isfinite(ai):
            ax.plot([ai], [r["R"].loc[ai]], marker="o", ms=6, color=st["c"], mew=0)
    ax.set_xlabel("accent strength α")
    ax.set_ylabel("recovery ratio $R(\\alpha)$", fontsize=9)
    ax.set_title("Recovery of the L1-prompt accent  ↑", fontsize=10)
    ax.margins(x=0.04)
    # Right-hand side: at alpha near 0 every curve is pinned to R = 0, so a label there
    # sits on top of the data.
    ax.annotate("L1-prompt ceiling", xy=(1.0, 1.0), xytext=(-2, 3), fontsize=8,
                color=MUTED, textcoords="offset points", va="bottom", ha="right")
    ax.annotate("GAE-prompt floor", xy=(1.0, 0.0), xytext=(-2, 3), fontsize=8,
                color=MUTED, textcoords="offset points", va="bottom", ha="right")

    # (b) the interval each ratio is a fraction of
    ys = np.arange(len(accs)) if axb is not None else []
    if axb is not None:
        for y, acc in zip(ys, accs):
            r, st = recs[acc], ACCENT_STYLE.get(acc, {"c": INK, "m": "o", "label": acc})
            faint = 1.0 if r["valid"] else 0.35
            axb.plot([r["floor"], r["ceiling"]], [y, y], color=st["c"], lw=3, alpha=0.35 * faint,
                     solid_capstyle="round")
            axb.plot([r["floor"]], [y], marker="|", ms=11, mew=2, color=st["c"], alpha=faint)
            axb.plot([r["ceiling"]], [y], marker="|", ms=11, mew=2, color=st["c"], alpha=faint)
            axb.plot([r["cs"].loc[r["alpha_star"]]], [y], marker="o", ms=8, mfc="none",
                     mew=1.4, color=st["c"], alpha=faint)
            if np.isfinite(r["alpha_int"]):
                axb.plot([r["cs"].loc[r["alpha_int"]]], [y], marker="o", ms=6, color=st["c"],
                         mew=0, alpha=faint)
            if not r["valid"]:
                axb.annotate(f"gap {r['gap']:.2f} < {min_gap:g}: $R$ not reported",
                             xy=(max(r["ceiling"], r["floor"]), y), xytext=(14, 0),
                             textcoords="offset points", fontsize=8, color=MUTED, va="center")
        axb.set_yticks(ys)
        axb.set_yticklabels([ACCENT_STYLE.get(a, {"label": a})["label"] for a in accs], fontsize=9)
        axb.set_ylim(len(accs) - 0.4, -0.6)
        axb.set_xlabel("AccentCS")
        axb.set_title("Floor–ceiling interval, and where α lands", fontsize=10)
        axb.margins(x=0.12)
        axb.grid(axis="y", visible=False)

    handles = [Line2D([], [], color=ACCENT_STYLE[a]["c"], ls=ACCENT_STYLE[a]["ls"],
                      marker=ACCENT_STYLE[a]["m"], ms=4, lw=1.8,
                      label=f"{ACCENT_STYLE[a]['label']} ({recs[a]['step']:,} steps)")
               for a in accs if a in ACCENT_STYLE and recs[a]["valid"]]
    # An accent with no line in panel (a) must not appear in the legend as though it had
    # one; it keeps its interval in panel (b), which is where its exclusion is legible.
    dropped = [ACCENT_STYLE.get(a, {"label": a})["label"] for a in accs if not recs[a]["valid"]]
    if dropped:
        handles.append(Line2D([], [], ls="none", marker="", label=(
            f"{', '.join(dropped)}: gap < {min_gap:g}, $R$ undefined")))
    handles += [Line2D([], [], ls="none", marker="o", ms=9, mfc="none", mew=1.4,
                       color=MUTED, label="α* (peak recovery)"),
                Line2D([], [], ls="none", marker="o", ms=6, color=MUTED, mew=0,
                       label=f"α† (peak with WER ≤ {100 * WER_USABLE:.0f}%)")]
    fig.legend(handles=handles, loc="lower center", ncol=3 if intervals else 2,
               fontsize=9, bbox_to_anchor=(0.5, -0.10 if intervals else -0.16))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, out)


# --- tables ----------------------------------------------------------------------
def _at(long_df, step, alpha, col, ref="native", speaker=None):
    """Speaker-pooled value of `col` at one (step, alpha) of the GAE sweep."""
    d = long_df[(long_df.step == step) & (long_df.ref_kind == ref)]
    if speaker is not None:
        d = d[d.speaker == speaker]
    d = d[np.isclose(d.alpha, alpha)].dropna(subset=[col]) if col in d.columns else d.iloc[:0]
    return float(d[col].mean()) if not d.empty else float("nan")


def _fmt(x, nd=3, plus=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"


def _num(x):
    """Thousands separator that survives LaTeX's maths mode."""
    return f"{int(x):,}".replace(",", "{,}")


def operating_point(data, min_gap=MIN_GAP):
    """One row per accent: the interval, the best-recovery alpha, and what it costs."""
    rows = []
    for acc, df in data.items():
        r = recovery(df, None, min_gap)
        if r is None:
            continue
        a, ai = r["alpha_star"], r["alpha_int"]
        row = {"accent": acc, "step": r["step"], "ceiling": r["ceiling"],
               "floor": r["floor"], "gap": r["gap"], "gap_ok": r["valid"],
               "alpha_star": a, "R_star": float(r["R"].loc[a]),
               "alpha_int": ai,
               "R_int": float(r["R"].loc[ai]) if np.isfinite(ai) else float("nan")}
        for col in ("wer", "utmos", "spk_sim"):
            row[f"{col}_0"] = _at(df, r["step"], 0.0, col)
            row[f"{col}_star"] = _at(df, r["step"], a, col)
            row[f"{col}_int"] = _at(df, r["step"], ai, col) if np.isfinite(ai) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def operating_point_tex(op, path, min_gap=MIN_GAP):
    """The main-text table, deliberately narrow.

    Four things the wide draft carried are gone, each recoverable elsewhere rather than
    lost: the unconstrained peak alpha* and its R (in the figure as a ring marker, and
    as a row of both appendix tables); UTMOS, which the overview figures already drop
    for carrying an unquantified bias against non-native accents; and the alpha=0 costs
    that each cost column used to repeat in brackets -- the floors sit in a narrow band
    across accents, so the caption states the band once. Everything stays in the CSV.
    """
    body = []
    for _, r in op.iterrows():
        lab = ACCENT_STYLE.get(r.accent, {"label": r.accent})["label"]
        body.append(
            f"{lab} & {_fmt(r.ceiling)} & {_fmt(r.floor)} & {_fmt(r.gap)} & "
            f"{r.alpha_int:.1f} & {_fmt(r.R_int, 2) if r.gap_ok else '--'} & "
            f"{100 * r.wer_int:.1f} & {_fmt(r.spk_sim_int)} \\\\")
    steps = ", ".join(f"{ACCENT_STYLE.get(r.accent, {'label': r.accent})['label']} "
                      f"{_num(r.step)}" for _, r in op.iterrows())
    tex = f"""% requires: \\usepackage{{booktabs}}
\\begin{{table}}[t]
\\centering\\small
\\caption{{Cross-accent prompting at each accent's final checkpoint ({steps} steps),
  prompt speakers pooled. The ceiling $\\mathrm{{CS}}_{{\\mathrm{{L1}}}}(0)$ and floor
  $\\mathrm{{CS}}_{{\\mathrm{{GAE}}}}(0)$ are the two anchors of Equation~\\ref{{eq:recovery}}
  and the gap between them is what $R$ divides by. Recovery peaks at
  $\\alpha \\geq {op.alpha_star.min():.1f}$ for every accent, where WER is
  {100 * op.wer_star.min():.0f}--{100 * op.wer_star.max():.0f}\\%; the table
  therefore reports $\\alpha^\\dagger$, the greatest recovery that keeps WER at or below
  {100 * WER_USABLE:.0f}\\%, and the two costs paid there. At $\\alpha=0$ every accent
  starts from WER below {100 * op.wer_0.max():.1f}\\% and speaker similarity in
  {op.spk_sim_0.min():.3f}--{op.spk_sim_0.max():.3f}, so the columns are read against
  those. Speaker similarity is measured against the GAE prompt speaker and so is
  expected to fall as the target accent is imposed. $R$ is not reported where the gap
  is below {min_gap:g} AccentCS: the L1 prompt buys too little accent over the GAE
  prompt there for a fraction of the interval to mean anything.}}
\\label{{tab:rq2-operating-point}}
\\begin{{tabular}}{{lrrrrrrr}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{AccentCS anchors}} & \\multicolumn{{2}}{{c}}{{Recovery}} & \\multicolumn{{2}}{{c}}{{Cost at $\\alpha^\\dagger$}} \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-6}}\\cmidrule(lr){{7-8}}
Accent & Ceil. & Floor & Gap & $\\alpha^\\dagger$ & $R(\\alpha^\\dagger)$ & WER (\\%) & Spk.\\ sim. \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(tex)
    print(f"[rq2] wrote {path}")


SPK_LABEL = {"m": "male", "f": "female"}


def appendix_tex(data, speaker, path, min_gap=MIN_GAP):
    """R(alpha) for one prompt speaker: alphas down the rows, accents across."""
    recs = {a: recovery(df, speaker, min_gap) for a, df in data.items()}
    accs = [a for a in recs if recs[a]]
    alphas = sorted({float(x) for a in accs for x in recs[a]["cs"].index})
    labs = [ACCENT_STYLE.get(a, {"label": a})["label"] for a in accs]

    def row(name, fn, nd=3):
        return f"{name} & " + " & ".join(_fmt(fn(recs[a]), nd) for a in accs) + " \\\\"

    body = [row("Ceiling $\\mathrm{CS}_{\\mathrm{L1}}(0)$", lambda r: r["ceiling"]),
            row("Floor $\\mathrm{CS}_{\\mathrm{GAE}}(0)$", lambda r: r["floor"]),
            row("Gap", lambda r: r["gap"]), "\\midrule"]
    for al in alphas:
        cells = []
        for a in accs:
            R = recs[a]["R"]
            cells.append(_fmt(R.loc[al], 2) if al in R.index else "--")
        body.append(f"$\\alpha={al:.1f}$ & " + " & ".join(cells) + " \\\\")
    body.append("\\midrule")
    body.append(row("$\\alpha^\\star$", lambda r: r["alpha_star"], 1))
    body.append(row("$\\alpha^\\dagger$", lambda r: r["alpha_int"], 1))
    steps = ", ".join(f"{ACCENT_STYLE.get(a, {'label': a})['label']} {_num(recs[a]['step'])}"
                      for a in accs)
    tex = f"""% requires: \\usepackage{{booktabs}}
\\begin{{table}}[t]
\\centering\\footnotesize
\\caption{{Recovery ratio $R(\\alpha)$ under the {SPK_LABEL[speaker]} GAE prompt, at each
  accent's final checkpoint ({steps} steps). The anchors of Equation~\\ref{{eq:recovery}}
  are recomputed within this prompt speaker, so the ceiling is the same-speaker
  L1-prompt condition. A dash marks an accent whose ceiling--floor gap is below
  {min_gap:g} AccentCS, where the ratio divides by a difference too small to interpret.
  $\\alpha^\\star$ is the accent strength of peak recovery and $\\alpha^\\dagger$ the peak
  that keeps WER at or below {100 * WER_USABLE:.0f}\\%, both for this prompt speaker.
  Compare with Table~\\ref{{tab:rq2-recovery-{'f' if speaker == 'm' else 'm'}}} for the
  {SPK_LABEL['f' if speaker == 'm' else 'm']} prompt.}}
\\label{{tab:rq2-recovery-{speaker}}}
\\begin{{tabular}}{{l{'r' * len(accs)}}}
\\toprule
 & {' & '.join(labs)} \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(tex)
    print(f"[rq2] wrote {path}")


# --- driver ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="results")
    p.add_argument("--tag", default="lr3e5_r16")
    p.add_argument("--accents", nargs="+",
                   default=["dutch", "bengali", "arabic", "hindi", "mandarin"])
    p.add_argument("--out-dir", default="results/figures/dissertation")
    p.add_argument("--table-dir", default="results/tables")
    p.add_argument("--appendix-dir", default="results/appendix")
    p.add_argument("--intervals", action="store_true",
                   help="add the floor-ceiling panel (redundant with the main table)")
    p.add_argument("--min-gap", type=float, default=MIN_GAP,
                   help="AccentCS ceiling-floor gap below which R is suppressed")
    a = p.parse_args()

    data = load_accents(a.results_root, a.tag, a.accents)
    if not data:
        raise SystemExit("no accent trees loaded")

    fig_recovery(data, Path(a.out_dir) / "ch3_recovery_gae", a.min_gap, a.intervals)

    long = recovery_long(data, a.min_gap)
    tdir, adir = Path(a.table_dir), Path(a.appendix_dir)
    tdir.mkdir(parents=True, exist_ok=True)
    long.to_csv(tdir / "ch3_recovery.csv", index=False)
    print(f"[rq2] wrote {tdir / 'ch3_recovery.csv'}")

    op = operating_point(data, a.min_gap)
    op.to_csv(tdir / "ch3_recovery_operating_point.csv", index=False)
    operating_point_tex(op, tdir / "ch3_recovery_operating_point.tex", a.min_gap)

    for spk in ("m", "f"):
        appendix_tex(data, spk, adir / f"rq2_recovery_prompt_{spk}.tex", a.min_gap)
        long[long.prompt == spk].to_csv(adir / f"rq2_recovery_prompt_{spk}.csv", index=False)
    print(f"[rq2] wrote {adir}/rq2_recovery_prompt_{{m,f}}.csv")


if __name__ == "__main__":
    main()
