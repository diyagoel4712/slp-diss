"""Visualise evaluation_results.csv (written by run_eval.py).

Produces, under SOTA_models_experiments/figures/:
  per_model_means.png        one bar panel per metric, mean over all clips, by model
  model_x_accent_heatmaps.png per-metric model x accent heatmaps (annotated)
  metric_distributions.png    per-metric box plots of the per-clip spread, by model
and writes summary_by_model.csv / summary_by_accent.csv next to the results CSV.

Each metric is labelled with its "better" direction so the panels are self-explanatory.
Run with the evaluation interpreter (the one with pandas + matplotlib); run_pipeline.sh
invokes it after run_eval.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eval_config as cfg

# metric -> (pretty label, +1 if higher is better else -1)
METRIC_META = {
    "utmos":       ("UTMOS (MOS)",            +1),
    "f0_rmse":     ("F0 RMSE (cents)",        -1),
    "mcd":         ("MCD (dB)",               -1),
    "wer":         ("WER",                    -1),
    "aid_correct": ("Accent-ID acc.",         +1),
    "cs_accent":   ("Accent emb. cos-sim",    +1),
    "ppg_kl":      ("PPG-KL",                 -1),
    "speaker_sim": ("Speaker sim (SECS)",     +1),
}

RESULTS = cfg.SOTA / "evaluation_results.csv"
FIGDIR = cfg.SOTA / "figures"


def title_for(metric):
    label, direction = METRIC_META[metric]
    return f"{label}  ({'↑' if direction > 0 else '↓'} better)"


def grid_shape(n):
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    return nrow, ncol


def per_model_means(df, metrics, models):
    nrow, ncol = grid_shape(len(metrics))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    means = df.groupby("model")[metrics].mean(numeric_only=True).reindex(models)
    for k, metric in enumerate(metrics):
        ax = axes[k // ncol][k % ncol]
        vals = means[metric]
        ax.bar(range(len(models)), vals.values, color="#4C72B0")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_title(title_for(metric), fontsize=10)
        for i, v in enumerate(vals.values):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.margins(y=0.18)
    for k in range(len(metrics), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Per-model means (all clips)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGDIR / "per_model_means.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


def model_x_accent_heatmaps(df, metrics, models, accents):
    nrow, ncol = grid_shape(len(metrics))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow), squeeze=False)
    for k, metric in enumerate(metrics):
        ax = axes[k // ncol][k % ncol]
        piv = (df.groupby(["model", "accent"])[metric].mean()
                 .unstack("accent").reindex(index=models, columns=accents))
        data = piv.values.astype(float)
        # normalise per-metric for colour only, oriented so brighter = better.
        direction = METRIC_META[metric][1]
        finite = data[np.isfinite(data)]
        if finite.size:
            lo, hi = np.nanmin(data), np.nanmax(data)
            norm = (data - lo) / (hi - lo) if hi > lo else np.zeros_like(data)
            if direction < 0:
                norm = 1 - norm
        else:
            norm = np.zeros_like(data)
        im = ax.imshow(norm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(accents))); ax.set_xticklabels(accents, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(models)));  ax.set_yticklabels(models, fontsize=8)
        ax.set_title(title_for(metric), fontsize=10)
        for i in range(len(models)):
            for j in range(len(accents)):
                v = data[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if norm[i, j] < 0.5 else "black")
    for k in range(len(metrics), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Model x accent (cell = mean; colour brighter = better)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGDIR / "model_x_accent_heatmaps.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


def metric_distributions(df, metrics, models):
    nrow, ncol = grid_shape(len(metrics))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for k, metric in enumerate(metrics):
        ax = axes[k // ncol][k % ncol]
        data = [df.loc[df["model"] == m, metric].dropna().values for m in models]
        ax.boxplot(data, tick_labels=models, showfliers=False)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_title(title_for(metric), fontsize=10)
    for k in range(len(metrics), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Per-clip distributions by model", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGDIR / "metric_distributions.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


def main():
    if not RESULTS.exists():
        raise SystemExit(f"no results CSV at {RESULTS} -- run run_eval.py first")
    df = pd.read_csv(RESULTS)
    if "aid_correct" in df.columns:
        df["aid_correct"] = df["aid_correct"].astype(float)

    metrics = [m for m in METRIC_META if m in df.columns and df[m].notna().any()]
    models = [m for m in cfg.MODELS if m in set(df["model"])]
    accents = [a for a in cfg.ACCENT_SPEAKERS if a in set(df["accent"])]
    print(f"{len(df)} clips | models={models} | metrics={metrics}")

    FIGDIR.mkdir(exist_ok=True)
    # summary tables
    df.groupby("model")[metrics].mean(numeric_only=True).to_csv(cfg.SOTA / "summary_by_model.csv")
    df.groupby("accent")[metrics].mean(numeric_only=True).to_csv(cfg.SOTA / "summary_by_accent.csv")

    for out in (per_model_means(df, metrics, models),
                model_x_accent_heatmaps(df, metrics, models, accents),
                metric_distributions(df, metrics, models)):
        print("wrote", out)


if __name__ == "__main__":
    main()
