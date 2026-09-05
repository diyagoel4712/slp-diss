"""Training-loss curves for the per-accent LoRA fine-tunes (A0).

Reads the TensorBoard event files the fork's trainer writes during fine-tuning
(``ckpts.logger=tensorboard`` -> ``runs/lora_<accent>/events.out.tfevents.*``,
relative to the launch dir, i.e. ``AccentVector/runs/`` under the Eddie wrapper)
and renders the loss curves for all accents in the dissertation house style.

Scalars logged by the trainer: ``loss`` (train, every update), ``valid_loss``
(every ``ckpts.last_per_updates``), plus ``lr`` / ``lora_norm`` / ``lora_cos_prev``.

Pull the runs down from Eddie first (they are gitignored, so they only live there):

    rsync -av --include='*/' --include='events.out.tfevents.*' --exclude='*' \
        s2247837@eddie.ecdf.ed.ac.uk:/exports/.../slp-diss/AccentVector/runs/ runs/

    python -m accent_vector.experiments.plot_loss_curves            # runs/ -> results/figures/training/
    python -m accent_vector.experiments.plot_loss_curves --runs-root runs --smooth 0.95

Figures:
  fig_train_loss.pdf   all accents on one axes: raw loss (faint) + EMA (bold)
  fig_loss_panels.pdf  one panel per accent: train EMA + validation loss
Also writes loss_curves.csv (subsampled train EMA + every valid_loss point) and
prints a per-accent summary (steps, final/best loss).

Resumed runs (a fine-tune chained across several qsub jobs) drop several event
files in the same dir; they are merged in wall-clock order and de-duplicated by
step, keeping the newest write for a repeated step.

No tensorboard/tbparse dependency: the event files are parsed here directly
(TFRecord frames + the handful of protobuf fields the scalar summaries use).
"""
import argparse
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from accent_vector.experiments.plot_dissertation import ACCENT_STYLE, INK, MUTED

DEFAULT_ACCENTS = ["dutch", "bengali", "arabic", "hindi", "mandarin"]


# --- minimal event-file reader --------------------------------------------------
# TFRecord frame: u64 length | u32 crc(length) | payload | u32 crc(payload).
# The CRCs are not checked (nothing here is corruption-sensitive); a short read
# just ends the file, which is what a job killed mid-write looks like.
def _tfrecords(path):
    with open(path, "rb") as fh:
        while True:
            head = fh.read(12)
            if len(head) < 12:
                return
            (length,) = struct.unpack("<Q", head[:8])
            payload = fh.read(length)
            if len(payload) < length or len(fh.read(4)) < 4:
                return
            yield payload


def _varint(buf, i):
    val = shift = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


def _fields(buf):
    """Yield (field_number, wire_type, payload) for one protobuf message.
    payload is an int (varint), bytes (length-delimited) or raw bytes (fixed)."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            val, i = _varint(buf, i)
        elif wt == 1:
            val, i = buf[i:i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            val, i = buf[i:i + ln], i + ln
        elif wt == 5:
            val, i = buf[i:i + 4], i + 4
        else:                                   # groups: not used by Event/Summary
            return
        yield fno, wt, val


def read_event_file(path):
    """[(wall_time, step, tag, value)] for every simple_value scalar in one file.
    Event{1: wall_time, 2: step, 5: Summary}, Summary{1: Value},
    Summary.Value{1: tag, 2: simple_value}."""
    rows = []
    for rec in _tfrecords(path):
        wall, step, summaries = float("nan"), 0, []
        for fno, wt, val in _fields(rec):
            if fno == 1 and wt == 1:
                wall = struct.unpack("<d", val)[0]
            elif fno == 2 and wt == 0:
                step = val
            elif fno == 5 and wt == 2:
                summaries.append(val)
        for summ in summaries:
            for fno, wt, val in _fields(summ):
                if fno != 1 or wt != 2:
                    continue
                tag, value = None, None
                for f2, w2, v2 in _fields(val):
                    if f2 == 1 and w2 == 2:
                        tag = v2.decode("utf-8", "replace")
                    elif f2 == 2 and w2 == 5:
                        value = struct.unpack("<f", v2)[0]
                if tag is not None and value is not None:
                    rows.append((wall, int(step), tag, float(value)))
    return rows


# --- run discovery + loading ----------------------------------------------------
def find_runs(runs_root, accents):
    """{accent: [event files]} -- any event file whose path mentions the accent
    (so lora_<accent>/, lora_<accent>_resume/, <accent>/run2/ all get picked up)."""
    runs_root = Path(runs_root)
    files = sorted(runs_root.rglob("events.out.tfevents.*"))
    out = {a: [] for a in accents}
    for f in files:
        rel = str(f.relative_to(runs_root)).lower()
        for a in accents:
            if a in rel:
                out[a].append(f)
                break
    return {a: fs for a, fs in out.items() if fs}


def load_run(files):
    """Merged long df (step, tag, value) for one accent's event files.
    Chained/resumed jobs repeat steps: keep the latest write per (tag, step)."""
    rows = []
    for f in sorted(files, key=lambda p: p.name):       # filename embeds the start time
        rows += read_event_file(f)
    if not rows:
        return pd.DataFrame(columns=["wall_time", "step", "tag", "value"])
    df = pd.DataFrame(rows, columns=["wall_time", "step", "tag", "value"])
    df = df.sort_values(["tag", "step", "wall_time"], kind="stable")
    df = df.drop_duplicates(subset=["tag", "step"], keep="last")
    return df[np.isfinite(df.value)].reset_index(drop=True)


def series(df, tag):
    d = df[df.tag == tag].sort_values("step")
    return d.step.to_numpy(float), d.value.to_numpy(float)


def ema(y, weight):
    """TensorBoard's smoothing: exponential moving average with bias correction."""
    out = np.empty_like(y, dtype=float)
    last = num = 0.0
    for i, v in enumerate(y):
        last = last * weight + (1 - weight) * v
        num = num * weight + (1 - weight)
        out[i] = last / num if num else v
    return out


def _style(acc):
    return ACCENT_STYLE.get(acc, {"c": "#0072B2", "ls": "-", "m": "o", "label": acc})


def _save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out.with_suffix('.pdf')} (+ .png)")


# --- figures --------------------------------------------------------------------
def fig_train_loss(runs, out, smooth=0.95, ylim=None):
    """All accents on one axes: raw per-update loss faint, EMA on top."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    used = {}
    for acc, df in runs.items():
        x, y = series(df, "loss")
        if not len(x):
            continue
        st = _style(acc)
        ax.plot(x, y, color=st["c"], lw=0.5, alpha=0.15, zorder=1)
        ax.plot(x, ema(y, smooth), color=st["c"], ls=st["ls"], lw=1.8, zorder=2)
        used[acc] = int(x.max())
    ax.set_xlabel("fine-tuning update")
    ax.set_ylabel("training loss")
    ax.set_title(f"LoRA fine-tuning loss per accent (EMA, α={smooth:g}; raw faint)", fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(handles=[Line2D([], [], color=_style(a)["c"], ls=_style(a)["ls"], lw=1.8,
                              label=f"{_style(a)['label']} ({n:,} updates)")
                       for a, n in used.items()],
              fontsize=8, loc="upper right")
    _save(fig, out)


def fig_loss_panels(runs, out, smooth=0.95, ncols=3):
    """One panel per accent: smoothed train loss + validation loss."""
    accs = list(runs)
    nrows = int(np.ceil(len(accs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.9 * nrows),
                             sharex=False, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, acc in zip(axes, accs):
        st, df = _style(acc), runs[acc]
        x, y = series(df, "loss")
        if len(x):
            ax.plot(x, y, color=st["c"], lw=0.5, alpha=0.15)
            ax.plot(x, ema(y, smooth), color=st["c"], ls="-", lw=1.6, label="train (EMA)")
        vx, vy = series(df, "valid_loss")
        if len(vx):
            ax.plot(vx, vy, color=INK, ls="--", lw=1.0, marker=st["m"], ms=3,
                    label="validation")
        ax.set_title(st["label"], fontsize=10)
        ax.set_xlabel("update", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("loss", fontsize=9)
    if len(axes) > ncols:
        axes[ncols].set_ylabel("loss", fontsize=9)
    for ax in axes[len(accs):]:
        ax.axis("off")
    handles = [Line2D([], [], color=MUTED, lw=1.6, label="train (EMA)"),
               Line2D([], [], color=INK, ls="--", lw=1.0, marker="o", ms=3,
                      label="validation")]
    axes[len(accs) - 1].legend(handles=handles, fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, out)


# --- table ----------------------------------------------------------------------
def summarise(runs, smooth):
    rows = []
    for acc, df in runs.items():
        x, y = series(df, "loss")
        vx, vy = series(df, "valid_loss")
        sm = ema(y, smooth) if len(y) else np.array([])
        rows.append({
            "accent": acc,
            "updates": int(x.max()) if len(x) else 0,
            "train_loss_final_ema": float(sm[-1]) if len(sm) else np.nan,
            "train_loss_min_ema": float(sm.min()) if len(sm) else np.nan,
            "valid_loss_final": float(vy[-1]) if len(vy) else np.nan,
            "valid_loss_best": float(vy.min()) if len(vy) else np.nan,
            "valid_best_update": int(vx[int(vy.argmin())]) if len(vy) else -1,
            "n_valid_points": int(len(vy)),
        })
    return pd.DataFrame(rows)


def long_csv(runs, smooth, stride):
    """Long CSV: train EMA every <stride> updates + every valid_loss point."""
    frames = []
    for acc, df in runs.items():
        x, y = series(df, "loss")
        if len(x):
            sm = ema(y, smooth)
            keep = np.arange(0, len(x), max(1, stride))
            frames.append(pd.DataFrame({"accent": acc, "step": x[keep],
                                        "split": "train", "loss": sm[keep]}))
        vx, vy = series(df, "valid_loss")
        if len(vx):
            frames.append(pd.DataFrame({"accent": acc, "step": vx,
                                        "split": "valid", "loss": vy}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs",
                   help="dir holding lora_<accent>/events.out.tfevents.* (default: runs)")
    p.add_argument("--accents", nargs="+", default=DEFAULT_ACCENTS)
    p.add_argument("--out-dir", default="results/figures/training")
    p.add_argument("--smooth", type=float, default=0.95, help="EMA weight (0=raw)")
    p.add_argument("--max-step", type=int, help="truncate every curve at this update")
    p.add_argument("--ylim", nargs=2, type=float, help="y-limits for the overview figure")
    p.add_argument("--csv-stride", type=int, default=100,
                   help="keep every Nth train point in loss_curves.csv (default 100)")
    a = p.parse_args()

    found = find_runs(a.runs_root, a.accents)
    missing = [acc for acc in a.accents if acc not in found]
    if missing:
        print(f"[plot] no event files for: {', '.join(missing)} (looked under {a.runs_root})")
    if not found:
        raise SystemExit(f"no event files under {a.runs_root} -- rsync runs/ down from Eddie first")

    runs = {}
    for acc in a.accents:                      # keep the dissertation accent order
        if acc not in found:
            continue
        df = load_run(found[acc])
        if a.max_step:
            df = df[df.step <= a.max_step]
        if df.empty:
            print(f"[plot] skip {acc}: no scalars in {len(found[acc])} event file(s)")
            continue
        runs[acc] = df
        n_upd = df[df.tag == "loss"].step.max()
        print(f"[plot] {acc}: {len(found[acc])} file(s), tags={sorted(df.tag.unique())}, "
              f"{int(n_upd) if pd.notna(n_upd) else 0:,} updates")

    out_dir = Path(a.out_dir)
    fig_train_loss(runs, out_dir / "fig_train_loss", smooth=a.smooth, ylim=a.ylim)
    fig_loss_panels(runs, out_dir / "fig_loss_panels", smooth=a.smooth)

    csv = long_csv(runs, a.smooth, a.csv_stride)
    if not csv.empty:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv.to_csv(out_dir / "loss_curves.csv", index=False)
        print(f"[plot] wrote {out_dir / 'loss_curves.csv'}")

    summary = summarise(runs, a.smooth)
    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    summary.to_csv(out_dir / "loss_summary.csv", index=False)
    print(f"[plot] wrote {out_dir / 'loss_summary.csv'}")


if __name__ == "__main__":
    main()
