"""[RQ6] Combined dashboard *video* of the accent vector forming during fine-tuning.

Reads the LoRA snapshots (lora_<step>.pt) written during training and animates three
synchronised panels, one frame per snapshot:

  left    tau_t traced through a 2-D MDS projection of {tau_t}, moving from the
          origin toward tau_final (does the DIRECTION settle early?)
  top-r   ||tau_t||            (magnitude keeps growing)
  bot-r   cos(tau_t, tau_final) with the convergence threshold line

The H6 story is visual: cos saturates well before magnitude => the accent direction
is learnable from little optimisation and alpha supplies the remaining intensity.

    python -m accent_vector.experiments.viz_temporal \
        --ckpt-dir exps/F5TTS_v1_LoRA_british/<run>/ckpts/snapshots \
        --out results/british/temporal.mp4
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from accent_vector.experiments import shared
from accent_vector.experiments.rq6_temporal import collect_checkpoints, _vector_1d_lora


def _load_trajectory(ckpt_dir, include, exclude):
    """[(step, tau_t 1-D array)] over the LoRA snapshots, step-sorted."""
    ckpts = collect_checkpoints(ckpt_dir, prefix="lora_")
    if not ckpts:
        raise SystemExit(f"no lora_<step>.pt snapshots found in {ckpt_dir}")
    traj = []
    keys0 = None
    for step, path in ckpts:
        keys, v = _vector_1d_lora(path, include, exclude)
        if keys0 is None:
            keys0 = keys
        elif keys != keys0:
            raise SystemExit(f"parameter-key mismatch at {path}; snapshots must share architecture")
        traj.append((step, v))
    return traj


def _geometry(traj):
    """MDS embedding of {tau_t}, magnitudes, and cos_to_final."""
    vecs = {str(step): v for step, v in traj}
    names, C = shared.cosine_matrix(vecs)          # names preserve step order
    emb = shared.classical_mds(C, 2)               # (N, 2)
    final = traj[-1][1]
    nfinal = float(np.linalg.norm(final)) or 1.0
    mags = np.array([float(np.linalg.norm(v)) for _, v in traj])
    cos_final = np.array([float(v @ final / (np.linalg.norm(v) * nfinal))
                          if np.linalg.norm(v) > 0 else np.nan for _, v in traj])
    return emb, mags, cos_final


def animate(ckpt_dir, out, threshold=0.95, include=None, exclude=None, fps=8):
    traj = _load_trajectory(ckpt_dir, include, exclude)
    steps = [s for s, _ in traj]
    emb, mags, cos_final = _geometry(traj)
    n = len(steps)
    print(f"[viz_temporal] {n} snapshots, steps {steps[0]}..{steps[-1]}, dim={traj[0][1].size}")

    fig = plt.figure(figsize=(12, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0])
    ax_mds = fig.add_subplot(gs[:, 0])
    ax_mag = fig.add_subplot(gs[0, 1])
    ax_cos = fig.add_subplot(gs[1, 1])

    # --- static framing ---
    pad = 0.1 * (np.ptp(emb, axis=0) + 1e-9)
    ax_mds.set_xlim(emb[:, 0].min() - pad[0], emb[:, 0].max() + pad[0])
    ax_mds.set_ylim(emb[:, 1].min() - pad[1], emb[:, 1].max() + pad[1])
    ax_mds.set_title(r"accent vector $\tau_t$ trajectory (MDS)")
    ax_mds.set_xlabel("MDS-1"); ax_mds.set_ylabel("MDS-2")
    ax_mds.scatter(*emb[-1], s=140, marker="*", color="crimson", zorder=5, label=r"$\tau_{final}$")
    ax_mds.scatter(*emb[0], s=40, color="0.5", zorder=4, label=r"$\tau_0$")
    ax_mds.legend(loc="best", fontsize=8)

    ax_mag.set_xlim(steps[0], steps[-1]); ax_mag.set_ylim(0, mags.max() * 1.08 + 1e-9)
    ax_mag.set_title(r"magnitude $\|\tau_t\|$"); ax_mag.set_xlabel("update")

    ax_cos.set_xlim(steps[0], steps[-1]); ax_cos.set_ylim(0, 1.02)
    ax_cos.set_title(r"direction $\cos(\tau_t, \tau_{final})$"); ax_cos.set_xlabel("update")
    ax_cos.axhline(threshold, ls="--", color="crimson", lw=1, label=f"{threshold:g}")
    ax_cos.legend(loc="lower right", fontsize=8)

    (path_line,) = ax_mds.plot([], [], "-", color="steelblue", lw=1.5, alpha=0.7)
    (head,) = ax_mds.plot([], [], "o", color="steelblue", ms=9, zorder=6)
    (mag_line,) = ax_mag.plot([], [], "-", color="darkorange", lw=2)
    (cos_line,) = ax_cos.plot([], [], "-", color="seagreen", lw=2)
    title = fig.suptitle("")

    def update(i):
        j = i + 1
        path_line.set_data(emb[:j, 0], emb[:j, 1])
        head.set_data([emb[i, 0]], [emb[i, 1]])
        mag_line.set_data(steps[:j], mags[:j])
        cos_line.set_data(steps[:j], cos_final[:j])
        c = cos_final[i]
        title.set_text(f"update {steps[i]}   |  ||tau||={mags[i]:.3f}   "
                       f"cos->final={c:.3f}" + ("  [direction converged]" if c >= threshold else ""))
        return path_line, head, mag_line, cos_line, title

    anim = animation.FuncAnimation(fig, update, frames=n, blit=False, interval=1000 / fps)

    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(str(out), writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
        print(f"[viz_temporal] wrote {out}")
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        gif = out.with_suffix(".gif")
        print(f"[viz_temporal] ffmpeg unavailable ({e}); falling back to GIF {gif}")
        anim.save(str(gif), writer=animation.PillowWriter(fps=fps))
        print(f"[viz_temporal] wrote {gif}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="RQ6 accent-vector trajectory dashboard video")
    p.add_argument("--ckpt-dir", required=True, help="dir of lora_<step>.pt snapshots")
    p.add_argument("--out", required=True, help="output .mp4 (falls back to .gif)")
    p.add_argument("--threshold", type=float, default=0.95, help="direction-convergence line")
    p.add_argument("--include", action="append", default=[], help="only these LoRA key substrings")
    p.add_argument("--exclude", action="append", default=[], help="drop these LoRA key substrings")
    p.add_argument("--fps", type=int, default=8)
    a = p.parse_args()
    animate(a.ckpt_dir, a.out, threshold=a.threshold,
            include=a.include or None, exclude=a.exclude or None, fps=a.fps)


if __name__ == "__main__":
    main()
