"""[E3.1-3.3] RQ3 (core) -- segmental vs suprasegmental decomposition.

As alpha increases, does the accent vector move SEGMENTAL structure (phone
realisation) and SUPRASEGMENTAL structure (pitch, rhythm, tempo) toward the
natural target accent, or is the shift segmental-dominated?

    segmental       : PPG-KL between synth(alpha) and natural target clips
    suprasegmental  : voicing-based rhythm + pitch descriptors (extract_f0)
    contrast        : fraction of the baseline(alpha=0)->natural gap closed at
                      alpha=1, segmental vs suprasegmental (H3: seg >> supra,
                      widest for a prosodically-distant accent)

    python -m accent_vector.experiments.rq3_decomposition \
        --sweep-dir results/british --natural-ref /data/vctk_england_clips \
        --out-csv results/british/rq3.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from accent_vector.experiments import shared

SUPRA_FEATURES = ["pct_voiced", "npvi_voiced", "artic_rate", "f0_mean", "f0_std", "f0_range"]


def _run_lengths(mask):
    """Lengths of consecutive-True runs in a boolean array (e.g. voiced runs)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    idx = np.flatnonzero(np.diff(mask.astype(int)))
    bounds = np.concatenate(([0], idx + 1, [mask.size]))
    return [int(bounds[i + 1] - bounds[i]) for i in range(len(bounds) - 1)
            if mask[bounds[i]]]


def _npvi(durations):
    """normalized Pairwise Variability Index over successive interval durations
    (a standard speech-rhythm descriptor). Higher = more stress-timed."""
    d = np.asarray(durations, dtype=float)
    if d.size < 2:
        return float("nan")
    num = np.abs(d[:-1] - d[1:])
    den = (d[:-1] + d[1:]) / 2.0
    ok = den > 0
    return float(100.0 / (d.size - 1) * np.sum(num[ok] / den[ok]))


def voicing_rhythm(f0, hop_s):
    """Alignment-free suprasegmental descriptors from an F0 track (extract_f0):
    voiced fraction (%V proxy), nPVI over voiced-run durations, and articulation
    rate (voiced runs per second). A voicing-based proxy for %V/nPVI that needs
    no forced aligner; swap in MFA vowel/consonant intervals for the rigorous
    version."""
    f0 = np.asarray(f0, dtype=float)
    voiced = ~np.isnan(f0)
    total = voiced.size
    runs = _run_lengths(voiced)
    dur_s = [r * hop_s for r in runs]
    duration_s = total * hop_s if total else 0.0
    return {
        "pct_voiced": float(voiced.mean()) if total else float("nan"),
        "npvi_voiced": _npvi(dur_s),
        "artic_rate": (len(runs) / duration_s) if duration_s else float("nan"),
        "f0_mean": float(np.nanmean(f0)) if voiced.any() else float("nan"),
        "f0_std": float(np.nanstd(f0)) if voiced.any() else float("nan"),
        "f0_range": (float(np.nanmax(f0) - np.nanmin(f0)) if voiced.any() else float("nan")),
    }


def gap_closure_scalar(base, alpha_val, natural):
    """Fraction of the baseline->natural gap a scalar feature closes at this
    alpha: (x_alpha - x_base) / (x_natural - x_base). ~1 => moved fully to the
    natural-accent value; ~0 => did not move."""
    denom = natural - base
    if abs(denom) < 1e-12:
        return float("nan")
    return float((alpha_val - base) / denom)


def gap_closure_distance(d_base_to_natural, d_alpha_to_natural):
    """Fraction of a distance-to-natural closed (e.g. PPG-KL): 1 - d_alpha/d_base."""
    if abs(d_base_to_natural) < 1e-12:
        return float("nan")
    return float(1.0 - d_alpha_to_natural / d_base_to_natural)



def _mean_ppg_kl(synth_wavs, natural_wavs, ef):
    """Mean symmetric PPG-KL, each synth clip paired against a cycled natural
    clip. (Content-matched pairing tightens this; DTW makes it defined either way.)"""
    pairs = [(s, natural_wavs[i % len(natural_wavs)]) for i, s in enumerate(synth_wavs)]
    return float(np.mean([ef.ppg_kl(s, g) for s, g in pairs]))


def _mean_supra(wavs, ef, hop_s):
    feats = {k: [] for k in SUPRA_FEATURES}
    for w in wavs:
        vr = voicing_rhythm(ef.extract_f0(w), hop_s)
        for k in SUPRA_FEATURES:
            feats[k].append(vr[k])
    return {k: float(np.nanmean(v)) for k, v in feats.items()}


def run(sweep_dir, natural_ref, out_csv, hop_s):
    ef = shared.load_eval()
    natural = shared.wavs_in(natural_ref)
    if not natural:
        raise SystemExit(f"no natural target-accent clips in {natural_ref}")
    natural_supra = _mean_supra(natural, ef, hop_s)

    grid = shared.alpha_dirs(sweep_dir)
    base_alpha, base_dir = grid[0]  # alpha=0 baseline (GAE anchor)
    base_wavs = shared.wavs_in(base_dir)
    base_seg = _mean_ppg_kl(base_wavs, natural, ef)
    base_supra = _mean_supra(base_wavs, ef, hop_s)

    rows = []
    for alpha, d in grid:
        wavs = shared.wavs_in(d)
        seg = _mean_ppg_kl(wavs, natural, ef)
        supra = _mean_supra(wavs, ef, hop_s)
        row = {"alpha": alpha, "seg_ppg_kl_to_natural": seg,
               "seg_closure": gap_closure_distance(base_seg, seg)}
        closures = []
        for k in SUPRA_FEATURES:
            row[k] = supra[k]
            c = gap_closure_scalar(base_supra[k], supra[k], natural_supra[k])
            row[f"{k}_closure"] = c
            if not np.isnan(c):
                closures.append(c)
        row["supra_closure_mean"] = float(np.mean(closures)) if closures else float("nan")
        rows.append(row)
        print(f"[rq3] alpha={alpha} seg_closure={row['seg_closure']:.3f} "
              f"supra_closure_mean={row['supra_closure_mean']:.3f}")

    top = rows[-1]  # alpha=1 (or max)
    print(f"[rq3] AT alpha={top['alpha']}: segmental closes {top['seg_closure']:.3f}, "
          f"suprasegmental {top['supra_closure_mean']:.3f} "
          f"-> {'segmental-dominated' if top['seg_closure'] > top['supra_closure_mean'] else 'balanced'}")

    fields = ["alpha", "seg_ppg_kl_to_natural", "seg_closure"] + \
             [c for k in SUPRA_FEATURES for c in (k, f"{k}_closure")] + ["supra_closure_mean"]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[rq3] wrote {out_csv}")


def main():
    p = argparse.ArgumentParser(description="RQ3 segmental/suprasegmental decomposition")
    p.add_argument("--sweep-dir", required=True)
    p.add_argument("--natural-ref", required=True, help="dir of natural target-accent clips")
    p.add_argument("--hop-s", type=float, default=256 / 16000,
                   help="F0 frame hop in seconds (extract_f0 default: 256/16000)")
    p.add_argument("--out-csv", required=True)
    a = p.parse_args()
    run(a.sweep_dir, a.natural_ref, a.out_csv, a.hop_s)


if __name__ == "__main__":
    main()
