"""[E1.1-1.2] RQ1 -- cross-backbone reproduction.

Over one accent's alpha sweep, measure whether accent strength rises
monotonically with alpha while speaker identity is retained, and track the
content/language-leakage signal (WER) that the missing language-ID token on F5
is predicted to worsen at high alpha.

    accent strength : cs_accent (GenAID embedding cosine to natural target)
                      + aid_acc (GenAID label accuracy)
    identity        : speaker_similarity to the fixed neutral reference
    leakage         : wer vs the held-out English transcripts

Monotonicity is summarised with Spearman rho over alpha (H1: rho > 0 for accent,
~0 for speaker similarity).

    python -m accent_vector.experiments.rq1_reproduction \
        --sweep-dir results/british --transcripts transcripts/eval_transcripts.txt \
        --ref-wav refs/neutral.wav --accent-ref /data/vctk_england_clips \
        --target-accent English --out-csv results/british/rq1.csv
"""

import argparse
import csv
from pathlib import Path

from accent_vector.experiments import common


def _spearman(xs, ys):
    import numpy as np
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = ~(np.isnan(xs) | np.isnan(ys))
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(xs[ok]))
    ry = np.argsort(np.argsort(ys[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def run(sweep_dir, transcripts, ref_wav, accent_refs, target_accent, device, out_csv):
    ef = common.load_eval()
    tx = [ln.strip() for ln in Path(transcripts).read_text().splitlines() if ln.strip()] if transcripts else []
    refs = common.wavs_in(accent_refs) if accent_refs else None

    rows = []
    for alpha, d in common.alpha_dirs(sweep_dir):
        wavs = common.wavs_in(d)
        row = {"alpha": alpha, "n": len(wavs)}
        if ref_wav:
            row["spk_sim"] = ef.speaker_similarity(wavs, [ref_wav] * len(wavs), device=device)
        if refs:
            paired = (refs * (len(wavs) // len(refs) + 1))[: len(wavs)]
            row["accent_cs"] = ef.cs_accent(wavs, paired)
        if target_accent:
            row["accent_acc"] = ef.aid_acc(wavs, [target_accent] * len(wavs))
        if tx:
            errs = [ef.wer(w, tx[common.utt_index(w)]) for w in wavs
                    if common.utt_index(w) is not None and common.utt_index(w) < len(tx)]
            row["wer"] = sum(errs) / len(errs) if errs else float("nan")
        rows.append(row)
        print(f"[rq1] alpha={alpha}: {row}")

    alphas = [r["alpha"] for r in rows]
    summary = {}
    for key in ("accent_cs", "accent_acc", "spk_sim", "wer"):
        if all(key in r for r in rows):
            summary[f"spearman_{key}"] = _spearman(alphas, [r[key] for r in rows])
    print(f"[rq1] monotonicity (Spearman vs alpha): {summary}")

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "alpha", k))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
        f.write("# " + ", ".join(f"{k}={v:.3f}" for k, v in summary.items()) + "\n")
    print(f"[rq1] wrote {out_csv}")


def main():
    p = argparse.ArgumentParser(description="RQ1 reproduction metrics")
    p.add_argument("--sweep-dir", required=True)
    p.add_argument("--transcripts")
    p.add_argument("--ref-wav")
    p.add_argument("--accent-ref", help="dir of natural target-accent clips (cs_accent)")
    p.add_argument("--target-accent")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-csv", required=True)
    a = p.parse_args()
    run(a.sweep_dir, a.transcripts, a.ref_wav, a.accent_ref, a.target_accent, a.device, a.out_csv)


if __name__ == "__main__":
    main()
