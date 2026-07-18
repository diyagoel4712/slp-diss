"""[E1.1-1.4] RQ1 -- cross-backbone reproduction, incl. RQ1b language leakage.

Over one accent's alpha sweep, measure whether accent strength rises
monotonically with alpha while speaker identity is retained, and instrument the
language-leakage that the missing language-ID token on F5 is predicted to worsen
at high alpha.

    accent strength : cs_accent (GenAID embedding cosine to natural target)
                      + aid_acc (GenAID label accuracy)
    identity        : speaker_similarity to the fixed native-language (L1) reference
    leakage [RQ1b]  : wer vs the held-out English transcripts, and -- when a LID
                      predictor is wired -- P(English) from a spoken-LID model
    leakage onset   : the alpha at which WER crosses a threshold (rising) or
                      P(English) drops below one (falling); compare to XTTS

Monotonicity is summarised with Spearman rho over alpha (H1: rho > 0 for accent,
~0 for speaker similarity). Leakage onset makes the "how far can you scale before
content leaves English" question a single comparable number (RQ1b).

The LID signal is an optional hook: if evaluation_functions exposes
``predict_lid_english(wavs) -> [{'p_english': float}, ...]`` (e.g. wrapping
speechbrain/lang-id-voxlingua107-ecapa in its isolated env) it is used; otherwise
the eng_lid column is nan and only the WER-based onset is reported.

    python -m accent_vector.experiments.rq1_reproduction \
        --sweep-dir results/british --transcripts transcripts/eval_transcripts.txt \
        --ref-wav refs/england.wav --accent-ref /data/vctk_england_clips \
        --target-accent English --lid --out-csv results/british/rq1.csv
"""

import argparse
import csv
import re
from pathlib import Path

from accent_vector.experiments import shared


def utt_index(path):
    """Recover the transcript index from a utt####.wav name (or None)."""
    m = re.search(r"utt(\d+)", Path(path).stem)
    return int(m.group(1)) if m else None


def _spearman(xs, ys):
    import numpy as np
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = ~(np.isnan(xs) | np.isnan(ys))
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(xs[ok]))
    ry = np.argsort(np.argsort(ys[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def _english_lid(ef, wavs):
    """Mean P(English) from a wired spoken-LID predictor, or nan if none is
    available. Wire ``ef.predict_lid_english(wavs) -> [{'p_english': float}]``
    (e.g. speechbrain/lang-id-voxlingua107-ecapa in the isolated env)."""
    import numpy as np
    fn = getattr(ef, "predict_lid_english", None)
    if fn is None:
        return float("nan")
    preds = fn(wavs)
    return float(np.mean([p["p_english"] for p in preds]))


def run(sweep_dir, transcripts, ref_wav, accent_refs, target_accent, device, out_csv,
        lid=False, wer_leak_threshold=0.5, lid_leak_threshold=0.5):
    ef = shared.load_eval()
    tx = [ln.strip() for ln in Path(transcripts).read_text().splitlines() if ln.strip()] if transcripts else []
    refs = shared.wavs_in(accent_refs) if accent_refs else None
    lid_available = lid and getattr(ef, "predict_lid_english", None) is not None
    if lid and not lid_available:
        print("[rq1] --lid requested but evaluation_functions has no predict_lid_english; "
              "eng_lid will be nan (WER-based onset still reported)")

    rows = []
    for alpha, d in shared.alpha_dirs(sweep_dir):
        wavs = shared.wavs_in(d)
        row = {"alpha": alpha, "n": len(wavs)}
        if ref_wav:
            row["spk_sim"] = ef.speaker_similarity(wavs, [ref_wav] * len(wavs), device=device)
        if refs:
            paired = (refs * (len(wavs) // len(refs) + 1))[: len(wavs)]
            row["accent_cs"] = ef.cs_accent(wavs, paired)
        if target_accent:
            row["accent_acc"] = ef.aid_acc(wavs, [target_accent] * len(wavs))
        if tx:
            errs = [ef.wer(w, tx[utt_index(w)]) for w in wavs
                    if utt_index(w) is not None and utt_index(w) < len(tx)]
            row["wer"] = sum(errs) / len(errs) if errs else float("nan")
        if lid:
            row["eng_lid"] = _english_lid(ef, wavs)  # P(English); falls as content leaks
        rows.append(row)
        print(f"[rq1] alpha={alpha}: {row}")

    alphas = [r["alpha"] for r in rows]
    summary = {}
    for key in ("accent_cs", "accent_acc", "spk_sim", "wer", "eng_lid"):
        if all(key in r for r in rows):
            summary[f"spearman_{key}"] = _spearman(alphas, [r[key] for r in rows])

    # RQ1b: leakage onset -- how far the vector scales before content leaves English.
    if all("wer" in r for r in rows):
        summary["wer_leak_onset"] = shared.leakage_onset(
            alphas, [r["wer"] for r in rows], wer_leak_threshold, rising=True)
    if lid_available:
        summary["lid_leak_onset"] = shared.leakage_onset(
            alphas, [r["eng_lid"] for r in rows], lid_leak_threshold, rising=False)
    print(f"[rq1] monotonicity + leakage onset: {summary}")

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
    p.add_argument("--lid", action="store_true",
                   help="measure P(English) per alpha via ef.predict_lid_english (RQ1b)")
    p.add_argument("--wer-leak-threshold", type=float, default=0.5,
                   help="WER above which content is treated as leaked (rising onset)")
    p.add_argument("--lid-leak-threshold", type=float, default=0.5,
                   help="P(English) below which content is treated as leaked (falling onset)")
    p.add_argument("--out-csv", required=True)
    a = p.parse_args()
    run(a.sweep_dir, a.transcripts, a.ref_wav, a.accent_ref, a.target_accent, a.device,
        a.out_csv, lid=a.lid, wer_leak_threshold=a.wer_leak_threshold,
        lid_leak_threshold=a.lid_leak_threshold)


if __name__ == "__main__":
    main()
