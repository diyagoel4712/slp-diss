"""Score an alpha-sweep tree with the repo's existing eval suite.

Reuses ``Evaluation/evaluation_functions.py`` unchanged (the plan's
step 4): the paper's Section 5 metrics map onto ours as

    WER            -> wer            (Whisper)
    speaker sim.   -> speaker_similarity   (ECAPA; paper uses wavlm-base-plus-sv)
    accent prob/sim-> aid_acc / cs_accent  (GenAID; paper uses VoxProfile)

For an ``infer_accent`` sweep laid out as ``<sweep>/alpha_<a>/utt####.wav`` this
produces one metrics row per alpha, so you can check the core claim directly:
accent similarity should rise with alpha while speaker similarity stays high.

Because the reference is fixed per accent across the sweep (the accent's
native-language L1 clip), speaker similarity is measured against that reference clip. Accent metrics are optional and need
real target-accent clips to compare against (``--accent-ref``), mirroring how
cs_accent stood in for VoxProfile accent-sim in our benchmark.

f0_rmse/mcd/ppg_kl are NOT computed unless ``--natural-ref`` is given: all three
(``evaluation_functions.py``'s own signatures) compare against a natural recording
of the SAME utterance content, not just the same accent -- they measure
reconstruction/pronunciation fidelity for a specific sentence, not general accent
similarity. Held-out ``--transcripts`` synthesised zero-shot have no such natural
counterpart unless you supply one (e.g. a bilingual speaker's natural English
recording of each eval transcript, index-paired like ``--accent-ref``). Without
that, these three columns are correctly omitted rather than computed against an
unrelated sentence, which would report a real-looking but meaningless number.

Usage
-----
    python -m accent_vector.experiments.evaluate \
        --sweep-dir results/per-accent/british \
        --transcripts data/transcripts/eval_transcripts.txt \
        --ref-wav refs/england.wav \
        --accent-ref /data/vctk_england_clips \
        --natural-ref /data/bilingual_natural_english_clips \
        --target-accent English \
        --out-csv results/per-accent/british/metrics.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "Evaluation"))

def alpha_dirs(sweep_dir):
    """Yield (alpha, dir) for alpha_<a> subdirs, sorted by alpha. Falls back to
    treating sweep_dir itself as a single (alpha=None) set of wavs."""
    sweep_dir = Path(sweep_dir)
    found = []
    for d in sweep_dir.iterdir() if sweep_dir.is_dir() else []:
        m = re.fullmatch(r"alpha_([0-9.]+)", d.name)
        if d.is_dir() and m:
            found.append((float(m.group(1)), d))
    if found:
        return sorted(found, key=lambda t: t[0])
    return [(None, sweep_dir)]


def wavs_in(d):
    return sorted(str(p) for p in Path(d).glob("*.wav"))


def score_dir(d, transcripts, ref_wav, accent_refs, target_accent, device, natural_refs=None):
    import evaluation_functions as ef

    wavs = wavs_in(d)
    row = {"n": len(wavs)}

    # --- WER (intelligibility): utt#### <-> transcripts[####] ---
    if transcripts:
        errs = []
        for w in wavs:
            m = re.search(r"utt(\d+)", Path(w).stem)
            if m and int(m.group(1)) < len(transcripts):
                errs.append(ef.wer(w, transcripts[int(m.group(1))]))
        row["wer"] = sum(errs) / len(errs) if errs else "n/a"

    # --- speaker similarity vs the fixed native-language (L1) reference ---
    if ref_wav:
        try:
            row["spk_sim"] = ef.speaker_similarity(wavs, [ref_wav] * len(wavs), device=device)
        except Exception as e:
            row["spk_sim"] = f"ERR: {e}"

    # --- accent similarity (cosine) vs real target-accent clips ---
    if accent_refs:
        try:
            refs = (accent_refs * ((len(wavs) // len(accent_refs)) + 1))[: len(wavs)]
            row["accent_cs"] = ef.cs_accent(wavs, refs)
        except Exception as e:
            row["accent_cs"] = f"ERR: {e}"

    # --- accent-ID accuracy vs the intended label ---
    if target_accent:
        try:
            row["accent_acc"] = ef.aid_acc(wavs, [target_accent] * len(wavs))
        except Exception as e:
            row["accent_acc"] = f"ERR: {e}"

    # --- f0_rmse / mcd / ppg_kl: need a NATURAL recording of the same utterance
    # content (not just the same accent) as ground truth -- only run if supplied.
    if natural_refs:
        refs = (natural_refs * ((len(wavs) // len(natural_refs)) + 1))[: len(wavs)]
        for name, fn in (("f0_rmse", ef.f0_rmse), ("mcd", ef.mcd), ("ppg_kl", ef.ppg_kl)):
            try:
                vals = [fn(w, r) for w, r in zip(wavs, refs)]
                row[name] = sum(vals) / len(vals) if vals else "n/a"
            except Exception as e:
                row[name] = f"ERR: {e}"

    return row


def main():
    parser = argparse.ArgumentParser(description="Score an alpha-sweep with the repo eval suite")
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--transcripts", help="same file fed to infer_accent")
    parser.add_argument("--ref-wav", help="native-language (L1) reference (for speaker similarity)")
    parser.add_argument("--accent-ref", help="dir of real target-accent clips (for cs_accent)")
    parser.add_argument("--natural-ref",
                        help="dir of natural recordings of the SAME utterances as "
                             "--transcripts, index-paired (for f0_rmse/mcd/ppg_kl); "
                             "omit to skip these three, since they need matching "
                             "content, not just matching accent")
    parser.add_argument("--target-accent", help="intended accent label (for aid_acc)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    transcripts = None
    if args.transcripts:
        with open(args.transcripts, encoding="utf-8") as f:
            transcripts = [ln.strip() for ln in f if ln.strip()]
    accent_refs = wavs_in(args.accent_ref) if args.accent_ref else None
    natural_refs = wavs_in(args.natural_ref) if args.natural_ref else None

    rows = []
    for alpha, d in alpha_dirs(args.sweep_dir):
        print(f"[eval] alpha={alpha} dir={d}")
        row = {"alpha": alpha}
        row.update(score_dir(d, transcripts, args.ref_wav, accent_refs,
                             args.target_accent, args.device, natural_refs=natural_refs))
        rows.append(row)

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "alpha", k))
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] wrote {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
