#!/usr/bin/env python3
"""
Turn the staged CGN subset into F5-TTS training clips + a metadata manifest.

Input layout (under --root):
    audio/wav/comp-*/nl/<id>.wav                 16 kHz mono recordings (whole files)
    annot/text/ort/comp-*/nl/<id>.ort.gz         Praat TextGrid, time-aligned orthography

For every non-empty interval in each recording's speaker tier we cut the audio,
write a clip at its NATIVE 16 kHz rate (F5 resamples to 24 kHz in the dataloader,
so pre-resampling would only be lossy upsampling), and append a manifest row.

The output is the "audio_file|text" CSV that this fork's
``accent_vector.data_preprocess prepare`` consumes directly (Phase B/C
"bring your own metadata.csv" path); prepare handles tokenization + Arrow build.

Output (under --out):
    wavs/<id>_<nnn>.wav
    metadata.csv        header "audio_file|text"; rows wavs/<id>_<nnn>.wav|<text>
    details.tsv         clip, recording, speaker, comp, start, end, dur, text  (bookkeeping)
"""

import argparse
import csv
import gzip
import re
import sys
from pathlib import Path

import torchaudio

SPK_TIER = re.compile(r"^[NV]\d{5}$")     # CGN speaker-ID tier names, e.g. N00703
SKIP_TOKENS = {"xxx", "ggg", "Xxx"}       # CGN: unintelligible / non-speech markers


# ---------- Praat short-format TextGrid parser ----------

def _unq(s):
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return s.replace('""', '"')


def parse_textgrid_short(text):
    """Return {tier_name: [(xmin, xmax, label), ...]} for interval tiers only."""
    lines = [ln for ln in (l.strip() for l in text.splitlines()) if ln != ""]
    # header: File type / "TextGrid" / xmin / xmax / <exists> / n_tiers
    i = 6
    n_tiers = int(lines[5])
    tiers = {}
    for _ in range(n_tiers):
        cls = _unq(lines[i]); name = _unq(lines[i + 1]); n = int(lines[i + 4])
        i += 5
        if cls == "IntervalTier":
            intervals = []
            for _ in range(n):
                xmin = float(lines[i]); xmax = float(lines[i + 1]); lab = _unq(lines[i + 2])
                intervals.append((xmin, xmax, lab))
                i += 3
            tiers[name] = intervals
        else:  # TextTier (point tier): time, mark -> skip, just advance
            i += 2 * n
    return tiers


def speaker_intervals(tiers):
    """Utterance intervals from the speaker tier (fallback: first interval tier)."""
    for name, iv in tiers.items():
        if SPK_TIER.match(name):
            return iv
    return next(iter(tiers.values()), [])


# ---------- text cleaning ----------

def clean_text(t):
    t = " ".join(t.split())
    if not t:
        return None
    if any(tok in SKIP_TOKENS for tok in t.split()):
        return None
    if not re.search(r"[A-Za-zÀ-ÿ]", t):     # no actual letters (pure punctuation/noise)
        return None
    return t


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="staged CGN dir (has audio/ and annot/)")
    ap.add_argument("--out", required=True, help="output dir for wavs/ + metadata.csv")
    ap.add_argument("--min-dur", type=float, default=3.0,
                    help="drop clips shorter than this (s); paper Section 4.2 uses 3.0")
    ap.add_argument("--max-dur", type=float, default=30.0, help="drop clips longer than this (s)")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)

    ort_files = sorted(root.glob("annot/text/ort/comp-*/nl/*.ort.gz"))
    print(f"found {len(ort_files)} transcripts", file=sys.stderr)

    n_clip = 0
    total_sec = 0.0
    meta_f = open(out / "metadata.csv", "w", newline="", encoding="utf-8")
    meta = csv.writer(meta_f, delimiter="|")            # quotes text if it contains | or "
    meta.writerow(["audio_file", "text"])               # header, as prepare expects
    det = open(out / "details.tsv", "w", encoding="utf-8")
    det.write("clip\trecording\tspeaker\tcomp\tstart\tend\tdur\ttext\n")

    for k, ort in enumerate(ort_files, 1):
        rid = ort.name[:-len(".ort.gz")]
        comp = ort.parts[-3]                              # comp-o
        wav_path = root / "audio" / "wav" / comp / "nl" / f"{rid}.wav"
        if not wav_path.exists():
            print(f"! missing wav for {rid}", file=sys.stderr)
            continue

        with gzip.open(ort, "rt", encoding="latin-1") as f:
            tiers = parse_textgrid_short(f.read())
        spk_name = next((n for n in tiers if SPK_TIER.match(n)), "?")
        utts = [(s, e, clean_text(t)) for (s, e, t) in speaker_intervals(tiers)]
        utts = [(s, e, t) for (s, e, t) in utts
                if t and args.min_dur <= (e - s) <= args.max_dur]
        if not utts:
            continue

        wav, sr = torchaudio.load(str(wav_path))         # native 16 kHz, no resample
        if wav.shape[0] > 1:                              # force mono
            wav = wav.mean(0, keepdim=True)

        for j, (s, e, t) in enumerate(utts):
            a, b = int(s * sr), int(e * sr)
            seg = wav[:, a:b]
            if seg.shape[1] == 0:
                continue
            name = f"{rid}_{j:03d}.wav"
            torchaudio.save(str(out / "wavs" / name), seg, sr,
                            encoding="PCM_S", bits_per_sample=16)
            meta.writerow([f"wavs/{name}", t])
            det.write(f"wavs/{name}\t{rid}\t{spk_name}\t{comp}\t{s:.3f}\t{e:.3f}\t{e-s:.3f}\t{t}\n")
            n_clip += 1
            total_sec += (e - s)

        if k % 200 == 0:
            print(f"  {k}/{len(ort_files)} recordings -> {n_clip} clips "
                  f"({total_sec/3600:.1f} h)", file=sys.stderr)

    meta_f.close()
    det.close()
    print(f"\ndone: {n_clip} clips, {total_sec/3600:.1f} h", file=sys.stderr)
    print(f"  {out/'metadata.csv'}\n  {out/'wavs'}/", file=sys.stderr)


if __name__ == "__main__":
    main()
