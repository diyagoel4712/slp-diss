#!/usr/bin/env python3
"""
Turn the staged CGN subset into F5-TTS training clips + a metadata manifest.

Input layout (under --root):
    audio/wav/comp-*/nl/<id>.wav                 16 kHz mono recordings (whole files)
    annot/text/ort/comp-*/nl/<id>.ort.gz         Praat TextGrid, time-aligned orthography

CGN's ort tier is a manual forced alignment, so its silences are exact pause
boundaries and its punctuation marks sentence boundaries. We segment each speaker's
speech at those pause/sentence boundaries into clips within a duration band (the
Emilia / LibriTTS approach), then cut the audio and write each clip at its NATIVE
16 kHz rate (F5 resamples to 24 kHz in the dataloader, so pre-resampling would only
be lossy upsampling).

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


# ---------- utterance segmentation ----------
# Long-form audio -> training utterances, following the Emilia / LibriTTS approach:
# segment one speaker's speech at PAUSE and SENTENCE boundaries, keeping each clip in
# a duration band the model trains on. CGN's ort tier is a manual forced alignment, so
# its silences are exact pause boundaries and its punctuation marks sentence boundaries
# -- no VAD/ASR needed. We accumulate whole sentences until a clip reaches target_dur,
# then close it at the next sentence/pause boundary; we never cross a silence longer
# than max_cross, and never exceed max_dur.

SENT_END = re.compile(r"[.!?]\s*$")                  # sentence-final punctuation


def segment_utterances(intervals, min_dur, target_dur, max_dur, max_cross, pause):
    """intervals: cleaned, non-empty (start, end, text) in time order.
    Returns (start, end, joined_text) clips segmented at pause/sentence boundaries."""
    clips, cur = [], None         # cur = [start, end, [texts]]
    for s, e, t in intervals:
        if cur is None:
            cur = [s, e, [t]]
            continue
        gap = s - cur[1]                          # silence before this interval
        dur = cur[1] - cur[0]                     # current clip length so far
        ends_sentence = bool(SENT_END.search(cur[2][-1]))
        if (e - cur[0]) > max_dur or gap > max_cross:
            # adding would overflow, or a real section break -> close here
            close = True
        elif dur >= target_dur and (ends_sentence or gap >= pause):
            # long enough; close at a natural (sentence or pause) boundary
            close = True
        else:
            close = False                         # keep building toward target
        if close:
            clips.append(cur)
            cur = [s, e, [t]]
        else:
            cur[1] = e
            cur[2].append(t)
    if cur:
        clips.append(cur)
    # keep only clips within the training duration band
    return [(c[0], c[1], " ".join(c[2])) for c in clips
            if min_dur <= (c[1] - c[0]) <= max_dur]


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="staged CGN dir (has audio/ and annot/)")
    ap.add_argument("--out", required=True, help="output dir for wavs/ + metadata.csv")
    ap.add_argument("--min-dur", type=float, default=3.0,
                    help="drop clips shorter than this (s); paper Section 4.2 uses 3.0")
    ap.add_argument("--max-dur", type=float, default=15.0,
                    help="hard cap on clip length (s)")
    ap.add_argument("--target-dur", type=float, default=10.0,
                    help="once a clip reaches this length, close it at the next sentence/pause boundary (s)")
    ap.add_argument("--max-cross", type=float, default=1.0,
                    help="never join across a silence longer than this (s); a longer pause forces a cut")
    ap.add_argument("--pause", type=float, default=0.3,
                    help="a silence >= this counts as a boundary where a target-length clip may close (s)")
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
        # cleaned, non-empty intervals in time order -> segment at pause/sentence boundaries
        clean = [(s, e, ct) for (s, e, t) in speaker_intervals(tiers)
                 if (ct := clean_text(t))]
        utts = segment_utterances(clean, args.min_dur, args.target_dur, args.max_dur,
                                  args.max_cross, args.pause)
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
