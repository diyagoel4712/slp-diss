#!/usr/bin/env python3
"""Staged AISHELL-1 MIC subset -> F5-TTS clips + metadata.csv, run ON EDDIE.

Input (from select_extract_aishell.py, rsynced to Eddie):
    <root>/<spk>/MC<spk>W<utt>.flac   16 kHz mono, high-fidelity mic
    <root>/<spk>/MC<spk>W<utt>.txt    Hanzi transcript (one line)

Decodes each FLAC -> wav and keeps the Hanzi text AS-IS -- no romanisation, because
F5's base vocab is the Emilia ZH+EN pinyin vocab and the fork's `prepare` runs
convert_char_to_pinyin on Hanzi. Emits metadata.csv for dnsmos -> vocab_check -> prepare.

Gender balance + the <3 s filter already happened in select_extract_aishell.py on DICE;
--min-dur here is just a safety net.
"""

import argparse
import csv
import sys
from pathlib import Path

import torchaudio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="staged aishell_mic dir (<spk>/*.flac + .txt)")
    ap.add_argument("--out", required=True, help="output dir for wavs/ + metadata.csv")
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=30.0)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    flacs = sorted(root.glob("*/*.flac"))
    print(f"{len(flacs)} flac files", file=sys.stderr)

    meta_f = open(out / "metadata.csv", "w", newline="", encoding="utf-8")
    meta = csv.writer(meta_f, delimiter="|"); meta.writerow(["audio_file", "text"])
    n, total, skip = 0, 0.0, 0
    for i, fl in enumerate(flacs, 1):
        txt = fl.with_suffix(".txt")
        if not txt.exists():
            skip += 1; continue
        text = txt.read_text(encoding="utf-8").strip()
        if not text:
            skip += 1; continue
        wav, sr = torchaudio.load(str(fl))
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        dur = wav.shape[1] / sr
        if not (args.min_dur <= dur <= args.max_dur):
            continue
        name = fl.stem + ".wav"
        torchaudio.save(str(out / "wavs" / name), wav, sr,
                        encoding="PCM_S", bits_per_sample=16)
        meta.writerow([f"wavs/{name}", text])
        n += 1; total += dur
        if i % 2000 == 0:
            print(f"  {i}/{len(flacs)} -> {n} clips ({total/3600:.1f} h)", file=sys.stderr)

    meta_f.close()
    print(f"\ndone: {n} clips, {total/3600:.1f} h ({skip} skipped: no/empty text)", file=sys.stderr)
    print(f"  {out/'metadata.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
