#!/usr/bin/env python3
"""AISHELL-1 (LDC2018S14 packaging) -> F5-TTS clips + metadata.csv, in ONE pass on Eddie,
reading the MIC (high-fidelity) FLAC straight out of the speaker zips -- no intermediate
extraction.

Gender-balances speakers from docs/spkrinfo.txt, keeps the MIC channel, drops utterances
outside [min-dur, max-dur] (duration read from the FLAC header, so short clips are skipped
without decoding), decodes the rest FLAC->wav, and keeps the Hanzi text AS-IS -- no
romanisation, because F5's base vocab is the Emilia ZH+EN pinyin vocab and the fork's
`prepare` runs convert_char_to_pinyin on Hanzi.

    python prep_aishell_from_zips.py \
        --root /exports/eddie/scratch/s2247837/data/aishell1_src \
        --out  /exports/eddie/scratch/s2247837/data/aishell_clips --min-dur 3.0
    # --root has data/C*.zip and docs/spkrinfo.txt

spkrinfo.txt columns: id  age  gender(M/F)  region  area(North/...).
Then: dnsmos_filter.py -> vocab_check.py -> prepare -> finetune (ACCENT_NAME=mandarin).
"""

import argparse
import csv
import io
import os
import sys
import zipfile
from pathlib import Path

import soundfile as sf


def flac_duration(b):
    """b: first >=42 bytes of a FLAC file -> seconds (from STREAMINFO, no decode)."""
    if b[:4] != b"fLaC":
        return None
    si = b[8:8 + 34]
    sr = (si[10] << 12) | (si[11] << 4) | (si[12] >> 4)
    total = ((si[13] & 0x0F) << 32) | (si[14] << 24) | (si[15] << 16) | (si[16] << 8) | si[17]
    return total / sr if sr else None


def read_spkrinfo(path):
    """-> {speaker_id: (gender, area)}."""
    info = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                info[p[0]] = (p[2].upper()[:1], p[4] if len(p) >= 5 else "")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="AISHELL-1 dir (data/C*.zip + docs/spkrinfo.txt)")
    ap.add_argument("--out", required=True, help="output dir for wavs/ + metadata.csv")
    ap.add_argument("--min-dur", type=float, default=3.0, help="drop clips shorter than this (s)")
    ap.add_argument("--max-dur", type=float, default=30.0, help="drop clips longer than this (s)")
    ap.add_argument("--max-per-gender", type=int, default=0,
                    help="cap speakers per gender (0 = all, balanced to the smaller gender)")
    ap.add_argument("--area", default="", help="optional: keep only this speaker area (e.g. North)")
    ap.add_argument("--exclude-speakers", default="",
                    help="comma-sep zip-stem ids to hold out of training (e.g. those used as L1 "
                         "prompts by extract_aishell_prompts.py) -- keeps the prompt speaker unseen")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    info = read_spkrinfo(root / "docs" / "spkrinfo.txt")

    avail = {z.stem: z for z in (root / "data").glob("C*.zip")}
    exclude = {s.strip() for s in args.exclude_speakers.split(",") if s.strip()}
    if exclude:
        print(f"excluding {len(exclude)} held-out speaker(s): {sorted(exclude)}", file=sys.stderr)
    by_g = {"M": [], "F": []}
    for spk in sorted(avail):
        if spk in exclude:
            continue
        g, area = info.get(spk, ("?", ""))
        if args.area and area != args.area:
            continue
        if g in by_g:
            by_g[g].append(spk)
    n = min(len(by_g["M"]), len(by_g["F"]))
    if args.max_per_gender:
        n = min(n, args.max_per_gender)
    selected = sorted(by_g["M"][:n] + by_g["F"][:n])
    print(f"speakers: M={len(by_g['M'])} F={len(by_g['F'])}"
          f"{' area=' + args.area if args.area else ''} -> balanced {n}+{n}={len(selected)}",
          file=sys.stderr)

    meta_f = open(out / "metadata.csv", "w", newline="", encoding="utf-8")
    meta = csv.writer(meta_f, delimiter="|"); meta.writerow(["audio_file", "text"])
    det = open(out / "details.tsv", "w", encoding="utf-8")
    det.write("clip\tspeaker\tgender\tdur\ttext\n")

    n_clip, total = 0, 0.0
    hrs = {"M": 0.0, "F": 0.0}
    for k, spk in enumerate(selected, 1):
        g = info[spk][0]
        with zipfile.ZipFile(avail[spk]) as zf:
            names = set(zf.namelist())
            for nm in sorted(names):
                if "/MIC/" not in nm or not nm.endswith(".flac"):
                    continue
                txt = nm[:-5] + ".txt"
                if txt not in names:
                    continue
                data = zf.read(nm)
                dur = flac_duration(data[:64])
                if dur is None or not (args.min_dur <= dur <= args.max_dur):
                    continue
                text = zf.read(txt).decode("utf-8", "replace").strip()
                if not text:
                    continue
                audio, sr = sf.read(io.BytesIO(data))          # MIC is mono 16 kHz
                name = os.path.basename(nm)[:-5] + ".wav"       # MC0002W0122.wav
                sf.write(str(out / "wavs" / name), audio, sr, subtype="PCM_16")
                meta.writerow([f"wavs/{name}", text])
                det.write(f"wavs/{name}\t{spk}\t{g}\t{dur:.2f}\t{text}\n")
                n_clip += 1; total += dur; hrs[g] += dur
        if k % 20 == 0:
            print(f"  {k}/{len(selected)} speakers -> {n_clip} clips ({total/3600:.1f} h)", file=sys.stderr)

    meta_f.close(); det.close()
    print(f"\ndone: {n_clip} clips, {total/3600:.1f} h "
          f"(M={hrs['M']/3600:.1f} h, F={hrs['F']/3600:.1f} h)", file=sys.stderr)
    print(f"  {out/'metadata.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
