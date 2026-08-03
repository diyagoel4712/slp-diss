#!/usr/bin/env python3
"""AISHELL-1 (LDC2018S14 packaging) select + extract, run ON DICE with stdlib only
(no audio libraries -- FLAC duration comes from the STREAMINFO header, no decoding).

Gender-balances speakers from docs/spkrinfo.txt, keeps only the MIC (high-fidelity)
channel, drops utterances < --min-dur, and extracts the kept .flac + paired .txt into
<out>/<spk>/ -- ready to rsync to Eddie (so you move ~what-you-want, not the 38 GB of
all-3-channels).

    python3 select_extract_aishell.py --root /group/corpora/large2/AISHELL-1 \
        --out ~/aishell_mic_sel --min-dur 3.0
    # then: rsync -avh ~/aishell_mic_sel/ <UUN>@eddie:/exports/eddie/scratch/<UUN>/data/aishell_mic/

spkrinfo.txt columns: id  age  gender(M/F)  region  area(North/...).
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path


def flac_duration(b):
    """b: first >=42 bytes of a FLAC file -> duration in seconds (from STREAMINFO)."""
    if b[:4] != b"fLaC":
        return None
    si = b[8:8 + 34]                                  # STREAMINFO body (block 0)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="AISHELL-1 dir (has data/ and docs/spkrinfo.txt)")
    ap.add_argument("--out", required=True, help="staging dir for the selected MIC subset")
    ap.add_argument("--min-dur", type=float, default=3.0, help="drop utterances shorter than this (s)")
    ap.add_argument("--max-per-gender", type=int, default=0,
                    help="cap speakers per gender (0 = all, balanced to the smaller gender)")
    ap.add_argument("--area", default="", help="optional: keep only this speaker area (e.g. North)")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = read_spkrinfo(root / "docs" / "spkrinfo.txt")

    avail = {z.stem: z for z in (root / "data").glob("C*.zip")}   # speakers with a data zip
    by_g = {"M": [], "F": []}
    for spk in sorted(avail):
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
          f"{' (area=' + args.area + ')' if args.area else ''} -> balanced {n}+{n}={len(selected)}",
          file=sys.stderr)

    n_utt, total, short = 0, 0.0, 0
    hrs = {"M": 0.0, "F": 0.0}
    for k, spk in enumerate(selected, 1):
        g = info[spk][0]
        d = out / spk
        d.mkdir(exist_ok=True)
        with zipfile.ZipFile(avail[spk]) as zf:
            names = set(zf.namelist())
            for nm in sorted(names):
                if "/MIC/" not in nm or not nm.endswith(".flac"):
                    continue
                with zf.open(nm) as f:
                    dur = flac_duration(f.read(64))
                if dur is None or dur < args.min_dur:
                    short += dur is not None
                    continue
                txt = nm[:-5] + ".txt"
                if txt not in names:
                    continue
                base = os.path.basename(nm)                       # MC0002W0122.flac
                (d / base).write_bytes(zf.read(nm))
                (d / (base[:-5] + ".txt")).write_bytes(zf.read(txt))
                n_utt += 1; total += dur; hrs[g] += dur
        if k % 20 == 0:
            print(f"  {k}/{len(selected)} speakers -> {n_utt} utts ({total/3600:.1f} h)", file=sys.stderr)

    print(f"\ndone: {n_utt} utts, {total/3600:.1f} h "
          f"(M={hrs['M']/3600:.1f} h, F={hrs['F']/3600:.1f} h; dropped {short} < {args.min_dur}s)",
          file=sys.stderr)
    print(f"  staged -> {out}   (now rsync to Eddie)", file=sys.stderr)


if __name__ == "__main__":
    main()
