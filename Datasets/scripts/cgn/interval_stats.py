#!/usr/bin/env python3
"""Measure CGN ort speaker-tier interval durations, to verify the sub-3 s pileup
and size the merge parameters. Reads the exact same tiers prep_cgn_f5.py uses.

    python interval_stats.py --root /exports/eddie/scratch/s2247837/data/cgn_dutch
"""
import argparse
import gzip
import sys
from pathlib import Path

from prep_cgn_f5 import SPK_TIER, parse_textgrid_short, speaker_intervals, clean_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="staged CGN dir (has annot/text/ort)")
    args = ap.parse_args()

    orts = sorted(Path(args.root).glob("annot/text/ort/comp-*/nl/*.ort.gz"))
    print(f"{len(orts)} transcripts", file=sys.stderr)

    bins = {"<1": 0, "1-2": 0, "2-3": 0, "3-5": 0, "5-10": 0, ">10": 0}
    n = 0
    tot = 0.0
    below3 = 0
    below3_sec = 0.0
    multi_spk_files = 0        # sanity-check the single-speaker assumption

    for ort in orts:
        with gzip.open(ort, "rt", encoding="latin-1") as f:
            tiers = parse_textgrid_short(f.read())
        if sum(1 for t in tiers if SPK_TIER.match(t)) > 1:
            multi_spk_files += 1
        for s, e, t in speaker_intervals(tiers):
            if not clean_text(t):
                continue
            d = e - s
            n += 1
            tot += d
            if d < 3:
                below3 += 1
                below3_sec += d
            k = ("<1" if d < 1 else "1-2" if d < 2 else "2-3" if d < 3
                 else "3-5" if d < 5 else "5-10" if d < 10 else ">10")
            bins[k] += 1

    print(f"\nfiles with >1 speaker tier: {multi_spk_files}  (should be 0)")
    print(f"non-empty intervals: {n}")
    print(f"total speech: {tot/3600:.1f} h")
    if n:
        print(f"< 3 s: {below3} intervals ({100*below3/n:.1f}%), "
              f"{below3_sec/3600:.1f} h ({100*below3_sec/tot:.1f}% of speech)")
        print("duration histogram (intervals):")
        for k in ["<1", "1-2", "2-3", "3-5", "5-10", ">10"]:
            print(f"  {k:>5} s: {bins[k]}")


if __name__ == "__main__":
    main()
