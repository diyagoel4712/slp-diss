#!/usr/bin/env python3
"""Split a Stella recording into N utterance clips at its largest pauses, using ONLY ffmpeg
(no espeak/aeneas -- espeak isn't installable on Eddie). ffmpeg `silencedetect` finds pauses;
we cut at the midpoints of the N-1 longest interior silences. Not text-aligned, so VERIFY BY
EAR and tune --noise / --min-sil if boundaries are off.

Needs ffmpeg + ffprobe on PATH (conda install -c conda-forge ffmpeg).

  python split_by_silence.py --in ground_truth_refs/arabic/male/arabic_m.wav \
      --out-dir ground_truth_refs/arabic/male --prefix arabic_m --n 5
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path


def duration(audio):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nk=1:nw=1", audio],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def silences(audio, noise_db, min_sil):
    """-> [(start, end, dur)] silence intervals from ffmpeg silencedetect."""
    p = subprocess.run(["ffmpeg", "-i", audio, "-af",
                        f"silencedetect=noise={noise_db}dB:d={min_sil}", "-f", "null", "-"],
                       capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", p.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", p.stderr)]
    return [(s, e, e - s) for s, e in zip(starts, ends)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_wav", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--n", type=int, default=5, help="number of utterances to split into")
    ap.add_argument("--noise", default="-30", help="silence threshold in dB (raise toward -25 to find more)")
    ap.add_argument("--min-sil", type=float, default=0.25, help="min silence length (s) to count as a pause")
    args = ap.parse_args()

    if not Path(args.in_wav).is_file():
        sys.exit(f"input not found: {args.in_wav}")
    total = duration(args.in_wav)
    sils = silences(args.in_wav, args.noise, args.min_sil)

    # interior pauses only (ignore leading/trailing silence), then take the N-1 longest
    interior = [s for s in sils if s[0] > 0.05 and s[1] < total - 0.05]
    interior.sort(key=lambda x: -x[2])
    chosen = sorted(interior[:args.n - 1], key=lambda x: x[0])
    if len(chosen) < args.n - 1:
        print(f"! only {len(chosen)} interior pauses found for {args.n} clips -- "
              f"lower --min-sil or raise --noise (e.g. -25), or cut the rest by hand", file=sys.stderr)

    cuts = [(c[0] + c[1]) / 2 for c in chosen]          # cut at pause midpoints
    bounds = [0.0] + cuts + [total]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(bounds) - 1):
        b, e = bounds[i], bounds[i + 1]
        out = out_dir / f"{args.prefix}_s{i + 1}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.in_wav,
                        "-ss", f"{b:.3f}", "-to", f"{e:.3f}", str(out)], check=True)
        print(f"  s{i + 1}  {b:6.2f}-{e:6.2f}s ({e - b:.2f}s) -> {out.name}")
    print(f"\nwrote {len(bounds) - 1} clips to {out_dir} -- VERIFY BY EAR; "
          f"tune --noise/--min-sil if a boundary landed mid-sentence", file=sys.stderr)


if __name__ == "__main__":
    main()
