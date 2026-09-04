#!/usr/bin/env python3
"""Convert mp3 (or any ffmpeg-readable audio) to the wav format the ground-truth refs use:
mono, 16 kHz, 16-bit PCM -- matching the existing ground_truth_refs/*/{male,female}/*.wav.
ffmpeg-only, so it runs on Eddie (conda install -c conda-forge ffmpeg).

Typical use is the step BEFORE split_by_silence.py: convert the downloaded paragraph
recording, then cut it into utterances.

  # one file, writes alongside as mandarin1.wav
  python mp3_to_wav.py ground_truth_refs/mandarin/female/mandarin1.mp3

  # a whole directory (non-recursive unless -r), explicit name, or a different rate
  python mp3_to_wav.py ground_truth_refs/mandarin -r
  python mp3_to_wav.py in.mp3 --out ground_truth_refs/mandarin/female/mandarin_f.wav
  python mp3_to_wav.py in.mp3 --sr 24000            # F5 prompt rate instead of eval's 16k
"""
import argparse
import subprocess
import sys
from pathlib import Path

AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".wav"}


def convert(src, dst, sr, channels, overwrite):
    if dst.resolve() == src.resolve():
        sys.exit(f"refusing to overwrite input in place: {src}")
    if dst.exists() and not overwrite:
        print(f"  skip (exists, pass --force): {dst}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-ac", str(channels), "-ar", str(sr), "-sample_fmt", "s16",
                    str(dst)], check=True)
    print(f"  {src.name} -> {dst}  (mono, {sr} Hz, 16-bit)" if channels == 1
          else f"  {src.name} -> {dst}  ({channels}ch, {sr} Hz, 16-bit)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="audio file(s) and/or directory(ies)")
    ap.add_argument("--out", help="output path (file input only); default: input with .wav")
    ap.add_argument("--out-dir", help="write all outputs here instead of beside each input")
    ap.add_argument("--sr", type=int, default=16000, help="output sample rate (default 16000, "
                                                          "matching the eval ground-truth refs)")
    ap.add_argument("--channels", type=int, default=1, help="output channels (default 1 = mono)")
    ap.add_argument("-r", "--recursive", action="store_true", help="recurse into input directories")
    ap.add_argument("--force", action="store_true", help="overwrite existing .wav outputs")
    args = ap.parse_args()

    if not any(Path(p).exists() for p in args.inputs):
        sys.exit(f"no such input: {args.inputs}")
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        sys.exit("ffmpeg not found on PATH (conda install -c conda-forge ffmpeg)")

    srcs = []
    for p in args.inputs:
        p = Path(p)
        if p.is_dir():
            found = sorted(f for f in (p.rglob("*") if args.recursive else p.iterdir())
                           if f.is_file() and f.suffix.lower() in AUDIO_EXTS
                           and f.suffix.lower() != ".wav")
            if not found:
                print(f"! no convertible audio in {p}"
                      f"{'' if args.recursive else ' (try -r)'}", file=sys.stderr)
            srcs += found
        elif p.is_file():
            srcs.append(p)
        else:
            sys.exit(f"input not found: {p}")

    if args.out and len(srcs) != 1:
        sys.exit(f"--out takes a single input file (got {len(srcs)}); use --out-dir")

    n = 0
    for src in srcs:
        if args.out:
            dst = Path(args.out)
        elif args.out_dir:
            dst = Path(args.out_dir) / (src.stem + ".wav")
        else:
            dst = src.with_suffix(".wav")
        n += convert(src, dst, args.sr, args.channels, args.force)
    print(f"\nconverted {n}/{len(srcs)} file(s). Next: split into utterances with "
          f"split_by_silence.py", file=sys.stderr)


if __name__ == "__main__":
    main()
