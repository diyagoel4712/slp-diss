#!/usr/bin/env python3
"""Split a Speech Accent Archive "Please call Stella" recording into per-utterance clips
via aeneas forced alignment against the known sentence list -- for use as rq3/cs_accent
ground truth (utterance-sized natural target-accent clips).

Why forced alignment (not transcription): the transcript is fixed and known, so aligning
the known text to the audio is robust to the speaker's accent -- unlike ASR, which would
mis-transcribe accented English. aeneas synthesises the text (espeak) and DTW-aligns it.

Requires aeneas + espeak-ng + ffmpeg (ffmpeg also does the cutting, so no python audio
deps):  pip install numpy && pip install aeneas

The eval reads GT from ground_truth_refs/<accent>/<male|female>/ (a per-gender POOL, cycled
+ DTW-aligned in rq3), so pooling several speakers' sentence clips there is fine; they just
need to be utterance-sized, not one long paragraph.

  # run once per gender (inputs live one level up from the eval path):
  python split_stella.py --in ground_truth_refs/male/arabic_m.wav \
      --out-dir ground_truth_refs/arabic/male   --prefix arabic_m
  python split_stella.py --in ground_truth_refs/female/arabic_f.wav \
      --out-dir ground_truth_refs/arabic/female --prefix arabic_f
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SENTENCES = HERE / "stella_sentences.txt"


def align(audio, text_file, lang):
    """aeneas forced alignment -> [(text, begin_s, end_s)] per non-empty line."""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    cfg = f"task_language={lang}|is_text_type=plain|os_task_file_format=json"
    subprocess.run([sys.executable, "-m", "aeneas.tools.execute_task",
                    audio, str(text_file), cfg, tmp], check=True)
    frags = json.load(open(tmp))["fragments"]
    os.unlink(tmp)
    out = []
    for f in frags:
        line = " ".join(f.get("lines", [])).strip()
        if line:                                    # skip aeneas HEAD/TAIL empty fragments
            out.append((line, float(f["begin"]), float(f["end"])))
    return out


def cut(audio, begin, end, out_path):
    """Precise cut with ffmpeg (re-encode; -ss/-to after -i for accuracy)."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio),
                    "-ss", f"{begin:.3f}", "-to", f"{end:.3f}", str(out_path)], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_wav", required=True, help="one SAA Stella recording")
    ap.add_argument("--out-dir", required=True, help="ground_truth_refs/<accent>/<male|female>")
    ap.add_argument("--prefix", required=True, help="clip name prefix, e.g. arabic_m")
    ap.add_argument("--sentences", default=str(DEFAULT_SENTENCES),
                    help="one utterance per line (default: shared Stella 5-line split)")
    ap.add_argument("--lang", default="eng", help="aeneas task_language (Stella text is English)")
    args = ap.parse_args()

    if not Path(args.in_wav).is_file():
        sys.exit(f"input not found: {args.in_wav}")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n_expected = sum(1 for ln in open(args.sentences, encoding="utf-8") if ln.strip())

    frags = align(args.in_wav, args.sentences, args.lang)
    if len(frags) != n_expected:
        print(f"! warning: aligned {len(frags)} fragments but sentences file has {n_expected} "
              f"-- check the split by ear", file=sys.stderr)
    for i, (line, b, e) in enumerate(frags, 1):
        out = out_dir / f"{args.prefix}_s{i}.wav"
        cut(args.in_wav, b, e, out)
        print(f"  s{i}  {b:6.2f}-{e:6.2f}s  ({e-b:.2f}s)  -> {out.name}   \"{line[:45]}\"")
    print(f"\nwrote {len(frags)} clips to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
