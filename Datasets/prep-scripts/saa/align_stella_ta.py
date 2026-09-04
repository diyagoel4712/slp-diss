#!/usr/bin/env python3
"""Forced-align a Speech Accent Archive "Please call Stella" recording to the known
sentence list and print the inter-sentence boundary times -- a torchaudio replacement for
the aeneas step in split_stella.py (aeneas + espeak-ng install cleanly on neither macOS
nor Eddie; torchaudio's MMS_FA aligner is already in the accentvector-eval env).

Same rationale as split_stella.py: the transcript is fixed and known, so aligning known
text is robust to the speaker's accent, unlike ASR or silence heuristics (Stella's commas
produce pauses LONGER than some sentence boundaries, so split_by_silence's "N-1 longest
pauses" rule misfires on these recordings).

Prints a comma-separated boundary list to stdout for split_by_silence.py --at:

  conda run -n accentvector-eval python align_stella_ta.py --in <wav>
  python split_by_silence.py --in <wav> --out-dir <dir> --prefix <p> --at "$(...)"
"""
import argparse
import re
import sys
from pathlib import Path

import torch
import torchaudio
from torchaudio.pipelines import MMS_FA as BUNDLE

HERE = Path(__file__).resolve().parent
DEFAULT_SENTENCES = HERE / "stella_sentences.txt"


def normalise(line):
    """MMS_FA's lexicon is lowercase a-z plus apostrophe."""
    return [w for w in re.sub(r"[^a-z' ]", " ", line.lower()).split() if w]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_wav", required=True)
    ap.add_argument("--sentences", default=str(DEFAULT_SENTENCES))
    ap.add_argument("--verbose", action="store_true", help="also print per-sentence spans to stderr")
    args = ap.parse_args()

    sents = [ln.strip() for ln in open(args.sentences, encoding="utf-8") if ln.strip()]
    words, owner = [], []                      # flat word list + which sentence each came from
    for i, s in enumerate(sents):
        for w in normalise(s):
            words.append(w); owner.append(i)

    wav, sr = torchaudio.load(args.in_wav)
    wav = wav.mean(0, keepdim=True)            # mono
    if sr != BUNDLE.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, BUNDLE.sample_rate)
    model = BUNDLE.get_model()
    tokenizer, aligner = BUNDLE.get_tokenizer(), BUNDLE.get_aligner()
    with torch.inference_mode():
        emission, _ = model(wav)
        spans = aligner(emission[0], tokenizer(words))

    ratio = wav.size(1) / emission.size(1) / BUNDLE.sample_rate      # frames -> seconds
    starts = [s[0].start * ratio for s in spans]
    ends = [s[-1].end * ratio for s in spans]

    cuts = []
    for i in range(len(words) - 1):            # boundary = midpoint of the gap at a sentence change
        if owner[i] != owner[i + 1]:
            cuts.append((ends[i] + starts[i + 1]) / 2)
    if args.verbose:
        for i, s in enumerate(sents):
            idx = [j for j, o in enumerate(owner) if o == i]
            print(f"  s{i+1}  {starts[idx[0]]:6.2f}-{ends[idx[-1]]:6.2f}s  \"{s[:50]}\"", file=sys.stderr)
    print(",".join(f"{c:.3f}" for c in cuts))


if __name__ == "__main__":
    main()
