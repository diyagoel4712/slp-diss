#!/usr/bin/env python3
"""Fetch held-out FLEURS L1 prompt candidates for a decoupled accent -- corpus-independent of
the accent-vector training data, so no speaker holdout / retraining needed.

Streams google/fleurs <config> with decode=False (raw WAV bytes -> no torchcodec), picks
N candidates per gender in a duration window, and writes into --out:
    <prefix>_<M|F>_<id>.wav        (FLEURS is 16 kHz mono)
    <prefix>_<M|F>_<id>_ref.txt    raw FLEURS transcription

REF-TEXT FORM:
  * Mandarin (cmn_hans_cn): the raw Hanzi transcription IS the ref-text -- F5 pinyin-converts
    it at infer, exactly as AISHELL Hanzi was at train. No romanisation.
  * Indic (hi_in, bn_in): romanise the _ref.txt afterwards to match training (indic-translit).

FLEURS gender is a ClassLabel: 0=male, 1=female. Some configs are single-gender in test/dev
(Arabic's test was female-only) -- default --split train usually has both; the run prints the
per-gender count so you can switch splits if one comes up short.

  python fetch_fleurs_prompts.py --config cmn_hans_cn --prefix mandarin \
      --out ../../AccentVector/prompts/mandarin
"""
import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset, Audio

GENDER = {0: "M", 1: "F"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="FLEURS config, e.g. cmn_hans_cn / hi_in / ar_eg")
    ap.add_argument("--prefix", required=True, help="output name prefix, e.g. mandarin")
    ap.add_argument("--out", required=True, help="prompt output dir (e.g. AccentVector/prompts/mandarin)")
    ap.add_argument("--split", default="train", help="FLEURS split (train usually has both genders)")
    ap.add_argument("--n-per-gender", type=int, default=2)
    ap.add_argument("--min-samples", type=int, default=96000, help="min length in samples (6 s @16k)")
    ap.add_argument("--max-samples", type=int, default=144000, help="max length in samples (9 s @16k)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("google/fleurs", args.config, split=args.split, streaming=True) \
            .cast_column("audio", Audio(decode=False))

    picked = {0: 0, 1: 0}
    for r in ds:
        g, n = r["gender"], r["num_samples"]
        if g in picked and picked[g] < args.n_per_gender and args.min_samples <= n <= args.max_samples:
            stem = f"{args.prefix}_{GENDER[g]}_{r['id']}"
            with open(out / f"{stem}.wav", "wb") as f:
                f.write(r["audio"]["bytes"])
            with open(out / f"{stem}_ref.txt", "w", encoding="utf-8") as f:
                f.write(r["transcription"].strip() + "\n")
            picked[g] += 1
            print(f"  {GENDER[g]} id={r['id']} {n/16000:.1f}s -> {stem}.wav  \"{r['transcription'][:30]}\"")
        if all(v >= args.n_per_gender for v in picked.values()):
            break

    print(f"\npicked M={picked[0]} F={picked[1]} -> {out}", file=sys.stderr)
    short = [GENDER[g] for g, v in picked.items() if v < args.n_per_gender]
    if short:
        print(f"! only found {short} short of {args.n_per_gender} in split '{args.split}' -- "
              f"try --split test/validation, or widen --min/--max-samples", file=sys.stderr)


if __name__ == "__main__":
    main()
