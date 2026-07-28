#!/usr/bin/env python3
"""
Answer one question empirically: are the IndicVoices-R transcripts native-script
(Devanagari / Bengali) or already romanised (Latin)? -> is romanize.py needed?

Samples rows from the manifest(s), classifies every transcript character by Unicode
block, and prints the mix + a verdict. Run BEFORE assuming romanisation is required.

    python inspect_script.py --manifest '/path/iv_r/hi/*.jsonl' --n 200

Reads the same manifests as select_indicvoices.py (.jsonl / .json / dir / glob).
"""

import argparse
import glob
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path


def iter_manifest(paths):
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for jf in sorted(p.rglob("*.json")):
                with open(jf, encoding="utf-8") as f:
                    yield json.load(f)
            continue
        with open(p, encoding="utf-8") as f:
            head = f.read(1); f.seek(0)
            if head == "[":
                for row in json.load(f):
                    yield row
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)


def expand(patterns):
    out = []
    for pat in patterns:
        hits = glob.glob(pat)
        out.extend(hits if hits else [pat])
    return out


def block(ch):
    o = ord(ch)
    if 0x0900 <= o <= 0x097F: return "devanagari"
    if 0x0980 <= o <= 0x09FF: return "bengali"
    if o < 0x80 and ch.isalpha(): return "latin"
    if ch.isalpha(): return "other-letter"
    return "non-letter"          # digits, punctuation, space


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, nargs="+")
    ap.add_argument("--n", type=int, default=200, help="rows to sample (default 200)")
    ap.add_argument("--field", default="normalized",
                    help="transcript field to inspect (normalized/verbatim/text)")
    ap.add_argument("--show", type=int, default=5, help="example transcripts to print")
    args = ap.parse_args()

    counts = Counter()
    examples = []
    seen = 0
    for row in iter_manifest(expand(args.manifest)):
        text = (row.get(args.field) or row.get("text") or "").strip()
        if not text:
            continue
        if len(examples) < args.show:
            examples.append(text)
        for ch in text:
            counts[block(ch)] += 1
        seen += 1
        if seen >= args.n:
            break

    if not seen:
        sys.exit("no non-empty transcripts found -- check --field / --manifest")

    letters = {k: counts[k] for k in ("devanagari", "bengali", "latin", "other-letter")}
    tot_letters = sum(letters.values()) or 1
    print(f"sampled {seen} transcripts (field='{args.field}')\n")
    print("examples:")
    for e in examples:
        print(f"  {e}")
    print("\nletter mix:")
    for k, v in sorted(letters.items(), key=lambda x: -x[1]):
        print(f"  {k:<13} {v:>8}  ({100*v/tot_letters:.1f}%)")

    native = letters["devanagari"] + letters["bengali"]
    latin = letters["latin"]
    print("\nverdict:")
    if native / tot_letters > 0.5:
        print("  NATIVE script dominant -> romanize.py IS needed before `prepare`.")
    elif latin / tot_letters > 0.9:
        print("  Already Latin/romanised -> romanize.py NOT needed; skip it.")
    else:
        print("  Mixed -> inspect manually; some native script present, romanize.py likely needed.")
    print("  (Either way, vocab_check.py is the final gate: ~0 OOV = safe to train.)")


if __name__ == "__main__":
    main()
