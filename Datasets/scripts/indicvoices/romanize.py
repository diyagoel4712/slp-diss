#!/usr/bin/env python3
"""
Romanise a native-script metadata.csv (Devanagari / Bengali) into Latin so F5's base
vocab (Latin + pinyin only) can tokenise it. Native transcripts otherwise tokenise to
id 0 -- see AccentVector PROPOSAL.md gotcha #2 / README "Non-Latin transcripts".

Reads  <clips>/metadata.csv         (audio_file|text, native script; from prep_indicvoices_f5)
Writes <clips>/metadata.roman.csv   (audio_file|text, romanised)
Then run vocab_check.py on the .roman.csv and point `prepare` at it.

Backends (--backend):
  indicxlit          ai4bharat IndicXlit -- natural, ASCII, phonetic roman (default).
                     pip install ai4bharat-transliteration ; needs --lang (hi/bn).
  indic-translit     rule-based sanscript. --scheme HK|ITRANS (ASCII) or IAST (diacritics;
                     IAST adds chars that may be OOV -- prefer HK/ITRANS, then vocab_check).
  uroman             universal romaniser (ASCII). needs the `uroman` python package.

By default an ASCII-fold safety net runs after transliteration (NFKD + drop non-ASCII):
the ASCII schemes target ASCII, so any leftover non-ASCII is untransliterated residue
(rare candra vowels, orphan nukta, stray Unicode noise) that would be OOV -- folding it
away guarantees vocab-safe output. Disable with --no-ascii-fold (e.g. for an IAST run
whose diacritics you've added to vocab.txt).

Idempotent per row; re-runnable. Non-script characters (Latin code-switches, digits,
punctuation) pass through untouched in every backend.
"""

import argparse
import csv
import sys
import unicodedata
from pathlib import Path

from tqdm import tqdm


def ascii_fold(s):
    """Guarantee vocab-safe output: NFKD-normalise and drop any non-ASCII. The ASCII
    romanisation schemes (HK/ITRANS, IndicXlit, uroman) target ASCII, so anything
    non-ASCII left over is untransliterated residue (rare candra vowels, orphan nukta,
    stray Unicode noise) that would be OOV against F5's Latin base vocab. Returns the
    folded string and the count of characters removed."""
    folded = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    folded = " ".join(folded.split())
    dropped = sum(1 for c in s if not c.isspace()) - sum(1 for c in folded if not c.isspace())
    return folded, max(0, dropped)


def make_indicxlit(lang):
    from ai4bharat.transliteration import XlitEngine
    eng = XlitEngine(lang, beam_width=4, src_script_type="indic")   # indic -> roman

    def f(text):
        # translit_sentence keeps punctuation/latin/digits, romanises indic words
        return eng.translit_sentence(text, lang_code=lang) if text else text
    return f


def make_indic_translit(lang, scheme):
    from indic_transliteration import sanscript
    src = {"hi": sanscript.DEVANAGARI, "bn": sanscript.BENGALI,
           "mr": sanscript.DEVANAGARI, "ne": sanscript.DEVANAGARI}.get(lang)
    if src is None:
        sys.exit(f"indic-translit: no source script mapping for lang '{lang}' "
                 "(add it to romanize.py)")
    dst = getattr(sanscript, scheme.upper())

    def f(text):
        return sanscript.transliterate(text, src, dst) if text else text
    return f


def make_uroman(lang):
    import uroman as ur
    uro = ur.Uroman()

    def f(text):
        return uro.romanize_string(text, lcode=lang) if text else text
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True,
                    help="dir containing metadata.csv (from prep_indicvoices_f5.py)")
    ap.add_argument("--lang", required=True, help="ISO code of the source script (hi, bn, ...)")
    ap.add_argument("--backend", choices=("indicxlit", "indic-translit", "uroman"),
                    default="indicxlit")
    ap.add_argument("--scheme", default="HK",
                    help="indic-translit output scheme: HK|ITRANS (ASCII) or IAST (diacritics)")
    ap.add_argument("--in-name", default="metadata.csv")
    ap.add_argument("--out-name", default="metadata.roman.csv")
    ap.add_argument("--no-ascii-fold", action="store_true",
                    help="skip the ASCII-fold safety net (keep whatever the backend emits, "
                         "incl. non-ASCII residue). Default is to fold -> guaranteed vocab-safe. "
                         "Use only with a diacritic scheme (IAST) whose chars you've added to vocab.")
    args = ap.parse_args()
    fold = not args.no_ascii_fold

    clips = Path(args.clips)
    src = clips / args.in_name
    out = clips / args.out_name

    romanise = {
        "indicxlit": lambda: make_indicxlit(args.lang),
        "indic-translit": lambda: make_indic_translit(args.lang, args.scheme),
        "uroman": lambda: make_uroman(args.lang),
    }[args.backend]()

    with open(src, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f, delimiter="|")
        header = next(r)
        rows = [row for row in r if len(row) >= 2]
    print(f"romanising {len(rows)} rows with {args.backend} (lang={args.lang})", file=sys.stderr)

    n_empty = n_folded_rows = n_folded_chars = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(header)
        for rel, text in tqdm(rows, unit="row"):
            roman = romanise(text)
            roman = " ".join((roman or "").split())
            if fold:
                folded, dropped = ascii_fold(roman)
                if dropped:
                    n_folded_rows += 1
                    n_folded_chars += dropped
                roman = folded
            if not roman:
                n_empty += 1
                continue
            w.writerow([rel, roman])

    print(f"\nwrote {len(rows) - n_empty}/{len(rows)} rows -> {out}", file=sys.stderr)
    if n_empty:
        print(f"  ({n_empty} rows romanised to empty and were dropped)", file=sys.stderr)
    if fold and n_folded_chars:
        print(f"  (ASCII-fold removed {n_folded_chars} non-ASCII residue chars "
              f"across {n_folded_rows} rows)", file=sys.stderr)
    print("  next: python vocab_check.py --metadata "
          f"{out}   (expect ~0 OOV; investigate any misses)", file=sys.stderr)


if __name__ == "__main__":
    main()
