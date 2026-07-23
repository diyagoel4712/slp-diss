#!/usr/bin/env python3
"""Check that the transcripts are covered by F5's pretrained vocab BEFORE training,
so Dutch diacritics (e ->  , etc.) aren't silently lost at tokenization.

Mirrors accent_vector.data_preprocess.prepare exactly: same convert_char_to_pinyin
tokenizer, same pretrained vocab.txt that prepare copies for finetuning. Any character
reported "missing" would be out-of-vocab at train time.

    python vocab_check.py --metadata /exports/.../cgn_dutch_clips/metadata.dnsmos.csv
"""

import argparse
import csv
import sys
from collections import Counter
from importlib.resources import files

from f5_tts.model.utils import convert_char_to_pinyin

# same path prepare uses for the pretrained (finetune) vocab
PRETRAINED_VOCAB_PATH = files("f5_tts").joinpath("../../data/vocab.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="audio_file|text CSV")
    args = ap.parse_args()

    texts = []
    with open(args.metadata, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f, delimiter="|")
        next(r)                                   # header
        for row in r:
            if len(row) >= 2:
                texts.append(row[1].strip())
    print(f"{len(texts)} transcripts", file=sys.stderr)

    # tokenize in batches of 100, exactly as prepare does
    conv = []
    for i in range(0, len(texts), 100):
        conv.extend(convert_char_to_pinyin(texts[i:i + 100], polyphone=True))

    char_freq = Counter()
    for conv_text in conv:                        # conv_text is a list of tokens
        char_freq.update(conv_text)

    vocab = set(files_read(PRETRAINED_VOCAB_PATH))

    missing = {c: n for c, n in char_freq.items() if c not in vocab}
    total = sum(char_freq.values())
    miss_tokens = sum(missing.values())

    print(f"pretrained vocab: {len(vocab)} tokens")
    print(f"distinct tokens in data: {len(char_freq)}  "
          f"(covered {len(char_freq) - len(missing)}, missing {len(missing)})")
    print(f"missing-token occurrences: {miss_tokens}/{total} "
          f"({100 * miss_tokens / total:.4f}% of all tokens)")
    if missing:
        print("\nMISSING (token  codepoint  count):")
        for c, n in sorted(missing.items(), key=lambda x: -x[1]):
            print(f"  {c!r:>6}  U+{ord(c):04X}  x{n}" if len(c) == 1 else f"  {c!r}  (multi) x{n}")
        print("\n-> normalize these (e.g. strip diacritics) or extend vocab.txt before training.")
    else:
        print("\nOK: every token is in the pretrained vocab.")


def files_read(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]   # keep spaces; drop only the newline


if __name__ == "__main__":
    main()
