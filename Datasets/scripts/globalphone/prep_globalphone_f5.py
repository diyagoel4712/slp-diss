#!/usr/bin/env python3
"""GlobalPhone Arabic -> F5-TTS clips + metadata.csv.

Layout (under --root):
    adc/<spk>/AR<spk>_<N>.adc.shn   shorten-compressed, headerless raw PCM,
                                    16 kHz / 16-bit / LITTLE-ENDIAN (verified by autocorr)
    trl/AR<spk>.trl (or rmn/)       GlobalPhone romanisation; "; N:" marks utterance N,
                                    followed by its transcript line
    spk/AR<spk>.spk                 has ";SEX: male|female"

Decodes each utterance with the `shorten` binary, pairs it with its "; N:" transcript,
normalises the romanisation to ASCII, and writes wavs/ + metadata.csv (audio_file|text)
for the usual dnsmos_filter -> vocab_check -> prepare chain.

GlobalPhone Arabic is ~35 h read MSA and that's ~the whole target, so we keep essentially
all of it -- no balanced subselection step (unlike CGN).

The clean_translit() rules are a FIRST PASS: run vocab_check.py on the output and refine
the mapping if there's meaningful OOV (see the note there).
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

SR = 16000
UTT = re.compile(r"^;\s*(\d+)\s*:\s*$")          # "; 3:" utterance marker


def decode_shn(shn, shorten):
    """shorten-decompress -> np.int16 samples (little-endian raw PCM)."""
    with tempfile.NamedTemporaryFile(suffix=".adc", delete=False) as tf:
        tmp = tf.name
    try:
        subprocess.run([shorten, "-x", str(shn), tmp],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raw = open(tmp, "rb").read()
    finally:
        os.path.exists(tmp) and os.unlink(tmp)
    return np.frombuffer(raw, dtype="<i2")


def parse_trl(path):
    """-> [(utt_num, text)]. A '; N:' line is followed by its transcript line(s)."""
    utts, cur, buf = [], None, []
    for line in open(path, encoding="latin-1"):
        line = line.rstrip("\n")
        m = UTT.match(line)
        if m:
            if cur is not None:
                utts.append((cur, " ".join(buf).strip()))
            cur, buf = int(m.group(1)), []
        elif line.startswith(";"):
            continue                              # ;SpRecheRID and other comments
        elif cur is not None and line.strip():
            buf.append(line.strip())
    if cur is not None:
        utts.append((cur, " ".join(buf).strip()))
    return utts


def clean_translit(t):
    """FIRST-PASS normalise GlobalPhone romanisation -> lowercase ASCII. Adjust per vocab_check.

    Strips foreign/markup, taa-marbuta and boundary notation, folds Arabic-chat digits
    (7=Ha, 3=ain, ...), lowercases, keeps letters+spaces. Deterministic, so consistent
    for training even where it merges phonemic distinctions (like unvocalised text)."""
    t = re.sub(r"<[^>]*>", " ", t)                # <-foreign-> markers
    t = re.sub(r"\(T?e\)", "a", t)                # (Te)/(e) taa marbuta -> a
    t = re.sub(r"\(t\)", "t", t)
    t = re.sub(r"[~_+\-]", "", t)                 # shadda / underscore / clitic-join boundaries
    t = t.replace("7", "h")                       # 7 = Haa -> h
    t = re.sub(r"[0-9]", "", t)                    # drop remaining chat-digits (3=ain, etc.)
    t = t.lower()
    t = re.sub(r"[^a-z\s]", " ", t)               # letters + space only
    return re.sub(r"\s+", " ", t).strip()


def gender(spk_file):
    if spk_file.exists():
        for line in open(spk_file, encoding="latin-1"):
            m = re.search(r";SEX:\s*(\w+)", line, re.I)
            if m:
                return m.group(1).lower()[0]      # 'm' / 'f'
    return "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="globalphone_arabic dir (has adc/ trl/ spk/)")
    ap.add_argument("--out", required=True, help="output dir for wavs/ + metadata.csv")
    ap.add_argument("--shorten", default=os.path.expanduser("~/.local/bin/shorten"),
                    help="path to the shorten binary")
    ap.add_argument("--tier", default="trl", choices=("trl", "rmn"),
                    help="which GlobalPhone romanisation to read (default trl)")
    ap.add_argument("--min-dur", type=float, default=3.0, help="drop clips shorter than this (s)")
    ap.add_argument("--max-dur", type=float, default=30.0, help="drop clips longer than this (s)")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    if not os.access(args.shorten, os.X_OK):
        sys.exit(f"shorten binary not executable: {args.shorten} (build it, or pass --shorten)")

    trls = sorted((root / args.tier).glob("AR*." + args.tier))
    print(f"{len(trls)} speakers", file=sys.stderr)

    meta_f = open(out / "metadata.csv", "w", newline="", encoding="utf-8")
    meta = csv.writer(meta_f, delimiter="|"); meta.writerow(["audio_file", "text"])
    det = open(out / "details.tsv", "w", encoding="utf-8")
    det.write("clip\tspeaker\tgender\tdur\ttext\n")

    n, total, miss = 0, 0.0, 0
    for k, trl in enumerate(trls, 1):
        spk = trl.stem                            # AR001
        num = spk[2:]                             # 001  (adc/<num>/)
        g = gender(root / "spk" / f"{spk}.spk")
        for utt_n, text in parse_trl(trl):
            shn = root / "adc" / num / f"{spk}_{utt_n}.adc.shn"
            ct = clean_translit(text)
            if not shn.exists() or not ct:
                miss += not shn.exists()
                continue
            try:
                samp = decode_shn(shn, args.shorten)
            except Exception as e:                # noqa: BLE001
                print(f"! decode failed {shn}: {e}", file=sys.stderr); miss += 1; continue
            dur = len(samp) / SR
            if not (args.min_dur <= dur <= args.max_dur):
                continue
            name = f"{spk}_{utt_n}.wav"
            w = wave.open(str(out / "wavs" / name), "wb")
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes(samp.astype("<i2").tobytes()); w.close()
            meta.writerow([f"wavs/{name}", ct])
            det.write(f"wavs/{name}\t{spk}\t{g}\t{dur:.2f}\t{ct}\n")
            n += 1; total += dur
        if k % 20 == 0:
            print(f"  {k}/{len(trls)} speakers -> {n} clips ({total/3600:.1f} h)", file=sys.stderr)

    meta_f.close(); det.close()
    print(f"\ndone: {n} clips, {total/3600:.1f} h ({miss} missing/failed)", file=sys.stderr)
    print(f"  {out/'metadata.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
