#!/usr/bin/env python3
"""Extract held-out AISHELL-1 speakers as F5 Mandarin L1 *prompts* (ref audio + ref text).

Mandarin needs NO romanisation: F5's prepare/infer run convert_char_to_pinyin on Hanzi, so
the prompt ref-text is just the raw Hanzi transcript -- exactly the form the model trained on
(unlike Arabic, which had to match GlobalPhone's romanisation). Pulls one MIC clip per chosen
speaker in [--min-dur,--max-dur] (nearest --target) + its .txt, into --out:
    mandarin_<M|F>_<spk>.wav        mono 16 kHz
    mandarin_<M|F>_<spk>_ref.txt    Hanzi transcript (matches the audio + training)

HELD-OUT CAVEAT: prep_aishell_from_zips.py consumes a balanced speaker set, so these are only
truly held out if you EXCLUDE them from training and retrain:
    prep_aishell_from_zips.py --exclude-speakers <ids printed below> ...
Then set l1base() mandarin/m|f in AccentVector/scripts/submit_indic_ckpt_grid.sh to the
mandarin_<M|F>_<spk> basenames.

  python extract_aishell_prompts.py --root <aishell1_src> \
      --out ../../../AccentVector/prompts/mandarin        # auto-pick --n-per-gender each
      [--speakers C0002,C0004]                             # or choose explicitly
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_aishell_from_zips import flac_duration, read_spkrinfo  # noqa: E402


def pick_clip(zf, names, lo, hi, target):
    """Return (flac_bytes, hanzi_text, dur) for the MIC clip nearest `target` in [lo,hi]."""
    best = None
    for nm in sorted(names):
        if "/MIC/" not in nm or not nm.endswith(".flac"):
            continue
        txt = nm[:-5] + ".txt"
        if txt not in names:
            continue
        data = zf.read(nm)
        dur = flac_duration(data[:64])
        if dur is None or not (lo <= dur <= hi):
            continue
        text = zf.read(txt).decode("utf-8", "replace").strip()
        if not text:
            continue
        score = abs(dur - target)
        if best is None or score < best[0]:
            best = (score, data, text, dur)
    return None if best is None else best[1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="AISHELL-1 dir (data/C*.zip + docs/spkrinfo.txt)")
    ap.add_argument("--out", required=True, help="prompt output dir (e.g. AccentVector/prompts/mandarin)")
    ap.add_argument("--speakers", default="", help="comma-sep zip-stem ids to use (else auto-pick)")
    ap.add_argument("--n-per-gender", type=int, default=2)
    ap.add_argument("--min-dur", type=float, default=6.0)
    ap.add_argument("--max-dur", type=float, default=9.0)
    ap.add_argument("--target", type=float, default=7.0, help="preferred clip length (s)")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = read_spkrinfo(root / "docs" / "spkrinfo.txt")
    avail = {z.stem: z for z in (root / "data").glob("C*.zip")}
    if not avail:
        sys.exit(f"no C*.zip under {root/'data'}")

    if args.speakers:
        chosen = [s.strip() for s in args.speakers.split(",") if s.strip()]
    else:                                            # auto-pick first N of each gender
        by_g = {"M": [], "F": []}
        for spk in sorted(avail):
            g = info.get(spk, ("?", ""))[0]
            if g in by_g and len(by_g[g]) < args.n_per_gender:
                by_g[g].append(spk)
            if all(len(v) >= args.n_per_gender for v in by_g.values()):
                break
        chosen = by_g["M"] + by_g["F"]

    picked = []
    for spk in chosen:
        if spk not in avail:
            print(f"! {spk}: no zip found", file=sys.stderr); continue
        g = info.get(spk, ("?", ""))[0]
        G = g if g in ("M", "F") else "X"
        with zipfile.ZipFile(avail[spk]) as zf:
            res = pick_clip(zf, set(zf.namelist()), args.min_dur, args.max_dur, args.target)
        if res is None:
            print(f"! {spk}: no MIC clip in [{args.min_dur},{args.max_dur}]s", file=sys.stderr); continue
        data, text, dur = res
        stem = f"mandarin_{G}_{spk}"
        audio, sr = sf.read(io.BytesIO(data))
        sf.write(str(out / f"{stem}.wav"), audio, sr, subtype="PCM_16")
        with open(out / f"{stem}_ref.txt", "w", encoding="utf-8") as f:
            f.write(text + "\n")
        picked.append((stem, spk, G, dur))
        print(f"  {stem}  {dur:.1f}s  \"{text[:30]}\"")

    print(f"\nwrote {len(picked)} prompts to {out}", file=sys.stderr)
    if picked:
        ids = ",".join(p[1] for p in picked)
        print(f"HELD-OUT speaker ids: {ids}", file=sys.stderr)
        print(f"-> exclude from training + retrain:\n"
              f"   python prep_aishell_from_zips.py --exclude-speakers {ids} --root <aishell> --out <train_out>",
              file=sys.stderr)
        print("-> then set l1base() mandarin/m|f in submit_indic_ckpt_grid.sh to these basenames",
              file=sys.stderr)


if __name__ == "__main__":
    main()
