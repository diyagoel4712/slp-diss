#!/usr/bin/env python3
"""Extract held-out GlobalPhone Arabic speakers as F5 L1 *prompts* (ref audio + ref text).

The L1 prompt just needs held-out native Arabic speech whose transcript is in the SAME
romanisation the accent vector trained on -- GlobalPhone's own trl/ text, ASCII-normalised
by clean_translit(). Reusing GP (not FLEURS) means ZERO romanisation mismatch, unlike a
generic romaniser (uroman) whose Arabic convention won't match GlobalPhone's.

For each chosen speaker it picks one clip in [--min-dur, --max-dur] (nearest --target) and
writes, into --out (e.g. AccentVector/prompts/arabic):
    ar_<M|F>_<spk>.wav        mono 16 kHz
    ar_<M|F>_<spk>_ref.txt    clean_translit transcript (matches the audio + training)

HELD-OUT CAVEAT: prep_globalphone_f5.py consumes ALL speakers, so a prompt speaker is only
truly held out if you also EXCLUDE it from the training metadata and retrain:
    prep_globalphone_f5.py --exclude-speakers <ids printed below> ...
(natural to do when moving Arabic to the paper-faithful hparams). Then update l1base() in
AccentVector/scripts/submit_indic_ckpt_grid.sh to the ar_<M|F>_<spk> basenames.

  python extract_gp_prompts.py --root <gp_arabic> \
      --out ../../../AccentVector/prompts/arabic \
      --speakers AR001,AR042            # or omit to auto-pick --n-per-gender per gender
"""
import argparse
import os
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_globalphone_f5 import parse_trl, decode_shn, clean_translit, gender, SR  # noqa: E402


def pick_clip(root, shorten, spk, tier, lo, hi, target):
    """Return the speaker's clip nearest `target` within [lo,hi] as (utt_n, samples, ct, dur)."""
    trl = root / tier / f"{spk}.{tier}"
    if not trl.exists():
        return None
    num = spk[2:]                                   # AR001 -> 001 (adc/<num>/)
    best = None
    for utt_n, text in parse_trl(trl):
        ct = clean_translit(text)
        shn = root / "adc" / num / f"{spk}_{utt_n}.adc.shn"
        if not ct or not shn.exists():
            continue
        try:
            samp = decode_shn(shn, shorten)
        except Exception:                           # noqa: BLE001
            continue
        dur = len(samp) / SR
        if lo <= dur <= hi:
            score = abs(dur - target)
            if best is None or score < best[0]:
                best = (score, utt_n, samp, ct, dur)
    return None if best is None else best[1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="globalphone_arabic dir (has adc/ trl/ spk/)")
    ap.add_argument("--out", required=True, help="prompt output dir (e.g. AccentVector/prompts/arabic)")
    ap.add_argument("--shorten", default=os.path.expanduser("~/.local/bin/shorten"))
    ap.add_argument("--tier", default="trl", choices=("trl", "rmn"))
    ap.add_argument("--speakers", default="", help="comma-sep AR ids to use (else auto-pick)")
    ap.add_argument("--n-per-gender", type=int, default=2)
    ap.add_argument("--min-dur", type=float, default=6.0)
    ap.add_argument("--max-dur", type=float, default=9.0)
    ap.add_argument("--target", type=float, default=7.0, help="preferred clip length (s)")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not os.access(args.shorten, os.X_OK):
        sys.exit(f"shorten binary not executable: {args.shorten}")

    all_spk = sorted(p.stem for p in (root / args.tier).glob(f"AR*.{args.tier}"))
    if not all_spk:
        sys.exit(f"no {args.tier} files under {root/args.tier}")

    if args.speakers:
        chosen = [s.strip() for s in args.speakers.split(",") if s.strip()]
    else:                                           # auto-pick first N of each gender
        by_g = {"m": [], "f": []}
        for spk in all_spk:
            g = gender(root / "spk" / f"{spk}.spk")
            if g in by_g and len(by_g[g]) < args.n_per_gender:
                by_g[g].append(spk)
            if all(len(v) >= args.n_per_gender for v in by_g.values()):
                break
        chosen = by_g["m"] + by_g["f"]

    picked = []
    for spk in chosen:
        g = gender(root / "spk" / f"{spk}.spk")
        G = g.upper() if g in ("m", "f") else "X"
        res = pick_clip(root, args.shorten, spk, args.tier, args.min_dur, args.max_dur, args.target)
        if res is None:
            print(f"! {spk}: no clip in [{args.min_dur},{args.max_dur}]s", file=sys.stderr)
            continue
        utt_n, samp, ct, dur = res
        stem = f"ar_{G}_{spk}"
        w = wave.open(str(out / f"{stem}.wav"), "wb")
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(samp.astype("<i2").tobytes()); w.close()
        with open(out / f"{stem}_ref.txt", "w", encoding="utf-8") as f:
            f.write(ct + "\n")
        picked.append((stem, spk, G, utt_n, dur))
        print(f"  {stem}  utt {utt_n}  {dur:.1f}s  \"{ct[:60]}\"")

    print(f"\nwrote {len(picked)} prompts to {out}", file=sys.stderr)
    if picked:
        ids = ",".join(p[1] for p in picked)
        print(f"HELD-OUT speaker ids: {ids}", file=sys.stderr)
        print(f"-> exclude from training + retrain:\n"
              f"   python prep_globalphone_f5.py --exclude-speakers {ids} --root <gp> --out <train_out>",
              file=sys.stderr)
        print("-> then set l1base() arabic/m|f in submit_indic_ckpt_grid.sh to these ar_<M|F>_<spk> basenames",
              file=sys.stderr)


if __name__ == "__main__":
    main()
