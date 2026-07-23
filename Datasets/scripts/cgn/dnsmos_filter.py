#!/usr/bin/env python3
"""Quality-filter a metadata.csv by DNSMOS, matching Accent Vector paper Section 4.2:
"selecting only utterances with a DNSMOS score higher than 3.4".

Scores each clip listed in <clips>/metadata.csv and writes a filtered CSV
(metadata.dnsmos.csv) plus a per-clip score table. Point the fork's `prepare`
stage at the filtered CSV afterwards. Re-runnable with a different --min without
re-cutting audio.

    python dnsmos_filter.py --clips /exports/.../cgn_dutch_clips --min 3.4

Needs DNSMOS:  pip install speechmos
DNSMOS operates at 16 kHz (our clips are already 16 kHz).
"""

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm


def load_dnsmos():
    try:
        from speechmos import dnsmos
    except ImportError:
        sys.exit("DNSMOS backend missing -- install it in this env:  pip install speechmos")
    return dnsmos


def score_metric(result, metric):
    """Pull one MOS out of the DNSMOS result dict, tolerant to key casing across
    speechmos versions (e.g. 'ovrl_mos' vs 'OVRL')."""
    wanted = {
        "ovrl": ("ovrl_mos", "OVRL", "ovrl", "OVRL_MOS"),
        "sig":  ("sig_mos", "SIG", "sig", "SIG_MOS"),
        "bak":  ("bak_mos", "BAK", "bak", "BAK_MOS"),
        "p808": ("p808_mos", "P808_MOS", "p808"),
    }[metric]
    for k in wanted:
        if k in result:
            return float(result[k])
    raise KeyError(f"'{metric}' not in DNSMOS output keys: {list(result)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True,
                    help="dir containing metadata.csv and wavs/ (from prep_cgn_f5.py)")
    ap.add_argument("--min", type=float, default=3.4,
                    help="DNSMOS threshold; paper Section 4.2 uses 3.4")
    ap.add_argument("--metric", choices=("ovrl", "sig", "bak", "p808"), default="p808",
                    help="which DNSMOS score to threshold on; default p808 = the single-score "
                         "DNSMOS (P.808) the paper cites [36]. All four are logged regardless.")
    ap.add_argument("--out", default=None,
                    help="filtered CSV path (default <clips>/metadata.dnsmos.csv)")
    args = ap.parse_args()

    clips = Path(args.clips)
    src = clips / "metadata.csv"
    out = Path(args.out) if args.out else clips / "metadata.dnsmos.csv"
    dnsmos = load_dnsmos()

    with open(src, newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="|")
        next(r)                                   # header
        rows = [row for row in r if len(row) >= 2]
    print(f"scoring {len(rows)} clips (DNSMOS {args.metric} >= {args.min})", file=sys.stderr)

    metrics = ("ovrl", "sig", "bak", "p808")       # log all so we can re-threshold cheaply
    kept = []
    n_drop = 0
    with open(clips / "dnsmos_scores.tsv", "w", encoding="utf-8") as sf:
        sf.write("audio_file\t" + "\t".join(metrics) + "\n")
        bar = tqdm(rows, desc=f"DNSMOS {args.metric}>= {args.min}", unit="clip")
        for rel, text in bar:
            wav = clips / rel
            try:
                # pass the path: speechmos reads + resamples to 16 kHz internally
                result = dnsmos.run(str(wav), 16000)
                scores = {m: score_metric(result, m) for m in metrics}
            except Exception as e:
                print(f"! scoring failed for {rel}: {e}", file=sys.stderr)
                n_drop += 1
                continue
            sf.write(rel + "\t" + "\t".join(f"{scores[m]:.3f}" for m in metrics) + "\n")
            if scores[args.metric] >= args.min:
                kept.append((rel, text))
            else:
                n_drop += 1
            bar.set_postfix(kept=len(kept), dropped=n_drop)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["audio_file", "text"])
        w.writerows(kept)

    print(f"\nkept {len(kept)}/{len(rows)} clips; dropped {n_drop}", file=sys.stderr)
    print(f"  filtered manifest -> {out}", file=sys.stderr)
    print(f"  per-clip scores   -> {clips/'dnsmos_scores.tsv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
