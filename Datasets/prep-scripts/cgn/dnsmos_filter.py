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
import multiprocessing as mp
import sys
from pathlib import Path

from tqdm import tqdm

METRICS = ("ovrl", "sig", "bak", "p808")   # log all so we can re-threshold cheaply


def load_dnsmos():
    try:
        from speechmos import dnsmos
    except ImportError:
        sys.exit("DNSMOS backend missing -- install it in this env:  pip install speechmos")
    return dnsmos


# --- parallel worker (DNSMOS is per-clip independent; one model per process) ---
_DNSMOS = None
_CLIPS = None


def _init_worker(clips_dir):
    global _DNSMOS, _CLIPS
    _DNSMOS = load_dnsmos()
    _CLIPS = clips_dir


def _score_one(row):
    """Return (rel, text, scores_dict|None, err|None). Runs in a worker (or inline)."""
    rel, text = row[0], row[1]
    try:
        # pass the path: speechmos reads + resamples to 16 kHz internally
        result = _DNSMOS.run(str(_CLIPS / rel), 16000)
        scores = {m: score_metric(result, m) for m in METRICS}
    except Exception as e:  # noqa: BLE001 -- log + drop the clip, keep going
        return (rel, text, None, str(e))
    return (rel, text, scores, None)


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
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes (default 1). DNSMOS is per-clip "
                         "independent, so set this to ~#cores for a big speedup on large "
                         "sets (~10h single-threaded for 100h of audio -> ~jobs x faster).")
    args = ap.parse_args()

    clips = Path(args.clips)
    src = clips / "metadata.csv"
    out = Path(args.out) if args.out else clips / "metadata.dnsmos.csv"
    jobs = max(1, args.jobs)

    with open(src, newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="|")
        next(r)                                   # header
        rows = [row for row in r if len(row) >= 2]
    print(f"scoring {len(rows)} clips (DNSMOS {args.metric} >= {args.min}, jobs={jobs})",
          file=sys.stderr)

    kept = []
    n_drop = 0
    pool = None
    if jobs == 1:
        _init_worker(clips)                       # load the model in-process
        results = map(_score_one, rows)
    else:
        pool = mp.Pool(jobs, initializer=_init_worker, initargs=(clips,))
        results = pool.imap(_score_one, rows, chunksize=16)   # imap keeps input order

    with open(clips / "dnsmos_scores.tsv", "w", encoding="utf-8") as sf:
        sf.write("audio_file\t" + "\t".join(METRICS) + "\n")
        bar = tqdm(results, total=len(rows), desc=f"DNSMOS {args.metric}>= {args.min}", unit="clip")
        for rel, text, scores, err in bar:
            if scores is None:
                print(f"! scoring failed for {rel}: {err}", file=sys.stderr)
                n_drop += 1
                continue
            sf.write(rel + "\t" + "\t".join(f"{scores[m]:.3f}" for m in METRICS) + "\n")
            if scores[args.metric] >= args.min:
                kept.append((rel, text))
            else:
                n_drop += 1
            bar.set_postfix(kept=len(kept), dropped=n_drop)

    if pool is not None:
        pool.close()
        pool.join()

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["audio_file", "text"])
        w.writerows(kept)

    print(f"\nkept {len(kept)}/{len(rows)} clips; dropped {n_drop}", file=sys.stderr)
    print(f"  filtered manifest -> {out}", file=sys.stderr)
    print(f"  per-clip scores   -> {clips/'dnsmos_scores.tsv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
