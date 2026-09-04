#!/usr/bin/env python3
"""
Select a balanced ~N-hour Netherlandic-Dutch subset of CGN for finetuning.

Pipeline (in order):
  1. recording-level gate : recordingID starts with "fn"  (Netherlandic; "fv" = Flemish)
  2. single-speaker gate  : exactly one ID in speakerIDs
  3. speaker NL gate       : birthPlace starts "NL", birthRegion starts "regN",
                             and residence/education region are NOT explicitly Flemish (regV)
  4. balanced greedy select: alternate sex1/sex2, take whole speakers, sum secCount -> target
  5. progressive tighten   : if the eligible pool exceeds the target, require firstLang == SD;
                             if it STILL exceeds the target, also require homeLang == SD

Outputs: a summary to stdout and one recordingID per line to --out.
"""

import argparse
import csv
import sys
from collections import defaultdict

ENC = "latin-1"          # CGN meta text files
SD = "SD"                # Standard Dutch code in firstLang / homeLang


def load_speakers(path):
    """ID -> dict of the speaker fields we care about."""
    spk = {}
    with open(path, encoding=ENC) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = (row.get("ID") or "").strip()
            if not sid:
                continue
            spk[sid] = {
                "sex":         (row.get("sex") or "").strip(),
                "birthPlace":  (row.get("birthPlace") or "").strip(),
                "birthRegion": (row.get("birthRegion") or "").strip(),
                "resRegion":   (row.get("resRegion") or "").strip(),
                "eduRegion":   (row.get("eduRegion") or "").strip(),
                "firstLang":   (row.get("firstLang") or "").strip(),
                "homeLang":    (row.get("homeLang") or "").strip(),
            }
    return spk


def is_nl_speaker(s):
    """Netherlandic by birth + region, and not explicitly raised/schooled in Flanders."""
    return (
        s["birthPlace"].startswith("NL")
        and s["birthRegion"].startswith("regN")
        and not s["resRegion"].startswith("regV")
        and not s["eduRegion"].startswith("regV")
    )


def load_recordings(path):
    """List of (recordingID, secCount, [speakerIDs]) for fn + single-speaker rows."""
    recs = []
    with open(path, encoding=ENC) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rid = (row.get("recordingID") or "").strip()
            if not rid.startswith("fn"):            # gate 1: Netherlandic
                continue
            raw = (row.get("speakerIDs") or "").strip()
            ids = [x.strip() for x in raw.split(",") if x.strip()]
            if len(ids) != 1:                       # gate 2: single speaker
                continue
            try:
                sec = int((row.get("secCount") or "").strip())
            except ValueError:
                continue
            recs.append((rid, sec, ids[0]))
    return recs


def build_pool(recs, spk, require_first_sd=False, require_home_sd=False):
    """
    Return speaker_id -> {"sex", "sec", "recs":[ids]} for speakers passing the NL gate
    plus any requested language tightening. Only single-speaker fn recordings feed in.
    """
    pool = defaultdict(lambda: {"sex": None, "sec": 0, "recs": []})
    for rid, sec, sid in recs:
        s = spk.get(sid)
        if not s or not is_nl_speaker(s):           # gate 3: NL speaker
            continue
        if require_first_sd and s["firstLang"] != SD:
            continue
        if require_home_sd and s["homeLang"] != SD:
            continue
        if s["sex"] not in ("sex1", "sex2"):
            continue
        pool[sid]["sex"] = s["sex"]
        pool[sid]["sec"] += sec
        pool[sid]["recs"].append(rid)
    return pool


def pool_hours(pool):
    return sum(v["sec"] for v in pool.values()) / 3600.0


def select_balanced(pool, target_sec, order):
    """
    Alternate sex1/sex2, taking whole speakers, until total >= target_sec.
    `order` controls within-sex speaker order: 'asc' (many small speakers -> max diversity),
    'desc' (few large speakers), 'id' (deterministic by speaker id).
    """
    keyfn = {
        "asc":  lambda sid: pool[sid]["sec"],
        "desc": lambda sid: -pool[sid]["sec"],
        "id":   lambda sid: sid,
    }[order]

    queues = {sx: sorted((sid for sid in pool if pool[sid]["sex"] == sx), key=keyfn)
              for sx in ("sex1", "sex2")}

    totals = {"sex1": 0, "sex2": 0}
    chosen = []
    while (totals["sex1"] + totals["sex2"]) < target_sec:
        # pick the currently-behind sex that still has speakers
        behind = sorted(totals, key=totals.get)      # ascending by accumulated seconds
        sx = next((s for s in behind if queues[s]), None)
        if sx is None:
            break                                    # both queues exhausted
        sid = queues[sx].pop(0)
        for rid in pool[sid]["recs"]:
            chosen.append(rid)
        totals[sx] += pool[sid]["sec"]
    return chosen, totals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speakers", required=True, help="path to speakers.txt")
    ap.add_argument("--recordings", required=True, help="path to recordings.txt")
    ap.add_argument("--hours", type=float, default=100.0, help="target hours (default 100)")
    ap.add_argument("--order", choices=("asc", "desc", "id"), default="asc",
                    help="within-sex speaker order: asc=max speaker diversity (default)")
    ap.add_argument("--out", default="selected_recordings.txt",
                    help="output file of selected recordingIDs")
    args = ap.parse_args()

    target_sec = args.hours * 3600.0

    spk = load_speakers(args.speakers)
    recs = load_recordings(args.recordings)          # fn + single-speaker only
    print(f"[gates 1-2] fn single-speaker recordings: {len(recs)}", file=sys.stderr)

    # gate 3 + progressive language tightening (step 5)
    pool = build_pool(recs, spk)
    print(f"[gate 3   ] NL speakers in pool: {len(pool)}  "
          f"({pool_hours(pool):.1f} h)", file=sys.stderr)

    if pool_hours(pool) > args.hours:
        tighter = build_pool(recs, spk, require_first_sd=True)
        print(f"[tighten  ] +firstLang==SD: {len(tighter)} speakers "
              f"({pool_hours(tighter):.1f} h)", file=sys.stderr)
        if pool_hours(tighter) < args.hours:
            print("  ! firstLang filter drops pool below target; applying anyway "
                  "(loosen if undesired)", file=sys.stderr)
        pool = tighter

        if pool_hours(pool) > args.hours:
            tighter2 = build_pool(recs, spk, require_first_sd=True, require_home_sd=True)
            print(f"[tighten  ] +homeLang==SD: {len(tighter2)} speakers "
                  f"({pool_hours(tighter2):.1f} h)", file=sys.stderr)
            if pool_hours(tighter2) < args.hours:
                print("  ! homeLang filter drops pool below target; applying anyway "
                      "(loosen if undesired)", file=sys.stderr)
            pool = tighter2

    if pool_hours(pool) < args.hours:
        print(f"! WARNING: eligible pool is only {pool_hours(pool):.1f} h "
              f"(< {args.hours} h target); selecting everything available.", file=sys.stderr)

    chosen, totals = select_balanced(pool, target_sec, args.order)
    chosen_spk = {sid for sid in pool if any(r in chosen for r in pool[sid]["recs"])}

    h1, h2 = totals["sex1"] / 3600.0, totals["sex2"] / 3600.0
    print("\n=== selection ===")
    print(f"recordings : {len(chosen)}")
    print(f"speakers   : {len(chosen_spk)}")
    print(f"hours      : {h1 + h2:.1f}  (sex1={h1:.1f} h, sex2={h2:.1f} h)")
    if h1 + h2:
        print(f"sex balance: sex1 {100*h1/(h1+h2):.0f}% / sex2 {100*h2/(h1+h2):.0f}%")

    with open(args.out, "w") as f:
        for rid in sorted(chosen):
            f.write(rid + "\n")
    print(f"wrote {len(chosen)} recordingIDs -> {args.out}")


if __name__ == "__main__":
    main()
