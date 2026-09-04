#!/usr/bin/env python3
"""
Select a balanced ~N-hour single-language subset of IndicVoices-R for finetuning
one accent (e.g. Hindi, Bengali). One accent vector per accent, so this runs once
per language with --lang.

IndicVoices-R differs from CGN: audio is ALREADY segmented into per-utterance clips,
and every clip carries its own speaker/gender/duration/scenario metadata in a JSONL
manifest. So there is no TextGrid segmentation here -- selection works directly on
clips and speakers.

Pipeline (in order):
  1. language gate       : row lang == --lang  (hi / bn ...)  [skipped if manifest is
                           already single-language and rows omit lang]
  2. clip duration gate  : --min-dur <= duration <= --max-dur  (paper floor 3.0 s)
  3. scenario tiering    : Tier-1 = read + extempore. If the Tier-1 pool is below the
                           target, ALSO admit conversational (Tier-2) and report it.
                           (matches the "read+extempore, else add single-speaker
                           conversational" decision.)
  4. optional quality gate: --min-snr / --min-c50 pre-filter using the manifest's own
                           snr/c50 fields (cheap; DNSMOS still runs later). Off by default.
  5. balanced greedy select: alternate the behind gender, take whole speakers, sum
                           clip duration -> target hours (same algorithm as select_cgn).

Because IndicVoices scenario/gender field *values* aren't perfectly documented, the
gates match case-insensitively on substrings and the script prints the observed
scenario/gender distributions so you can adjust --read-tokens / --extempore-tokens /
--conv-tokens if a corpus uses different labels.

Manifest input is flexible: a .jsonl (one JSON object per line), a .json array, or a
directory of per-utterance .json files. Pass --manifest one or more times (or a glob).

Outputs: a summary to stdout and one clip id per line to --out (the manifest's
`filename`/`chunk_name`, whichever identifies the wav). prep_indicvoices_f5.py consumes
--out + the same manifest(s).
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

# --- field-name knobs (override if a manifest release renames things) ---
F_LANG = "lang"
F_DUR = "duration"
F_SPK = "speaker_id"
F_GENDER = "gender"
F_SCENARIO = "scenario"          # read-speech / extempore / conversation
F_TASK = "task_name"             # fallback if scenario is absent
F_SNR = "snr"
F_C50 = "c50"
# clip identity: prefer `filename`, fall back to `chunk_name`
F_ID = ("filename", "chunk_name")

DEFAULT_READ = ("read",)
DEFAULT_EXTEMPORE = ("extempore", "extempo")
DEFAULT_CONV = ("conversation", "conversational", "dialog")


def clip_id(row):
    for k in F_ID:
        v = row.get(k)
        if v:
            return str(v).strip()
    return None


def norm_gender(g):
    """Map the manifest gender to two buckets; unknown/other -> None (dropped)."""
    g = (g or "").strip().lower()
    if g.startswith("m"):
        return "male"
    if g.startswith("f"):
        return "female"
    return None


def scenario_of(row):
    return (row.get(F_SCENARIO) or row.get(F_TASK) or "").strip().lower()


def matches(tokens, text):
    return any(tok in text for tok in tokens)


# ---------- manifest loading (jsonl / json array / dir of json) ----------

def _iter_file(p):
    """Yield rows from one manifest file: .jsonl (line-delimited, streamed),
    .json array, or a single .json object (per-utterance style). Extension-less
    files are sniffed, falling back to line mode."""
    p = Path(p)
    if p.suffix.lower() == ".jsonl":
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    with open(p, encoding="utf-8") as f:
        if f.read(2048).lstrip().startswith("["):
            f.seek(0)
            yield from json.load(f)
            return
        f.seek(0)
        try:
            obj = json.load(f)
        except json.JSONDecodeError:              # extension-less jsonl
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
            return
    yield from (obj if isinstance(obj, list) else [obj])


def iter_manifest(paths):
    for p in paths:
        p = Path(p)
        if p.is_dir():                            # pick up both .json and .jsonl
            for jf in sorted(list(p.rglob("*.jsonl")) + list(p.rglob("*.json"))):
                yield from _iter_file(jf)
        else:
            yield from _iter_file(p)


def expand(patterns):
    out = []
    for pat in patterns:
        hits = glob.glob(pat)
        out.extend(hits if hits else [pat])
    return out


# ---------- clip loading + gating ----------

def gate_clips(rows, lang, min_dur, max_dur, min_snr, min_c50):
    """Return (clips, stats) from an iterable of row dicts (JSONL- or parquet-derived).
    clips: list of dicts with id/spk/gender/dur/scenario. Applies the language +
    duration + optional quality gates; scenario tiering is applied later in build_pool
    so both tiers share one scan."""
    clips = []
    stats = {"rows": 0, "no_id": 0, "wrong_lang": 0, "bad_dur": 0,
             "no_gender": 0, "low_qual": 0, "scenario": defaultdict(int),
             "gender": defaultdict(int)}
    for row in rows:
        stats["rows"] += 1
        if lang and (row.get(F_LANG) or "").strip().lower() not in ("", lang.lower()):
            stats["wrong_lang"] += 1
            continue
        cid = clip_id(row)
        if not cid:
            stats["no_id"] += 1
            continue
        try:
            dur = float(row.get(F_DUR))
        except (TypeError, ValueError):
            stats["bad_dur"] += 1
            continue
        if not (min_dur <= dur <= max_dur):
            stats["bad_dur"] += 1
            continue
        g = norm_gender(row.get(F_GENDER))
        if g is None:
            stats["no_gender"] += 1
            continue
        if min_snr is not None:
            try:
                if float(row.get(F_SNR)) < min_snr:
                    stats["low_qual"] += 1
                    continue
            except (TypeError, ValueError):
                pass                              # missing snr -> don't drop on it
        if min_c50 is not None:
            try:
                if float(row.get(F_C50)) < min_c50:
                    stats["low_qual"] += 1
                    continue
            except (TypeError, ValueError):
                pass
        scen = scenario_of(row)
        stats["scenario"][scen or "<empty>"] += 1
        stats["gender"][g] += 1
        clips.append({"id": cid, "spk": (row.get(F_SPK) or "").strip(),
                      "gender": g, "dur": dur, "scenario": scen})
    return clips, stats


def load_clips(manifests, lang, min_dur, max_dur, min_snr, min_c50):
    """gate_clips over rows read from JSONL/JSON/dir manifests."""
    return gate_clips(iter_manifest(manifests), lang, min_dur, max_dur, min_snr, min_c50)


def build_pool(clips, read_tok, ext_tok, conv_tok, include_conv):
    """speaker_id -> {gender, sec, ids:[clip ids]} over the allowed scenarios.
    Tier-1 = read + extempore; +conversational when include_conv."""
    allowed_read_ext = lambda s: matches(read_tok, s) or matches(ext_tok, s)
    pool = defaultdict(lambda: {"gender": None, "sec": 0.0, "ids": []})
    for c in clips:
        s = c["scenario"]
        ok = allowed_read_ext(s) or (include_conv and matches(conv_tok, s))
        if not ok:
            continue
        sid = c["spk"] or f"__anon__{c['id']}"    # keep going even if speaker id missing
        pool[sid]["gender"] = c["gender"]
        pool[sid]["sec"] += c["dur"]
        pool[sid]["ids"].append(c["id"])
    return pool


def pool_hours(pool):
    return sum(v["sec"] for v in pool.values()) / 3600.0


def select_balanced(pool, target_sec, order):
    """Alternate the behind gender, taking whole speakers, until total >= target_sec.
    order controls within-gender speaker order: 'asc' (many small speakers -> max
    diversity, default), 'desc' (few large), 'id' (deterministic)."""
    keyfn = {
        "asc":  lambda sid: pool[sid]["sec"],
        "desc": lambda sid: -pool[sid]["sec"],
        "id":   lambda sid: sid,
    }[order]
    queues = {gx: sorted((sid for sid in pool if pool[sid]["gender"] == gx), key=keyfn)
              for gx in ("male", "female")}
    totals = {"male": 0.0, "female": 0.0}
    chosen_ids, chosen_spk = [], []
    while (totals["male"] + totals["female"]) < target_sec:
        behind = sorted(totals, key=totals.get)
        gx = next((g for g in behind if queues[g]), None)
        if gx is None:
            break
        sid = queues[gx].pop(0)
        chosen_ids.extend(pool[sid]["ids"])
        chosen_spk.append(sid)
        totals[gx] += pool[sid]["sec"]
    return chosen_ids, chosen_spk, totals


# ---------- shared driver (reused by the parquet ingest) ----------

def print_scan_stats(clips, stats, has_qual):
    print(f"[scan] {stats['rows']} rows -> {len(clips)} clips pass lang/dur/gender"
          f"{'/quality' if has_qual else ''} gates", file=sys.stderr)
    print(f"       dropped: wrong_lang={stats['wrong_lang']} bad_dur={stats['bad_dur']} "
          f"no_gender={stats['no_gender']} no_id={stats['no_id']} low_qual={stats['low_qual']}",
          file=sys.stderr)
    print("       scenario distribution (post-gate, by clip count):", file=sys.stderr)
    for s, n in sorted(stats["scenario"].items(), key=lambda x: -x[1]):
        print(f"         {s:<24} {n}", file=sys.stderr)
    print(f"       gender: {dict(stats['gender'])}", file=sys.stderr)


def run_selection(clips, hours, order, read_tok, ext_tok, conv_tok, no_conv_fallback):
    """Scenario tiering (read+extempore, then +conversational if short) + balanced
    greedy select. Returns (chosen_ids, chosen_spk, totals). Prints tier/warn lines."""
    pool = build_pool(clips, read_tok, ext_tok, conv_tok, include_conv=False)
    print(f"[tier1] read+extempore pool: {len(pool)} speakers ({pool_hours(pool):.1f} h)",
          file=sys.stderr)
    if pool_hours(pool) < hours and not no_conv_fallback:
        pool = build_pool(clips, read_tok, ext_tok, conv_tok, include_conv=True)
        print(f"[tier2] +conversational:    {len(pool)} speakers ({pool_hours(pool):.1f} h)"
              "  (read+extempore alone was below target)", file=sys.stderr)
    if pool_hours(pool) < hours:
        print(f"! WARNING: eligible pool is only {pool_hours(pool):.1f} h "
              f"(< {hours} h target); selecting everything available.", file=sys.stderr)
    return select_balanced(pool, hours * 3600.0, order)


def print_selection_summary(chosen_ids, chosen_spk, totals):
    hm, hf = totals["male"] / 3600.0, totals["female"] / 3600.0
    print("\n=== selection ===")
    print(f"clips    : {len(chosen_ids)}")
    print(f"speakers : {len(chosen_spk)}")
    print(f"hours    : {hm + hf:.1f}  (male={hm:.1f} h, female={hf:.1f} h)")
    if hm + hf:
        print(f"gender balance: male {100*hm/(hm+hf):.0f}% / female {100*hf/(hm+hf):.0f}%")


def toks(s):
    return tuple(t for t in s.split(",") if t)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, nargs="+",
                    help="IndicVoices-R manifest(s): .jsonl / .json / dir / glob")
    ap.add_argument("--lang", default="",
                    help="ISO code to keep (e.g. hi, bn). Blank = trust manifest is single-language")
    ap.add_argument("--hours", type=float, default=100.0, help="target hours (default 100)")
    ap.add_argument("--min-dur", type=float, default=3.0,
                    help="drop clips shorter than this (s); paper floor 3.0")
    ap.add_argument("--max-dur", type=float, default=30.0,
                    help="drop clips longer than this (s)")
    ap.add_argument("--min-snr", type=float, default=None,
                    help="optional manifest-snr pre-filter (dB); off by default (DNSMOS runs later)")
    ap.add_argument("--min-c50", type=float, default=None,
                    help="optional manifest-c50 pre-filter; off by default")
    ap.add_argument("--order", choices=("asc", "desc", "id"), default="asc",
                    help="within-gender speaker order: asc=max speaker diversity (default)")
    ap.add_argument("--read-tokens", default=",".join(DEFAULT_READ))
    ap.add_argument("--extempore-tokens", default=",".join(DEFAULT_EXTEMPORE))
    ap.add_argument("--conv-tokens", default=",".join(DEFAULT_CONV))
    ap.add_argument("--no-conv-fallback", action="store_true",
                    help="do NOT admit conversational even if read+extempore is below target")
    ap.add_argument("--out", default="selected_clips.txt",
                    help="output file of selected clip ids (filename/chunk_name)")
    args = ap.parse_args()

    manifests = expand(args.manifest)
    clips, stats = load_clips(manifests, args.lang, args.min_dur, args.max_dur,
                              args.min_snr, args.min_c50)
    print_scan_stats(clips, stats, has_qual=bool(args.min_snr or args.min_c50))

    chosen_ids, chosen_spk, totals = run_selection(
        clips, args.hours, args.order,
        toks(args.read_tokens), toks(args.extempore_tokens), toks(args.conv_tokens),
        args.no_conv_fallback)
    print_selection_summary(chosen_ids, chosen_spk, totals)

    with open(args.out, "w", encoding="utf-8") as f:
        for cid in sorted(set(chosen_ids)):
            f.write(cid + "\n")
    print(f"wrote {len(set(chosen_ids))} clip ids -> {args.out}")


if __name__ == "__main__":
    main()
