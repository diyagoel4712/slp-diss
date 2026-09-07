#!/usr/bin/env python3
"""Select Speech Accent Archive speakers per accent + gender, filtered on biographical
metadata, and (optionally) fetch their "Please call Stella" recordings.

Why: the eval's natural targets are currently ONE speaker per gender per accent, so every
rq3 distance is "distance to this particular person", not "distance to this accent". This
picks a POOL of speakers so the target becomes an accent centroid and the between-speaker
SD becomes measurable (which is also the correct scale for the prosodic distance).

Site structure (verified 2026-08-21, the Hugo rebuild of accent.gmu.edu):
    index    https://accent.gmu.edu/language/<language>/     -> links to /samples/<id>/
    speaker  https://accent.gmu.edu/samples/<id>/            -> <h3 id="biographical-data"> + table
    audio    https://accent.gmu.edu/audio/<id>.mp3
The archive is CC-BY-NC-SA 4.0. Bulk audio also lives on OSF (osf.io/e2j5y mp3,
osf.io/yh23d flac) -- prefer OSF if you ever want the whole corpus rather than a selection.

Every page is cached under --cache-dir, so re-runs cost no requests. First full run over
five languages is ~470 pages: at the default 1.0 s delay that is roughly 8-10 minutes.

    # what is actually available (no audio downloaded)
    python saa_select.py --languages dutch mandarin hindi arabic bengali

    # commit to a selection and pull the mp3s into the eval's layout
    # (--audio-root defaults to AccentVector/data/ground_truth_refs, resolved from
    #  this file, so it lands in the right place from any working directory)
    python saa_select.py --languages dutch mandarin hindi arabic bengali \
        --per-gender 20 --max-residence 5 --download-audio
"""
import argparse
import csv
import random
import re
import sys
import time
import urllib.error
import urllib.request
from html import unescape
from pathlib import Path

BASE = "https://accent.gmu.edu"
UA = "saa_select/1.0 (MSc dissertation; contact via github)"
FIELDS = ["birth place", "native language", "other language(s)", "age, sex",
          "age of english onset", "english learning method", "english residence",
          "length of english residence"]


# --- fetching -------------------------------------------------------------------
def fetch(url, cache_dir, sleep, retries=3):
    """GET with an on-disk cache. The cache is the politeness mechanism: re-runs and
    parameter sweeps never touch the server again."""
    key = re.sub(r"[^A-Za-z0-9]+", "_", url.replace(BASE, "")).strip("_") or "index"
    path = Path(cache_dir) / f"{key}.html"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read().decode("utf-8", errors="replace")
            path.write_text(body, encoding="utf-8")
            time.sleep(sleep)
            return body
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last = e
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                return ""                      # a real absence, not a transient failure
            time.sleep(sleep * (2 ** attempt))  # back off before retrying
    print(f"  ! giving up on {url}: {last}", file=sys.stderr)
    return ""


def sample_ids(language, cache_dir, sleep):
    """Every sample id filed under one language, e.g. ['mandarin1', 'mandarin30', ...]."""
    html = fetch(f"{BASE}/language/{language}/", cache_dir, sleep)
    ids = re.findall(rf'href="{BASE}/samples/({re.escape(language)}\d+)/"', html)
    return sorted(set(ids), key=lambda s: int(s[len(language):]))


# --- parsing --------------------------------------------------------------------
def strip_tags(s):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def parse_bio(html):
    """The biographical table -> {field: value}. Returns {} if the page has no table."""
    m = re.search(r'<h3[^>]*id="biographical-data".*?<table.*?</table>', html, re.S)
    if not m:
        return {}
    out = {}
    for k, v in re.findall(r"<td>(.*?)</td>\s*<td>(.*?)</td>", m.group(0), re.S):
        out[strip_tags(k).lower()] = strip_tags(v)
    return out


def parse_age_sex(value):
    """'38, male' -> (38, 'male'). Tolerates a missing half."""
    age, sex = None, None
    if not value:
        return age, sex
    parts = [p.strip().lower() for p in value.split(",")]
    for p in parts:
        if re.fullmatch(r"\d+(\.\d+)?", p):
            age = float(p)
        elif p in ("male", "female"):
            sex = p
    return age, sex


def parse_years(value):
    """'0.1 years' -> 0.1, '2 years' -> 2.0, '' -> None. Missing is NOT treated as 0:
    an unknown residence must not silently pass a '< 5 years' filter."""
    if not value:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", value)
    return float(m.group(1)) if m else None


def scrape_speaker(sid, cache_dir, sleep):
    html = fetch(f"{BASE}/samples/{sid}/", cache_dir, sleep)
    if not html:
        return None
    bio = parse_bio(html)
    if not bio:
        return None
    age, sex = parse_age_sex(bio.get("age, sex", ""))
    return {
        "sample_id": sid,
        "language": re.match(r"[a-z_]+", sid).group(0),
        "speaker_num": int(re.search(r"\d+$", sid).group(0)),
        "sex": sex,
        "age": age,
        "birth_place": bio.get("birth place", ""),
        "native_language": bio.get("native language", ""),
        "other_languages": bio.get("other language(s)", ""),
        "age_of_english_onset": parse_years(bio.get("age of english onset", "")),
        "english_learning_method": bio.get("english learning method", ""),
        "english_residence": bio.get("english residence", ""),
        "length_english_residence_years": parse_years(bio.get("length of english residence", "")),
        "has_ipa": "ipa-transcription" in html,
        "page_url": f"{BASE}/samples/{sid}/",
        "audio_url": f"{BASE}/audio/{sid}.mp3",
    }


# --- selection ------------------------------------------------------------------
# Tiers, best first. The point of relaxing is NOT to abandon the criterion but to turn
# it from a cutoff into a ranking: you still get the shortest-residence speakers first,
# you just keep going past the threshold until the cell is full.
TIERS = ["primary", "relaxed_optional", "relaxed_residence", "unknown_residence"]


def classify(rec, args):
    """(tier_index, note) for one speaker, or None if it cannot fill a gender cell."""
    if rec is None or rec["sex"] not in ("male", "female"):
        return None, "no sex recorded"
    yrs = rec["length_english_residence_years"]
    onset = rec["age_of_english_onset"]
    opt_fail = ((args.min_onset is not None and (onset is None or onset < args.min_onset))
                or (args.require_ipa and not rec["has_ipa"]))
    if yrs is None:
        return 3, "residence not recorded"
    if yrs < args.max_residence:
        if opt_fail:
            return 1, f"residence {yrs:g}y, fails optional filter"
        return 0, f"residence {yrs:g}y"
    return 2, f"residence {yrs:g}y (over {args.max_residence:g}y threshold)"


def order_key(tier, rec):
    """Within primary, a seeded shuffle keeps the pool unbiased. Everywhere else, rank by
    residence ascending so the least-resident (most accented) candidates come first."""
    yrs = rec["length_english_residence_years"]
    return (rec["_shuffle"] if tier == 0 else
            (yrs if yrs is not None else float("inf"), rec["speaker_num"]))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--languages", nargs="+", required=True,
                   help="SAA language labels, e.g. dutch mandarin hindi arabic bengali")
    p.add_argument("--per-gender", type=int, default=20)
    p.add_argument("--max-residence", type=float, default=5.0,
                   help="keep speakers with length of english residence STRICTLY below this")
    p.add_argument("--min-onset", type=float, default=None,
                   help="optional: minimum age of english onset (tightens accentedness spread)")
    p.add_argument("--strict", action="store_true",
                   help="hard filter instead of the default fill-to-quota ranking: drop every "
                        "speaker that fails, even if the cell ends up short")
    p.add_argument("--listen-html", default=None,
                   help="write a self-contained audition page: one row per speaker with an "
                        "audio player, its metadata and why it was selected")
    p.add_argument("--require-ipa", action="store_true",
                   help="optional: keep only speakers with a published IPA transcription")
    p.add_argument("--seed", type=int, default=0,
                   help="selection is a seeded shuffle, so it is reproducible but not biased "
                        "toward low sample numbers (which track submission era)")
    p.add_argument("--cache-dir", default=".saa_cache")
    p.add_argument("--sleep", type=float, default=1.0, help="delay between live requests (s)")
    p.add_argument("--out-csv", default="saa_selection.csv")
    p.add_argument("--rejects-csv", default=None,
                   help="optional: write every scraped speaker with its keep/reject reason")
    p.add_argument("--download-audio", action="store_true")
    # Resolved from this file, not the cwd: the eval reads GT from exactly this
    # path, and a bare relative default silently created a stray dir wherever
    # the script happened to be run from.
    p.add_argument("--audio-root",
                   default=str(Path(__file__).resolve().parents[3]
                               / "AccentVector" / "data" / "ground_truth_refs"),
                   help="mp3s land in <audio-root>/<language>/<male|female>/<sample_id>.mp3")
    args = p.parse_args()

    chosen, everything = [], []
    for lang in args.languages:
        ids = sample_ids(lang, args.cache_dir, args.sleep)
        print(f"\n[{lang}] {len(ids)} speakers listed; fetching metadata...")
        pool = {"male": [], "female": []}
        for i, sid in enumerate(ids, 1):
            rec = scrape_speaker(sid, args.cache_dir, args.sleep)
            if rec is None:
                everything.append({"sample_id": sid, "tier": "unparseable", "note": ""})
                continue
            tier, note = classify(rec, args)
            everything.append({**rec, "tier": TIERS[tier] if tier is not None else "excluded",
                               "note": note})
            if tier is not None and not (args.strict and tier != 0):
                rec["_tier"], rec["_note"] = tier, note
                pool[rec["sex"]].append(rec)
            if i % 50 == 0:
                print(f"    ...{i}/{len(ids)}")
        for sex in ("male", "female"):
            cands = pool[sex]
            rng = random.Random(f"{args.seed}:{lang}:{sex}")
            for r in cands:
                r["_shuffle"] = rng.random()
            cands.sort(key=lambda r: (r["_tier"], order_key(r["_tier"], r)))
            take = cands[: args.per_gender]
            for r in take:
                r["tier"], r["selection_note"] = TIERS[r["_tier"]], r["_note"]
                for k in ("_tier", "_note", "_shuffle"):
                    r.pop(k, None)
            chosen.extend(take)
            breakdown = ", ".join(f"{TIERS[t]}={sum(1 for r in take if r['tier'] == TIERS[t])}"
                                  for t in range(len(TIERS))
                                  if any(r["tier"] == TIERS[t] for r in take))
            short = "" if len(take) >= args.per_gender else \
                    f"   *** POOL EXHAUSTED: only {len(take)} of {args.per_gender} ***"
            print(f"  {sex:>6}: {len(cands):>3} available -> taking {len(take)}"
                  f"  [{breakdown}]{short}")

    if not chosen:
        sys.exit("nothing selected -- loosen --max-residence or check the language labels")

    cols = list(chosen[0].keys())
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(chosen)
    print(f"\n[saa] wrote {args.out_csv}  ({len(chosen)} speakers)")

    if args.rejects_csv:
        keys = sorted({k for r in everything for k in r if not k.startswith("_")})
        everything = [{k: v for k, v in r.items() if not k.startswith("_")} for r in everything]
        with open(args.rejects_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(everything)
        print(f"[saa] wrote {args.rejects_csv}  ({len(everything)} scraped, with reasons)")

    if args.listen_html:
        rows = []
        for r in chosen:
            meta = (f"{r['sex']}, age {r['age']:.0f}" if r["age"] else r["sex"] or "?")
            rows.append(
                f'<tr><td class="id">{r["sample_id"]}</td>'
                f'<td>{r["language"]}</td><td>{meta}</td>'
                f'<td>{r["birth_place"]}</td>'
                f'<td class="num">{r["length_english_residence_years"]}</td>'
                f'<td class="num">{r["age_of_english_onset"]}</td>'
                f'<td class="{r["tier"]}">{r["tier"]}</td>'
                f'<td><audio controls preload="none" src="{r["audio_url"]}"></audio></td>'
                f'<td><input type="checkbox" data-id="{r["sample_id"]}"></td></tr>')
        Path(args.listen_html).write_text(
            "<!doctype html><meta charset=utf-8><title>SAA audition</title>"
            "<style>body{font:14px/1.5 system-ui;margin:24px;max-width:1200px}"
            "table{border-collapse:collapse;width:100%}"
            "th,td{padding:6px 9px;border-bottom:1px solid #ddd;text-align:left}"
            "th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#555}"
            ".id{font-family:ui-monospace,monospace;font-weight:600}"
            ".num{font-variant-numeric:tabular-nums;text-align:right}"
            ".relaxed_residence{color:#b45309}.unknown_residence{color:#b91c1c}"
            ".primary{color:#047857}audio{height:32px}"
            "#out{position:sticky;bottom:0;background:#fff;border-top:2px solid #333;"
            "padding:10px;font-family:ui-monospace,monospace;font-size:12px}</style>"
            "<h1>SAA audition</h1><p>Tick the speakers to keep; the list below updates. "
            "Audio streams from accent.gmu.edu.</p>"
            "<table><thead><tr><th>sample id</th><th>language</th><th>age, sex</th>"
            "<th>birth place</th><th>residence (y)</th><th>onset</th><th>tier</th>"
            "<th>audio</th><th>keep</th></tr></thead><tbody>"
            + "".join(rows) +
            "</tbody></table><div id=out>keep: (none)</div>"
            "<script>const o=document.getElementById('out');"
            "document.addEventListener('change',()=>{const k=[...document.querySelectorAll("
            "'input:checked')].map(i=>i.dataset.id);"
            "o.textContent='keep ('+k.length+'): '+(k.join(' ')||'(none)');});</script>",
            encoding="utf-8")
        print(f"[saa] wrote {args.listen_html}  (open in a browser to audition)")

    if args.download_audio:
        root = Path(args.audio_root)
        print(f"\n[saa] downloading {len(chosen)} mp3s under {root}/ ...")
        for rec in chosen:
            dst = root / rec["language"] / rec["sex"] / f"{rec['sample_id']}.mp3"
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                req = urllib.request.Request(rec["audio_url"], headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=60) as r:
                    dst.write_bytes(r.read())
                time.sleep(args.sleep)
            except Exception as e:
                print(f"  ! {rec['sample_id']}: {e}", file=sys.stderr)
        print("[saa] audio done")
        print("\nnext:  python mp3_to_wav.py <audio-root>/<accent> -r"
              "\n       # per speaker: align to the known sentences, then cut at those times"
              "\n       B=$(conda run -n accentvector-eval python align_stella_ta.py --in <wav>)"
              "\n       python split_by_silence.py --in <wav> --out-dir <accent>/<gender> "
              "--prefix <sample_id> --at \"$B\"")


if __name__ == "__main__":
    main()
