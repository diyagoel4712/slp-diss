"""[E5.1-5.2] RQ5 -- evaluation-bias audit.

Two runnable checks (the "sources of harm" flowchart is a written deliverable):

  relative WER : WER(synth accent) - WER(natural, same accent). Subtracting the
                 ASR's inherent error on real accented speech isolates
                 accent-induced error from the ASR's own accent bias -- the
                 "error margins, not absolute values" idea.
  gender split : every metric disaggregated by speaker gender, to test whether
                 quality/accent scores differ for female vs male speakers.

    python -m accent_vector.experiments.rq5_bias \
        --synth-dir results/british/alpha_1.0 --transcripts transcripts/eval_transcripts.txt \
        --natural-dir /data/vctk_england_clips --natural-meta england.csv \
        --vctk-root /data/VCTK-Corpus-0.92 --ref-wav refs/neutral.wav \
        --out-csv results/british/rq5.csv

``--natural-meta`` is an audio_file|text CSV giving transcripts for the natural
clips (so their WER is defined).
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from accent_vector.experiments import common


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _read_meta(path):
    out = {}
    if not path:
        return out
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f, delimiter="|")
        next(r, None)
        for row in r:
            if len(row) >= 2:
                out[Path(row[0]).stem] = row[1].strip()
    return out


def run(synth_dir, transcripts, natural_dir, natural_meta, vctk_root, ref_wav, device, out_csv):
    ef = common.load_eval()
    tx = [ln.strip() for ln in Path(transcripts).read_text().splitlines() if ln.strip()] if transcripts else []
    synth = common.wavs_in(synth_dir)

    # --- synth WER ---
    synth_wer = _mean([ef.wer(w, tx[common.utt_index(w)]) for w in synth
                       if common.utt_index(w) is not None and common.utt_index(w) < len(tx)])

    # --- natural WER (for the relative-WER correction) ---
    nat_wer = float("nan")
    if natural_dir and natural_meta:
        meta = _read_meta(natural_meta)
        nat = common.wavs_in(natural_dir)
        nat_wer = _mean([ef.wer(w, meta[Path(w).stem]) for w in nat if Path(w).stem in meta])

    rows = [{
        "metric": "wer_absolute", "value": synth_wer,
        "note": "raw WER on synthesized accent",
    }, {
        "metric": "wer_natural_accent", "value": nat_wer,
        "note": "WER on real speech of the same accent (ASR's own bias)",
    }, {
        "metric": "wer_relative", "value": synth_wer - nat_wer if not np.isnan(nat_wer) else float("nan"),
        "note": "synth minus natural: accent-induced error above the ASR baseline",
    }]

    # --- gender-disaggregated metrics ---
    genders = common.vctk_gender_map(vctk_root) if vctk_root else {}
    if genders:
        by_g = defaultdict(list)
        for w in synth:
            g = genders.get(common.speaker_from_wav(w))
            if g:
                by_g[g].append(w)
        for g, wavs in sorted(by_g.items()):
            if ref_wav:
                rows.append({"metric": f"spk_sim[{g}]",
                             "value": ef.speaker_similarity(wavs, [ref_wav] * len(wavs), device=device),
                             "note": f"{len(wavs)} clips"})
            gw = _mean([ef.wer(w, tx[common.utt_index(w)]) for w in wavs
                        if common.utt_index(w) is not None and common.utt_index(w) < len(tx)])
            rows.append({"metric": f"wer[{g}]", "value": gw, "note": f"{len(wavs)} clips"})
    else:
        rows.append({"metric": "gender_split", "value": float("nan"),
                     "note": "no --vctk-root gender map; skipped"})

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value", "note"])
        w.writeheader()
        for r in rows:
            r["value"] = f"{r['value']:.4f}" if isinstance(r["value"], float) else r["value"]
            w.writerow(r)
    print(f"[rq5] wrote {out_csv}")
    for r in rows:
        print(f"    {r['metric']}: {r['value']}  ({r['note']})")


def main():
    p = argparse.ArgumentParser(description="RQ5 evaluation-bias checks")
    p.add_argument("--synth-dir", required=True)
    p.add_argument("--transcripts", required=True)
    p.add_argument("--natural-dir")
    p.add_argument("--natural-meta", help="audio_file|text CSV for natural clips")
    p.add_argument("--vctk-root", help="for the speaker->gender map")
    p.add_argument("--ref-wav")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-csv", required=True)
    a = p.parse_args()
    run(a.synth_dir, a.transcripts, a.natural_dir, a.natural_meta, a.vctk_root,
        a.ref_wav, a.device, a.out_csv)


if __name__ == "__main__":
    main()
