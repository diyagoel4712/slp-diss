#!/usr/bin/env python3
"""
Ingest IndicVoices-R from its HuggingFace release (per-language folders of parquet
shards with the audio embedded) into F5-TTS training clips + a metadata manifest.

The E2E tar mirror (wav + JSONL) is decommissioned; HF ships parquet instead, so this
replaces select_indicvoices.py + prep_indicvoices_f5.py in one pass for the parquet
layout. It reuses the SAME balanced selection as select_indicvoices.py (imported), so
behaviour matches the tar path.

Two passes over the shards:
  A. metadata only (audio column skipped -> fast, low memory): gate + balanced
     selection -> the ~target-hours set of clip ids.
  B. audio: decode ONLY the selected clips' embedded audio, downmix to mono, and
     write wavs/ + metadata.csv (NATIVE script text) + details.tsv.

Native sample rate is kept by default (IndicVoices-R is 48 kHz; F5's dataloader
resamples to 24 k at load time) -- pass --sr 24000 to pre-resample. Transcripts are
NATIVE script; run romanize.py on metadata.csv next (F5 base vocab is Latin+pinyin).

    python prep_from_parquet.py --parquet-dir /data/iv_r/Hindi --lang hi \
        --out /data/iv_hi/clips --hours 100
    python prep_from_parquet.py --parquet-dir /data/iv_r/Hindi --inspect   # schema peek

Downstream (unchanged): dnsmos_filter.py -> romanize.py -> vocab_check.py -> prepare.
"""

import argparse
import csv
import glob
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

# reuse the exact gating + selection from the sibling module
sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_indicvoices as sel   # noqa: E402

TEXT_CANDIDATES = ("normalized", "verbatim", "text")
AUDIO_NAME_HINTS = ("audio", "wav", "speech")


def synth_id(shard_path, idx):
    """The HF release has no filename/chunk_name column, so identify a clip by its
    (shard, row-position). Deterministic across passes as long as both iterate the
    same shards in the same order (parquet preserves row order)."""
    return f"{Path(shard_path).stem}_{idx:06d}"


def shard_paths(patterns):
    out = []
    for pat in patterns:
        p = Path(pat)
        if p.is_dir():
            out.extend(sorted(str(x) for x in p.rglob("*.parquet")))
        else:
            hits = sorted(glob.glob(pat))
            out.extend(hits if hits else [pat])
    return out


def detect_audio_col(schema):
    """HF Audio is stored as a struct with a 'bytes' (and/or 'array') child. Find it,
    falling back to any column named like audio."""
    import pyarrow as pa
    for f in schema:
        if pa.types.is_struct(f.type):
            children = {f.type.field(i).name for i in range(f.type.num_fields)}
            if children & {"bytes", "array"}:
                return f.name
    for f in schema:
        if any(h in f.name.lower() for h in AUDIO_NAME_HINTS):
            return f.name
    return None


def pick(schema_names, candidates):
    for c in candidates:
        if c in schema_names:
            return c
    return None


def decode_audio(a):
    """a is the parquet audio cell (dict). Return (float32 mono ndarray, sample_rate)."""
    if isinstance(a, dict) and a.get("array") is not None:
        arr = np.asarray(a["array"], dtype=np.float32)
        sr = int(a["sampling_rate"])
    else:
        raw = a["bytes"] if isinstance(a, dict) else a
        if raw is None and isinstance(a, dict) and a.get("path"):
            arr, sr = sf.read(a["path"], dtype="float32")
        else:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if arr.ndim > 1:                       # (n, ch) -> mono
        arr = arr.mean(axis=1)
    return arr.astype(np.float32), sr


def resample(arr, sr, out_sr):
    if out_sr == sr:
        return arr
    import torch
    import torchaudio
    t = torch.from_numpy(arr).unsqueeze(0)
    t = torchaudio.functional.resample(t, sr, out_sr)
    return t.squeeze(0).numpy()


def meta_rows(shards, audio_col):
    """Yield metadata dicts (audio column skipped) across all shards -- pass A. Injects
    a synthetic 'filename' (shard+position) so sel.clip_id/gating have a stable id."""
    for f in shards:
        cols = [n for n in pq.read_schema(f).names if n != audio_col]
        t = pq.read_table(f, columns=cols)
        idx = 0
        for batch in t.to_batches():
            for row in batch.to_pylist():
                row["filename"] = synth_id(f, idx)
                idx += 1
                yield row


def inspect(shards):
    f = shards[0]
    schema = pq.read_schema(f)
    names = list(schema.names)
    audio_col = detect_audio_col(schema)
    print(f"shard: {f}\n")
    print("schema:")
    for fld in schema:
        print(f"  {fld.name:<22} {fld.type}")
    print(f"\ndetected -> id=<synthesized: shard+row-position (no filename column)>  "
          f"text={pick(names, TEXT_CANDIDATES)}  audio={audio_col}")
    t = pq.read_table(f, columns=[n for n in names if n != audio_col])
    row0 = t.slice(0, 1).to_pylist()[0]
    print("\nrow0 (no audio):")
    for k, v in row0.items():
        v = (v[:80] + "…") if isinstance(v, str) and len(v) > 80 else v
        print(f"  {k:<22} {v!r}")
    if audio_col:
        a = pq.read_table(f, columns=[audio_col]).slice(0, 1).to_pylist()[0][audio_col]
        print(f"\n{audio_col} cell keys: {list(a.keys()) if isinstance(a, dict) else type(a)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", nargs="+", required=True,
                    help="language parquet folder(s) / glob(s) (e.g. .../Hindi)")
    ap.add_argument("--inspect", action="store_true",
                    help="print schema + detected columns + a sample row, then exit")
    ap.add_argument("--out", help="output dir for wavs/ + metadata.csv (required unless --inspect)")
    ap.add_argument("--lang", default="", help="ISO code to keep (hi, bn); blank = keep all")
    ap.add_argument("--hours", type=float, default=100.0)
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--min-snr", type=float, default=None)
    ap.add_argument("--min-c50", type=float, default=None)
    ap.add_argument("--order", choices=("asc", "desc", "id"), default="asc")
    ap.add_argument("--read-tokens", default=",".join(sel.DEFAULT_READ))
    ap.add_argument("--extempore-tokens", default=",".join(sel.DEFAULT_EXTEMPORE))
    ap.add_argument("--conv-tokens", default=",".join(sel.DEFAULT_CONV))
    ap.add_argument("--no-conv-fallback", action="store_true")
    ap.add_argument("--sr", type=int, default=0,
                    help="output sample rate; 0 (default) = keep native, let F5 resample")
    ap.add_argument("--text-field", default=None,
                    help="transcript column (default: first of normalized/verbatim/text)")
    args = ap.parse_args()

    shards = shard_paths(args.parquet_dir)
    if not shards:
        sys.exit("no .parquet shards found under --parquet-dir")
    print(f"{len(shards)} parquet shards", file=sys.stderr)

    if args.inspect:
        inspect(shards)
        return
    if not args.out:
        sys.exit("--out is required unless --inspect")

    schema = pq.read_schema(shards[0])
    names = list(schema.names)
    audio_col = detect_audio_col(schema)
    text_col = args.text_field or pick(names, TEXT_CANDIDATES)
    if not (audio_col and text_col):
        sys.exit(f"could not detect columns (text={text_col} audio={audio_col}); "
                 f"run --inspect and pass --text-field / check the audio column")
    print(f"columns -> id=<synthesized> text={text_col} audio={audio_col}", file=sys.stderr)

    # ---- pass A: metadata -> gate + balanced selection ----
    clips, stats = sel.gate_clips(meta_rows(shards, audio_col), args.lang,
                                  args.min_dur, args.max_dur, args.min_snr, args.min_c50)
    sel.print_scan_stats(clips, stats, has_qual=bool(args.min_snr or args.min_c50))
    chosen_ids, chosen_spk, totals = sel.run_selection(
        clips, args.hours, args.order,
        sel.toks(args.read_tokens), sel.toks(args.extempore_tokens),
        sel.toks(args.conv_tokens), args.no_conv_fallback)
    sel.print_selection_summary(chosen_ids, chosen_spk, totals)
    wanted = set(chosen_ids)

    # ---- pass B: decode + write ONLY the selected clips ----
    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    keep_cols = [c for c in dict.fromkeys(
        [text_col, audio_col, sel.F_SPK, sel.F_GENDER,
         sel.F_SCENARIO, sel.F_TASK, sel.F_DUR]) if c in names]

    meta_f = open(out / "metadata.csv", "w", newline="", encoding="utf-8")
    meta = csv.writer(meta_f, delimiter="|")
    meta.writerow(["audio_file", "text"])
    det = open(out / "details.tsv", "w", encoding="utf-8")
    det.write("clip\tspeaker\tgender\tscenario\tdur\ttext\n")

    n_clip, total_sec, n_bad, seen = 0, 0.0, 0, set()
    for f in shards:
        t = pq.read_table(f, columns=keep_cols)
        idx = -1                       # per-shard row position, matching meta_rows
        for batch in t.to_batches():
            for r in batch.to_pylist():
                idx += 1
                cid = synth_id(f, idx)
                if cid not in wanted or cid in seen:
                    continue
                seen.add(cid)
                text = " ".join((r.get(text_col) or "").split())
                if not text:
                    n_bad += 1
                    continue
                try:
                    arr, sr = decode_audio(r[audio_col])
                    out_sr = args.sr or sr
                    arr = resample(arr, sr, out_sr)
                except Exception as e:
                    print(f"! decode failed for {cid}: {e}", file=sys.stderr)
                    n_bad += 1
                    continue
                dur = len(arr) / out_sr
                if dur < args.min_dur:
                    n_bad += 1
                    continue
                name = f"{Path(cid).stem}.wav"
                sf.write(str(out / "wavs" / name), arr, out_sr, subtype="PCM_16")
                meta.writerow([f"wavs/{name}", text])
                scen = (r.get(sel.F_SCENARIO) or r.get(sel.F_TASK) or "").strip()
                det.write(f"wavs/{name}\t{r.get(sel.F_SPK,'')}\t{r.get(sel.F_GENDER,'')}\t"
                          f"{scen}\t{dur:.3f}\t{text}\n")
                n_clip += 1
                total_sec += dur
                if n_clip % 500 == 0:
                    print(f"  wrote {n_clip} clips ({total_sec/3600:.1f} h)", file=sys.stderr)

    meta_f.close()
    det.close()
    print(f"\ndone: {n_clip} clips, {total_sec/3600:.1f} h  "
          f"(skipped/bad {n_bad}, not-found {len(wanted) - len(seen)})", file=sys.stderr)
    print(f"  {out/'metadata.csv'}\n  {out/'wavs'}/", file=sys.stderr)


if __name__ == "__main__":
    main()
