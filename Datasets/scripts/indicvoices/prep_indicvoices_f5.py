#!/usr/bin/env python3
"""
Turn a selected IndicVoices-R subset into F5-TTS training clips + a metadata manifest.

IndicVoices-R clips are ALREADY per-utterance segments, so (unlike CGN) there is no
TextGrid segmentation: this stage just downmixes the selected clips to mono and writes
the "audio_file|text" manifest that this fork's
``accent_vector.data_preprocess prepare`` consumes.

By default we write clips at their NATIVE rate (IndicVoices-R is 48 kHz) and do NOT
resample: F5's dataloader resamples every clip to 24 kHz at load time
(model/dataset.py, `source_sample_rate != target_sample_rate`), so pre-resampling here
would only duplicate work F5 already does -- the same reason CGN wrote native 16 kHz.
Pass --sr to pre-resample anyway (e.g. --sr 24000 to roughly halve on-disk size and
make the dataloader's resample a no-op); it is a lossy-but-harmless downsample.

Transcripts are emitted in the NATIVE script (Devanagari / Bengali). Run romanize.py
on the resulting metadata.csv before `prepare` -- F5's base vocab is Latin+pinyin only,
so native script must be romanised (or the vocab extended) first (see romanize.py).

Input:
    --manifest   the same IndicVoices-R manifest(s) fed to select_indicvoices.py
    --selected   selected_clips.txt from select_indicvoices.py (one clip id per line)
    --audio-root directory the manifest's filename/chunk_name paths are relative to

Output (under --out):
    wavs/<clip>.wav      mono, --sr Hz, PCM16
    metadata.csv         header "audio_file|text"; rows wavs/<clip>.wav|<native text>
    details.tsv          clip, speaker, gender, scenario, dur, text  (bookkeeping)
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import torchaudio

# keep field names aligned with select_indicvoices.py
F_ID = ("filename", "chunk_name")
F_SPK, F_GENDER, F_SCEN, F_TASK, F_DUR = "speaker_id", "gender", "scenario", "task_name", "duration"


def clip_id(row):
    for k in F_ID:
        v = row.get(k)
        if v:
            return str(v).strip()
    return None


def _iter_file(p):
    """Yield rows from one manifest file: .jsonl (streamed), .json array, or a single
    .json object; extension-less files are sniffed with a jsonl fallback."""
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
        except json.JSONDecodeError:
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
        if p.is_dir():
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


def resolve_wav(cid, audio_root, basename_index):
    """cid may be a path relative to audio_root, or just a basename. Try the direct
    join first, then a basename lookup built by walking audio_root once."""
    direct = audio_root / cid
    if direct.exists():
        return direct
    return basename_index.get(Path(cid).name)


def build_basename_index(audio_root):
    idx = {}
    for w in audio_root.rglob("*.wav"):
        idx.setdefault(w.name, w)
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, nargs="+",
                    help="IndicVoices-R manifest(s) (same as select step)")
    ap.add_argument("--selected", required=True, help="selected_clips.txt from select step")
    ap.add_argument("--audio-root", required=True, help="root the clip paths resolve under")
    ap.add_argument("--out", required=True, help="output dir for wavs/ + metadata.csv")
    ap.add_argument("--sr", type=int, default=0,
                    help="output sample rate; 0 (default) = keep native rate and let F5's "
                         "dataloader resample to 24k. Set e.g. 24000 to pre-resample.")
    ap.add_argument("--text-field", choices=("normalized", "verbatim"), default="normalized",
                    help="which transcript to use (default normalized)")
    ap.add_argument("--min-dur", type=float, default=3.0,
                    help="re-check floor after load (s); paper Section 4.2 uses 3.0")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    audio_root = Path(args.audio_root)

    with open(args.selected, encoding="utf-8") as f:
        wanted = {ln.strip() for ln in f if ln.strip()}
    print(f"{len(wanted)} selected clip ids", file=sys.stderr)

    print("indexing audio_root by basename (one walk)...", file=sys.stderr)
    basename_index = build_basename_index(audio_root)

    meta_f = open(out / "metadata.csv", "w", newline="", encoding="utf-8")
    meta = csv.writer(meta_f, delimiter="|")
    meta.writerow(["audio_file", "text"])
    det = open(out / "details.tsv", "w", encoding="utf-8")
    det.write("clip\tspeaker\tgender\tscenario\tdur\ttext\n")

    n_clip = 0
    total_sec = 0.0
    seen = set()
    n_missing_wav = n_empty_text = n_too_short = 0
    resampler_cache = {}

    for row in iter_manifest(expand(args.manifest)):
        cid = clip_id(row)
        if cid not in wanted or cid in seen:
            continue
        seen.add(cid)
        text = (row.get(args.text_field) or row.get("text") or "").strip()
        text = " ".join(text.split())
        if not text:
            n_empty_text += 1
            continue
        wav_path = resolve_wav(cid, audio_root, basename_index)
        if wav_path is None:
            n_missing_wav += 1
            print(f"! missing wav for {cid}", file=sys.stderr)
            continue

        wav, sr = torchaudio.load(str(wav_path))
        if wav.shape[0] > 1:                              # force mono
            wav = wav.mean(0, keepdim=True)
        out_sr = args.sr or sr                            # 0 => keep native rate
        if sr != out_sr:
            rs = resampler_cache.get(sr)
            if rs is None:
                rs = torchaudio.transforms.Resample(sr, out_sr)
                resampler_cache[sr] = rs
            wav = rs(wav)
        dur = wav.shape[1] / out_sr
        if dur < args.min_dur:
            n_too_short += 1
            continue

        name = f"{Path(cid).stem}.wav"
        torchaudio.save(str(out / "wavs" / name), wav, out_sr,
                        encoding="PCM_S", bits_per_sample=16)
        meta.writerow([f"wavs/{name}", text])
        spk = (row.get(F_SPK) or "").strip()
        scen = (row.get(F_SCEN) or row.get(F_TASK) or "").strip()
        det.write(f"wavs/{name}\t{spk}\t{row.get(F_GENDER,'')}\t{scen}\t{dur:.3f}\t{text}\n")
        n_clip += 1
        total_sec += dur
        if n_clip % 500 == 0:
            print(f"  {n_clip} clips ({total_sec/3600:.1f} h)", file=sys.stderr)

    meta_f.close()
    det.close()
    missing_ids = wanted - seen
    print(f"\ndone: {n_clip} clips, {total_sec/3600:.1f} h", file=sys.stderr)
    print(f"  skipped: missing_wav={n_missing_wav} empty_text={n_empty_text} "
          f"too_short={n_too_short} not_in_manifest={len(missing_ids)}", file=sys.stderr)
    print(f"  {out/'metadata.csv'}\n  {out/'wavs'}/", file=sys.stderr)


if __name__ == "__main__":
    main()
