#!/usr/bin/env python3
"""How much data did each accent actually fine-tune on?

Two sources, same table:

  --provenance  (best) $CKPT_ROOT/provenance, where eddie_finetune_lora.sh records
           the accent + METADATA_CSV + AUDIO_ROOT of every job it ran. Each language
           is fine-tuned from a DIFFERENTLY NAMED csv -- Dutch/Mandarin/Arabic from
           metadata.dnsmos.csv, Hindi/Bengali from metadata.roman.csv -- so this is
           the only source that needs no guessing about which file to count.

  --clips  a prep output dir on scratch, e.g.
           /exports/eddie/scratch/s2247837/data/aishell_clips
           Picks the last-stage manifest present (see MANIFEST_PREFERENCE) and joins
           it to details.tsv on the clip path -- so durations and speaker counts come
           for free, no audio decoding. Rows dropped by the DNSMOS filter (and, for
           indic, by romanisation) are excluded, exactly as at training time.

  --data-root  the *prepared* datasets ($F5_ROOT/data/<accent>_pinyin), whose
           {train,valid}_duration.json are what the trainer literally loaded. Use
           these to double-check the above: they additionally exclude the handful of
           rows prepare drops for missing/unreadable audio.

    python scripts/dataset_stats.py --provenance $CKPT_ROOT/provenance
    python scripts/dataset_stats.py --clips /exports/eddie/scratch/s2247837/data/*_clips
    python scripts/dataset_stats.py --clips hindi=/path/iv_hindi_clips \
        --manifest hindi=/path/iv_hindi_clips/metadata.roman.csv
    python scripts/dataset_stats.py --provenance $CKPT_ROOT/provenance --steps dutch=60000
    python scripts/dataset_stats.py --data-root $F5_ROOT/data

The `manifest` column always reports which csv each row was counted from.

With --steps it also reports how much audio each run *consumed*: the frame batch
sampler packs clips into ~102 s batches, so a step count alone does not tell you
how many passes over the corpus you made.
"""

import argparse
import csv
import json
import os
import statistics as stats
from pathlib import Path

# defaults mirror scripts/F5TTS_v1_LoRA_accent.yaml + its mel_spec block
FRAMES_PER_BATCH = 9600
MAX_SAMPLES = 32
HOP_LENGTH = 256
SAMPLE_RATE = 24000
SEC_PER_FRAME = HOP_LENGTH / SAMPLE_RATE  # 10.67 ms
VAL_FRAC = 0.1  # accent_vector.data_preprocess.prepare_dataset default

# last pipeline stage first -- see from_clips()
MANIFEST_PREFERENCE = ("metadata.roman.csv", "metadata.dnsmos.csv", "metadata.csv")


def read_manifest(path):
    """Ordered clip paths from a `audio_file|text` manifest (prepare's row order)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f, delimiter="|")
        next(r, None)  # header
        return [row[0].strip() for row in r if len(row) >= 2]


def from_clips(clips_dir, manifest=None):
    """(durations in manifest order, speaker set, n_missing_from_details, manifest path).

    details.tsv is written by every prep script (cgn / indicvoices / aishell /
    globalphone) with at least clip, speaker, dur -- keyed on the same
    `wavs/<name>` path the manifest uses.
    """
    clips_dir = Path(clips_dir)
    if manifest:
        man_path = Path(manifest)
        if not man_path.exists():
            raise FileNotFoundError(f"no such manifest: {man_path}")
    else:
        # The finetune is handed a DIFFERENT manifest per language, so guessing
        # wrong silently changes the answer. Probe in the order the prep pipelines
        # produce them, last-stage first: indic runs romanise AFTER DNSMOS and
        # drops rows that romanise to empty (romanize.py), so metadata.roman.csv
        # -- not metadata.dnsmos.csv -- is what those finetunes consumed.
        # Prefer --provenance to skip guessing entirely.
        for name in MANIFEST_PREFERENCE:
            man_path = clips_dir / name
            if man_path.exists():
                break
        else:
            raise FileNotFoundError(f"no manifest under {clips_dir}")

    det_path = clips_dir / "details.tsv"
    if not det_path.exists():
        raise FileNotFoundError(f"no details.tsv under {clips_dir}")

    info = {}
    with open(det_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            clip = (row.get("clip") or "").strip()
            try:
                dur = float(row.get("dur") or 0)
            except ValueError:
                dur = 0.0
            info[clip] = (dur, (row.get("speaker") or "").strip())

    durations, speakers, missing = [], set(), 0
    for clip in read_manifest(man_path):
        hit = info.get(clip)
        if hit is None or hit[0] <= 0:
            missing += 1
            continue
        durations.append(hit[0])
        if hit[1]:
            speakers.add(hit[1])
    if not durations:
        raise RuntimeError(f"{clips_dir}: manifest and details.tsv share no clips")
    return durations, speakers, missing, man_path


def read_provenance(prov_dir):
    """{accent: (metadata_csv, clips_dir)} from record_provenance.sh output.

    eddie_finetune_lora.sh writes one ft.<job>.<stamp>.txt per run under
    $CKPT_ROOT/provenance/, holding the exact accent= / METADATA_CSV= / AUDIO_ROOT=
    the job ran with. Newest file per accent wins; conflicts are reported.
    """
    prov_dir = Path(prov_dir)
    files = sorted(prov_dir.glob("ft.*.txt"))
    if not files:
        raise FileNotFoundError(f"no ft.*.txt under {prov_dir}")

    runs, conflicts = {}, {}
    for f in files:  # sorted by name = job/timestamp order, so later runs overwrite
        fields = {}
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            for token in ("accent=", "METADATA_CSV=", "AUDIO_ROOT="):
                if line.startswith(token) or f"  {token}" in line:
                    val = line.split(token, 1)[1].split("  ")[0].strip()
                    fields[token.rstrip("=")] = val
        accent, csv_path = fields.get("accent"), fields.get("METADATA_CSV")
        if not accent or accent == "?" or not csv_path or csv_path == "?":
            continue
        audio_root = fields.get("AUDIO_ROOT")
        clips = Path(audio_root) if audio_root and audio_root != "?" else Path(csv_path).parent
        prev = runs.get(accent)
        if prev and prev[0] != csv_path:
            conflicts.setdefault(accent, {prev[0]}).add(csv_path)
        runs[accent] = (csv_path, clips)
    return runs, conflicts


def from_prepared(data_dir):
    """(durations, split sizes) from a prepared <accent>_pinyin dir."""
    got = {}
    for split in ("train", "valid"):
        f = data_dir / f"{split}_duration.json"
        if f.exists():
            got[split] = json.loads(f.read_text(encoding="utf-8"))["duration"]
    if not got:
        raise FileNotFoundError(f"no *_duration.json under {data_dir}")
    train, valid = got.get("train", []), got.get("valid", [])
    if train and train == valid:  # 1-clip corpus: prepare reuses rows for both splits
        valid = []
    return valid + train, len(train), len(valid)


def split_like_prepare(durations):
    """prepare_dataset takes the FIRST val_frac of rows as valid, rest as train."""
    n = len(durations)
    n_valid = max(1, round(n * VAL_FRAC)) if n > 1 else 0
    return durations[n_valid:], durations[:n_valid]


def pack_batches(durations, frames_per_batch=FRAMES_PER_BATCH, max_samples=MAX_SAMPLES):
    """Replicate F5's DynamicBatchSampler: sort by length, greedily pack until the
    summed frames exceed the threshold or max_samples is hit (drop_residual=False,
    so the tail batch is kept). Clips longer than the threshold are dropped by the
    sampler and never seen -- reported separately.

    Returns (n_batches, frames_packed, n_oversized).
    """
    frames = sorted(int(d * SAMPLE_RATE / HOP_LENGTH) for d in durations)
    oversized = sum(1 for f in frames if f > frames_per_batch)
    frames = [f for f in frames if f <= frames_per_batch]

    batches, packed, batch_n, batch_frames = 0, 0, 0, 0
    for f in frames:
        if batch_frames + f <= frames_per_batch and batch_n < max_samples:
            batch_n += 1
            batch_frames += f
        else:
            batches += 1
            packed += batch_frames
            batch_n, batch_frames = 1, f
    if batch_n:
        batches += 1
        packed += batch_frames
    return batches, packed, oversized


def describe(name, durations, train, valid, speakers=None, steps=None,
             frames_per_batch=FRAMES_PER_BATCH, max_samples=MAX_SAMPLES):
    row = {
        "accent": name,
        "clips": len(durations),
        "hours": sum(durations) / 3600,
        "train_h": sum(train) / 3600,
        "valid_h": sum(valid) / 3600,
        "mean_s": stats.mean(durations),
        "median_s": stats.median(durations),
        "min_s": min(durations),
        "max_s": max(durations),
    }
    if speakers is not None:
        row["speakers"] = len(speakers)

    n_batches, packed, oversized = pack_batches(train, frames_per_batch, max_samples)
    row["steps_per_epoch"] = n_batches
    row["oversized"] = oversized
    # audio actually fed per pass: sampler-dropped clips excluded, padding excluded
    row["epoch_h"] = packed * SEC_PER_FRAME / 3600

    if steps is not None:
        row["steps"] = steps
        row["epochs"] = steps / n_batches if n_batches else float("nan")
        row["seen_h"] = row["epochs"] * row["epoch_h"]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=[], metavar="[ACCENT=]DIR",
                    help="prep output dir(s); bare paths are named from the dir "
                         "(aishell_clips -> aishell). Globs fine.")
    ap.add_argument("--provenance", default=None, metavar="DIR",
                    help="$CKPT_ROOT/provenance -- take each accent's METADATA_CSV "
                         "(and clips dir) from what the finetune job recorded. "
                         "Best source: no guessing about per-language CSV names.")
    ap.add_argument("--manifest", nargs="*", default=[], metavar="ACCENT=PATH",
                    help="override the manifest for an accent (default: first of "
                         + ", ".join(MANIFEST_PREFERENCE) + " found in the clips dir)")
    ap.add_argument("--data-root", default=None,
                    help="dir holding <accent>_pinyin/ (default: $F5_ROOT/data)")
    ap.add_argument("--accents", nargs="*", default=None,
                    help="with --data-root: which accents (default: all found)")
    ap.add_argument("--steps", nargs="*", default=[], metavar="ACCENT=N",
                    help="trained step count per accent -> epochs + audio hours seen")
    ap.add_argument("--frames-per-batch", type=int, default=FRAMES_PER_BATCH,
                    help=f"datasets.batch_size_per_gpu (default {FRAMES_PER_BATCH})")
    ap.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                    help=f"datasets.max_samples (default {MAX_SAMPLES})")
    ap.add_argument("--csv", default=None, help="also write the table here")
    args = ap.parse_args()

    def kv(pairs, cast=str):
        out = {}
        for p in pairs:
            k, _, v = p.partition("=")
            if not v:
                ap.error(f"expected ACCENT=VALUE, got {p!r}")
            out[k] = cast(v)
        return out

    steps, manifests = kv(args.steps, int), kv(args.manifest)
    rows, notes = [], []

    # accent -> clips dir, from --clips (explicit or named after the dir)
    targets = {}
    for spec in args.clips:
        name, _, path = spec.partition("=")
        if not path:
            path, name = name, Path(name.rstrip("/")).name
            for suffix in ("_clips", "_wavs"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
        targets[name] = path

    if args.provenance:
        try:
            runs, conflicts = read_provenance(args.provenance)
        except FileNotFoundError as e:
            ap.error(str(e))
        for accent, (csv_path, clips) in runs.items():
            manifests.setdefault(accent, csv_path)  # --manifest still wins
            targets.setdefault(accent, str(clips))  # as does an explicit --clips dir
        for accent, seen in conflicts.items():
            notes.append(f"{accent}: runs used different manifests ({', '.join(sorted(seen))}); "
                         f"took the newest -- pass --manifest {accent}=PATH to pin another")
        if not runs:
            notes.append(f"no usable accent=/METADATA_CSV= records under {args.provenance}")

    for name in sorted(targets):
        try:
            durations, speakers, missing, man_path = from_clips(targets[name], manifests.get(name))
        except (FileNotFoundError, RuntimeError) as e:
            print(f"! {name}: {e}")
            continue
        train, valid = split_like_prepare(durations)
        row = describe(name, durations, train, valid, speakers, steps.get(name),
                       args.frames_per_batch, args.max_samples)
        # the manifest IS the answer to "how much data" -- always show which one
        row["manifest"] = man_path.name
        rows.append(row)
        if name not in manifests:
            notes.append(f"{name}: manifest auto-detected as {man_path.name} "
                         "(not from a job record -- confirm with --provenance)")
        if missing:
            notes.append(f"{name}: {missing} manifest clips absent from details.tsv, excluded")

    if args.data_root or not (args.clips or args.provenance):
        root = Path(args.data_root or os.environ.get("F5_ROOT")
                    or Path(__file__).resolve().parents[2] / "F5-TTS")
        data_root = root if root.name == "data" else root / "data"
        if data_root.is_dir():
            names = args.accents or sorted(p.name[: -len("_pinyin")]
                                           for p in data_root.glob("*_pinyin") if p.is_dir())
            for name in names:
                try:
                    durations, n_train, _ = from_prepared(data_root / f"{name}_pinyin")
                except FileNotFoundError as e:
                    print(f"! {name}: {e}")
                    continue
                # from_prepared returns valid rows first, then train (prepare's order)
                valid, train = durations[: len(durations) - n_train], durations[len(durations) - n_train:]
                rows.append(describe(f"{name} (prepared)", durations, train, valid, None,
                                     steps.get(name), args.frames_per_batch, args.max_samples))
        elif not (args.clips or args.provenance):
            ap.error(f"no such data dir: {data_root} (pass --clips/--provenance or --data-root)")

    if not rows:
        return

    cols = ["accent", "clips", "hours", "train_h", "valid_h", "mean_s", "median_s",
            "min_s", "max_s", "steps_per_epoch", "epoch_h", "oversized"]
    if any("speakers" in r for r in rows):
        cols.insert(2, "speakers")
    if any("manifest" in r for r in rows):
        cols.append("manifest")
    if any("steps" in r for r in rows):
        cols += ["steps", "epochs", "seen_h"]

    def fmt(v):
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    widths = [max(len(c), *(len(fmt(r.get(c, "-"))) for r in rows)) for c in cols]
    print("  ".join(c.rjust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(fmt(r.get(c, "-")).rjust(w) for c, w in zip(cols, widths)))

    if any(r.get("oversized") for r in rows):
        limit = args.frames_per_batch * SEC_PER_FRAME
        notes.append(f"oversized = clips > {limit:.1f}s; the frame sampler skips these "
                     "entirely, so they never reached the model")
    for n in notes:
        print(f"note: {n}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
