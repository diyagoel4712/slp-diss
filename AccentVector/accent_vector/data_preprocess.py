"""Fine-tuning data preparation for Accent Vector on F5-TTS.

Two stages:

1. ``build-vctk`` -- scan a VCTK-0.92 tree, keep one representative accent
   (default the England/"English" speakers, the paper's British variety;
   VCTK 4.2), filter out clips shorter than 3 s (paper Section 4.2), and emit
   a ``metadata.csv`` in F5's ``audio_file|text`` format.

   For non-English target languages (Phases B/C) you supply your own
   ``audio_file|text`` CSV directly and skip straight to ``prepare``.

2. ``prepare`` -- tokenize transcripts and write the F5 Arrow dataset
   (``raw.arrow`` + ``duration.json`` + ``vocab.txt``) that ``finetune_cli.py``
   consumes. This mirrors F5's own ``prepare_csv_wavs.py`` and reuses the
   pretrained vocab for fine-tuning.

The tokenizer is F5's default ``pinyin`` map; for English text
``convert_char_to_pinyin`` passes letters through unchanged, so it is the right
choice for the British/English phase. Non-Latin target scripts (Hindi,
Arabic, Korean) are NOT covered by the base vocab -- romanize the transcripts
or extend the vocab first (see ADAPTATION_PLAN.md, gotcha #2).

Usage
-----
    python -m accent_vector.data_preprocess build-vctk \
        --vctk-root /data/VCTK-Corpus-0.92 \
        --accent English \
        --out-csv data/british/metadata.csv

    python -m accent_vector.data_preprocess prepare \
        --audio-root /data/VCTK-Corpus-0.92 \
        --metadata   data/british/metadata.csv \
        --out-dir    data/british_pinyin
"""

import argparse
import csv
import json
import multiprocessing
import shutil
from importlib.resources import files
from pathlib import Path

import torchaudio
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from f5_tts.model.utils import convert_char_to_pinyin

PRETRAINED_VOCAB_PATH = files("f5_tts").joinpath("../../data/vocab.txt")
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)
MIN_DURATION_S = 3.0  # paper Section 4.2: discard utterances < 3 s


# --- stage 1: VCTK -> audio_file|text metadata ------------------------------
def load_vctk_accents(vctk_root):
    """speaker_id -> accent, parsed from VCTK's speaker-info.txt."""
    info_path = Path(vctk_root) / "speaker-info.txt"
    if not info_path.exists():
        raise FileNotFoundError(f"speaker-info.txt not found under {vctk_root}")
    accents = {}
    with open(info_path, encoding="utf-8") as f:
        header = True
        for line in f:
            parts = line.split()
            if header:  # first row is the column header (ID AGE GENDER ACCENTS ...)
                header = False
                continue
            if len(parts) < 4:
                continue
            speaker_id, _age, _gender, accent = parts[0], parts[1], parts[2], parts[3]
            accents[speaker_id] = accent
    return accents


def find_audio(vctk_root, speaker, utt_id):
    """Locate the clip for a VCTK utterance across the corpus's layout variants."""
    root = Path(vctk_root)
    candidates = [
        root / "wav48_silence_trimmed" / speaker / f"{utt_id}_mic1.flac",
        root / "wav48_silence_trimmed" / speaker / f"{utt_id}_mic2.flac",
        root / "wav48" / speaker / f"{utt_id}.wav",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def build_vctk_metadata(vctk_root, accent, out_csv, min_duration=MIN_DURATION_S):
    vctk_root = Path(vctk_root)
    accents = load_vctk_accents(vctk_root)
    speakers = sorted(s for s, a in accents.items() if a.lower() == accent.lower())
    if not speakers:
        raise ValueError(
            f"No speakers with accent '{accent}'. Present accents: "
            f"{sorted(set(accents.values()))}"
        )
    print(f"[build-vctk] accent '{accent}': {len(speakers)} speakers")

    txt_root = vctk_root / "txt"
    rows, kept, skipped_short, skipped_missing = [], 0, 0, 0
    for speaker in tqdm(speakers, desc="speakers"):
        spk_txt_dir = txt_root / speaker
        if not spk_txt_dir.is_dir():
            continue
        for txt_file in sorted(spk_txt_dir.glob(f"{speaker}_*.txt")):
            utt_id = txt_file.stem
            audio = find_audio(vctk_root, speaker, utt_id)
            if audio is None:
                skipped_missing += 1
                continue
            try:
                info = torchaudio.info(str(audio))
                duration = info.num_frames / info.sample_rate
            except Exception:
                skipped_missing += 1
                continue
            if duration < min_duration:
                skipped_short += 1
                continue
            text = txt_file.read_text(encoding="utf-8").strip()
            if not text:
                continue
            # store audio path relative to vctk_root so --audio-root stays portable
            rows.append((audio.relative_to(vctk_root).as_posix(), text))
            kept += 1

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text"])
        writer.writerows(rows)
    print(
        f"[build-vctk] kept {kept} clips (>= {min_duration}s); "
        f"skipped {skipped_short} short, {skipped_missing} missing -> {out_csv}"
    )
    return out_csv


# --- stage 2: metadata -> F5 Arrow dataset ----------------------------------
def read_audio_text_pairs(csv_path, audio_root):
    pairs = []
    audio_root = Path(audio_root)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="|")
        next(reader)  # header
        for row in reader:
            if len(row) >= 2:
                pairs.append(((audio_root / row[0].strip()).as_posix(), row[1].strip()))
    return pairs


def prepare_dataset(metadata_csv, audio_root, out_dir, is_finetune=True, lora_label=None):
    pairs = read_audio_text_pairs(metadata_csv, audio_root)
    if not pairs:
        raise RuntimeError(f"No usable rows in {metadata_csv}")

    result, durations, vocab_set = [], [], set()
    texts = [t for _, t in pairs]
    converted = []
    for i in range(0, len(texts), 100):  # tokenize in batches
        converted.extend(convert_char_to_pinyin(texts[i : i + 100], polyphone=True))

    for (audio_path, _), conv_text in tqdm(
        list(zip(pairs, converted)), desc="collecting"
    ):
        if not Path(audio_path).exists():
            print(f"[prepare] missing audio, skipping: {audio_path}")
            continue
        try:
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
        except Exception as e:
            print(f"[prepare] cannot read {audio_path}: {e}")
            continue
        if duration <= 0:
            continue
        row = {"audio_path": audio_path, "text": conv_text, "duration": duration}
        if lora_label is not None:
            # per-sample LoRA label; a single-accent run uses one constant label.
            # The LoRA dataset/sampler (dataset.py) reads row["lora_label"].
            row["lora_label"] = int(lora_label)
        result.append(row)
        durations.append(duration)
        vocab_set.update(list(conv_text))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with ArrowWriter(path=(out_dir / "raw.arrow").as_posix(), writer_batch_size=100) as writer:
        for line in tqdm(result, desc="writing raw.arrow"):
            writer.write(line)
    with open(out_dir / "duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f, ensure_ascii=False)

    vocab_out = out_dir / "vocab.txt"
    if is_finetune:
        assert Path(PRETRAINED_VOCAB_PATH).exists(), (
            f"pretrained vocab.txt not found: {PRETRAINED_VOCAB_PATH}"
        )
        shutil.copy2(PRETRAINED_VOCAB_PATH, vocab_out)
    else:
        with open(vocab_out, "w", encoding="utf-8") as f:
            for v in sorted(vocab_set):
                f.write(v + "\n")

    print(
        f"[prepare] {out_dir.name}: {len(result)} samples, "
        f"{sum(durations) / 3600:.2f} h, vocab {len(vocab_set)} -> {out_dir}"
    )
    return out_dir


def _build_parser():
    parser = argparse.ArgumentParser(description="Accent Vector data preparation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_v = sub.add_parser("build-vctk", help="VCTK tree -> audio_file|text metadata")
    p_v.add_argument("--vctk-root", required=True)
    p_v.add_argument("--accent", default="English", help="VCTK ACCENTS value to keep")
    p_v.add_argument("--out-csv", required=True)
    p_v.add_argument("--min-duration", type=float, default=MIN_DURATION_S)

    p_p = sub.add_parser("prepare", help="metadata -> F5 Arrow dataset")
    p_p.add_argument("--metadata", required=True, help="audio_file|text CSV")
    p_p.add_argument("--audio-root", required=True, help="prefix joined to CSV audio paths")
    p_p.add_argument("--out-dir", required=True)
    p_p.add_argument("--no-finetune-vocab", action="store_true",
                     help="build a fresh vocab instead of copying the pretrained one")
    p_p.add_argument("--lora-label", type=int, default=None,
                     help="write a constant per-sample lora_label (required for LoRA "
                          "training; a single-accent run uses one label, e.g. 0)")
    return parser


def main():
    args = _build_parser().parse_args()
    if args.command == "build-vctk":
        build_vctk_metadata(args.vctk_root, args.accent, args.out_csv, args.min_duration)
    elif args.command == "prepare":
        prepare_dataset(
            args.metadata, args.audio_root, args.out_dir,
            is_finetune=not args.no_finetune_vocab,
            lora_label=args.lora_label,
        )


if __name__ == "__main__":
    main()
