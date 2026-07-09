"""
data/alignment.py — MFA forced alignment wrapper and phone token extraction.

Produces a JSONL file of phone tokens:
  {
    "utterance_id": "spk001_utt042",
    "speaker_id":   "spk001",
    "variety":      "scottish_english",
    "audio_path":   "data/audio/spk001_utt042.wav",
    "phone":        "AE",           # ARPAbet, stress stripped
    "phone_raw":    "AE1",          # with stress digit
    "t_start":      1.234,
    "t_end":        1.310,
    "word":         "TRAP",
    "prev_phone":   "T",
    "next_phone":   "P",
    "context":      "intervocalic"  # or "word_final", "word_initial", "other"
  }

Only phones in DIAGNOSTIC_PHONES are emitted by default (configurable).
"""

import os
import json
import subprocess
import warnings
from pathlib import Path
from typing import Optional

from praatio import textgrid

from utils.phones import (
    strip_stress,
    DIAGNOSTIC_PHONES,
    ALL_PHONES,
    is_intervocalic,
    is_postvocalic_r_context,
    is_word_final,
)
from utils.audio import load_audio, extract_segment, is_too_quiet


# ---------------------------------------------------------------------------
# MFA runner
# ---------------------------------------------------------------------------

def run_mfa(
    audio_dir: str,
    transcript_dir: str,
    out_dir: str,
    acoustic_model: str = "english_us_arpa",
    dictionary: str = "english_us_arpa",
    jobs: int = 4,
    clean: bool = False,
):
    """
    Run Montreal Forced Aligner on a corpus directory.

    Requires MFA installed: conda install -c conda-forge montreal-forced-aligner
    Requires models:
        mfa models download acoustic english_us_arpa
        mfa models download dictionary english_us_arpa

    NOTE: MFA's English acoustic model is trained on standard American English.
    Alignment quality degrades for heavily-accented speech; this is a known
    limitation. Consider using the multilingual acoustic model for non-native
    speakers or the English MFA model trained on multi-accent data if available.
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "mfa", "align",
        audio_dir,
        dictionary,
        acoustic_model,
        out_dir,
        "--jobs", str(jobs),
        "--output_format", "long_textgrid",
    ]
    if clean:
        cmd.append("--clean")

    print(f"Running MFA: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"MFA failed:\n{result.stderr}")
    print("MFA alignment complete.")


# ---------------------------------------------------------------------------
# TextGrid parsing → phone token dicts
# ---------------------------------------------------------------------------

def parse_textgrid(tg_path: str):
    """
    Parse MFA TextGrid. Returns (phone_entries, word_entries) where each
    entry is (t_start, t_end, label).
    """
    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)

    phone_entries = []
    if "phones" in tg.tierNameList:
        phone_entries = list(tg.getTier("phones").entries)

    word_entries = []
    if "words" in tg.tierNameList:
        word_entries = list(tg.getTier("words").entries)

    return phone_entries, word_entries


def find_word_for_phone(t_mid: float, word_entries: list) -> str:
    """Return the word label for the phone whose midpoint is t_mid."""
    for (ws, we, wl) in word_entries:
        if ws <= t_mid <= we:
            return wl.upper()
    return ""


def classify_context(
    prev_phone: str,
    curr_phone: str,
    next_phone: str,
) -> str:
    """Classify the phonological context of a phone token."""
    if is_intervocalic(prev_phone, next_phone):
        return "intervocalic"
    if is_postvocalic_r_context(prev_phone, curr_phone):
        return "postvocalic_r"
    if is_word_final(next_phone):
        return "word_final"
    if prev_phone in ("SIL", "SP", "", "sil", "sp"):
        return "word_initial"
    return "other"


def extract_phone_tokens(
    audio_path: str,
    tg_path: str,
    utterance_id: str,
    speaker_id: str,
    variety: str,
    diagnostic_only: bool = True,
    min_duration_ms: float = 20.0,
    silence_filter: bool = True,
) -> list[dict]:
    """
    Extract phone tokens from a single utterance.

    Args:
        diagnostic_only: if True, only emit phones in DIAGNOSTIC_PHONES
        min_duration_ms: discard tokens shorter than this
        silence_filter:  discard tokens below energy threshold

    Returns list of phone token dicts.
    """
    phone_entries, word_entries = parse_textgrid(tg_path)
    if not phone_entries:
        return []

    # Load audio for energy filtering
    try:
        waveform = load_audio(audio_path)
    except Exception as e:
        warnings.warn(f"Could not load {audio_path}: {e}")
        return []

    tokens = []
    n = len(phone_entries)

    for i, (t_start, t_end, label_raw) in enumerate(phone_entries):
        label = strip_stress(label_raw).upper()

        # Skip silence tiers
        if label in ("SIL", "SP", "", "spn"):
            continue

        # Skip unknown phones
        if label not in ALL_PHONES:
            continue

        # Diagnostic filter
        if diagnostic_only and label not in DIAGNOSTIC_PHONES:
            continue

        # Duration filter
        duration_ms = (t_end - t_start) * 1000
        if duration_ms < min_duration_ms:
            continue

        # Energy filter (catches stop closures, near-silent segments)
        if silence_filter:
            seg = extract_segment(waveform, t_start, t_end, pad_ms=0)
            if seg is None or is_too_quiet(seg):
                continue

        # Context
        prev_label = strip_stress(phone_entries[i-1][2]).upper() if i > 0 else "SIL"
        next_label = strip_stress(phone_entries[i+1][2]).upper() if i < n-1 else "SIL"
        context = classify_context(prev_label, label, next_label)

        t_mid = (t_start + t_end) / 2
        word = find_word_for_phone(t_mid, word_entries)

        tokens.append({
            "utterance_id": utterance_id,
            "speaker_id":   speaker_id,
            "variety":      variety,
            "audio_path":   str(audio_path),
            "phone":        label,
            "phone_raw":    label_raw,
            "t_start":      round(t_start, 6),
            "t_end":        round(t_end, 6),
            "word":         word,
            "prev_phone":   prev_label,
            "next_phone":   next_label,
            "context":      context,
        })

    return tokens


# ---------------------------------------------------------------------------
# Corpus-level processing
# ---------------------------------------------------------------------------

def process_corpus(
    audio_dir: str,
    tg_dir: str,
    variety_map: dict,           # {speaker_id: variety_label}
    out_path: str,
    diagnostic_only: bool = True,
    speaker_map_path: Optional[str] = None,
):
    """
    Process an entire aligned corpus → phone_tokens.jsonl

    variety_map: {"spk001": "scottish_english", "spk002": "general_american", ...}
    """
    audio_dir = Path(audio_dir)
    tg_dir    = Path(tg_dir)

    all_tokens = []
    speakers_seen = set()
    varieties_seen = set()

    for tg_path in sorted(tg_dir.glob("**/*.TextGrid")):
        stem = tg_path.stem
        wav_path = audio_dir / (stem + ".wav")
        if not wav_path.exists():
            wav_path = audio_dir / (tg_path.parent.name) / (stem + ".wav")
        if not wav_path.exists():
            warnings.warn(f"No audio for {stem}, skipping.")
            continue

        # Infer speaker_id from filename convention: spk_utt or just use directory
        speaker_id = tg_path.parent.name if tg_path.parent.name != tg_dir.name else stem.split("_")[0]
        variety = variety_map.get(speaker_id, "unknown")

        if variety == "unknown":
            warnings.warn(f"No variety for speaker {speaker_id}")

        tokens = extract_phone_tokens(
            audio_path=str(wav_path),
            tg_path=str(tg_path),
            utterance_id=stem,
            speaker_id=speaker_id,
            variety=variety,
            diagnostic_only=diagnostic_only,
        )
        all_tokens.extend(tokens)
        speakers_seen.add(speaker_id)
        varieties_seen.add(variety)

        print(f"  {stem}: {len(tokens)} tokens")

    with open(out_path, "w") as f:
        for tok in all_tokens:
            f.write(json.dumps(tok) + "\n")

    # Summary
    phone_counts = {}
    for tok in all_tokens:
        phone_counts[tok["phone"]] = phone_counts.get(tok["phone"], 0) + 1

    print(f"\n{'='*60}")
    print(f"Total tokens:   {len(all_tokens)}")
    print(f"Speakers:       {len(speakers_seen)}")
    print(f"Varieties:      {len(varieties_seen)}: {sorted(varieties_seen)}")
    print(f"Phone breakdown:")
    for ph, cnt in sorted(phone_counts.items(), key=lambda x: -x[1]):
        print(f"  {ph:6s}: {cnt:6d}")
    print(f"Saved → {out_path}")

    if speaker_map_path:
        speaker_map = {s: i for i, s in enumerate(sorted(speakers_seen))}
        variety_label_map = {v: i for i, v in enumerate(sorted(varieties_seen))}
        with open(speaker_map_path, "w") as f:
            json.dump({"speakers": speaker_map, "varieties": variety_label_map}, f, indent=2)
        print(f"Speaker/variety maps → {speaker_map_path}")

    return all_tokens


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir",    required=True)
    p.add_argument("--tg_dir",       required=True)
    p.add_argument("--variety_map",  required=True, help="JSON: {speaker_id: variety}")
    p.add_argument("--out",          default="data/phone_tokens.jsonl")
    p.add_argument("--speaker_map",  default="data/speaker_map.json")
    p.add_argument("--all_phones",   action="store_true",
                   help="Include all phones, not just diagnostic ones")
    args = p.parse_args()

    with open(args.variety_map) as f:
        variety_map = json.load(f)

    process_corpus(
        audio_dir=args.audio_dir,
        tg_dir=args.tg_dir,
        variety_map=variety_map,
        out_path=args.out,
        diagnostic_only=not args.all_phones,
        speaker_map_path=args.speaker_map,
    )
