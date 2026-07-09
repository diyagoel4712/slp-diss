"""
features.py — Phonological feature extraction from force-aligned audio.

Pipeline:
  1. Load forced alignment TextGrid (MFA output)
  2. For each phone segment, extract measurable phonological features
  3. Aggregate to utterance-level means (what the model predicts)

Features extracted:
  - F1, F2, F3 formants (via Praat / parselmouth) — vowel quality & rhoticity
  - VOT — voice onset time for stop consonants
  - F0 mean — over vowel nucleus
  - Binary: is_rhotic, is_flapped, is_glottalled

References:
  - Montreal Forced Aligner: McAuliffe et al. (2017, Interspeech)
  - Praat: Boersma & Weenink (2023)
  - Lobanov normalisation: Lobanov (1971)
  - Rhoticity from F3: Espy-Wilson et al. (2000)
"""

import os
import json
import numpy as np
import parselmouth
from parselmouth.praat import call
from praatio import textgrid
from pathlib import Path
from typing import Optional
import warnings

# IPA-style phone sets (ARPAbet mapped to categories)
VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
}
RHOTIC_VOWEL = "ER"            # rhotacised mid-central vowel
STOPS = {"P", "T", "K", "B", "D", "G"}
VOICELESS_STOPS = {"P", "T", "K"}


def load_textgrid(tg_path: str):
    """Load MFA TextGrid and return word + phone tiers."""
    tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=False)
    phone_tier = tg.getTier("phones")
    return phone_tier


def extract_formants(
    snd: parselmouth.Sound,
    t_start: float,
    t_end: float,
    max_formant: float = 5500.0,
    n_formants: int = 3,
) -> dict:
    """
    Extract mean F1, F2, F3 over a vowel segment using Praat's Burg method.
    max_formant should be ~5500 for female, ~5000 for male — consider
    normalising after extraction (Lobanov / Nearey NORM).
    """
    duration = t_end - t_start
    if duration < 0.025:
        return {"f1": np.nan, "f2": np.nan, "f3": np.nan}

    formant = call(snd, "To Formant (burg)", 0, n_formants, max_formant, 0.025, 50)
    mid = (t_start + t_end) / 2.0

    f1 = call(formant, "Get value at time", 1, mid, "Hertz", "Linear")
    f2 = call(formant, "Get value at time", 2, mid, "Hertz", "Linear")
    f3 = call(formant, "Get value at time", 3, mid, "Hertz", "Linear")

    return {"f1": f1, "f2": f2, "f3": f3}


def extract_vot(
    snd: parselmouth.Sound,
    t_burst: float,
    window: float = 0.08,
) -> Optional[float]:
    """
    Estimate VOT by detecting voicing onset after stop burst.
    Simplified: measure zero-crossing rate to detect periodic voicing onset.
    For production use, replace with VoiceSauce or AutoVOT
    (Keshet et al. 2014, JASA).

    Returns VOT in milliseconds or None if undetectable.
    """
    t_end = t_burst + window
    segment = snd.extract_part(from_time=t_burst, to_time=t_end, preserve_times=False)
    samples = segment.values[0]
    sr = snd.sampling_frequency

    frame_size = int(0.005 * sr)  # 5ms frames
    zcrs = []
    for i in range(0, len(samples) - frame_size, frame_size):
        frame = samples[i:i + frame_size]
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_size)
        zcrs.append(zcr)

    # Voicing onset ≈ first frame where ZCR drops (periodic voicing begins)
    zcrs = np.array(zcrs)
    threshold = zcrs[:3].mean() * 0.6
    onsets = np.where(zcrs < threshold)[0]
    if len(onsets) == 0:
        return None
    vot_ms = onsets[0] * 5.0  # each frame = 5ms
    return vot_ms


def extract_f0(
    snd: parselmouth.Sound,
    t_start: float,
    t_end: float,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
) -> Optional[float]:
    """Mean F0 over a vowel nucleus (central 60%)."""
    duration = t_end - t_start
    if duration < 0.04:
        return None
    margin = duration * 0.2
    t0 = t_start + margin
    t1 = t_end - margin

    pitch = call(snd, "To Pitch", 0, pitch_floor, pitch_ceiling)
    f0_vals = [
        call(pitch, "Get value at time", t, "Hertz", "Linear")
        for t in np.linspace(t0, t1, 10)
    ]
    f0_vals = [v for v in f0_vals if not (np.isnan(v) or v == 0)]
    return float(np.mean(f0_vals)) if f0_vals else None


def is_glottalled(phone_label: str, snd, t_start, t_end) -> bool:
    """
    Detect /t/ → [ʔ] glottalisation: presence of creaky/laryngealised voice.
    Heuristic: high jitter + low amplitude in closure phase.
    For production use VoiceSauce (Shue et al. 2011).
    """
    if phone_label != "T":
        return False
    duration = t_end - t_start
    if duration < 0.02:
        return False
    segment = snd.extract_part(from_time=t_start, to_time=t_start + min(0.03, duration))
    rms = np.sqrt(np.mean(segment.values[0] ** 2))
    # Glottal stop: very low amplitude closure
    return bool(rms < 0.005)


def extract_utterance_features(audio_path: str, tg_path: str) -> dict:
    """
    Main entry point. Given an audio file and its MFA TextGrid,
    returns a dict of utterance-level phonological features
    (averaged over all qualifying segments).

    These become the training targets for the phonological prediction heads.
    """
    snd = parselmouth.Sound(audio_path)
    phone_tier = load_textgrid(tg_path)

    f1_vals, f2_vals, f3_vals, f0_vals, vot_vals = [], [], [], [], []
    rhotic_count, total_rhotic_ctx = 0, 0
    flap_count, flap_ctx = 0, 0
    glottal_count, glottal_ctx = 0, 0

    intervals = phone_tier.entries  # list of (start, end, label)

    for i, (t_start, t_end, label) in enumerate(intervals):
        label_base = label.rstrip("0123456789").upper()  # strip stress marks

        # --- Vowel features ---
        if label_base in VOWELS:
            formants = extract_formants(snd, t_start, t_end)
            if not np.isnan(formants["f1"]):
                f1_vals.append(formants["f1"])
                f2_vals.append(formants["f2"])
                f3_vals.append(formants["f3"])

            f0 = extract_f0(snd, t_start, t_end)
            if f0 is not None:
                f0_vals.append(f0)

            # Rhoticity: /ER/ present → rhotic variety
            total_rhotic_ctx += 1
            if label_base == RHOTIC_VOWEL:
                rhotic_count += 1

        # --- Stop features ---
        if label_base in VOICELESS_STOPS:
            vot = extract_vot(snd, t_start)
            if vot is not None:
                vot_vals.append(vot)

        # --- /t/ realisation (flap vs glottal) ---
        if label_base == "T":
            glottal_ctx += 1
            if is_glottalled(label_base, snd, t_start, t_end):
                glottal_count += 1
            # Flapping: /t/ between vowels, unstressed context (simplified)
            if i > 0 and i < len(intervals) - 1:
                prev = intervals[i-1][2].rstrip("0123456789").upper()
                nxt  = intervals[i+1][2].rstrip("0123456789").upper()
                if prev in VOWELS and nxt in VOWELS:
                    flap_ctx += 1
                    # Detect flap: short duration + voiced
                    duration = t_end - t_start
                    if duration < 0.035:
                        flap_count += 1

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    def safe_rate(num, denom):
        return num / denom if denom > 0 else float("nan")

    return {
        # Continuous (regression targets — normalised later)
        "f1_mean": safe_mean(f1_vals),
        "f2_mean": safe_mean(f2_vals),
        "f3_mean": safe_mean(f3_vals),
        "f0_mean": safe_mean(f0_vals),
        "vot_mean": safe_mean(vot_vals),
        # Rates (used for binary classification heads)
        "rhoticity_rate":  safe_rate(rhotic_count, total_rhotic_ctx),
        "flap_rate":       safe_rate(flap_count, flap_ctx),
        "glottal_rate":    safe_rate(glottal_count, glottal_ctx),
        # Raw counts (useful for debugging)
        "_n_vowels": len(f1_vals),
        "_n_stops":  len(vot_vals),
    }


def lobanov_normalise(features_list: list[dict]) -> list[dict]:
    """
    Lobanov (1971) speaker normalisation for F1/F2:
      F*_i = (F_i - mu_speaker) / sigma_speaker
    Applied across all utterances of each speaker to remove
    speaker-level differences in vocal tract length.

    features_list: list of feature dicts, each with "speaker_id" key.
    """
    from collections import defaultdict
    speaker_vals = defaultdict(lambda: {"f1": [], "f2": [], "f3": []})

    for f in features_list:
        sid = f["speaker_id"]
        for k in ("f1_mean", "f2_mean", "f3_mean"):
            v = f.get(k, float("nan"))
            if not np.isnan(v):
                speaker_vals[sid][k.split("_")[0]].append(v)

    speaker_stats = {}
    for sid, vals in speaker_vals.items():
        speaker_stats[sid] = {
            k: (np.mean(v), np.std(v) + 1e-6)
            for k, v in vals.items()
        }

    normalised = []
    for f in features_list:
        f = dict(f)
        sid = f["speaker_id"]
        stats = speaker_stats[sid]
        for raw_key, fkey in [("f1_mean", "f1"), ("f2_mean", "f2"), ("f3_mean", "f3")]:
            if not np.isnan(f.get(raw_key, float("nan"))):
                mu, sigma = stats[fkey]
                f[raw_key + "_norm"] = (f[raw_key] - mu) / sigma
            else:
                f[raw_key + "_norm"] = float("nan")
        normalised.append(f)

    return normalised


def process_corpus(audio_dir: str, tg_dir: str, out_path: str, speaker_map: dict = None):
    """
    Process an entire corpus directory.
    audio_dir:   directory of .wav files
    tg_dir:      directory of .TextGrid files (same basenames)
    out_path:    output JSON path
    speaker_map: {"basename": "speaker_id"} — if None, basename is used
    """
    audio_dir = Path(audio_dir)
    tg_dir = Path(tg_dir)
    results = []

    for wav_path in sorted(audio_dir.glob("*.wav")):
        stem = wav_path.stem
        tg_path = tg_dir / (stem + ".TextGrid")
        if not tg_path.exists():
            warnings.warn(f"No TextGrid for {stem}, skipping.")
            continue

        try:
            feats = extract_utterance_features(str(wav_path), str(tg_path))
            feats["utterance_id"] = stem
            feats["speaker_id"] = speaker_map.get(stem, stem) if speaker_map else stem
            results.append(feats)
            print(f"✓ {stem}")
        except Exception as e:
            warnings.warn(f"Failed on {stem}: {e}")

    results = lobanov_normalise(results)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} utterances → {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--tg_dir", required=True)
    p.add_argument("--out", default="features.json")
    args = p.parse_args()
    process_corpus(args.audio_dir, args.tg_dir, args.out)
