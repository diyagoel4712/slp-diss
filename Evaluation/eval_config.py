"""Single source of truth for the L2-ARCTIC accent-TTS evaluation grid.

Imported by both `synthesis_driver.py` (generation) and `run_eval.py`
(scoring) so the synthesised file layout always matches what the metrics look for.
Change the grid here, not in two places.
"""
from pathlib import Path
import re
import zipfile

# repo layout
REPO      = Path(__file__).resolve().parents[1]
L2_ARCTIC = REPO / "Datasets" / "L2-ARCTIC"
SOTA      = REPO / "SOTA_models_experiments"
PROMPTS   = SOTA / "arctic.data"          # CMU ARCTIC id -> prompt text

# --- the 10 evaluation utterances: CMU ARCTIC id -> reference text ---
UTTERANCES = {
    "arctic_a0525": "The Oligarchy wanted violence, and it set its agents provocateurs to work.",
    "arctic_a0027": "To my surprise he began to show actual enthusiasm in my favor.",
    "arctic_a0002": "Not at this particular case, Tom, apologized Whittemore.",
    "arctic_b0362": "We threaten to be of the one mind before the voyage is completed.",
    "arctic_a0007": "And you always want to see it in the superlative degree.",
    "arctic_a0023": "A combination of Canadian capital quickly organized and petitioned for the same privileges.",
    "arctic_a0268": "Now go ahead and tell me in a straightforward way what has happened.",
    "arctic_a0319": "And the Edinburgh Evening News says, with editorial gloom.",
    "arctic_a0323": "He considered the victory already his and stepped forward to the meat.",
    "arctic_a0344": "He seemed to fill it with his tremendous vitality.",
}

# --- accent -> L2-ARCTIC speakers (4 accents x 4 speakers) ---
# NB: L2-ARCTIC labels these speakers by L1 (Hindi); we name the accent "Indian" so it
# lines up with GenAID's `southasian` -> `Indian` map (the only L2 accent GenAID covers).
ACCENT_SPEAKERS = {
    "Arabic":     ["ABA", "SKA", "YBAA", "ZHAA"],
    "Indian":     ["ASI", "RRBI", "SVBI", "TNI"],
    "Korean":     ["HJK", "HKK", "YDCK", "YKWK"],
    "Vietnamese": ["HQTV", "PNV", "THV", "TLV"],
}

# speaker -> gender (from L2-ARCTIC README), used to build Parler descriptions.
SPEAKER_GENDER = {
    "ABA": "M", "SKA": "F", "YBAA": "M", "ZHAA": "F",
    "ASI": "M", "RRBI": "M", "SVBI": "F", "TNI": "F",
    "HJK": "F", "HKK": "M", "YDCK": "F", "YKWK": "M",
    "HQTV": "M", "PNV": "F", "THV": "F", "TLV": "M",
}

# --- reference / enrollment utterance for the zero-shot CLONING models ---
# Held-out cloning prompt per speaker (same speaker => same accent + voice). It MUST stay
# OUTSIDE `UTTERANCES` (cloning from the eval clip itself would be leakage). Change here to
# use a different / longer prompt.
REFERENCE_UTT = "arctic_a0001"

# --- model -> root dir of its synthesised output tree ---
# synth wav convention:  <out_root>/<accent>/<speaker>/<utt_id>.wav
MODELS = {
    "xtts":       SOTA / "coqui"      / "outputs" / "xtts",
    "vits":       SOTA / "vits"       / "outputs" / "vits",
    "f5tts":      SOTA / "f5-tts"     / "outputs" / "f5tts",
    "cosyvoice3": SOTA / "CosyVoice3" / "outputs" / "cosyvoice3",
    "parler":     SOTA / "parler-tts" / "outputs" / "parler",
}

# how each model is told the target accent (documentation + driver dispatch)
MODEL_FAMILY = {
    "xtts": "clone", "f5tts": "clone", "cosyvoice3": "clone",  # reference-clip cloning
    "vits": "speaker_id",                                       # VCTK sid baseline
    "parler": "description",                                    # NL-description baseline
}

# VITS (vctk checkpoint) can only emit a trained VCTK speaker -> native-accent baseline.
VITS_SID = 4


def ref_path(speaker, utt_id):
    """Natural L2-ARCTIC reference recording for a speaker/utterance."""
    return L2_ARCTIC / speaker / "wav" / f"{utt_id}.wav"


def synth_path(model, accent, speaker, utt_id):
    return MODELS[model] / accent / speaker / f"{utt_id}.wav"


def speaker_accent():
    """speaker code -> accent label."""
    return {spk: acc for acc, spks in ACCENT_SPEAKERS.items() for spk in spks}


_PROMPT_RE = re.compile(r'\(\s*(\S+)\s+"([^"]+)"\s*\)')

def load_prompts():
    """CMU ARCTIC id -> prompt text (used for cloning-model reference transcripts)."""
    out = {}
    with open(PROMPTS, encoding="utf-8") as f:
        for line in f:
            m = _PROMPT_RE.search(line)
            if m:
                out[m.group(1)] = m.group(2).strip()
    return out


def ensure_wavs(speaker, utt_ids):
    """Extract specific reference wavs for a speaker straight out of its L2-ARCTIC zip.

    Returns the list of wav paths that exist afterwards (skips utts the speaker never
    recorded -- see the README "Notes" section).
    """
    got = []
    zpath = L2_ARCTIC / f"{speaker}.zip"
    have_zip = zpath.exists()
    with (zipfile.ZipFile(zpath) if have_zip else _nullcm()) as z:
        for u in utt_ids:
            w = ref_path(speaker, u)
            if w.exists():
                got.append(w)
                continue
            if not have_zip:
                continue
            try:
                z.extract(f"{speaker}/wav/{u}.wav", L2_ARCTIC)
                got.append(w)
            except KeyError:
                pass
    return got


class _nullcm:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def enrollment(speaker, prompts=None):
    """(ref_wav_path, ref_text) for a speaker's held-out cloning reference.

    Extracts the reference wav from the zip on demand. ref_text is the matching CMU
    prompt, required by F5-TTS / CosyVoice3.
    """
    prompts = prompts or load_prompts()
    ensure_wavs(speaker, [REFERENCE_UTT])
    return ref_path(speaker, REFERENCE_UTT), prompts[REFERENCE_UTT]


def grid():
    """Yield one dict per (accent, speaker, utterance) cell of the evaluation matrix."""
    for accent, speakers in ACCENT_SPEAKERS.items():
        for speaker in speakers:
            for utt_id, text in UTTERANCES.items():
                yield {"accent": accent, "speaker": speaker,
                       "utt_id": utt_id, "text": text}
