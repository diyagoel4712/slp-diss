# SOTA models — evaluation

The goal of this folder is to run evaluation across a range of open-source models to assess their accent-generation capabilities, and thereby define the problem space that current models face when generating text in different accents. 

We prompt each model with 4 utterances, each to be synthesised in 4 different accents by 5 different speakers. Each speaker-utterance combination is evaluated against a reference speech sample.

All models are zero-shot. Some models (XTTS-2, VITS and F5) require reference speech, while others (CosyVoice3 and Parler-TTS) require natural language descriptions of desired speaker characteristics.

The chosen accents are: Arabic, Indian, Vietnamese and Korean.

`evaluation_functions.py` implements eight objective metrics for comparing
synthesised speech against natural reference speech (VCTK).

| # | Function | Measures | Backend | Env |
|---|----------|----------|---------|-----|
| 1 | `utmos` | naturalness (MOS) | utmosv2 | root `.venv` |
| 2 | `f0_rmse` | prosody / pitch (mel-scaled) | librosa pyin | `.conda` |
| 3 | `mcd` | spectral envelope | librosa MFCC + DTW | `.conda` |
| 4 | `wer` | intelligibility | Whisper + jiwer | `.conda` |
| 5 | `aid_acc` | accent-ID accuracy | GenAID (+ CommonAccent) | `genaid` |
| 6 | `cs_accent` | accent-embedding cosine sim | GenAID embeddings | `genaid` |
| 7 | `ppg_kl` | segmental pronunciation | wav2vec2 phoneme-CTC | `.conda` |
| 8 | `speaker_similarity` | speaker identity (SECS) | ECAPA-TDNN | `genaid` |

## Why two environments

The metrics span two Python environments because GenAID requires SpeechBrain
0.5.x, whose old pins conflict with the modern `transformers`/`torch` used by the
other metrics. `evaluation_functions.py` runs in **`.conda`** and calls the
**`genaid`** env as a subprocess for metrics 5/6/8.

- **`.conda`** (Python 3.11) — F0/MCD/WER/PPG-KL. See `requirements-eval.txt`.
- **`genaid`** (Python 3.10) — accent-ID + speaker embeddings. See `requirements-genaid.txt`.
- Root **`.venv`** (uv, Python 3.13) — UTMOS only (`utmosv2`, declared in `../pyproject.toml`).

`_GENAID_PYTHON` in `evaluation_functions.py` hardcodes the genaid interpreter path
— update it if your conda prefix differs.

## .conda env setup (metrics 2/3/4/7)

```bash
uv pip install --python /path/to/.conda/bin/python -r requirements-eval.txt
```

First runs download model weights (cached after): Whisper `base.en` (~140 MB) for
WER; `facebook/wav2vec2-lv-60-espeak-cv-ft` (~1.2 GB) for PPG-KL.

## genaid env setup (metrics 5/6/8)

> ⚠️ `GenAID/` is gitignored (third-party clone, like the other model dirs), so the
> steps below — including our wrapper scripts and source patches — are NOT in version
> control and must be reapplied on a fresh checkout.

```bash
# 1. Clone the GenAID fork (a SpeechBrain v0.5.16 fork)
git clone https://github.com/jzmzhong/GenAID.git SOTA_models_experiments/GenAID

# 2. Create the env + install (CPU torch; use the pytorch CUDA index on Linux/GPU)
conda create -n genaid python=3.10 -y
conda run -n genaid pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
conda run -n genaid pip install -r SOTA_models_experiments/requirements-genaid.txt
conda run -n genaid pip install --editable SOTA_models_experiments/GenAID

# 3. Download the trained GenAID checkpoint (Google Drive, ~1.1 GB) and unzip
#    into recipes/CommonAccent/GenAID_v6/ (contains save/<CKPT...>/{model,wav2vec2}.ckpt
#    and save/accent_encoder.txt)
cd SOTA_models_experiments/GenAID/recipes/CommonAccent
gdown "https://drive.google.com/uc?id=1slGrpZSu5g-nF7R-QMCmtGcjN3kw7lQj" -O GenAID_ckpt.zip
unzip GenAID_ckpt.zip && rm GenAID_ckpt.zip
```

CommonAccent (#5 secondary) and ECAPA (#8) auto-download from the HuggingFace Hub on
first run. The XLSR backbone for GenAID also downloads on first run.

### Wrapper scripts (place in `recipes/CommonAccent/`)

Copy these from version control / this dissertation's records into the clone:

- `predict_GenAID.py` — GenAID accent label + posteriors + embedding per wav.
- `predict_commonaccent.py` — CommonAccent ECAPA secondary classifier.
- `predict_speaker_embeddings.py` — ECAPA-TDNN speaker embeddings (#8).

### Required patches to the SpeechBrain fork

The fork targets a 2023-era SpeechBrain; modern `huggingface_hub`/`torchaudio` need
three edits:

1. `speechbrain/pretrained/fetching.py` — in the `hf_hub_download(...)` call, rename
   `use_auth_token=use_auth_token` → `token=use_auth_token` (arg was renamed).
2. `speechbrain/pretrained/interfaces.py` — in `from_hparams`, broaden the optional
   pymodule fetch `except ValueError:` → `except Exception:` (modern hub raises
   `EntryNotFoundError`, not `ValueError`, when the optional `custom.py` is absent).
3. In `predict_commonaccent.py` / `predict_speaker_embeddings.py`, load audio with
   `librosa` and call `classify_batch` / `encode_batch` instead of `classify_file`
   (avoids torchaudio's `torchcodec` backend dependency).

Also note `predict_GenAID.py` passes `device` to `pretrainer.load_collected(device=...)`
(checkpoint was saved on CUDA) and includes the unused speaker-adversarial head so the
checkpoint's `ModuleList` loads positionally.

## Notes / caveats for the writeup

- **Accent taxonomy**: GenAID's 13 classes don't fully cover VCTK — `Welsh` and
  `NorthernIrish` have no GenAID class (see `GENAID_TO_VCTK`). Exclude or handle them
  explicitly. CommonAccent has 16 classes (incl. `wales`).
- **Model bias**: WER, PPG-KL, accent-ID and speaker-sim all inherit their backbone
  models' biases — declare the models used.
- **CPU**: all envs verified on macOS CPU (2026-06-15); GPU is faster but optional.
