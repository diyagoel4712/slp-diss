# Phonological Accent Embedding Encoder

Fine-tunes WavLM/wav2vec2 to produce speaker-invariant phonological accent embeddings.

## Architecture

```
Audio → WavLM Backbone → Mean Pool → Projection → Accent Embedding (256-d)
                                           ↓                    ↓
                              Phonological Heads        GRL → Speaker Classifier
                              (F1/F2, VOT, rhoticity…)        (adversarial)
```

## Files

- `model.py`         — AccentEncoder architecture (backbone + heads + GRL)
- `features.py`      — Phonological feature extraction from aligned audio
- `dataset.py`       — Dataset class expecting forced-aligned data
- `train.py`         — Training loop with combined loss
- `embed.py`         — Inference: audio → embedding
- `align.py`         — MFA wrapper for forced alignment
- `requirements.txt` — Dependencies

## Quick Start

```bash
pip install -r requirements.txt

# 1. Run forced alignment on your corpus
python align.py --audio_dir data/audio --transcript_dir data/transcripts --out_dir data/aligned

# 2. Extract phonological features
python features.py --aligned_dir data/aligned --out_dir data/features

# 3. Train
python train.py --features_dir data/features --audio_dir data/audio

# 4. Embed new audio
python embed.py --checkpoint checkpoints/best.pt --audio path/to/clip.wav
```
