# PACEM
## Phone-conditioned Accent Contrastive Embedding Model

A revised methodology for phonologically-grounded accent embeddings, addressing
the core failures of utterance-level regression + GRL approaches.

---

## Why This Architecture Instead

The previous approach had three fundamental problems:

1. **Utterance-level averaging destroys the signal.** The TRAP-BATH split is
   about where /æ/ sits relative to /ɑː/ — not the mean formant of all vowels.
   The accent-informative structure is the *conditional distribution* F1|phone_category,
   not the marginal F1 across all phones.

2. **GRL disentanglement is theoretically insufficient.** It removes linear
   speaker signal from the marginal distribution but cannot remove nonlinear
   speaker information. More critically, accent and speaker are causally
   entangled: a speaker's rhoticity *is* a property of that speaker, not
   separable from them via gradient tricks.

3. **Phonetic surface measurements ≠ phonological system.** VOT varies with
   speaking rate; F0 mean is almost entirely a speaker property. Regressing
   on these conflates phonological accent features with prosodic and physiological
   variation.

---

## The New Architecture

```
Audio
  ↓
WavLM-base (frozen CNN, fine-tuned transformer)
  ↓  [frame-level representations, ~50fps]
MFA forced alignment → phone token boundaries
  ↓
Phone-token pooling: mean-pool WavLM frames within each phone segment
  → (N_phones, H) phone token embeddings per utterance
  ↓
Phone-Conditioned Contrastive Learning (training objective)
  ┌─────────────────────────────────────────────────────┐
  │  For each anchor phone token (e.g. /æ/ from speaker A): │
  │    ATTRACT: /æ/ from same speaker, same variety      │
  │    ATTRACT: /æ/ from different speaker, same variety  │  ← key signal
  │    REPEL:   /æ/ from different variety               │
  │    REPEL:   /ɑː/, /ɛ/, other phones (cross-phone)   │
  └─────────────────────────────────────────────────────┘
  ↓
Phone-Category Pooling (at inference)
  Group all phone tokens by ARPAbet category
  → per-category mean embedding
  → concatenate diagnostically-selected categories
  → linear projection → L2-normalised accent embedding
  ↓
Evaluation: Spearman ρ with Speech Accent Archive perceptual distances
            Minimal-pair ABX discrimination (Bartelds et al. 2022 metric)
```

### Key differences from the previous approach

| Aspect | Previous (broken) | PACEM (this) |
|--------|-------------------|--------------|
| Granularity | Utterance-level | Phone-token-level |
| Target | Acoustic measurements (F1, VOT) | Contrastive phonological similarity |
| Speaker disentanglement | GRL (marginal, insufficient) | Implicit via same-phone cross-speaker positives |
| Phonological validity | Phonetic surface measures | Conditional distributions per phone category |
| Evaluation | Speaker classification accuracy | Perceptual distance correlation |
| Formant normalisation | Lobanov (applied inconsistently) | Not needed — contrastive objective normalises implicitly |

---

## Project Structure

```
pacem/
  configs/
    base.yaml          — default hyperparameters
  data/
    corpus.py          — corpus loading (VCTK, L2-ARCTIC, Speech Accent Archive)
    alignment.py       — MFA wrapper + phone token extraction
    phone_tokens.py    — phone-token dataset for contrastive learning
    sampler.py         — contrastive batch sampler (controls positives/negatives)
  models/
    backbone.py        — WavLM with weighted layer aggregation
    projector.py       — phone-token projector head
    pooler.py          — phone-category pooling → accent embedding
    pacem.py           — full model (backbone + projector + pooler)
  training/
    losses.py          — phone-conditioned NT-Xent + cross-phone repulsion
    trainer.py         — training loop
  evaluation/
    abx.py             — ABX minimal-pair discrimination score
    perceptual.py      — correlation with Speech Accent Archive distances
    probing.py         — linear probe for accent variety classification
  utils/
    phones.py          — ARPAbet definitions, diagnostic phone sets
    audio.py           — audio loading, resampling utilities
  train.py             — entry point
  embed.py             — inference: audio → accent embedding
  evaluate.py          — run evaluation suite
  requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# You need MFA installed: conda install -c conda-forge montreal-forced-aligner
# And an MFA English acoustic model:
#   mfa models download acoustic english_us_arpa
#   mfa models download dictionary english_us_arpa

# 1. Align your corpus
python -m pacem.data.alignment \
    --audio_dir data/audio \
    --transcript_dir data/transcripts \
    --out_dir data/aligned \
    --variety_map data/variety_map.json   # {"speaker_id": "variety_label"}

# 2. Extract phone tokens
python -m pacem.data.phone_tokens \
    --audio_dir data/audio \
    --aligned_dir data/aligned \
    --variety_map data/variety_map.json \
    --out data/phone_tokens.jsonl

# 3. Train
python train.py --config configs/base.yaml

# 4. Embed
python embed.py --checkpoint checkpoints/best.pt --audio clip.wav

# 5. Evaluate
python evaluate.py --checkpoint checkpoints/best.pt \
    --accent_archive data/accent_archive_distances.csv
```

---

## Recommended Corpora

| Corpus | Speakers | Varieties | Notes |
|--------|----------|-----------|-------|
| VCTK | 108 | ~20 UK/US/Commonwealth | Free, already has transcripts |
| L2-ARCTIC | 24 | 6 L1 backgrounds | Small but clean |
| Speech Accent Archive | 2600+ | 177 countries | Has perceptual ratings — use for eval only |
| CORAAL | ~150 | AAL longitudinal | Rich sociolinguistic metadata |
| IViE | ~100 | 9 British varieties | Intonation-focused but good segmental data |

Minimum recommended: VCTK + L2-ARCTIC for training (~130 speakers).
The Speech Accent Archive should be held out entirely for evaluation.

---

## References

- Bartelds et al. (2022). A new acoustic-based pronunciation distance measure. *Frontiers in AI*.
- Shon et al. (2018). Emotionless: domain-invariant representations for transfer learning. *Interspeech*.
- Chen et al. (2022). WavLM: Large-scale self-supervised pre-training. *IEEE JSTSP*.
- Pasad et al. (2021). Layer-wise analysis of a self-supervised speech representation model. *ASRU*.
- Yang et al. (2021). SUPERB: Speech processing universal performance benchmark. *Interspeech*.
- Kharitonov et al. (2021). Data augmenting contrastive learning of speech representations. *SLT*.
- Locatello et al. (2019). Challenging common assumptions in disentangled representations. *ICML*.
- Labov (1994). Principles of linguistic change, Vol. 1. Blackwell.
- McAuliffe et al. (2017). Montreal forced aligner. *Interspeech*.
