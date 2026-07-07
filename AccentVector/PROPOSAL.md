# Dissertation Proposal — What Do Accent Vectors Encode?

**Probing segmental vs. suprasegmental accent control in flow-matching TTS.**

A 2-month MSc dissertation building on the [Accent Vector on F5-TTS](README.md)
port. Experiment code lives in [`accent_vector/experiments/`](accent_vector/experiments);
the runnable mapping is in [EXPERIMENTS.md](EXPERIMENTS.md).

## Motivation

Lertpetchpun et al. (2026) show that fine-tuning a multilingual TTS model on
native L1 speech and taking the parameter shift `τ = θ_ft − θ_pre` as an "accent
vector" gives controllable, composable accent manipulation without accented
English data. But their own numbers expose a crack: **Mandarin gains least**
(+23% vs +140% for British), attributed to prosodic distance. Central question:
*do accent vectors capture accent, or only its segmental (phoneme-level) shadow,
leaving suprasegmental structure (F0, duration, rhythm, tone) largely
untouched?* This matters because the method claims to shift "duration, rhythm
and prosody" but never measures whether it does — and its evaluation instruments
(VoxProfile, Whisper, UTMOS) are themselves phoneme- and English-biased.

## Research questions & hypotheses

- **RQ1 — Cross-backbone generalisation.** Does task-vector accent control
  transfer from XTTS (autoregressive codec) to **F5-TTS (flow-matching)**, which
  has **no language-ID token**? *H1:* mechanism transfers (α-monotonic accent,
  speaker retained), but content/language leakage at high α is worse than XTTS
  without the language anchor.
- **RQ2 — Geometry.** Do vectors cluster by linguistic family, and how much is
  accent vs training-corpus confound? *H2:* weight-space similarity partially
  recovers accent relatedness; RSA vs output-space (GenAID) is positive but
  imperfect.
- **RQ3 — Segmental vs suprasegmental (core).** As α increases, do segmental
  (phone) and suprasegmental (F0/rhythm/tempo) features both move toward the
  natural target? *H3:* the vector is **segmental-dominated**, and the gap is
  widest for a prosodically-distant accent — explaining the Mandarin result.
- **RQ4 — Intervention (stretch).** Can layer-targeted scaling or
  prosody-matched reference retrieval improve suprasegmental transfer? *H4:*
  yes, without collapsing speaker similarity.
- **RQ5 — Evaluation bias.** Where does bias enter, and does a fairer protocol
  (relative WER, gender-disaggregated, familiarity-baselined) change conclusions?

## Method

F5-TTS with **LoRA** (rank 16, all linear layers — matches the paper, ~30 MB
vectors, cleaner geometry). Accents: British (VCTK-England control) + Spanish +
Vietnamese (Latin-script) + one prosodically-distant (Hindi romanised, or
Mandarin) as the H3 stress test. Measurement reuses
`SOTA_models_experiments/evaluation_functions.py`:

- *Segmental:* `ppg_kl` — KL between synth and natural-accent phone posteriorgrams across α.
- *Suprasegmental:* `extract_f0` → pitch + voicing-based rhythm proxy (%V, nPVI, articulation rate); MFA alignment as the rigorous upgrade.
- *Identity/utility:* `speaker_similarity`, `wer`, `utmos`; *accent:* `cs_accent`, `aid_acc`.
- *Geometry:* per-layer RMS norms → weight-space cosine → MDS; RSA (Mantel) vs GenAID output-space matrix.
- *Intervention:* masked composition (`extract_vector compose --include`) to scale layer subsets and localise prosody.

## Timeline (8 weeks; writing runs continuously from week 3)

| Wk | Focus | Milestone |
|----|-------|-----------|
| 1 | GPU setup, reproduce Phase A (British) end-to-end | working port; α-sweep |
| 2 | Validate RQ1 (α-monotonicity, speaker retention); eval harness | **go/no-go gate** |
| 3 | Train Spanish + Vietnamese vectors; start geometry (RQ2) | vector library; first accent map |
| 4 | Add distant accent; finalise RQ1/RQ2 | RQ1+RQ2 results; methods draft |
| 5 | **RQ3 decomposition** (segmental vs suprasegmental across α) | core figures |
| 6 | Layer localisation; interpret failure mode | RQ3 conclusion; analysis draft |
| 7 | Stretch RQ4 intervention + RQ5 bias audit/flowchart | intervention or negative result; bias chapter |
| 8 | Consolidate, buffer, polish | submitted dissertation |

## Scope tiers

- **Minimum viable:** RQ1 reproduction + RQ3 decomposition (≥3 accents) + RQ5 audit.
- **Target:** + RQ2 geometry/RSA + the distant accent.
- **Stretch:** RQ4 intervention (a negative result is a valid finding).

## Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| GPU-hours limited | LoRA (paper's own choice; ~8M params) |
| Non-Latin script won't tokenise (Hindi) | romanise transcripts, or use Mandarin/Latin-only |
| English-biased metrics | RQ5 makes this a *finding*; report relative/margin metrics |
| Forced alignment fails on accented synthesis | fall back to voicing-based F0 rhythm proxy (already implemented) |
| No native listeners for mixed-accent subjective eval | keep objective; frame subjective eval as future work |

## Contributions

1. First port of Accent Vector to a **flow-matching** backbone, isolating the
   role of the language-ID token the original relies on.
2. First **quantitative decomposition** of what accent task vectors encode
   (segmental vs suprasegmental), explaining the paper's Mandarin weakness.
3. A **weight-space accent geometry** validated against perceptual embeddings (RSA).
4. A **bias audit and fairer evaluation protocol** for accent-TTS.
