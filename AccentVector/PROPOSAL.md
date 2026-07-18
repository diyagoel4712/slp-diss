# Dissertation Proposal

**Examining the generalisability of the task vector approach for accented zero-shot TTS**

An MSc dissertation building on the [Accent Vector on F5-TTS](README.md)
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

**Goal**: replicate the **Accent Vector** method (Lertpetchpun et al., 2026 — task-vector accent
control via LoRA) by adapting **Expressive-Vectors** (github.com/the-bird-F/Expressive-Vectors,
Apache-2.0), but on **F5-TTS** as the backbone instead of the paper's XTTS-v2.

**Why F5-TTS**: it was the best model in our own benchmark (MCD 7.2, accent-sim 0.73, spk-sim
0.62), we already run it (`Preliminary_test_results/f5-tts`, weights cached), and Expressive-
Vectors already implements the LoRA-fine-tune → extract-vector → scale/interpolate → infer
pipeline on F5-TTS. The task-vector method is backbone-agnostic: `θ = θ_pre + Σ αᵢ·τᵢ`, where
`τᵢ = θ_LoRA(i)` (the i-th accent's LoRA delta).

The paper's L2 recipe on XTTS = "fine-tune on the target language's native speech while
pinning the **language-ID token to English**", so the LoRA delta is an *accent* shift, not a
*language* switch. **F5-TTS has no language-ID token**; it conditions only on (reference audio, 
reference text, generation text), with a `char`/`pinyin` tokenizer. Consequences:

1. **No lang-ID knob to pin** → the anchor that keeps content "English" on XTTS doesn't exist
   on F5. Instead, content language is set purely by the **generation text** you feed at
   inference. So the F5 mapping is: LoRA-fine-tune on target-language audio+transcript (delta
   captures that language's acoustics/prosody); at inference feed **English** text so content
   stays English while the merged delta pushes acoustics toward the accent. Conceptually this
   is *cleaner* on F5 (no competing language token), but it is NOT the paper's exact procedure —
   document the deviation.

2. **Script / vocab coverage** F5's base vocab covers Latin (incl. accented) + pinyin. For L2 
   accents with non-Latin native scripts, native transcripts won't tokenise (unknown chars → id 
   0), so they need romanized/transliterated transcripts, or a vocab extension, before fine-tuning. 
   **Verify the base vocab before assuming.**

3. **Reference audio carries accent — and we keep it.** F5 clones the reference clip, so its 
   accent feeds the output. Following the paper's cloning setup, we provide a **native-language 
   (L1) reference of the target accent** at inference (e.g. Hindi
   speech for the Indian accent), held **fixed per speaker across the α-sweep** so the vector
   is the only thing varying within a sweep. The sweep runs between two exact anchors:
   **α=0 = θ_pre** (the pretrained model, no fine-tuning, cloning the accent from the reference
   alone) and **α=1 = θ_pre + τ = θ_ft** (the fully fine-tuned model, full accent-vector impact).
   So it measures the fine-tuning's contribution as accent strength climbs from the base
   cloning level to the full fine-tune, with speaker identity expected to hold across α.


## Research questions & hypotheses

- **RQ1 — Cross-backbone generalisation.** Does task-vector accent control
  transfer from XTTS (autoregressive codec) to **F5-TTS (flow-matching)**, which
  has **no language-ID token**? *H1:* mechanism transfers (α-monotonic accent,
  speaker retained). Tested by the accent-strength-vs-α monotonicity + speaker
  retention — the shape of the paper's Fig. 3 reproduced on a new backbone.
- **RQ1b — Language leakage and the language-ID anchor.** Because F5 has no
  language-ID token to hold content in English, does content drift toward the
  target *language* (not just accent) sooner than on XTTS? *H1b:* leakage sets in
  at lower α on F5. Measured by (i) WER vs α, (ii) P(English) from a spoken-LID
  model vs α — the direct drift signal, distinct from accent — and (iii) a single
  **leakage-onset α** (where WER crosses / P(English) drops below a threshold),
  compared to the paper's XTTS numbers. WER alone conflates drift with the ASR's
  accent penalty, so **relative WER** (RQ5) and the LID signal disambiguate.
  *Confound (stated as a limitation):* F5-vs-XTTS varies backbone **and** token
  together; the clean isolation — ablating the token *within* XTTS — needs XTTS
  re-stood-up and is out of scope, so the onset gap is evidence, not proof.
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
- **RQ6 — Fine-tuning trajectory (optional, Tier 1).** How does the accent vector
  form over training? Track `‖τ_t‖` and `cos(τ_t, τ_final)` across checkpoints.
  *H6:* the accent **direction** stabilises well before magnitude — so the
  direction is learnable from little optimisation and α supplies the remaining
  intensity. This is the *optimisation* trajectory (near-free: CPU vector math
  over checkpoints already saved); the *data-efficiency* question (separate LoRAs
  on data fractions, with error bounds) is Tier 2/3 and **out of scope** — step ≠
  data amount, since F5 fine-tunes many epochs over one corpus.

## Method

F5-TTS with **LoRA** (rank 16, all linear layers — matches the paper, ~30 MB
vectors, cleaner geometry). Accents: British (VCTK-England control) + Spanish +
Vietnamese (Latin-script) + one prosodically-distant (Hindi romanised, or
Mandarin) as the H3 stress test. Measurement reuses
`Evaluation/evaluation_functions.py`:

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
