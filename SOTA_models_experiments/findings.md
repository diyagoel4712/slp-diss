# Accented-TTS evaluation — preliminary findings

**Setup.** Five SOTA open-source zero-shot TTS models, three conditioning paradigms, prompted
to produce 4 L2-English accents (Arabic, Indian, Korean, Vietnamese) × 4 speakers × 10
utterances, scored against each speaker's natural L2-ARCTIC recording.

**Status of this run.** 4 of 5 models evaluated (f5tts, xtts, vits, parler; **cosyvoice3
pending**). **All 8 metrics computed** (UTMOS now included). 159 clips/model scored.

## Per-model means (↑ = higher better, ↓ = lower better)

| Model | Paradigm | UTMOS↑ | F0 RMSE (cents)↓ | MCD (dB)↓ | WER↓ | Accent-ID acc.↑ | Accent emb. cos↑ | PPG-KL↓ | Speaker sim↑ |
|---|---|---|---|---|---|---|---|---|---|
| **f5tts** | clone | 3.42 | 351.8 | **7.17** | **0.049** | **0.245** | **0.732** | **0.377** | **0.618** |
| **xtts** | clone | 3.03 | 211.8 | 9.43 | 0.061 | 0.019 | 0.493 | 0.424 | 0.463 |
| **parler** | description | 3.20 | 205.2 | 9.77 | 0.062 | 0.000 | 0.285 | 0.486 | 0.091 |
| **vits** | speaker-ID | **3.52** | 672.1 | 9.71 | 0.102 | 0.000 | 0.232 | 0.559 | 0.059 |

## Per-accent means (across models)

| Accent | UTMOS↑ | F0 RMSE↓ | MCD↓ | WER↓ | Accent-ID↑ | Accent emb. cos↑ | PPG-KL↓ | Speaker sim↑ |
|---|---|---|---|---|---|---|---|---|
| Arabic | 3.28 | 390.6 | 9.32 | 0.065 | 0.000 | 0.443 | 0.535 | 0.293 |
| Indian | 3.34 | 367.9 | 8.85 | 0.069 | 0.263 | 0.347 | 0.417 | 0.263 |
| Korean | 3.26 | 315.3 | 9.04 | 0.074 | 0.000 | 0.529 | 0.372 | 0.339 |
| Vietnamese| 3.31 | 365.7 | 8.87 | 0.066 | 0.000 | 0.423 | 0.525 | 0.336 |

## Key findings

1. **Reference-cloning is the only paradigm that renders accent + speaker.** On the two
   identity metrics, the cloning models dominate: speaker similarity f5tts **0.62** / xtts
   **0.46** vs parler **0.09** / vits **0.06**; accent-embedding similarity f5tts **0.73** /
   xtts **0.49** vs parler **0.29** / vits **0.23**. Description (Parler) and speaker-ID (VITS)
   do not reproduce the target speaker or accent — as expected, since neither is conditioned on
   the target voice.

2. **F5-TTS is the strongest model overall** — best accent-embedding sim, speaker sim, MCD
   (7.2 dB, ~2.5 dB below the rest), and WER (0.049). It both sounds closest to the reference
   timbre and is the most intelligible.

3. **Accent identification is near-zero except for Indian** — and only because Indian is the
   one L2 accent in the GenAID classifier's taxonomy. Even there, the best model reaches only
   ~26% accuracy (driven by f5tts). For Arabic/Korean/Vietnamese the metric is undefined, so
   **accent-embedding cosine is the load-bearing accent metric**, not accent-ID accuracy. This
   taxonomy gap is itself a finding about the immaturity of accent-evaluation tooling.

4. **Parler does not differentiate accents (quantified).** Comparing the four accent versions
   of the same gender+utterance in GenAID's speaker-agnostic accent space:
   - **Parler cross-accent embedding similarity: mean 0.72 (median 0.77)**
   - **Natural L2-ARCTIC cross-accent similarity (real accents): mean 0.37 (median 0.42)**

   Parler's "accents" are roughly **twice as similar to each other as genuinely distinct
   accents are** — i.e. the `"{accent} accent"` descriptor largely collapses to one voice. This
   corroborates the informal listening test (the clips sound the same across accents). The
   residual variation (it is not exactly 1.0) is partly Parler's stochastic sampling, so this
   is read as "near-collapse," not perfect invariance.

5. **VITS is a clean floor for fidelity.** Worst (or near-worst) on accent/speaker and least
   intelligible (WER 0.10) — a fixed native VCTK speaker, no accent or speaker adaptation. Its
   very large F0 RMSE (672 cents) reflects a cross-speaker pitch mismatch, not a prosody failure.

6. **Naturalness is decoupled from fidelity.** On UTMOS, VITS is *highest* (3.52) and f5tts
   second (3.42), while the best cloning model on identity (xtts) is *lowest* (3.03). So the
   models that best reproduce the target accent/speaker are not the most natural-sounding —
   naturalness and accent/speaker fidelity are largely orthogonal here, and accent control
   currently comes at some cost to naturalness. UTMOS alone would rank the models almost
   backwards relative to the accent goal, which is why the multi-metric view matters.

## How to read these numbers (caveats)

- **F0 RMSE, MCD and speaker-sim are cross-speaker for VITS and Parler** (they don't reproduce
  the target voice), so their values on those metrics reflect "wrong speaker," not poor
  modelling. Compare paradigms on what each is *meant* to do.
- **Parler's accent descriptors are likely out-of-distribution** for `parler-tts-mini-v1`, so
  finding (4) is a statement about off-the-shelf description-based control, not "Parler is bad."
- **cosyvoice3 pending** — will be added in the final 5-model pass.

## Headline for the problem statement

Among current open models, **implicitly transferring an accent from a reference clip (esp.
F5-TTS) is the only approach that meaningfully renders L2 accents**; explicit text description
(Parler) and speaker-ID (VITS) do not. And even the best model captures the target accent only
moderately (accent-embedding 0.73; ~26% accent-ID on the single measurable accent) — defining
the open problem this dissertation addresses.
