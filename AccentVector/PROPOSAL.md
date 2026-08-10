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

3. **Reference audio carries accent — so the reference is an experimental variable, not a
   constant.** F5 clones the reference clip, and — crucially — because it has no language-ID
   token or speaker/perceiver factorisation (unlike XTTS), it clones *whatever accent is in
   that clip* directly into the output. A direct consequence, confirmed in our own listening:
   **pretrained F5 (α=0) already produces convincingly accented English when cloning an L1
   reference.** This is not the method working — it is the reference supplying the accent. It
   makes the reference kind the load-bearing variable, so we sweep α under **two reference
   conditions** (`scripts/eddie_infer_sweep.sh REF_KIND`, sibling output trees
   `results/<accent>/{l1,native}/`):

   - **L1 reference** (paper-faithful): a native-language clip of the target accent (e.g.
     Hindi speech for the Indian accent), fixed within the sweep. **α=0 = θ_pre** (pretrained,
     cloning accent from the reference alone) → **α=1 = θ_pre + τ = θ_ft** (fully fine-tuned).
     Because the reference already carries accent, this condition measures the fine-tuning's
     *marginal* contribution on top of an already-accented baseline — and the honest empirical
     question is the **shape of the α-curve** (monotone accent gain vs leakage/degradation),
     not whether α=0 sounds accented (it does, by construction).
   - **Neutral native-English reference** (decoupling control): a clip whose accent is *not*
     the target, so α=0 is neutral English and any target-accent signal emerging as α climbs
     is attributable to **the vector alone**. This isolates the fine-tuning's contribution from
     reference cloning — the discriminating test of whether the task vector does anything the
     base model cannot, and the only condition under which the composition/mixing claims (which
     a fixed accented reference can never demonstrate) are testable. Speaker identity is
     expected to hold against *this* reference across α.

4. **F5 is pretrained on two languages only (Emilia ZH+EN); XTTS-v2 covers 17.** Every accent
   language in the paper (Spanish, German, French, Hindi, Mandarin) is inside XTTS's pretraining
   distribution. On F5 our accents **split**: British and Mandarin are in-distribution, while
   Dutch, Hindi, Bengali and Arabic are not. Fine-tuning on an unseen language must do generic
   work — phonotactics, and the pressure of a tokenizer built for EN+ZH — that fine-tuning on a
   seen one does not, so we expect systematically larger `‖τ‖` for the out-of-distribution four.
   Three consequences, all of which are *measured* rather than assumed:

   - This is a strong candidate for the shared `μ` component in RQ5-H5a: not accent, but "move
     off the EN/ZH manifold". So RSA is run as a **partial** Mantel controlling for a binary
     in-distribution indicator and for corpus hours, and the coefficient kernels are compared
     three ways — typology-only, distribution-only, and both. If distribution beats typology,
     H5b fails for an interesting reason rather than a boring one.
   - `τ_mandarin` may be **near-degenerate**: if the base already speaks Mandarin, LoRA on
     AISHELL captures studio read-speech style more than an accent shift. Cheaply falsifiable —
     compare `‖τ_mandarin‖` against the others, and check whether α-sweeping it on English text
     raises Mandarin-accentedness at all in the existing RQ1 sweep.
   - If it *is* degenerate, that is a finding, not a failure: it offers an **alternative account
     of the paper's own Mandarin weakness** (their +23% vs +140% for British). Their explanation
     is prosodic distance; the competing explanation is that Mandarin sits inside the backbone's
     pretraining distribution, leaving less accent signal to extract. The two are separable here
     because F5's coverage differs from XTTS's — the same asymmetry that makes this port a
     deviation also makes it an instrument.


## Research questions & hypotheses

- **RQ1 — Cross-backbone generalisation (incl. language leakage).** Does
  task-vector accent control transfer from XTTS (autoregressive codec) to
  **F5-TTS (flow-matching)**, which has **no language-ID token**? *H1:* the
  mechanism transfers (α-monotonic accent, speaker retained) — **but**, lacking the
  language-ID anchor, content leakage toward the target *language* sets in at lower α
  than on XTTS. Tested by the accent-strength-vs-α monotonicity + speaker retention —
  the shape of the paper's Fig. 3 reproduced on a new backbone. Because a non-factorised
  backbone clones accent straight from the reference, "does α=0 sound accented" is *not*
  the question (it does, trivially, under an L1 reference); the test is the **α-curve
  shape**, read jointly across the two reference conditions (Method §3): does accent rise
  with α under the **neutral-reference** control (vector doing real, reference-independent
  work) or only under the L1 reference (accent was cloning)? A flat/leaky neutral-reference
  curve is a valid *negative* transfer result — the sharpest statement that on flow-matching
  F5 the vector is largely redundant to cloning. The seed signal is already visible in a
  British smoke test (`results/british/`, n=4): accent-ID, WER and UTMOS all *degrade* as α→1.

  *Language leakage (the language-ID anchor).* Because F5 has no language-ID token to hold
  content in English, does content drift toward the target *language* (not just accent)
  sooner than on XTTS? Measured by (i) WER vs α, (ii) P(English) from a spoken-LID model vs
  α — the direct drift signal, distinct from accent — and (iii) a single **leakage-onset α**
  (where WER crosses / P(English) drops below a threshold), compared to the paper's XTTS
  numbers. WER alone conflates drift with the ASR's accent penalty, so **relative WER** (see
  Limitations) and the LID signal disambiguate. *Confound (stated as a limitation):* F5-vs-XTTS
  varies backbone **and** token together; the clean isolation — ablating the token *within*
  XTTS — needs XTTS re-stood-up and is out of scope, so the onset gap is evidence, not proof.
- **RQ2 — Fine-tuning trajectory.** How does the accent vector form over training?
  Track `‖τ_t‖` and `cos(τ_t, τ_final)` across checkpoints (weight space), and
  compare the α-sweep at matched α across checkpoints (output space). *H2:* the accent
  **direction** stabilises well before magnitude — so the direction is learnable from
  little optimisation and α supplies the remaining intensity — and in output space
  accent saturates with training earlier than language leakage worsens (an earlier
  checkpoint at moderate α is the fluent-accented sweet spot). This is the
  *optimisation* trajectory (near-free: CPU vector math over checkpoints already saved,
  plus one synthesis grid); the *data-efficiency* question (separate LoRAs on data
  fractions, with error bounds) is Tier 2/3 and **out of scope** — step ≠ data amount,
  since F5 fine-tunes many epochs over one corpus.
- **RQ3 — Segmental vs suprasegmental (core).** As α increases, do segmental
  (phone) and suprasegmental (F0/rhythm/tempo) features both move toward the
  natural target? *H3:* the vector is **segmental-dominated**, and the gap is
  widest for a prosodically-distant accent — explaining the Mandarin result.
  This is untouched by the "F5 already sounds accented" observation: a clip *sounding*
  accented says nothing about *which* structure carries it. The decomposition (segmental
  `ppg_kl` vs suprasegmental F0/nPVI/articulation-rate gap-closure) is precisely what an
  ear cannot separate, and remains the core contribution regardless of the RQ1 outcome.
- **RQ4 — Intervention (stretch).** Can layer-targeted scaling of the vector
  (up-weighting prosody-carrying layers / excluding the content-language path)
  improve suprasegmental transfer? *H4:* yes, without collapsing speaker similarity.
- **RQ5 — Compositional generalisation (can geometry *predict* a new accent?).**
  Can a usable accent vector for a target accent with **no or very little** training
  data be constructed as a weighted combination `τ̂_target = Σᵢ wᵢ·τᵢ` of an existing
  vector library — and does linguistic structure predict the coefficients? This turns
  the weight-space geometry from a *descriptive* claim (do vectors cluster by family?)
  into a *predictive* one, and it is the only RQ whose scope grows without more
  fine-tuning: training an accent is needed only to obtain its **oracle**, whereas
  *scoring* a composed vector needs nothing but a handful of natural clips of the
  target, since `cs_accent` / GenAID centroid cosine are reference-based.

  *H5a (geometry).* Accent vectors decompose as `τᵢ = μ + rᵢ` — a large shared
  fine-tuning/corpus-domain component plus an accent-discriminative residual — and a
  held-out vector is partially spanned by the rest (least-squares reconstruction R²
  well above a matched-norm random-direction null). The *residuals*, not the raw
  vectors, carry accent identity; this is where the training-corpus confound is
  measured rather than merely acknowledged. `μ` has a named rival interpretation —
  "move off F5's EN/ZH pretraining manifold" (Motivation §4) — so it is tested against a
  binary in-distribution indicator and against corpus hours, not just asserted to be
  domain noise.
  *H5b (zero-data).* Typology-weighted coefficients (lang2vec/URIEL phonology and
  inventory distance, softmax kernel) beat the uniform mean and match or beat
  nearest-neighbour — i.e. the weight-space geometry is predictively, not just
  correlationally, aligned with linguistic relatedness. The competing hypothesis is
  explicit and pre-registered: if what dominates τ is *in- vs out-of-pretraining
  distribution* rather than family, typology will lose to a distribution-only kernel and
  the uniform mean will do suspiciously well. Kernels are therefore compared three ways
  (typology-only / distribution-only / both) and RSA is run as a **partial** Mantel.
  *H5c (few-shot).* ~30 s of target-accent audio, via gradient-free coefficient search
  over the simplex (no gradients through F5), closes a substantial fraction of the gap
  between nearest-neighbour and the true fine-tuned vector.
  *H5d (limits — ties to RQ3).* Composition recovers the **segmental** component better
  than the suprasegmental one, inheriting and amplifying the bias RQ3 identifies: a
  composed vector sounds *foreign-accented* more than it sounds like *that* accent.

  Two evaluation tiers. **Tier A — leave-one-accent-out with oracle** over the six
  trained accents (Method §Vector library): the held-out accent's true `τ` gives exact
  weight-space error and an end-to-end ceiling. The 2+2+1+1 family structure makes the
  prediction crisp — Germanic and Indic are the positive-control folds, where a correct
  kernel must load the in-family neighbour, while Arabic and Mandarin are family
  singletons and so the stress folds. **Tier B — genuinely unseen, no oracle**: accents
  never fine-tuned (audio only, from FLEURS), scored against natural speech. Tier B is
  what demonstrates the headline claim at scale for near-zero training cost. Every method
  is bracketed by a floor (`θ_pre`) and a ceiling (true `θ_ft`), and compared against
  uniform-mean and nearest-neighbour baselines so a positive result cannot reduce to
  "any accent vector helps".

  A negative result is a finding, and the framing carries it: "the accent subspace is
  not linearly spanned by six vectors" is a sharp statement about the method's
  compositionality, and the weight-space arm (H5a) needs no synthesis at all, so it
  stands even if synthesis quality disappoints.

  *Positioning.* Lertpetchpun et al. mix exactly two **trained** accents at equal
  weights and report no geometry analysis and no unseen-accent experiment. Constructing
  a vector for a task with no data for it is established elsewhere — task analogies
  (Ilharco et al., 2023), coefficient search over a LoRA library (LoraHub, Huang et al.,
  2024), typology-generated language adapters (MAD-G, Ansell et al., 2021), language
  arithmetic (Klimaszewski et al., 2024) and adapter ensembling for unseen languages
  (ZGUL) — and task-vector merging is known to transfer to low-resource speech
  (LoRS-Merging, 2025). None of it has been done for accent in TTS, or on a
  flow-matching backbone.

## Method

F5-TTS with **LoRA** (rank 16, all linear layers — matches the paper, ~30 MB
vectors, cleaner geometry). Vector library — six accents over four families, all from
corpora whose prep is already written, so RQ5 costs no new data work:

| Accent | Family | Corpus | Prep | Status |
|---|---|---|---|---|
| British | Germanic | VCTK-England (control) | done | trained |
| Dutch | Germanic | CGN | `scripts/cgn/` | trained |
| Hindi | Indic | IndicVoices-R (romanised) | `scripts/indicvoices/` | trained |
| Bengali | Indic | IndicVoices-R (romanised) | `scripts/indicvoices/` | trained |
| Arabic | Semitic | GlobalPhone MSA (~35 h, romanised) | `scripts/globalphone/` | to train |
| Mandarin | Sinitic | AISHELL-1 (Hanzi → pinyin) | `scripts/aishell/` | to train |

The two in-family pairs (Germanic, Indic) are the positive-control folds for RQ5 — a
correct typological kernel must put its mass on the in-family neighbour. Arabic and
Mandarin are family singletons and therefore the stress folds, and Mandarin is doubly
loaded: it is the prosodically-distant H3 test case *and* the accent the original paper
gains least on, so it is where H5d (composition recovers segmental structure but not
suprasegmental) should bite hardest. Hours and speaker counts are matched across accents
as far as the corpora allow and recorded in provenance — unmatched corpus size inflates
the shared `μ` component of RQ5-H5a and contaminates every coefficient.

*Geometry caveat from the tokenizer.* AISHELL keeps Hanzi and `prepare` converts it to
pinyin, so the Mandarin vector carries mass in text-embedding rows the Latin/romanised
accents never touch, deflating its raw cosine to them for a reason that is tokenisation,
not phonology. All RQ5 geometry is therefore reported twice: over all parameters, and
with `text_embed`/`input_embed` excluded — the mask RQ4 already uses, supported by the
existing `--include`/`--exclude` flags. Measurement reuses
`Evaluation/evaluation_functions.py`:

- *Segmental:* `ppg_kl` — KL between synth and natural-accent phone posteriorgrams across α.
- *Suprasegmental:* `extract_f0` → pitch + voicing-based rhythm proxy (%V, nPVI, articulation rate); MFA alignment as the rigorous upgrade.
- *Identity/utility:* `speaker_similarity`, `wer`, `utmos`; *accent:* `cs_accent`, `aid_acc`.
- *Geometry (RQ5-H5a):* per-layer RMS norms → weight-space cosine → MDS; RSA (Mantel) vs
  GenAID output-space matrix; the `τᵢ = μ + rᵢ` split with cosines on raw vectors *and*
  residuals; leave-one-out least-squares reconstruction R² against a matched-norm
  random-direction null. Cosine is computed on the **materialised** `ΔW = (α_lora/r)·B·A`
  per module, never on the LoRA `A`/`B` factors — that factorisation is gauge-invariant
  (`A→GA, B→BG⁻¹` leaves `ΔW` fixed), so a cosine over it is not a well-defined geometry.
- *Composition (RQ5-H5b–d):* `extract_vector compose` with per-vector coefficients from
  four sources — uniform mean, nearest-neighbour, a lang2vec/URIEL typology kernel
  (zero-data), and gradient-free simplex search on ~30 s of target audio (few-shot) —
  plus the oracle least-squares fit as the ceiling. Composed output is then run through
  the RQ3 decomposition to test H5d.
- *Intervention:* masked composition (`extract_vector compose --include`) to scale layer subsets and localise prosody.

## Timeline (8 weeks; writing runs continuously from week 3)

| Wk | Focus | Milestone |
|----|-------|-----------|
| 1 | GPU setup, reproduce Phase A (British) end-to-end | working port; α-sweep |
| 2 | Validate RQ1 (α-monotonicity, speaker retention); eval harness | **go/no-go gate** |
| 3 | Train Arabic + Mandarin vectors (prep already written); trajectory mapping (RQ2) | 6-accent vector library; trajectory curves |
| 4 | RQ5-H5a geometry: μ/residual split, LOO reconstruction, partial Mantel; finalise RQ1 | RQ1+RQ2+H5a results (CPU-only); methods draft |
| 5 | **RQ3 decomposition** (segmental vs suprasegmental across α) | core figures |
| 6 | RQ5 Tier A composition (H5b typology + H5c few-shot); layer localisation | LOO composition table vs floor/ceiling |
| 7 | Stretch RQ4 intervention + bias-audit limitation | intervention or negative result; limitations chapter |
| 8 | Consolidate, buffer, polish | submitted dissertation |

## Scope tiers

- **Minimum viable:** RQ1 reproduction + RQ2 trajectory + RQ3 decomposition (≥3 accents) +
  RQ5-H5a geometry + the bias-audit limitation. H5a is in the minimum tier because it is
  CPU-only vector maths over checkpoints that already exist — it cannot be blocked by GPU
  contention or by a disappointing α-sweep.
- **Target:** + RQ5 Tier A (leave-one-out composition with oracle: H5b typology kernel,
  H5c few-shot search) over all six accents.
- **Stretch:** RQ5 Tier B (genuinely unseen accents, no oracle) + H5d segmental/
  suprasegmental decomposition of composed output + RQ4 intervention. Negative results are
  valid findings throughout — for RQ5 especially, "the accent subspace is not linearly
  spanned" is a sharp claim about the method's compositionality.

## Evaluation bias (discussion / limitations)

The evaluation instruments are themselves phoneme- and English-biased (VoxProfile,
Whisper, UTMOS), so a recurring limitation — threaded through the results rather than
posed as a standalone RQ — is *where evaluation bias enters and whether a fairer
protocol changes the conclusions*: relative (not absolute) WER, gender-disaggregated
scores, and familiarity-baselined accent similarity. A concrete instance: the gap
between *ear-perceived* accent (subjectively "convincing") and *measured* accent
(best-case only ~0.73 accent-embedding / ~26% accent-ID in our benchmark) shows
perceptual conviction overstates measured fidelity — which is why the study uses a
metric-first protocol throughout and reports its instruments' bias as a limitation on
every claim.

## Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| GPU-hours limited | LoRA (paper's own choice; ~8M params) |
| Non-Latin script won't tokenise (Hindi) | romanise transcripts, or use Mandarin/Latin-only |
| English-biased metrics | treat as a limitation and *finding*; report relative/margin metrics |
| Forced alignment fails on accented synthesis | fall back to voicing-based F0 rhythm proxy (already implemented) |
| No native listeners for mixed-accent subjective eval | keep objective; frame subjective eval as future work |
| Single vectors already degrade as α→1 (British smoke test), so composed ones cannot do better | score every RQ5 method at its **best α along a scale sweep**, not at α=1, so composition isn't penalised for a scaling failure that afflicts the baseline equally; and keep H5a (no synthesis at all) in the minimum tier |
| Six accents is too few to *fit* a typology→coefficient regression | use a one-hyperparameter similarity kernel, not a learned regression; fit its temperature by **nested** LOO on the remaining accents; report as a case study and state the n explicitly |
| `τ_mandarin` degenerate because Mandarin is in F5's pretraining set | falsify cheaply via `‖τ_mandarin‖` and the existing RQ1 sweep; if degenerate, report as an alternative account of the paper's Mandarin weakness (Motivation §4) |

## Contributions

1. First port of Accent Vector to a **flow-matching** backbone, isolating the
   role of the language-ID token the original relies on.
2. First **quantitative decomposition** of what accent task vectors encode
   (segmental vs suprasegmental), explaining the paper's Mandarin weakness.
3. A **predictive** weight-space accent geometry: the first attempt to *construct* an
   accent vector for an accent with no or minimal training data by composing an existing
   vector library, with typology-derived (zero-data) and few-shot coefficients bracketed
   by a floor and an oracle ceiling. The original paper mixes only accents it trained, at
   equal weights, and reports no geometry at all.
4. A **bias audit and fairer evaluation protocol** for accent-TTS.
